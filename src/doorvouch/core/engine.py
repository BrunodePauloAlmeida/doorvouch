"""The V0 rule: an admin's confirmed thumbs-up pre-registers the unknown faces in view."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from doorvouch.core.association import find_owner
from doorvouch.core.embeddings import Embedding, cosine_similarity
from doorvouch.core.gallery import SFACE_COSINE_THRESHOLD, Gallery
from doorvouch.core.geometry import Box
from doorvouch.core.gestures import THUMB_UP, GestureConfirmer
from doorvouch.core.observations import FaceObservation, FrameObservation
from doorvouch.core.people import PendingVisitor, Person

FaceStatus = Literal["admin", "known", "pending", "uncertain", "unknown"]
SnapshotFn = Callable[[Box], bytes | None]


class Repository(Protocol):
    """What the engine needs from storage."""

    def load_gallery(self) -> list[tuple[Person, Embedding]]: ...

    def list_pending(self) -> list[PendingVisitor]: ...

    def create_pending(
        self,
        embedding: Embedding,
        *,
        vouched_by: int | None,
        snapshot: bytes | None = None,
        now: datetime | None = None,
    ) -> PendingVisitor: ...

    def touch_pending(self, pending_id: int, *, now: datetime | None = None) -> None: ...

    def log_event(
        self,
        kind: str,
        *,
        person_id: int | None = None,
        pending_id: int | None = None,
        detail: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> int: ...

    def data_version(self) -> int: ...


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Thresholds of the vouching rule. The defaults are starting points to calibrate."""

    # Faces at least this similar to an enrolled sample are that person.
    match_threshold: float = SFACE_COSINE_THRESHOLD
    # A face is pre-registered only if it is less similar than this to everyone
    # enrolled. The gap up to match_threshold is a "not sure" zone: a known
    # person caught at a bad angle must not become a new visitor.
    visitor_max_similarity: float = 0.30
    min_face_width: float = 64.0
    min_face_score: float = 0.8
    min_gesture_score: float = 0.6
    gesture_frames: int = 5
    gesture_gap_frames: int = 1
    gesture_cooldown_seconds: float = 3.0
    hand_max_reach: float = 4.0
    hand_ambiguity_ratio: float = 0.8
    presence_cooldown_seconds: float = 60.0
    ignored_gesture_cooldown_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be in (0, 1]")
        if not 0.0 < self.visitor_max_similarity <= self.match_threshold:
            raise ValueError("visitor_max_similarity must be in (0, match_threshold]")
        if not 0.0 <= self.min_gesture_score <= 1.0:
            raise ValueError("min_gesture_score must be in [0, 1]")
        if not 0.0 <= self.min_face_score <= 1.0:
            raise ValueError("min_face_score must be in [0, 1]")
        if self.min_face_width < 0:
            raise ValueError("min_face_width must not be negative")
        if self.presence_cooldown_seconds < 0 or self.ignored_gesture_cooldown_seconds < 0:
            raise ValueError("cooldowns must not be negative")


@dataclass(frozen=True, slots=True)
class IdentifiedFace:
    """A detected face together with what the engine concluded about it."""

    observation: FaceObservation
    status: FaceStatus
    similarity: float
    person: Person | None = None
    pending_id: int | None = None

    @property
    def label(self) -> str:
        if self.person is not None:
            return self.person.name
        if self.pending_id is not None:
            return f"pending #{self.pending_id}"
        return self.status


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """Something worth logging that happened while processing a frame."""

    kind: str
    message: str
    person_id: int | None = None
    pending_id: int | None = None
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Outcome of one frame: who is in view, what happened, and gesture progress per admin."""

    faces: tuple[IdentifiedFace, ...] = ()
    events: tuple[EngineEvent, ...] = ()
    gesture_progress: Mapping[int, float] = field(default_factory=dict)


_SKIP_MESSAGES = {
    "uncertain": "Visitor skipped: the face is too similar to someone already enrolled",
    "low_score": "Visitor skipped: the face is not clear enough",
    "too_small": "Visitor skipped: the face is too small, ask them to come closer",
}


class VouchingEngine:
    """Applies the vouching rule to a stream of frame observations.

    1. Every face is compared with the enrolled people and the pending visitors.
    2. A thumbs-up counts only when its hand belongs to the face of an admin.
    3. Once an admin has held the gesture for enough frames, every clearly
       unknown face in view becomes a pending visitor, to be named later.

    Unknown faces are never stored unless an admin vouches for them.
    """

    def __init__(
        self,
        repository: Repository,
        config: EngineConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repo = repository
        self._config = config or EngineConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._confirmer = GestureConfirmer(
            required_frames=self._config.gesture_frames,
            max_gap_frames=self._config.gesture_gap_frames,
            cooldown_seconds=self._config.gesture_cooldown_seconds,
        )
        self._gallery = Gallery((), threshold=self._config.match_threshold)
        self._people: dict[int, Person] = {}
        self._pending: list[PendingVisitor] = []
        self._last_seen: dict[int, float] = {}
        self._last_ignored: dict[str, float] = {}
        self._data_version = -1
        self.reload()

    @property
    def config(self) -> EngineConfig:
        return self._config

    def reload(self) -> None:
        """Read the enrolled people and the pending visitors again."""
        self._gallery = Gallery(self._repo.load_gallery(), threshold=self._config.match_threshold)
        self._people = self._gallery.people
        self._pending = self._repo.list_pending()
        self._data_version = self._repo.data_version()

    def refresh_if_changed(self) -> bool:
        """Reload when another process changed the database, e.g. ``doorvouch enroll``."""
        if self._repo.data_version() == self._data_version:
            return False
        self.reload()
        return True

    def process(
        self, observation: FrameObservation, *, snapshot: SnapshotFn | None = None
    ) -> FrameResult:
        """Apply the rule to one frame and record what happened.

        ``snapshot`` returns a JPEG of the given face box; it is called only for
        faces that become pending visitors.
        """
        faces = [self._identify(face) for face in observation.faces]
        boxes = [face.observation.box for face in faces]
        now = observation.timestamp
        events: list[EngineEvent] = []

        gesturing: set[int] = set()
        for hand in observation.hands:
            if hand.gesture != THUMB_UP or hand.score < self._config.min_gesture_score:
                continue
            owner = find_owner(
                hand.center,
                boxes,
                max_reach=self._config.hand_max_reach,
                ambiguity_ratio=self._config.hand_ambiguity_ratio,
            )
            if owner is None:
                self._ignore(events, "no_owner", "it is not clear whose hand it is", now)
                continue
            person = faces[owner].person
            if person is not None and person.is_admin:
                gesturing.add(person.id)
            else:
                self._ignore(events, "not_admin", "it does not come from an admin", now, person)

        for admin_id in self._confirmer.observe(gesturing, now):
            events.extend(self._vouch(self._people[admin_id], faces, snapshot))
        events.extend(self._presence(faces, now))

        for event in events:
            self._repo.log_event(
                event.kind,
                person_id=event.person_id,
                pending_id=event.pending_id,
                detail=event.detail,
                now=self._clock(),
            )
        progress = {admin_id: self._confirmer.progress(admin_id) for admin_id in gesturing}
        return FrameResult(tuple(faces), tuple(events), progress)

    def _identify(self, face: FaceObservation) -> IdentifiedFace:
        match = self._gallery.match(face.embedding)
        if match.person is not None:
            status: FaceStatus = "admin" if match.person.is_admin else "known"
            return IdentifiedFace(face, status, match.similarity, person=match.person)
        pending = self._match_pending(face.embedding)
        if pending is not None:
            return IdentifiedFace(face, "pending", match.similarity, pending_id=pending.id)
        if match.similarity >= self._config.visitor_max_similarity:
            return IdentifiedFace(face, "uncertain", match.similarity)
        return IdentifiedFace(face, "unknown", match.similarity)

    def _match_pending(self, embedding: Embedding) -> PendingVisitor | None:
        best: PendingVisitor | None = None
        best_similarity = self._config.match_threshold
        for pending in self._pending:
            similarity = cosine_similarity(embedding, pending.embedding)
            if similarity >= best_similarity:
                best, best_similarity = pending, similarity
        return best

    def _vouch(
        self, admin: Person, faces: list[IdentifiedFace], snapshot: SnapshotFn | None
    ) -> list[EngineEvent]:
        visitors = [face for face in faces if face.person is None]
        if not visitors:
            message = f"{admin.name} gave a thumbs-up, but there is no visitor in view"
            return [EngineEvent("vouch_without_visitor", message, person_id=admin.id)]

        events: list[EngineEvent] = []
        for face in visitors:
            if face.pending_id is not None:
                self._repo.touch_pending(face.pending_id, now=self._clock())
                message = f"Pending visitor #{face.pending_id} is already waiting for a name"
                events.append(
                    EngineEvent(
                        "pending_seen", message, person_id=admin.id, pending_id=face.pending_id
                    )
                )
                continue
            reason = self._skip_reason(face)
            if reason is not None:
                detail = {"reason": reason, "similarity": round(face.similarity, 3)}
                events.append(
                    EngineEvent(
                        "visitor_skipped", _SKIP_MESSAGES[reason], person_id=admin.id, detail=detail
                    )
                )
                continue
            image = snapshot(face.observation.box) if snapshot is not None else None
            pending = self._repo.create_pending(
                face.observation.embedding,
                vouched_by=admin.id,
                snapshot=image,
                now=self._clock(),
            )
            self._pending.append(pending)
            message = f"{admin.name} vouched for a visitor: pending #{pending.id} created"
            events.append(
                EngineEvent("pending_created", message, person_id=admin.id, pending_id=pending.id)
            )
        return events

    def _skip_reason(self, face: IdentifiedFace) -> str | None:
        if face.status == "uncertain":
            return "uncertain"
        if face.observation.score < self._config.min_face_score:
            return "low_score"
        if face.observation.box.w < self._config.min_face_width:
            return "too_small"
        return None

    def _ignore(
        self,
        events: list[EngineEvent],
        reason: str,
        explanation: str,
        now: float,
        person: Person | None = None,
    ) -> None:
        last = self._last_ignored.get(reason)
        if last is not None and now - last < self._config.ignored_gesture_cooldown_seconds:
            return
        self._last_ignored[reason] = now
        events.append(
            EngineEvent(
                "gesture_ignored",
                f"Thumbs-up ignored: {explanation}",
                person_id=person.id if person is not None else None,
                detail={"reason": reason},
            )
        )

    def _presence(self, faces: list[IdentifiedFace], now: float) -> list[EngineEvent]:
        events: list[EngineEvent] = []
        for face in faces:
            person = face.person
            if person is None:
                continue
            last = self._last_seen.get(person.id)
            if last is not None and now - last < self._config.presence_cooldown_seconds:
                continue
            self._last_seen[person.id] = now
            events.append(
                EngineEvent("person_seen", f"{person.name} is in view", person_id=person.id)
            )
        return events
