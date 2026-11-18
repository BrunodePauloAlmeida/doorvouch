"""Enrolling a person: collect face samples from the camera and save them."""

from __future__ import annotations

from dataclasses import dataclass

from doorvouch.apps.live import Display, Frames
from doorvouch.core.embeddings import Embedding, cosine_similarity, mean_embedding
from doorvouch.core.engine import IdentifiedFace
from doorvouch.core.gallery import SFACE_COSINE_THRESHOLD, Gallery
from doorvouch.core.people import Person
from doorvouch.errors import DoorVouchError
from doorvouch.storage import SqliteStore
from doorvouch.vision.base import FaceAnalyzer

MIN_SAMPLES = 3


@dataclass(frozen=True, slots=True)
class Enrollment:
    """Outcome of saving samples: who they belong to and whether that person is new."""

    person: Person
    samples_added: int
    created: bool


def collect_samples(
    frames: Frames,
    *,
    faces: FaceAnalyzer,
    samples: int = 8,
    interval_seconds: float = 0.4,
    min_face_width: float = 80.0,
    min_score: float = 0.8,
    same_person_threshold: float = SFACE_COSINE_THRESHOLD,
    timeout_seconds: float = 60.0,
    display: Display | None = None,
    name: str = "",
) -> list[Embedding]:
    """Capture up to ``samples`` embeddings of the one person in front of the camera.

    A sample is taken only while exactly one clear, large enough face is in
    view, at most once per ``interval_seconds`` so that the samples differ a
    little. A face that does not match the first sample is not recorded.
    """
    collected: list[Embedding] = []
    first: float | None = None
    last_capture: float | None = None

    for timestamp, frame in frames:
        if first is None:
            first = timestamp
        if timestamp - first > timeout_seconds:
            break

        observations = faces.analyze(frame)
        hint = "Look at the camera"
        if len(observations) != 1:
            hint = "Exactly one face must be in view"
        else:
            face = observations[0]
            if face.score < min_score or face.box.w < min_face_width:
                hint = "Come closer and face the camera"
            elif (
                collected
                and cosine_similarity(face.embedding, collected[0]) < same_person_threshold
            ):
                hint = "That looks like someone else"
            elif last_capture is None or timestamp - last_capture >= interval_seconds:
                collected.append(face.embedding)
                last_capture = timestamp
                hint = "Good, now turn your head a little"

        if display is not None:
            shown = [IdentifiedFace(face, "unknown", 0.0) for face in observations]
            lines = [f"Enrolling {name}: {len(collected)}/{samples}", hint]
            if not display.show_faces(frame, shown, lines):
                break
        if len(collected) >= samples:
            break
    return collected


def save_enrollment(
    store: SqliteStore,
    embeddings: list[Embedding],
    *,
    name: str,
    is_admin: bool = False,
    match_threshold: float = SFACE_COSINE_THRESHOLD,
) -> Enrollment:
    """Store the samples under ``name``, adding to an existing person of that name.

    Refuses to give one face two identities: if the samples match somebody
    enrolled under another name, nothing is saved.
    """
    if len(embeddings) < MIN_SAMPLES:
        raise DoorVouchError(
            f"Only {len(embeddings)} usable face sample(s) were captured and at least "
            f"{MIN_SAMPLES} are needed. Improve the lighting, come closer and try again."
        )
    existing = store.find_person_by_name(name)
    match = Gallery(store.load_gallery(), threshold=match_threshold).match(
        mean_embedding(embeddings)
    )
    if match.person is not None and (existing is None or match.person.id != existing.id):
        raise DoorVouchError(
            f"This face is already enrolled as {match.person.name!r} "
            f"(similarity {match.similarity:.2f}). Use that name to add more samples."
        )

    if existing is None:
        person = store.add_person(name, is_admin=is_admin)
    else:
        sample_counts = {summary.person.id: summary.samples for summary in store.list_people()}
        if match.person is None and sample_counts.get(existing.id, 0) > 0:
            raise DoorVouchError(
                f"Someone else is already enrolled as {existing.name!r}. Pick another name."
            )
        if is_admin != existing.is_admin:
            role = "an admin" if existing.is_admin else "not an admin"
            raise DoorVouchError(
                f"{existing.name!r} is already enrolled and is {role}. To change that, remove "
                "the person with `doorvouch people remove` and enroll again."
            )
        person = existing

    for embedding in embeddings:
        store.add_face_sample(person.id, embedding)
    store.log_event(
        "person_enrolled",
        person_id=person.id,
        detail={"samples": len(embeddings), "new": existing is None},
    )
    return Enrollment(person, len(embeddings), created=existing is None)
