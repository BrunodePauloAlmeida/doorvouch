from __future__ import annotations

from pathlib import Path

import pytest

from doorvouch.core.engine import EngineConfig, VouchingEngine
from doorvouch.core.observations import HandObservation
from doorvouch.storage import SqliteStore
from helpers import axis, face, frame, hand_below, toward

ADMIN_FACE = axis(0)
MEMBER_FACE = axis(1)
VISITOR_FACE = axis(2)
OTHER_VISITOR_FACE = axis(3)

# Two people side by side, as a door camera would see them.
ADMIN = face(ADMIN_FACE, x=100)
VISITOR = face(VISITOR_FACE, x=450)
MEMBER = face(MEMBER_FACE, x=450)


class Scene:
    """Feeds frames to the engine at 10 frames per second and collects what happens."""

    def __init__(self, engine: VouchingEngine) -> None:
        self.engine = engine
        self.time = 0.0
        self.events = []
        self.snapshot_requests = []

    def play(self, frames: int, faces=(), hands=()):
        result = None
        for _ in range(frames):
            result = self.engine.process(frame(self.time, faces, hands), snapshot=self._snapshot)
            self.events.extend(result.events)
            self.time += 0.1
        return result

    def wait(self, seconds: float) -> None:
        self.time += seconds

    def kinds(self, *, without: tuple[str, ...] = ("person_seen",)) -> list[str]:
        return [event.kind for event in self.events if event.kind not in without]

    def only(self, kind: str):
        return [event for event in self.events if event.kind == kind]

    def _snapshot(self, box):
        self.snapshot_requests.append(box)
        return b"jpeg-bytes"


@pytest.fixture
def people(store):
    admin = store.add_person("Alice", is_admin=True)
    store.add_face_sample(admin.id, ADMIN_FACE)
    member = store.add_person("Ruth")
    store.add_face_sample(member.id, MEMBER_FACE)
    return admin, member


@pytest.fixture
def scene(store, people) -> Scene:
    return Scene(VouchingEngine(store))


def test_admin_thumbs_up_pre_registers_the_visitor(scene, store, people):
    admin, _ = people

    scene.play(4, [ADMIN, VISITOR], [hand_below(ADMIN)])
    assert store.list_pending() == []

    scene.play(1, [ADMIN, VISITOR], [hand_below(ADMIN)])

    (pending,) = store.list_pending()
    assert pending.vouched_by == admin.id
    assert pending.embedding == pytest.approx(VISITOR_FACE)
    assert pending.snapshot_path is not None
    assert pending.snapshot_path.read_bytes() == b"jpeg-bytes"
    assert scene.kinds() == ["pending_created"]
    assert scene.snapshot_requests == [VISITOR.box]


def test_holding_the_gesture_does_not_create_duplicates(scene, store):
    scene.play(40, [ADMIN, VISITOR], [hand_below(ADMIN)])
    assert len(store.list_pending()) == 1
    assert scene.kinds() == ["pending_created"]


def test_visitor_is_labelled_pending_on_the_following_frames(scene, store):
    scene.play(5, [ADMIN, VISITOR], [hand_below(ADMIN)])
    result = scene.play(1, [ADMIN, VISITOR])
    (pending,) = store.list_pending()
    statuses = {identified.label: identified.status for identified in result.faces}
    assert statuses == {"Alice": "admin", f"pending #{pending.id}": "pending"}


def test_vouching_again_for_the_same_visitor_only_refreshes_the_entry(scene, store):
    scene.play(5, [ADMIN, VISITOR], [hand_below(ADMIN)])
    scene.play(3, [ADMIN, VISITOR])
    scene.wait(10)
    scene.play(5, [ADMIN, VISITOR], [hand_below(ADMIN)])

    (pending,) = store.list_pending()
    assert pending.seen_count == 2
    assert scene.kinds() == ["pending_created", "pending_seen"]
    assert len(scene.snapshot_requests) == 1


def test_two_visitors_get_one_entry_each(scene, store):
    second = face(OTHER_VISITOR_FACE, x=800)
    scene.play(5, [ADMIN, VISITOR, second], [hand_below(ADMIN)])
    assert [p.embedding for p in store.list_pending()] == [
        pytest.approx(VISITOR_FACE),
        pytest.approx(OTHER_VISITOR_FACE),
    ]


def test_thumbs_up_from_the_visitor_is_ignored(scene, store):
    scene.play(30, [ADMIN, VISITOR], [hand_below(VISITOR)])
    assert store.list_pending() == []
    (ignored,) = scene.only("gesture_ignored")
    assert ignored.detail == {"reason": "not_admin"}
    assert ignored.person_id is None


def test_thumbs_up_from_a_member_who_is_not_admin_is_ignored(scene, store, people):
    _, member = people
    scene.play(30, [MEMBER, face(VISITOR_FACE, x=800)], [hand_below(MEMBER)])
    assert store.list_pending() == []
    assert [event.person_id for event in scene.only("gesture_ignored")] == [member.id]


def test_ignored_gestures_are_reported_again_after_a_while(scene):
    scene.play(30, [ADMIN, VISITOR], [hand_below(VISITOR)])
    scene.wait(5)
    scene.play(1, [ADMIN, VISITOR], [hand_below(VISITOR)])
    assert scene.kinds() == ["gesture_ignored", "gesture_ignored"]


def test_too_few_frames_do_nothing(scene, store):
    scene.play(4, [ADMIN, VISITOR], [hand_below(ADMIN)])
    scene.play(10, [ADMIN, VISITOR])
    assert store.list_pending() == []
    assert scene.kinds() == []


def test_one_dropped_frame_does_not_break_the_gesture(scene, store):
    scene.play(3, [ADMIN, VISITOR], [hand_below(ADMIN)])
    scene.play(1, [ADMIN, VISITOR])
    scene.play(2, [ADMIN, VISITOR], [hand_below(ADMIN)])
    assert len(store.list_pending()) == 1


def test_gesture_does_not_count_while_the_admin_face_is_not_recognised(scene, store):
    stranger_face = face(axis(5), x=100)
    scene.play(10, [stranger_face, VISITOR], [hand_below(stranger_face)])
    assert store.list_pending() == []


def test_hand_between_two_people_is_ambiguous_and_ignored(scene, store):
    between = HandObservation((325.0, 300.0), "Thumb_Up", 0.9)
    scene.play(10, [ADMIN, VISITOR], [between])
    assert store.list_pending() == []
    assert [event.detail for event in scene.only("gesture_ignored")] == [{"reason": "no_owner"}]


def test_hand_with_nobody_near_is_ignored(scene, store):
    lonely = HandObservation((150.0, 900.0), "Thumb_Up", 0.9)
    scene.play(10, [ADMIN, VISITOR], [lonely])
    assert store.list_pending() == []
    assert [event.detail for event in scene.only("gesture_ignored")] == [{"reason": "no_owner"}]


@pytest.mark.parametrize(
    "hand",
    [
        hand_below(ADMIN, gesture="Victory"),
        hand_below(ADMIN, gesture=None),
        hand_below(ADMIN, score=0.3),
    ],
)
def test_other_gestures_and_weak_detections_are_not_a_thumbs_up(scene, store, hand):
    scene.play(10, [ADMIN, VISITOR], [hand])
    assert store.list_pending() == []
    assert scene.kinds() == []


def test_vouching_with_nobody_else_in_view_is_only_logged(scene, store):
    scene.play(5, [ADMIN], [hand_below(ADMIN)])
    assert store.list_pending() == []
    assert scene.kinds() == ["vouch_without_visitor"]


def test_known_people_are_never_pre_registered(scene, store):
    scene.play(5, [ADMIN, MEMBER], [hand_below(ADMIN)])
    assert store.list_pending() == []
    assert scene.kinds() == ["vouch_without_visitor"]


def test_face_that_resembles_an_enrolled_person_is_not_pre_registered(scene, store):
    lookalike = face(toward(MEMBER_FACE, axis(6), 0.33), x=450)
    result = scene.play(5, [ADMIN, lookalike], [hand_below(ADMIN)])

    assert store.list_pending() == []
    assert [identified.status for identified in result.faces] == ["admin", "uncertain"]
    (skipped,) = scene.only("visitor_skipped")
    assert skipped.detail == {"reason": "uncertain", "similarity": 0.33}


@pytest.mark.parametrize(
    ("visitor", "reason"),
    [
        (face(VISITOR_FACE, x=450, width=40), "too_small"),
        (face(VISITOR_FACE, x=450, score=0.5), "low_score"),
    ],
)
def test_poor_views_of_the_visitor_are_skipped(scene, store, visitor, reason):
    scene.play(5, [ADMIN, visitor], [hand_below(ADMIN)])
    assert store.list_pending() == []
    assert [event.detail["reason"] for event in scene.only("visitor_skipped")] == [reason]


def test_unknown_faces_leave_no_trace_without_a_vouch(scene, store):
    scene.play(50, [VISITOR])
    assert store.list_pending() == []
    assert store.list_events() == []
    assert scene.snapshot_requests == []


def test_presence_of_known_people_is_logged_once_per_cooldown(scene, people):
    admin, member = people
    scene.play(100, [ADMIN, MEMBER])
    assert [event.person_id for event in scene.only("person_seen")] == [admin.id, member.id]

    scene.wait(60)
    scene.play(1, [ADMIN])
    assert [event.person_id for event in scene.only("person_seen")] == [
        admin.id,
        member.id,
        admin.id,
    ]


def test_events_are_written_to_the_store(scene, store, people):
    admin, _ = people
    scene.play(5, [ADMIN, VISITOR], [hand_below(ADMIN)])
    (pending,) = store.list_pending()
    logged = {event.kind: event for event in store.list_events()}
    assert logged["pending_created"].person_id == admin.id
    assert logged["pending_created"].pending_id == pending.id
    assert logged["person_seen"].person_id == admin.id


def test_gesture_progress_is_reported_for_the_overlay(scene, people):
    admin, _ = people
    result = scene.play(2, [ADMIN, VISITOR], [hand_below(ADMIN)])
    assert result.gesture_progress == {admin.id: pytest.approx(0.4)}
    assert scene.play(1, [ADMIN, VISITOR]).gesture_progress == {}


def test_custom_thresholds_are_respected(store, people):
    engine = VouchingEngine(store, EngineConfig(gesture_frames=2, min_face_width=150.0))
    scene = Scene(engine)
    scene.play(2, [ADMIN, VISITOR], [hand_below(ADMIN)])
    assert scene.kinds() == ["visitor_skipped"]


def test_engine_notices_people_enrolled_by_another_process(tmp_path: Path):
    database = tmp_path / "doorvouch.db"
    with SqliteStore(database) as running, SqliteStore(database) as other_terminal:
        engine = VouchingEngine(running)
        before = engine.process(frame(0.0, [VISITOR]))
        assert [identified.status for identified in before.faces] == ["unknown"]
        assert engine.refresh_if_changed() is False

        leo = other_terminal.add_person("Leo")
        other_terminal.add_face_sample(leo.id, VISITOR_FACE)

        assert engine.refresh_if_changed() is True
        after = engine.process(frame(0.1, [VISITOR]))
        assert [identified.label for identified in after.faces] == ["Leo"]
        assert engine.refresh_if_changed() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"match_threshold": 0.0},
        {"visitor_max_similarity": 0.5},
        {"min_gesture_score": 1.5},
        {"min_face_score": -0.1},
        {"min_face_width": -1.0},
        {"presence_cooldown_seconds": -1.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)
