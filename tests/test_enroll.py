from __future__ import annotations

import pytest

from doorvouch.apps.enroll import MIN_SAMPLES, collect_samples, save_enrollment
from doorvouch.errors import DoorVouchError
from fakes import RecordingDisplay, ScriptedFaces, ScriptedFrame, clip
from helpers import axis, face, toward

ALICE = axis(0)
ALICE_TURNED = toward(ALICE, axis(1), 0.8)
STRANGER = axis(5)

ALONE = ScriptedFrame(faces=(face(ALICE, x=500),))
TURNED = ScriptedFrame(faces=(face(ALICE_TURNED, x=500),))
EMPTY = ScriptedFrame()
TWO_PEOPLE = ScriptedFrame(faces=(face(ALICE, x=100), face(STRANGER, x=700)))
SOMEONE_ELSE = ScriptedFrame(faces=(face(STRANGER, x=500),))
FAR_AWAY = ScriptedFrame(faces=(face(ALICE, x=500, width=40),))
BLURRY = ScriptedFrame(faces=(face(ALICE, x=500, score=0.5),))


def collect(frames, **options):
    return collect_samples(clip(frames), faces=ScriptedFaces(), **options)


def test_samples_are_spaced_out_in_time():
    # 8 FPS and half a second between samples: frames 0, 4, 8 and 12 are captured.
    samples = collect([ALONE] * 13, samples=8, interval_seconds=0.5)
    assert len(samples) == 4


def test_collection_stops_once_enough_samples_were_taken():
    samples = collect([ALONE, TURNED] * 50, samples=3, interval_seconds=0.0)
    assert samples == [ALICE, ALICE_TURNED, ALICE]


@pytest.mark.parametrize("frame", [EMPTY, TWO_PEOPLE, FAR_AWAY, BLURRY])
def test_unusable_views_are_not_sampled(frame):
    assert collect([frame] * 20, interval_seconds=0.0) == []


def test_a_different_person_stepping_in_is_not_mixed_into_the_samples():
    samples = collect([ALONE, SOMEONE_ELSE, ALONE], samples=8, interval_seconds=0.0)
    assert samples == [ALICE, ALICE]


def test_collection_gives_up_after_the_timeout():
    samples = collect([EMPTY] * 100, timeout_seconds=2.0)
    assert samples == []


def test_progress_and_hints_are_shown():
    display = RecordingDisplay()
    collect([EMPTY, ALONE], samples=1, display=display, name="Alice")
    assert display.captions == [
        ["Enrolling Alice: 0/1", "Exactly one face must be in view"],
        ["Enrolling Alice: 1/1", "Good, now turn your head a little"],
    ]


def test_user_can_cancel_from_the_display():
    display = RecordingDisplay(stop_after=2)
    samples = collect([ALONE] * 50, interval_seconds=0.0, display=display)
    assert len(samples) == 2


def test_saving_creates_the_person_with_all_samples(store):
    enrollment = save_enrollment(store, [ALICE, ALICE_TURNED, ALICE], name="Alice", is_admin=True)

    assert enrollment.created is True
    assert enrollment.samples_added == 3
    assert enrollment.person.is_admin is True
    assert store.has_admin()
    assert len(store.load_gallery()) == 3
    (event,) = store.list_events()
    assert event.kind == "person_enrolled"
    assert event.detail == {"new": True, "samples": 3}


def test_enrolling_the_same_name_again_adds_samples(store):
    first = save_enrollment(store, [ALICE] * 3, name="Alice", is_admin=True)
    second = save_enrollment(store, [ALICE_TURNED] * 4, name="alice", is_admin=True)

    assert second.created is False
    assert second.person == first.person
    assert len(store.load_gallery()) == 7
    assert len(store.list_people()) == 1


def test_too_few_samples_are_refused(store):
    with pytest.raises(DoorVouchError, match=f"at least {MIN_SAMPLES}"):
        save_enrollment(store, [ALICE] * (MIN_SAMPLES - 1), name="Alice")
    assert store.list_people() == []


def test_one_face_cannot_get_a_second_identity(store):
    save_enrollment(store, [ALICE] * 3, name="Alice")
    with pytest.raises(DoorVouchError, match="already enrolled as 'Alice'"):
        save_enrollment(store, [ALICE_TURNED] * 3, name="Beth")
    assert [summary.person.name for summary in store.list_people()] == ["Alice"]


def test_a_name_in_use_cannot_be_taken_by_another_face(store):
    save_enrollment(store, [ALICE] * 3, name="Alice")
    with pytest.raises(DoorVouchError, match="Someone else is already enrolled as 'Alice'"):
        save_enrollment(store, [STRANGER] * 3, name="Alice")
    assert len(store.load_gallery()) == 3


def test_role_is_not_changed_by_enrolling_again(store):
    save_enrollment(store, [ALICE] * 3, name="Alice")
    with pytest.raises(DoorVouchError, match="is not an admin"):
        save_enrollment(store, [ALICE] * 3, name="Alice", is_admin=True)


def test_a_person_added_without_samples_can_be_completed_later(store):
    store.add_person("Alice", is_admin=True)
    enrollment = save_enrollment(store, [ALICE] * 3, name="Alice", is_admin=True)
    assert enrollment.created is False
    assert store.has_admin()
