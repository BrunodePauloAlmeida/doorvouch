from __future__ import annotations

import pytest

from doorvouch.apps.live import run_live
from doorvouch.core.engine import VouchingEngine
from fakes import RecordingDisplay, ScriptedFaces, ScriptedFrame, ScriptedGestures, clip
from helpers import axis, face, hand_below

ADMIN = face(axis(0), x=100)
VISITOR = face(axis(2), x=450)

QUIET = ScriptedFrame(faces=(ADMIN, VISITOR))
VOUCHING = ScriptedFrame(faces=(ADMIN, VISITOR), hands=(hand_below(ADMIN),))


@pytest.fixture
def engine(store) -> VouchingEngine:
    admin = store.add_person("Alice", is_admin=True)
    store.add_face_sample(admin.id, axis(0))
    return VouchingEngine(store)


def run(frames, engine, **options):
    return run_live(
        frames, faces=ScriptedFaces(), gestures=ScriptedGestures(), engine=engine, **options
    )


def test_frames_flow_through_the_models_into_the_engine(engine, store):
    seen = []
    stats = run(clip([VOUCHING] * 10), engine, on_event=seen.append)

    assert len(store.list_pending()) == 1
    assert [event.kind for event in seen] == ["person_seen", "pending_created"]
    assert stats.events == {"person_seen": 1, "pending_created": 1}
    assert stats.frames == 10
    assert stats.stopped_by_user is False
    assert stats.timed_out is False


def test_frame_rate_is_measured_from_the_timestamps(engine):
    stats = run(clip([QUIET] * 17, fps=16.0), engine)
    assert stats.seconds == 1.0
    assert stats.fps == 16.0


def test_empty_stream_gives_empty_stats(engine):
    stats = run(clip([]), engine)
    assert (stats.frames, stats.seconds, stats.fps, stats.events) == (0, 0.0, 0.0, {})
    assert stats.timed_out is False


def test_max_seconds_stops_the_loop(engine):
    stats = run(clip([QUIET] * 100, fps=8.0), engine, max_seconds=2.0)
    assert stats.frames == 16
    assert stats.timed_out is True
    assert stats.stopped_by_user is False


def test_a_stream_shorter_than_max_seconds_does_not_time_out(engine):
    stats = run(clip([QUIET] * 16, fps=8.0), engine, max_seconds=2.0)
    assert stats.frames == 16
    assert stats.timed_out is False


def test_display_gets_every_result_and_event_messages(engine):
    display = RecordingDisplay()
    run(clip([VOUCHING] * 6), engine, display=display)

    assert len(display.results) == 6
    assert display.banners[0] == "Alice is in view"
    assert "pending #1 created" in display.banners[1]


def test_user_can_stop_from_the_display(engine):
    display = RecordingDisplay(stop_after=3)
    stats = run(clip([QUIET] * 50), engine, display=display)
    assert stats.frames == 3
    assert stats.stopped_by_user is True
    assert stats.timed_out is False


def test_snapshot_is_taken_from_the_frame_that_triggered_it(engine, store):
    requests = []

    def snapshot(frame, box):
        requests.append((frame, box))
        return b"jpeg"

    run(clip([VOUCHING] * 10), engine, snapshot=snapshot)

    assert requests == [(VOUCHING, VISITOR.box)]
    (pending,) = store.list_pending()
    assert pending.snapshot_path.read_bytes() == b"jpeg"


def test_database_changes_are_checked_periodically(store):
    class CountingEngine(VouchingEngine):
        refreshes = 0

        def refresh_if_changed(self) -> bool:
            CountingEngine.refreshes += 1
            return super().refresh_if_changed()

    # Five seconds of video with a check every two seconds: at 2 s and at 4 s.
    run(clip([QUIET] * 41, fps=8.0), CountingEngine(store), refresh_seconds=2.0)
    assert CountingEngine.refreshes == 2
