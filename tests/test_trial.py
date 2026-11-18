from __future__ import annotations

from pathlib import Path

import pytest

from doorvouch.apps.trial import Attempt, TrialReport, judge, run_trial
from doorvouch.core.engine import EngineConfig
from doorvouch.errors import DoorVouchError
from fakes import RecordingDisplay, ScriptedFaces, ScriptedFrame, ScriptedGestures, clip
from helpers import axis, face, hand_below

ADMIN = face(axis(0), x=100)
VISITOR = face(axis(2), x=450)

VOUCHING = ScriptedFrame(faces=(ADMIN, VISITOR), hands=(hand_below(ADMIN),))
IMPOSTOR = ScriptedFrame(faces=(ADMIN, VISITOR), hands=(hand_below(VISITOR),))
QUIET = ScriptedFrame(faces=(ADMIN, VISITOR))
ADMIN_ALONE_VOUCHING = ScriptedFrame(faces=(ADMIN,), hands=(hand_below(ADMIN),))

# A one-second attempt at 8 FPS processes 8 frames, and a 9th frame ends the window.
WINDOW = 9


@pytest.fixture
def household(store):
    admin = store.add_person("Alice", is_admin=True)
    store.add_face_sample(admin.id, axis(0))
    return store


def trial(frames, store, tmp_path: Path, **options):
    settings = {"attempts": 3, "seconds": 1.0, "pause_seconds": 0.0}
    settings.update(options)
    return run_trial(
        clip(frames),
        store=store,
        config=EngineConfig(),
        faces=ScriptedFaces(),
        gestures=ScriptedGestures(),
        workdir=tmp_path / "scratch",
        **settings,
    )


def verdicts(report: TrialReport) -> list[str]:
    return [attempt.verdict for attempt in report.attempts]


@pytest.mark.parametrize(
    ("scenario", "events", "expected"),
    [
        ("vouch", {}, "fail"),
        ("vouch", {"pending_created": 1}, "pass"),
        ("vouch", {"pending_created": 2}, "fail"),
        ("vouch", {"vouch_without_visitor": 1}, "fail"),
        ("impostor", {}, "void"),
        ("impostor", {"person_seen": 2}, "void"),
        ("impostor", {"gesture_ignored": 2}, "pass"),
        ("impostor", {"gesture_ignored": 1, "pending_created": 1}, "fail"),
        ("impostor", {"gesture_ignored": 1, "vouch_without_visitor": 1}, "fail"),
        ("impostor", {"visitor_skipped": 1}, "fail"),
        ("impostor", {"pending_seen": 1}, "fail"),
    ],
)
def test_an_attempt_is_judged_from_its_events(scenario, events, expected):
    assert judge(scenario, events) == expected


def test_vouch_scenario_passes_when_one_pending_visitor_appears(household, tmp_path):
    report = trial([VOUCHING] * WINDOW * 3, household, tmp_path, scenario="vouch")

    assert verdicts(report) == ["pass", "pass", "pass"]
    assert [attempt.pending_created for attempt in report.attempts] == [1, 1, 1]
    assert all(attempt.vouch_confirmed for attempt in report.attempts)
    assert [attempt.frames for attempt in report.attempts] == [8, 8, 8]
    assert report.passed == 3
    assert report.complete is True
    assert report.accepted is True
    assert report.mean_fps == 8.0


def test_vouch_scenario_fails_an_attempt_without_a_gesture(household, tmp_path):
    frames = [VOUCHING] * WINDOW + [QUIET] * WINDOW + [VOUCHING] * WINDOW
    report = trial(frames, household, tmp_path, scenario="vouch")

    assert verdicts(report) == ["pass", "fail", "pass"]
    assert report.accepted is False  # 2 of 3 is below the 80% the scenario requires


def test_impostor_scenario_passes_when_the_gesture_is_seen_and_ignored(household, tmp_path):
    report = trial([IMPOSTOR] * WINDOW * 3, household, tmp_path, scenario="impostor")

    assert verdicts(report) == ["pass", "pass", "pass"]
    assert all(attempt.gestures_ignored == 1 for attempt in report.attempts)
    assert not any(attempt.vouch_confirmed for attempt in report.attempts)
    assert report.accepted is True


def test_impostor_attempt_without_any_gesture_is_void(household, tmp_path):
    report = trial([QUIET] * WINDOW * 3, household, tmp_path, scenario="impostor")

    assert verdicts(report) == ["void", "void", "void"]
    assert report.void == 3
    assert report.passed == 0
    assert report.accepted is False


def test_impostor_scenario_needs_every_attempt_to_pass(household, tmp_path):
    frames = [IMPOSTOR] * WINDOW * 2 + [VOUCHING] * WINDOW
    report = trial(frames, household, tmp_path, scenario="impostor")

    assert verdicts(report) == ["pass", "pass", "fail"]
    assert report.accepted is False


def test_impostor_scenario_fails_on_a_confirmed_vouch_that_created_nothing(household, tmp_path):
    frames = [ADMIN_ALONE_VOUCHING] * WINDOW
    report = trial(frames, household, tmp_path, scenario="impostor", attempts=1)

    (attempt,) = report.attempts
    assert attempt.verdict == "fail"
    assert attempt.vouch_confirmed is True
    assert attempt.pending_created == 0


def test_the_real_database_is_never_touched(household, tmp_path):
    trial([VOUCHING] * WINDOW * 3, household, tmp_path, scenario="vouch")
    assert household.list_pending() == []
    assert household.list_events() == []


def test_visitors_already_pending_do_not_spoil_the_attempts(household, tmp_path):
    household.create_pending(axis(2), vouched_by=None)
    report = trial([VOUCHING] * WINDOW * 3, household, tmp_path, scenario="vouch")
    assert report.passed == 3
    assert len(household.list_pending()) == 1


def test_an_attempt_cut_short_by_the_camera_is_not_counted(household, tmp_path):
    frames = [VOUCHING] * (WINDOW + 4)
    report = trial(frames, household, tmp_path, scenario="vouch")

    assert [attempt.number for attempt in report.attempts] == [1]
    assert report.complete is False
    assert report.accepted is False


def test_impostor_attempts_never_pass_by_running_out_of_frames(household, tmp_path):
    report = trial([IMPOSTOR] * WINDOW, household, tmp_path, scenario="impostor")

    assert verdicts(report) == ["pass"]
    assert report.requested == 3
    assert report.accepted is False


def test_trial_ends_when_the_camera_stops_during_a_countdown(household, tmp_path):
    # Five frames of countdown, then nine frames of attempt 1; nothing is left for attempt 2.
    frames = [VOUCHING] * (5 + WINDOW)
    report = trial(frames, household, tmp_path, scenario="vouch", pause_seconds=0.5)

    assert verdicts(report) == ["pass"]
    assert report.complete is False


def test_an_attempt_the_user_stops_is_not_counted(household, tmp_path):
    display = RecordingDisplay(stop_after=3)
    report = trial([VOUCHING] * WINDOW * 3, household, tmp_path, scenario="vouch", display=display)

    assert report.attempts == ()
    assert report.accepted is False


def test_countdown_is_shown_before_each_attempt(household, tmp_path):
    display = RecordingDisplay()
    frames = [QUIET] * 17 + [VOUCHING] * WINDOW
    report = trial(
        frames,
        household,
        tmp_path,
        scenario="vouch",
        attempts=1,
        pause_seconds=2.0,
        display=display,
    )

    countdown = [caption for (caption,) in display.captions]
    assert countdown == ["Attempt 1/1 starts in 2 s"] * 8 + ["Attempt 1/1 starts in 1 s"] * 8
    assert display.banners[0] == "Attempt 1/1: go"
    assert len(display.results) == 8
    assert report.accepted is True


def test_progress_callback_receives_each_attempt(household, tmp_path):
    seen: list[Attempt] = []
    trial(
        [VOUCHING] * WINDOW * 2,
        household,
        tmp_path,
        scenario="vouch",
        attempts=2,
        on_attempt=seen.append,
    )
    assert [attempt.number for attempt in seen] == [1, 2]


@pytest.mark.parametrize(
    "options",
    [
        {"scenario": "party"},
        {"scenario": "vouch", "attempts": 0},
        {"scenario": "vouch", "seconds": 0},
    ],
)
def test_invalid_trials_are_refused(household, tmp_path, options):
    with pytest.raises(DoorVouchError):
        trial([QUIET], household, tmp_path, **options)


def test_report_is_a_markdown_table_with_a_verdict():
    report = TrialReport(
        "vouch",
        (
            Attempt(1, "pass", True, 1, 0, 80, 9.9),
            Attempt(2, "fail", False, 0, 1, 79, 10.1),
        ),
        requested=2,
    )
    assert report.to_markdown().splitlines() == [
        "### Scenario: vouch",
        "",
        "| Attempt | Vouch confirmed | Pending created | Thumbs-up ignored | Frames | FPS "
        "| Result |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | :--- |",
        "| 1 | yes | 1 | 0 | 80 | 9.9 | pass |",
        "| 2 | no | 0 | 1 | 79 | 10.1 | fail |",
        "",
        "Passed 1 of 2 attempts: not accepted (80% must pass). Mean frame rate: 10.0 FPS.",
    ]


def test_report_explains_void_and_missing_attempts():
    report = TrialReport(
        "impostor",
        (
            Attempt(1, "pass", False, 0, 1, 80, 10.0),
            Attempt(2, "void", False, 0, 0, 80, 10.0),
        ),
        requested=3,
    )
    assert report.to_markdown().splitlines()[-5:] == [
        "Passed 1 of 2 attempts: not accepted (every attempt must pass). "
        "Mean frame rate: 10.0 FPS.",
        "",
        "Void attempts: 1. No thumbs-up was detected in them, so they tested nothing. "
        "Repeat the trial with the gesture held in clear view.",
        "",
        "The trial is incomplete: 3 attempts were requested and 2 ran to the end.",
    ]


def test_an_accepted_report_says_so():
    attempts = tuple(Attempt(n, "pass", True, 1, 0, 80, 10.0) for n in range(1, 11))
    report = TrialReport("vouch", attempts, requested=10)
    assert report.accepted is True
    assert report.to_markdown().splitlines()[-1] == (
        "Passed 10 of 10 attempts: accepted (80% must pass). Mean frame rate: 10.0 FPS."
    )


def test_a_report_without_attempts_is_not_accepted():
    assert TrialReport("impostor", (), requested=3).accepted is False
