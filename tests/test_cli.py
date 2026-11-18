from __future__ import annotations

from pathlib import Path

import pytest

from doorvouch import __version__, cli, models
from doorvouch.storage import SqliteStore
from fakes import ScriptedFaces, ScriptedFrame, ScriptedGestures, camera_playing
from helpers import axis, face, hand_below

ALICE = face(axis(0), x=100)
VISITOR = face(axis(2), x=450)

ALICE_ALONE = ScriptedFrame(faces=(ALICE,))
NOBODY = ScriptedFrame()
QUIET = ScriptedFrame(faces=(ALICE, VISITOR))
VOUCHING = ScriptedFrame(faces=(ALICE, VISITOR), hands=(hand_below(ALICE),))
IMPOSTOR = ScriptedFrame(faces=(ALICE, VISITOR), hands=(hand_below(VISITOR),))


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def run(home: Path):
    def invoke(*arguments: str) -> int:
        return cli.main(["--home", str(home), *arguments])

    return invoke


@pytest.fixture
def camera(monkeypatch):
    """Replace the models with scripted ones; call the fixture to choose the footage."""
    gestures = ScriptedGestures()
    monkeypatch.setattr(
        models,
        "require",
        lambda directory, specs=models.ALL_MODELS: {
            spec.key: Path(spec.filename) for spec in specs
        },
    )
    monkeypatch.setattr(
        "doorvouch.vision.opencv_faces.OpenCvFaceAnalyzer.from_models",
        classmethod(lambda cls, detector, recognizer: ScriptedFaces()),
    )
    monkeypatch.setattr(
        "doorvouch.vision.mediapipe_gestures.MediaPipeGestureDetector.from_model",
        classmethod(lambda cls, model: gestures),
    )
    monkeypatch.setattr("doorvouch.vision.opencv_faces.face_jpeg", lambda frame, box: b"jpeg")

    def play(frames) -> ScriptedGestures:
        monkeypatch.setattr("doorvouch.vision.camera.Camera", camera_playing(frames))
        return gestures

    return play


def enroll_alice(home: Path) -> None:
    with SqliteStore.open(home) as store:
        admin = store.add_person("Alice", is_admin=True)
        store.add_face_sample(admin.id, axis(0))


# Commands that do not need a camera ---------------------------------------------


def test_without_a_command_the_help_is_shown(capsys):
    assert cli.main([]) == 0
    assert "usage: doorvouch" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"doorvouch {__version__}"


def test_home_can_come_from_the_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOORVOUCH_HOME", str(tmp_path / "elsewhere"))
    assert cli.main(["people"]) == 0
    assert (tmp_path / "elsewhere" / "doorvouch.db").is_file()


def test_people_on_a_new_home_explains_how_to_start(run, capsys):
    assert run("people") == 0
    assert "Nobody is enrolled yet" in capsys.readouterr().out


def test_people_are_listed_with_role_and_samples(run, home, capsys):
    with SqliteStore.open(home) as store:
        admin = store.add_person("Alice", is_admin=True)
        store.add_face_sample(admin.id, axis(0))
        store.add_face_sample(admin.id, axis(1))
        store.add_person("Ruth")

    assert run("people") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["ID", "Name", "Role", "Samples", "Since"]
    assert lines[2].split()[:4] == ["1", "Alice", "admin", "2"]
    assert lines[3].split()[:4] == ["2", "Ruth", "member", "0"]


def test_removing_a_person_needs_confirmation(run, home, monkeypatch, capsys):
    enroll_alice(home)

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert run("people", "remove", "1") == 1
    assert "Nothing was removed." in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert run("people", "remove", "1") == 0
    assert "Removed Alice." in capsys.readouterr().out
    with SqliteStore.open(home) as store:
        assert store.list_people() == []


def test_removing_without_a_terminal_removes_nothing(run, home, monkeypatch):
    enroll_alice(home)

    def no_terminal(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    assert run("people", "remove", "1") == 1
    with SqliteStore.open(home) as store:
        assert len(store.list_people()) == 1


def test_removing_with_yes_does_not_ask(run, home, monkeypatch):
    enroll_alice(home)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("must not ask"))
    assert run("people", "remove", "1", "--yes") == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("people", "remove"), "Say who to remove"),
        (("people", "remove", "9"), "There is no person #9."),
        (("pending", "name", "1"), "Usage: `doorvouch pending name ID NAME`."),
        (("pending", "name", "9", "Carol"), "#9"),
        (("pending", "discard"), "Usage: `doorvouch pending discard ID`."),
        (("pending", "discard", "9"), "There is no pending visitor #9."),
    ],
)
def test_mistakes_are_explained_and_exit_with_2(run, capsys, arguments, message):
    assert run(*arguments) == 2
    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert message in error


def test_pending_visitors_are_listed_named_and_discarded(run, home, capsys):
    enroll_alice(home)
    with SqliteStore.open(home) as store:
        store.create_pending(axis(2), vouched_by=1, snapshot=b"jpeg")
        store.create_pending(axis(3), vouched_by=1)

    assert run("pending") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:3] == ["ID", "Vouched", "by"]
    assert lines[2].split()[:3] == ["1", "Alice", "1"]
    assert lines[2].endswith("pending-000001.jpg")
    assert lines[-1] == "Give one a name with `doorvouch pending name ID NAME`."

    assert run("pending", "name", "1", "Carol") == 0
    assert "Pending visitor #1 is now Carol (person #2)." in capsys.readouterr().out

    assert run("pending", "discard", "2") == 0
    assert "Pending visitor #2 was forgotten." in capsys.readouterr().out

    assert run("pending") == 0
    assert capsys.readouterr().out.strip() == "No pending visitors."
    with SqliteStore.open(home) as store:
        assert [summary.person.name for summary in store.list_people()] == ["Alice", "Carol"]
        assert not list((home / "snapshots").glob("*.jpg"))


def test_events_are_listed_newest_first(run, home, capsys):
    assert run("events") == 0
    assert capsys.readouterr().out.strip() == "No events yet."

    enroll_alice(home)
    with SqliteStore.open(home) as store:
        store.create_pending(axis(2), vouched_by=1)
        store.name_pending(1, "Carol")
        store.log_event("person_seen", person_id=1)

    assert run("events", "--limit", "1") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["When", "Event", "Person", "Pending", "Detail"]
    assert len(lines) == 3
    assert lines[2].split()[2:4] == ["person_seen", "Alice"]


def test_models_status_lists_what_is_missing(run, capsys):
    assert run("models") == 1
    output = capsys.readouterr().out
    for spec in models.ALL_MODELS:
        assert spec.filename in output
    assert "missing" in output
    assert "Run `doorvouch models download`" in output


def test_models_download_reports_each_file(run, home, monkeypatch, capsys):
    calls = []

    def download(spec, directory, *, force=False):
        calls.append((spec.key, directory, force))
        return models.Inspection(directory / spec.filename, None, 1234567, "ab" * 32)

    monkeypatch.setattr(models, "download", download)
    assert run("models", "download", "--force") == 0

    assert calls == [(spec.key, home / "models", True) for spec in models.ALL_MODELS]
    output = capsys.readouterr().out
    first = "face_detection_yunet_2026may.onnx ... ok (1,234,567 bytes, sha256 abababababababab)"
    assert first in output
    assert f"Models are in {home / 'models'}" in output


def test_ctrl_c_exits_with_130(run, monkeypatch, capsys):
    def interrupted(home):
        raise KeyboardInterrupt

    monkeypatch.setattr(SqliteStore, "open", classmethod(lambda cls, home: interrupted(home)))
    assert run("people") == 130
    assert "Interrupted." in capsys.readouterr().err


def test_doctor_reports_through_its_exit_code(run, home, monkeypatch):
    seen = []

    def fake_doctor(settings, *, camera_index):
        seen.append((settings.home, camera_index))
        return camera_index is None

    monkeypatch.setattr("doorvouch.apps.doctor.run_doctor", fake_doctor)

    assert run("doctor", "--no-camera") == 0
    assert run("doctor", "--camera", "2") == 1
    assert seen == [(home, None), (home, 2)]


# Commands that use the camera, with scripted footage -----------------------------


def test_the_whole_v0_story(run, home, camera, capsys):
    """Enroll an admin, vouch for a visitor, name the visitor, see them recognised."""
    camera([ALICE_ALONE] * 40)
    assert run("enroll", "--name", " Alice ", "--admin", "--samples", "3", "--no-window") == 0
    assert "Enrolled Alice (admin, person #1): 3 samples." in capsys.readouterr().out

    gestures = camera([VOUCHING] * 10)
    assert run("run", "--no-window") == 0
    output = capsys.readouterr().out
    assert "Camera 0 at 1280x720. Press Ctrl+C to stop." in output
    assert "Alice is in view" in output
    assert "Alice vouched for a visitor: pending #1 created" in output
    assert "10 frames in 1.1 s: 8.0 FPS." in output
    assert gestures.closed is True
    assert (home / "snapshots" / "pending-000001.jpg").read_bytes() == b"jpeg"

    assert run("pending", "name", "1", "Carol") == 0
    capsys.readouterr()

    camera([VOUCHING] * 10)
    assert run("run", "--no-window", "--seconds", "1") == 0
    output = capsys.readouterr().out
    assert "Carol is in view" in output
    assert "Alice gave a thumbs-up, but there is no visitor in view" in output
    assert "pending #" not in output

    assert run("events") == 0
    kinds = [line.split()[2] for line in capsys.readouterr().out.splitlines()[2:]]
    assert kinds == [
        "vouch_without_visitor",
        "person_seen",
        "person_seen",
        "pending_named",
        "pending_created",
        "person_seen",
        "person_enrolled",
    ]


def test_enrolling_again_adds_samples(run, home, camera, capsys):
    camera([ALICE_ALONE] * 40)
    assert run("enroll", "--name", "Alice", "--samples", "3", "--no-window") == 0
    assert run("enroll", "--name", "alice", "--samples", "4", "--no-window") == 0

    assert "Added samples to Alice (member, person #1): 4 samples." in capsys.readouterr().out
    with SqliteStore.open(home) as store:
        (summary,) = store.list_people()
        assert summary.samples == 7


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--name", "   "), "A person needs a name."),
        (("--name", "Alice", "--samples", "2"), "Capture at least 3 samples."),
    ],
)
def test_enroll_refuses_bad_arguments(run, camera, capsys, arguments, message):
    camera([ALICE_ALONE] * 40)
    assert run("enroll", *arguments, "--no-window") == 2
    assert message in capsys.readouterr().err


def test_enroll_fails_when_no_face_was_captured(run, home, camera, capsys):
    camera([NOBODY] * 20)
    assert run("enroll", "--name", "Alice", "--no-window") == 2
    assert "Only 0 usable face sample(s) were captured" in capsys.readouterr().err
    with SqliteStore.open(home) as store:
        assert store.list_people() == []


def test_commands_explain_missing_models(run, capsys):
    assert run("run", "--no-window") == 2
    error = capsys.readouterr().err
    assert "Model files are not ready" in error
    assert "Run `doorvouch models download`." in error


def test_run_warns_when_nobody_can_vouch(run, camera, capsys):
    camera([QUIET] * 4)
    assert run("run", "--no-window") == 0
    assert "warning: no admin is enrolled" in capsys.readouterr().err


def test_trial_needs_an_admin(run, camera, capsys):
    camera([VOUCHING] * 9)
    assert run("trial", "--scenario", "vouch", "--no-window") == 2
    assert "Enroll an admin first" in capsys.readouterr().err


def test_trial_writes_a_report_and_reports_acceptance(run, home, camera, tmp_path, capsys):
    enroll_alice(home)
    report = tmp_path / "report.md"
    options = ("--attempts", "2", "--seconds", "1", "--pause", "0", "--no-window")

    camera([VOUCHING] * 18)
    assert run("trial", "--scenario", "vouch", *options, "--output", str(report)) == 0
    output = capsys.readouterr().out
    assert "Attempt 1: pass (pending created: 1, thumbs-up ignored: 0, 8.0 FPS)" in output
    assert "Passed 2 of 2 attempts: accepted" in output
    assert "### Scenario: vouch" in report.read_text(encoding="utf-8")
    assert "Run on " in report.read_text(encoding="utf-8")

    camera([IMPOSTOR] * 18)
    assert run("trial", "--scenario", "impostor", *options, "--output", str(report)) == 0
    assert report.read_text(encoding="utf-8").count("### Scenario:") == 2

    camera([QUIET] * 18)
    assert run("trial", "--scenario", "impostor", *options) == 1
    assert "Void attempts: 2." in capsys.readouterr().out

    with SqliteStore.open(home) as store:
        assert store.list_pending() == []
