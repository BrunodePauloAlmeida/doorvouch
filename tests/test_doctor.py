from __future__ import annotations

from pathlib import Path

import pytest

from doorvouch import models
from doorvouch.apps import doctor
from doorvouch.config import Settings
from doorvouch.errors import DoorVouchError
from fakes import ScriptedFrame, ScriptedGestures, camera_playing


class HealthyFaces:
    def self_test(self) -> int:
        return 128


@pytest.fixture
def healthy(monkeypatch):
    """Pretend the packages, the models and the camera are all in place."""
    monkeypatch.setattr(doctor, "_check_package", lambda report, module, label: True)
    monkeypatch.setattr(
        models,
        "inspect",
        lambda spec, directory: models.Inspection(directory / spec.filename, None, 1234, "ab" * 32),
    )
    monkeypatch.setattr(
        "doorvouch.vision.opencv_faces.OpenCvFaceAnalyzer.from_models",
        classmethod(lambda cls, detector, recognizer: HealthyFaces()),
    )
    monkeypatch.setattr(
        "doorvouch.vision.mediapipe_gestures.MediaPipeGestureDetector.from_model",
        classmethod(lambda cls, model: ScriptedGestures()),
    )
    monkeypatch.setattr(doctor, "_blank_frame", ScriptedFrame)
    monkeypatch.setattr("doorvouch.vision.camera.Camera", camera_playing([ScriptedFrame()] * 40))


def examine(tmp_path: Path, **options) -> tuple[bool, list[str]]:
    lines: list[str] = []
    result = doctor.run_doctor(Settings(home=tmp_path / "home"), write=lines.append, **options)
    return result, lines


def test_a_healthy_computer_passes_every_check(tmp_path, healthy):
    result, lines = examine(tmp_path)

    assert result is True
    assert lines[-1] == "Everything looks good."
    assert not [line for line in lines if line.startswith(("[FAIL]", "[warn]"))]
    assert "[ ok ] Face detection and recognition run (embedding size 128)" in lines
    assert "[ ok ] Gesture recognition runs" in lines
    assert "[ ok ] Camera 0: 1280x720 at 8 FPS" in lines
    assert (tmp_path / "home").is_dir()


def test_missing_models_are_reported_with_the_fix(tmp_path):
    result, lines = examine(tmp_path, camera_index=None)

    assert result is False
    assert (
        "[FAIL] Model face_detection_yunet_2026may.onnx: missing. "
        "Run `doorvouch models download`." in lines
    )
    assert lines[-1].endswith("problem(s) found.")


def test_missing_packages_are_reported_not_raised(tmp_path, monkeypatch):
    real_import = doctor.importlib.import_module

    def without_vision(name: str, package: str | None = None):
        if name in {"cv2", "mediapipe"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, package)

    monkeypatch.setattr(doctor.importlib, "import_module", without_vision)
    result, lines = examine(tmp_path, camera_index=None)

    assert result is False
    assert any(line.startswith("[FAIL] OpenCV is not installed") for line in lines)
    assert any(line.startswith("[FAIL] MediaPipe is not installed") for line in lines)
    assert all("uv sync" in line for line in lines if "is not installed" in line)


def test_the_camera_check_can_be_skipped(tmp_path, healthy, monkeypatch):
    def unreachable(*_arguments, **_options):
        raise AssertionError("the camera must not be opened")

    monkeypatch.setattr("doorvouch.vision.camera.Camera", unreachable)
    result, lines = examine(tmp_path, camera_index=None)

    assert result is True
    assert not [line for line in lines if "Camera" in line]


def test_a_camera_that_does_not_open_is_a_failure(tmp_path, healthy, monkeypatch):
    def broken(index: int = 0, **_options):
        raise DoorVouchError(f"Could not open camera {index}.")

    monkeypatch.setattr("doorvouch.vision.camera.Camera", broken)
    result, lines = examine(tmp_path, camera_index=2)

    assert result is False
    assert "[FAIL] Camera 2: Could not open camera 2." in lines
    assert lines[-1] == "1 problem(s) found."


def test_a_low_resolution_camera_gets_a_warning(tmp_path, healthy, monkeypatch):
    low = camera_playing([ScriptedFrame()] * 40, resolution=(320, 240))
    monkeypatch.setattr("doorvouch.vision.camera.Camera", low)
    result, lines = examine(tmp_path)

    assert result is True
    assert "[warn] The camera resolution is low; faces may be too small to recognise." in lines


def test_a_crashing_model_is_reported_not_raised(tmp_path, healthy, monkeypatch):
    def explode(cls, detector, recognizer):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "doorvouch.vision.opencv_faces.OpenCvFaceAnalyzer.from_models", classmethod(explode)
    )
    result, lines = examine(tmp_path, camera_index=None)

    assert result is False
    assert "[FAIL] Face models crashed: RuntimeError: boom" in lines


def test_a_model_the_library_rejects_is_reported_with_its_message(tmp_path, healthy, monkeypatch):
    def reject(cls, model):
        raise DoorVouchError("MediaPipe could not load gesture_recognizer.task: bad file")

    monkeypatch.setattr(
        "doorvouch.vision.mediapipe_gestures.MediaPipeGestureDetector.from_model",
        classmethod(reject),
    )
    result, lines = examine(tmp_path, camera_index=None)

    assert result is False
    assert (
        "[FAIL] Gesture model: MediaPipe could not load gesture_recognizer.task: bad file" in lines
    )


def test_the_gesture_detector_is_closed_after_the_check(tmp_path, healthy, monkeypatch):
    detector = ScriptedGestures()
    monkeypatch.setattr(
        "doorvouch.vision.mediapipe_gestures.MediaPipeGestureDetector.from_model",
        classmethod(lambda cls, model: detector),
    )
    examine(tmp_path, camera_index=None)
    assert detector.closed is True


def test_accented_data_folder_gets_a_warning_on_windows_only(tmp_path):
    lines: list[str] = []
    report = doctor._Report(lines.append)
    home = tmp_path / "Résumé"

    doctor._check_home(home, report, windows=False)
    assert lines == [f"[ ok ] Data folder: {home}"]

    lines.clear()
    doctor._check_home(home, report, windows=True)
    assert lines[0] == f"[ ok ] Data folder: {home}"
    assert lines[1].startswith("[warn] The data folder path has accented characters.")
    assert report.failures == 0


def test_a_data_folder_that_cannot_be_created_is_a_failure(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    lines: list[str] = []
    report = doctor._Report(lines.append)

    doctor._check_home(blocker / "home", report)

    assert report.failures == 1
    assert lines[0].startswith("[FAIL] Cannot create the data folder")
