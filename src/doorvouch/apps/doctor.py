"""`doorvouch doctor`: check that this computer can run DoorVouch, and say what is wrong if not."""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

from doorvouch import __version__, models
from doorvouch.config import Settings
from doorvouch.errors import DoorVouchError
from doorvouch.fps import FpsMeter

_CAMERA_FRAMES = 30
_WARM_UP_FRAMES = 5


class _Report:
    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self.failures = 0

    def ok(self, message: str) -> None:
        self._write(f"[ ok ] {message}")

    def warn(self, message: str) -> None:
        self._write(f"[warn] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        self._write(f"[FAIL] {message}")


def run_doctor(
    settings: Settings,
    *,
    camera_index: int | None = 0,
    write: Callable[[str], None] = print,
) -> bool:
    """Run every check and report through ``write``. Returns ``True`` when nothing failed.

    Pass ``camera_index=None`` to skip the camera.
    """
    report = _Report(write)
    report.ok(f"DoorVouch {__version__}, Python {platform.python_version()}, {platform.system()}")
    _check_home(settings.home, report)
    has_opencv = _check_package(report, "cv2", "OpenCV")
    has_mediapipe = _check_package(report, "mediapipe", "MediaPipe")
    paths = _check_models(settings.models_dir, report)
    if paths is not None and has_opencv:
        _check_faces(paths, report)
    if paths is not None and has_opencv and has_mediapipe:
        _check_gestures(paths, report)
    if camera_index is not None and has_opencv:
        _check_camera(camera_index, report)

    if report.failures:
        write(f"{report.failures} problem(s) found.")
    else:
        write("Everything looks good.")
    return report.failures == 0


def _check_home(home: Path, report: _Report, *, windows: bool = os.name == "nt") -> None:
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        report.fail(f"Cannot create the data folder {home}: {error}")
        return
    report.ok(f"Data folder: {home}")
    if windows and not str(home).isascii():
        report.warn(
            "The data folder path has accented characters. If a model fails to load, "
            "point DOORVOUCH_HOME at a simple folder such as C:\\doorvouch."
        )


def _check_package(report: _Report, module: str, label: str) -> bool:
    try:
        imported = importlib.import_module(module)
    except ImportError as error:
        report.fail(f"{label} is not installed ({error}). Run `uv sync` in the project folder.")
        return False
    report.ok(f"{label} {getattr(imported, '__version__', 'unknown version')}")
    return True


def _check_models(directory: Path, report: _Report) -> dict[str, Path] | None:
    paths: dict[str, Path] = {}
    for spec in models.ALL_MODELS:
        found = models.inspect(spec, directory)
        if found.ok:
            paths[spec.key] = found.path
            report.ok(f"Model {spec.filename} ({found.size:,} bytes)")
        else:
            report.fail(f"Model {spec.filename}: {found.problem}. Run `doorvouch models download`.")
    return paths if len(paths) == len(models.ALL_MODELS) else None


def _check_faces(paths: dict[str, Path], report: _Report) -> None:
    from doorvouch.vision.opencv_faces import OpenCvFaceAnalyzer

    try:
        analyzer = OpenCvFaceAnalyzer.from_models(paths["face_detector"], paths["face_recognizer"])
        size = analyzer.self_test()
    except DoorVouchError as error:
        report.fail(f"Face models: {error}")
    except Exception as error:  # a diagnosis tool must report, not crash
        report.fail(f"Face models crashed: {type(error).__name__}: {error}")
    else:
        report.ok(f"Face detection and recognition run (embedding size {size})")


def _check_gestures(paths: dict[str, Path], report: _Report) -> None:
    from doorvouch.vision.mediapipe_gestures import MediaPipeGestureDetector

    try:
        detector = MediaPipeGestureDetector.from_model(paths["gesture_recognizer"])
        try:
            detector.detect(_blank_frame(), 0.0)
        finally:
            detector.close()
    except DoorVouchError as error:
        report.fail(f"Gesture model: {error}")
    except Exception as error:  # a diagnosis tool must report, not crash
        report.fail(f"Gesture model crashed: {type(error).__name__}: {error}")
    else:
        report.ok("Gesture recognition runs")


def _check_camera(index: int, report: _Report) -> None:
    from doorvouch.vision.camera import Camera

    try:
        with Camera(index) as camera:
            width, height = camera.resolution
            meter = FpsMeter(window=_CAMERA_FRAMES)
            for count, (timestamp, _frame) in enumerate(camera.frames(), start=1):
                if count > _WARM_UP_FRAMES:
                    meter.tick(timestamp)
                if count >= _CAMERA_FRAMES + _WARM_UP_FRAMES:
                    break
    except DoorVouchError as error:
        report.fail(f"Camera {index}: {error}")
    except Exception as error:  # a diagnosis tool must report, not crash
        report.fail(f"Camera {index} crashed: {type(error).__name__}: {error}")
    else:
        report.ok(f"Camera {index}: {width}x{height} at {meter.fps:.0f} FPS")
        if width < 640:
            report.warn("The camera resolution is low; faces may be too small to recognise.")


def _blank_frame() -> Any:
    import numpy

    return numpy.zeros((480, 640, 3), dtype=numpy.uint8)
