"""Interfaces the applications depend on, so models can be swapped or faked."""

from __future__ import annotations

from typing import Any, Protocol

from doorvouch.core.observations import FaceObservation, HandObservation
from doorvouch.errors import DoorVouchError

# A camera frame: a BGR image as OpenCV delivers it (a NumPy array).
Frame = Any


class FaceAnalyzer(Protocol):
    def analyze(self, frame: Frame) -> list[FaceObservation]:
        """Detect the faces in a frame and compute an embedding for each."""
        ...


class GestureDetector(Protocol):
    def detect(self, frame: Frame, timestamp: float) -> list[HandObservation]:
        """Detect the hands in a frame and the gesture each one shows."""
        ...


def load_cv2() -> Any:
    """Import OpenCV, or explain how to install it."""
    try:
        import cv2
    except ImportError as error:
        raise DoorVouchError(
            "OpenCV is not installed. Run `uv sync` in the project folder."
        ) from error
    return cv2
