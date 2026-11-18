"""Stand-ins for the camera, the models and the preview window."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptedFrame:
    """A "camera frame" that already knows what the models would see in it."""

    faces: tuple = ()
    hands: tuple = ()
    shape: tuple[int, int, int] = (720, 1280, 3)


class ScriptedFaces:
    def analyze(self, frame: ScriptedFrame):
        return list(frame.faces)


class ScriptedGestures:
    def __init__(self) -> None:
        self.closed = False

    def detect(self, frame: ScriptedFrame, timestamp: float):
        return list(frame.hands)

    def close(self) -> None:
        self.closed = True


def clip(
    frames: Iterable[ScriptedFrame], *, fps: float = 8.0, start: float = 100.0
) -> Iterator[tuple[float, ScriptedFrame]]:
    """Timestamp frames as a camera running at ``fps`` would.

    Keep ``fps`` a power of two: the timestamps are then exact binary fractions,
    so a test that expects "the frame one second later" is not at the mercy of
    floating-point rounding.
    """
    for index, frame in enumerate(frames):
        yield start + index / fps, frame


def camera_playing(
    frames: Iterable[ScriptedFrame], *, fps: float = 8.0, resolution: tuple[int, int] = (1280, 720)
) -> type:
    """A replacement for ``doorvouch.vision.camera.Camera`` that plays ``frames``."""
    recorded = list(frames)

    class ScriptedCamera:
        def __init__(self, index: int = 0, **_options: object) -> None:
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *_exception: object) -> None:
            return None

        @property
        def resolution(self) -> tuple[int, int]:
            return resolution

        def frames(self) -> Iterator[tuple[float, ScriptedFrame]]:
            return clip(recorded, fps=fps)

    return ScriptedCamera


class RecordingDisplay:
    """Remembers what it was asked to show; can ask to stop after some frames."""

    def __init__(self, *, stop_after: int | None = None) -> None:
        self.results = []
        self.captions = []
        self.banners = []
        self._stop_after = stop_after
        self._calls = 0

    def show(self, frame, result, fps: float) -> bool:
        self.results.append(result)
        return self._keep_going()

    def show_faces(self, frame, faces, lines) -> bool:
        self.captions.append(list(lines))
        return self._keep_going()

    def announce(self, message: str) -> None:
        self.banners.append(message)

    def _keep_going(self) -> bool:
        self._calls += 1
        return self._stop_after is None or self._calls < self._stop_after
