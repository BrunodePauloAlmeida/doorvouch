"""A webcam opened through OpenCV."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Self

from doorvouch.errors import DoorVouchError
from doorvouch.vision.base import Frame, load_cv2

_MAX_CONSECUTIVE_MISSES = 30


def _backends(cv2: Any) -> list[int]:
    """Capture backends to try, in order."""
    if os.name == "nt":
        # DirectShow opens webcams much faster than the default backend on Windows.
        return [cv2.CAP_DSHOW, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


class Camera:
    """Yields ``(timestamp, frame)`` pairs; the timestamp is ``time.monotonic()`` at capture."""

    def __init__(self, index: int = 0, *, width: int = 1280, height: int = 720) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cv2: Any = None
        self._capture: Any = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        cv2 = load_cv2()
        capture = None
        for backend in _backends(cv2):
            candidate = cv2.VideoCapture(self._index, backend)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is None:
            raise DoorVouchError(
                f"Could not open camera {self._index}. Check that it is connected and that no "
                "other program is using it, or try another one with --camera 1."
            )
        # MJPG lets USB webcams deliver 720p at full frame rate.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cv2 = cv2
        self._capture = capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Width and height the camera actually delivers, which may differ from the request."""
        if self._capture is None:
            raise DoorVouchError("The camera is not open.")
        cv2 = self._cv2
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self) -> Iterator[tuple[float, Frame]]:
        if self._capture is None:
            raise DoorVouchError("The camera is not open.")
        misses = 0
        while True:
            grabbed, frame = self._capture.read()
            if not grabbed or frame is None:
                misses += 1
                if misses >= _MAX_CONSECUTIVE_MISSES:
                    raise DoorVouchError("The camera stopped delivering frames.")
                time.sleep(0.01)
                continue
            misses = 0
            yield time.monotonic(), frame
