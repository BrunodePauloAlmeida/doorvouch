"""The preview window: boxes around faces, names, frame rate and the latest event."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from doorvouch.core.engine import FrameResult, IdentifiedFace
from doorvouch.core.observations import FaceObservation
from doorvouch.text import ascii_label
from doorvouch.vision.base import Frame, load_cv2

# Colours are blue, green, red, as OpenCV expects.
_COLORS = {
    "admin": (90, 200, 90),
    "known": (90, 200, 90),
    "pending": (0, 200, 255),
    "uncertain": (0, 140, 255),
    "unknown": (70, 70, 230),
}
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_BANNER_SECONDS = 4.0


def describe(face: IdentifiedFace) -> str:
    """The caption drawn above a face."""
    role = " (admin)" if face.status == "admin" else ""
    return ascii_label(f"{face.label}{role} {face.similarity:.2f}")


def draw_faces(cv2: Any, canvas: Frame, faces: Sequence[IdentifiedFace]) -> None:
    for face in faces:
        _draw_box(cv2, canvas, face.observation, describe(face), _COLORS[face.status])


def draw_lines(cv2: Any, canvas: Frame, lines: Sequence[str], *, bottom: bool = False) -> None:
    """Draw text lines in the top-left corner, or in the bottom-left one."""
    height = canvas.shape[0]
    for index, line in enumerate(lines):
        y = height - 16 - 28 * (len(lines) - 1 - index) if bottom else 28 + 28 * index
        _draw_text(cv2, canvas, ascii_label(line), (12, y), _WHITE)


def _draw_box(
    cv2: Any, canvas: Frame, face: FaceObservation, caption: str, color: tuple[int, int, int]
) -> None:
    box = face.box
    left, top = round(box.x), round(box.y)
    right, bottom = round(box.x + box.w), round(box.y + box.h)
    cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
    _draw_text(cv2, canvas, caption, (left, max(20, top - 8)), color)


def _draw_text(
    cv2: Any, canvas: Frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, origin, font, 0.6, _BLACK, 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, font, 0.6, color, 1, cv2.LINE_AA)


class OpenCvDisplay:
    """A window showing the camera with DoorVouch's conclusions drawn on top.

    Press ``q`` or ``Esc``, or close the window, to stop.
    """

    def __init__(self, title: str = "DoorVouch") -> None:
        self._cv2 = load_cv2()
        self._title = title
        self._shown = False
        self._banner = ""
        self._banner_until = 0.0

    def announce(self, message: str) -> None:
        """Show a message at the bottom of the window for a few seconds."""
        self._banner = message
        self._banner_until = time.monotonic() + _BANNER_SECONDS

    def show(self, frame: Frame, result: FrameResult, fps: float) -> bool:
        lines = [f"{fps:4.1f} FPS"]
        for progress in result.gesture_progress.values():
            lines.append(f"Thumbs-up {progress:.0%}")
        return self.show_faces(frame, result.faces, lines)

    def show_faces(
        self, frame: Frame, faces: Sequence[IdentifiedFace], lines: Sequence[str]
    ) -> bool:
        """Draw and display one frame. Returns ``False`` when the user asked to stop."""
        cv2 = self._cv2
        canvas = frame.copy()
        draw_faces(cv2, canvas, faces)
        draw_lines(cv2, canvas, lines)
        if self._banner and time.monotonic() < self._banner_until:
            draw_lines(cv2, canvas, [self._banner], bottom=True)
        cv2.imshow(self._title, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        if self._shown and cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) < 1:
            return False
        self._shown = True
        return True

    def close(self) -> None:
        self._cv2.destroyAllWindows()
