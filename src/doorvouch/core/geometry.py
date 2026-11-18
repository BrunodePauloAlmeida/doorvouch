"""Small geometric helpers shared by the core and the vision adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned rectangle in pixel coordinates, origin at the top-left corner."""

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Point:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def detection_scale(width: int, height: int, max_side: int) -> float:
    """Factor that shrinks a frame so its longest side is at most ``max_side``.

    Frames that already fit are left alone (the factor is never above 1.0).
    """
    longest = max(width, height)
    if longest <= 0 or max_side <= 0:
        raise ValueError("frame size and max_side must be positive")
    return min(1.0, max_side / longest)


def crop_region(
    box: Box, width: int, height: int, *, margin: float = 0.25
) -> tuple[int, int, int, int] | None:
    """Integer ``(left, top, right, bottom)`` of ``box`` grown by ``margin`` and clamped.

    ``margin`` is a fraction of the box size added on every side. Returns ``None``
    when nothing of the box lies inside a ``width`` x ``height`` image.
    """
    if margin < 0:
        raise ValueError("margin must not be negative")
    left = math.floor(box.x - box.w * margin)
    top = math.floor(box.y - box.h * margin)
    right = math.ceil(box.x + box.w * (1.0 + margin))
    bottom = math.ceil(box.y + box.h * (1.0 + margin))
    left, top = max(left, 0), max(top, 0)
    right, bottom = min(right, width), min(bottom, height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
