"""Frames-per-second measurement over a sliding window."""

from __future__ import annotations

from collections import deque


class FpsMeter:
    """Average frame rate over the last ``window`` frames."""

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self._stamps: deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float) -> None:
        self._stamps.append(timestamp)

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._stamps) - 1) / elapsed
