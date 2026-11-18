"""Turn per-frame gesture sightings into deliberate confirmations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

# Label MediaPipe's built-in gesture classifier gives to a thumbs-up.
THUMB_UP = "Thumb_Up"


@dataclass(slots=True)
class _Streak:
    frames: int = 0
    gap: int = 0
    fired: bool = False


class GestureConfirmer:
    """Confirm a gesture once it has been held for several frames in a row.

    A single frame is not enough: detectors flicker, and a hand passing by must
    not count. Each key (an admin's id) confirms at most once per sustained
    gesture, and not again until ``cooldown_seconds`` have passed.
    """

    def __init__(
        self,
        *,
        required_frames: int = 5,
        max_gap_frames: int = 1,
        cooldown_seconds: float = 3.0,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames must not be negative")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        self._required = required_frames
        self._max_gap = max_gap_frames
        self._cooldown = cooldown_seconds
        self._streaks: dict[int, _Streak] = {}
        self._cooldown_until: dict[int, float] = {}

    def observe(self, active: Collection[int], timestamp: float) -> list[int]:
        """Record which keys show the gesture in this frame.

        Returns the keys whose gesture became confirmed in this very frame.
        """
        current = set(active)
        confirmed: list[int] = []
        for key in sorted(current):
            streak = self._streaks.setdefault(key, _Streak())
            streak.frames += 1
            streak.gap = 0
            if streak.fired or streak.frames < self._required:
                continue
            if timestamp < self._cooldown_until.get(key, float("-inf")):
                continue
            streak.fired = True
            self._cooldown_until[key] = timestamp + self._cooldown
            confirmed.append(key)
        for key in [k for k in self._streaks if k not in current]:
            streak = self._streaks[key]
            streak.gap += 1
            if streak.gap > self._max_gap:
                del self._streaks[key]
        return confirmed

    def progress(self, key: int) -> float:
        """How far ``key`` is towards a confirmation, from 0.0 to 1.0."""
        streak = self._streaks.get(key)
        if streak is None:
            return 0.0
        return min(1.0, streak.frames / self._required)
