"""The live loop: camera frames in, decisions and events out."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from doorvouch.core.engine import EngineEvent, FrameResult, IdentifiedFace, VouchingEngine
from doorvouch.core.geometry import Box
from doorvouch.core.observations import FrameObservation
from doorvouch.fps import FpsMeter
from doorvouch.vision.base import FaceAnalyzer, Frame, GestureDetector

Frames = Iterable[tuple[float, Frame]]
Snapshotter = Callable[[Frame, Box], bytes | None]


class Display(Protocol):
    """Somewhere to show frames. ``show`` returns ``False`` when the user wants to stop."""

    def show(self, frame: Frame, result: FrameResult, fps: float) -> bool: ...

    def show_faces(
        self, frame: Frame, faces: Sequence[IdentifiedFace], lines: Sequence[str]
    ) -> bool: ...

    def announce(self, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class LiveStats:
    """Summary of one run of the live loop."""

    frames: int = 0
    seconds: float = 0.0
    fps: float = 0.0
    events: dict[str, int] = field(default_factory=dict)
    stopped_by_user: bool = False
    # True when the loop ended because ``max_seconds`` passed, not because the
    # frames ran out or the user stopped.
    timed_out: bool = False


def run_live(
    frames: Frames,
    *,
    faces: FaceAnalyzer,
    gestures: GestureDetector,
    engine: VouchingEngine,
    snapshot: Snapshotter | None = None,
    display: Display | None = None,
    on_event: Callable[[EngineEvent], None] | None = None,
    max_seconds: float | None = None,
    refresh_seconds: float = 2.0,
) -> LiveStats:
    """Process frames until they run out, ``max_seconds`` pass or the user stops.

    Every ``refresh_seconds`` the engine checks whether another process changed
    the database, so people enrolled or named in a second terminal show up
    without a restart.
    """
    meter = FpsMeter()
    counts: Counter[str] = Counter()
    processed = 0
    first: float | None = None
    last = 0.0
    refreshed = 0.0
    stopped = False
    timed_out = False

    for timestamp, frame in frames:
        if first is None:
            first = refreshed = timestamp
        if max_seconds is not None and timestamp - first >= max_seconds:
            timed_out = True
            break
        if timestamp - refreshed >= refresh_seconds:
            engine.refresh_if_changed()
            refreshed = timestamp

        observation = FrameObservation(
            timestamp,
            tuple(faces.analyze(frame)),
            tuple(gestures.detect(frame, timestamp)),
        )
        take_snapshot = None
        if snapshot is not None:
            take_snapshot = _bind(snapshot, frame)
        result = engine.process(observation, snapshot=take_snapshot)

        processed += 1
        last = timestamp
        meter.tick(timestamp)
        for event in result.events:
            counts[event.kind] += 1
            if on_event is not None:
                on_event(event)
            if display is not None:
                display.announce(event.message)
        if display is not None and not display.show(frame, result, meter.fps):
            stopped = True
            break

    seconds = last - first if first is not None and processed > 1 else 0.0
    fps = (processed - 1) / seconds if seconds > 0 else 0.0
    return LiveStats(processed, seconds, fps, dict(counts), stopped, timed_out)


def _bind(snapshot: Snapshotter, frame: Frame) -> Callable[[Box], bytes | None]:
    def take(box: Box) -> bytes | None:
        return snapshot(frame, box)

    return take
