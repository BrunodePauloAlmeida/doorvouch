"""What the vision adapters report for one camera frame."""

from __future__ import annotations

from dataclasses import dataclass

from doorvouch.core.embeddings import Embedding
from doorvouch.core.geometry import Box, Point


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One detected face: where it is, its unit-length embedding and the detector score."""

    box: Box
    embedding: Embedding
    score: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    """One detected hand: its centre in pixels and the recognised gesture, if any."""

    center: Point
    gesture: str | None
    score: float


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Everything seen in one frame. ``timestamp`` is in seconds on a monotonic clock."""

    timestamp: float
    faces: tuple[FaceObservation, ...] = ()
    hands: tuple[HandObservation, ...] = ()
