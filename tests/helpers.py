"""Builders for synthetic observations. No real face ever appears in the tests."""

from __future__ import annotations

import math

from doorvouch.core.embeddings import Embedding, normalize
from doorvouch.core.geometry import Box
from doorvouch.core.observations import FaceObservation, FrameObservation, HandObservation

DIMENSIONS = 8


def axis(index: int) -> Embedding:
    """A unit embedding along one axis. Different axes have similarity 0."""
    return tuple(1.0 if i == index else 0.0 for i in range(DIMENSIONS))


def toward(base: Embedding, other: Embedding, similarity: float) -> Embedding:
    """A unit embedding whose cosine similarity to ``base`` is ``similarity``.

    ``base`` and ``other`` must be orthogonal unit vectors.
    """
    rest = math.sqrt(1.0 - similarity * similarity)
    return normalize(similarity * b + rest * o for b, o in zip(base, other, strict=True))


def face(
    embedding: Embedding, *, x: float, y: float = 100.0, width: float = 100.0, score: float = 0.95
) -> FaceObservation:
    return FaceObservation(Box(x, y, width, width * 1.3), embedding, score)


def hand_below(target: FaceObservation, *, gesture: str | None = "Thumb_Up", score: float = 0.9):
    """A hand held in front of the chest of ``target``: about 1.8 face widths below the face."""
    center_x, center_y = target.box.center
    return HandObservation((center_x, center_y + 1.8 * target.box.w), gesture, score)


def frame(timestamp: float, faces=(), hands=()) -> FrameObservation:
    return FrameObservation(timestamp, tuple(faces), tuple(hands))
