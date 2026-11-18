"""Face embeddings as plain tuples of floats, compared by cosine similarity."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Embedding = tuple[float, ...]


def normalize(vector: Iterable[float]) -> Embedding:
    """Return ``vector`` scaled to unit length.

    Raises ``ValueError`` for an empty vector or one whose length is zero or not
    finite, because such a vector cannot stand for a face.
    """
    values = tuple(float(v) for v in vector)
    if not values:
        raise ValueError("embedding is empty")
    norm = math.sqrt(math.fsum(v * v for v in values))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("embedding has a zero or non-finite length")
    return tuple(v / norm for v in values)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between ``a`` and ``b``, from -1.0 to 1.0."""
    if len(a) != len(b):
        raise ValueError(f"embeddings differ in size: {len(a)} and {len(b)}")
    if not a:
        raise ValueError("embedding is empty")
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("embedding has a zero length")
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def mean_embedding(vectors: Iterable[Sequence[float]]) -> Embedding:
    """Unit-length average of several embeddings of the same face."""
    rows = [tuple(v) for v in vectors]
    if not rows:
        raise ValueError("no embeddings to average")
    size = len(rows[0])
    if any(len(row) != size for row in rows):
        raise ValueError("embeddings differ in size")
    return normalize(math.fsum(row[i] for row in rows) / len(rows) for i in range(size))
