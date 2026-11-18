"""Match a face embedding against the enrolled people."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from doorvouch.core.embeddings import Embedding, cosine_similarity
from doorvouch.core.people import Person

# Cosine similarity at or above which OpenCV's SFace tutorial treats two faces
# as the same identity. Calibrate it for your camera in the V0 acceptance run.
SFACE_COSINE_THRESHOLD = 0.363


@dataclass(frozen=True, slots=True)
class Match:
    """Result of a gallery lookup.

    ``person`` is ``None`` when nobody is similar enough. ``similarity`` is the
    best score found, or -1.0 for an empty gallery.
    """

    person: Person | None
    similarity: float


class Gallery:
    """Enrolled face samples, searched by brute force.

    A household has a handful of people and a few samples each, so comparing
    against every sample is fast enough and keeps the core free of dependencies.
    """

    def __init__(
        self,
        samples: Iterable[tuple[Person, Embedding]],
        *,
        threshold: float = SFACE_COSINE_THRESHOLD,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0")
        self._samples = list(samples)
        self._threshold = threshold

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def people(self) -> dict[int, Person]:
        return {person.id: person for person, _ in self._samples}

    def match(self, embedding: Embedding) -> Match:
        best_person: Person | None = None
        best = -1.0
        for person, sample in self._samples:
            similarity = cosine_similarity(embedding, sample)
            if similarity > best:
                best_person, best = person, similarity
        if best_person is not None and best >= self._threshold:
            return Match(best_person, best)
        return Match(None, best)
