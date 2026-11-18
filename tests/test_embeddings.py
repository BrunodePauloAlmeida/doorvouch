from __future__ import annotations

import math

import pytest

from doorvouch.core.embeddings import cosine_similarity, mean_embedding, normalize


def test_normalize_returns_a_unit_vector():
    unit = normalize([3.0, 4.0])
    assert unit == pytest.approx((0.6, 0.8))
    assert math.sqrt(sum(v * v for v in unit)) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [[], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_normalize_rejects_vectors_that_cannot_be_a_face(bad):
    with pytest.raises(ValueError):
        normalize(bad)


def test_cosine_similarity_of_known_angles():
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)
    assert cosine_similarity((1.0, 0.0), (1.0, 1.0)) == pytest.approx(math.sqrt(0.5))


def test_cosine_similarity_ignores_vector_length():
    assert cosine_similarity((2.0, 0.0), (0.5, 0.0)) == pytest.approx(1.0)


def test_cosine_similarity_rejects_mismatched_or_empty_vectors():
    with pytest.raises(ValueError, match="differ in size"):
        cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="empty"):
        cosine_similarity((), ())
    with pytest.raises(ValueError, match="zero length"):
        cosine_similarity((0.0, 0.0), (1.0, 0.0))


def test_mean_embedding_sits_between_its_inputs_and_has_unit_length():
    mean = mean_embedding([(1.0, 0.0), (0.0, 1.0)])
    assert mean == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))


def test_mean_embedding_rejects_empty_or_ragged_input():
    with pytest.raises(ValueError, match="no embeddings"):
        mean_embedding([])
    with pytest.raises(ValueError, match="differ in size"):
        mean_embedding([(1.0, 0.0), (1.0, 0.0, 0.0)])
