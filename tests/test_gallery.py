from __future__ import annotations

import pytest

from doorvouch.core.gallery import SFACE_COSINE_THRESHOLD, Gallery
from doorvouch.core.people import Person
from helpers import axis, toward

ALICE = Person(1, "Alice", is_admin=True)
RUTH = Person(2, "Ruth")


def test_empty_gallery_matches_nobody():
    match = Gallery([]).match(axis(0))
    assert match.person is None
    assert match.similarity == -1.0


def test_identical_embedding_matches_its_person():
    gallery = Gallery([(ALICE, axis(0)), (RUTH, axis(1))])
    match = gallery.match(axis(1))
    assert match.person == RUTH
    assert match.similarity == pytest.approx(1.0)


def test_similarity_exactly_at_the_threshold_is_a_match():
    gallery = Gallery([(ALICE, axis(0))], threshold=0.5)
    probe = toward(axis(0), axis(1), 0.5)
    assert gallery.match(probe).person == ALICE


def test_similarity_just_below_the_threshold_is_not_a_match_but_is_reported():
    gallery = Gallery([(ALICE, axis(0))])
    probe = toward(axis(0), axis(1), SFACE_COSINE_THRESHOLD - 0.01)
    match = gallery.match(probe)
    assert match.person is None
    assert match.similarity == pytest.approx(SFACE_COSINE_THRESHOLD - 0.01)


def test_best_of_several_samples_of_the_same_person_wins():
    frontal, profile = axis(0), axis(1)
    gallery = Gallery([(ALICE, frontal), (ALICE, profile)])
    probe = toward(profile, axis(2), 0.9)
    match = gallery.match(probe)
    assert match.person == ALICE
    assert match.similarity == pytest.approx(0.9)


def test_the_more_similar_person_wins():
    gallery = Gallery([(ALICE, axis(0)), (RUTH, toward(axis(0), axis(1), 0.6))])
    assert gallery.match(axis(0)).person == ALICE


def test_people_lists_each_person_once():
    gallery = Gallery([(ALICE, axis(0)), (ALICE, axis(1)), (RUTH, axis(2))])
    assert gallery.people == {1: ALICE, 2: RUTH}
    assert len(gallery) == 3


def test_threshold_must_be_a_valid_cosine():
    with pytest.raises(ValueError):
        Gallery([], threshold=1.5)
