"""Decide whose hand is making a gesture."""

from __future__ import annotations

from collections.abc import Sequence

from doorvouch.core.geometry import Box, Point, distance


def find_owner(
    hand: Point,
    faces: Sequence[Box],
    *,
    max_reach: float = 4.0,
    ambiguity_ratio: float = 0.8,
) -> int | None:
    """Index of the face a hand most plausibly belongs to, or ``None``.

    Distance runs from the hand to each face centre and is measured in widths of
    that face, which makes it roughly independent of how far the person stands
    from the camera. A hand further than ``max_reach`` face widths from every
    face belongs to nobody.

    When two faces are both within reach and the nearest is not clearly nearer
    (its distance is more than ``ambiguity_ratio`` times the runner-up's), the
    answer is ``None``: a thumbs-up carries authority, so a doubtful owner must
    not be guessed. See ``docs/decisions/0002-gesture-to-face-association.md``.
    """
    if max_reach <= 0:
        raise ValueError("max_reach must be positive")
    if not 0.0 < ambiguity_ratio <= 1.0:
        raise ValueError("ambiguity_ratio must be in (0, 1]")
    in_reach = sorted(
        (distance(hand, box.center) / box.w, index)
        for index, box in enumerate(faces)
        if box.w > 0 and distance(hand, box.center) / box.w <= max_reach
    )
    if not in_reach:
        return None
    nearest, owner = in_reach[0]
    if len(in_reach) > 1 and nearest > ambiguity_ratio * in_reach[1][0]:
        return None
    return owner
