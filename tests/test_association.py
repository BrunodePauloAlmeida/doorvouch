from __future__ import annotations

import pytest

from doorvouch.core.association import find_owner
from doorvouch.core.geometry import Box

# Two people side by side: face centres 350 px apart, faces 100 px wide.
LEFT = Box(100, 100, 100, 130)  # centre (150, 165)
RIGHT = Box(450, 100, 100, 130)  # centre (500, 165)


def test_no_faces_means_no_owner():
    assert find_owner((100, 100), []) is None


def test_hand_in_front_of_the_chest_belongs_to_that_person():
    assert find_owner((150, 345), [LEFT, RIGHT]) == 0
    assert find_owner((500, 345), [LEFT, RIGHT]) == 1


def test_hand_out_of_reach_of_everyone_belongs_to_nobody():
    assert find_owner((150, 900), [LEFT, RIGHT]) is None


def test_reach_is_measured_in_face_widths():
    near_camera = Box(0, 0, 200, 260)  # centre (100, 130)
    far_from_camera = Box(0, 0, 50, 65)  # centre (25, 32.5)
    assert find_owner((100, 730), [near_camera]) == 0  # 600 px = 3 widths of 200
    assert find_owner((25, 632.5), [far_from_camera]) is None  # 600 px = 12 widths of 50


def test_hand_halfway_between_two_people_is_ambiguous():
    assert find_owner((325, 300), [LEFT, RIGHT]) is None


def test_hand_slightly_towards_one_person_is_still_ambiguous():
    # 1.95 widths from LEFT and 2.36 from RIGHT: a ratio of 0.83, not clearly nearer.
    assert find_owner((300, 290), [LEFT, RIGHT]) is None


def test_hand_clearly_nearer_to_one_person_is_theirs():
    # 1.72 widths from LEFT and 2.61 from RIGHT: a ratio of 0.66.
    assert find_owner((270, 290), [LEFT, RIGHT]) == 0


def test_ambiguity_can_be_disabled():
    assert find_owner((300, 290), [LEFT, RIGHT], ambiguity_ratio=1.0) == 0


def test_a_second_face_out_of_reach_does_not_make_the_hand_ambiguous():
    far_right = Box(1200, 100, 100, 130)
    assert find_owner((300, 290), [LEFT, far_right]) == 0


def test_a_smaller_face_further_back_needs_the_hand_closer():
    behind = Box(300, 120, 50, 65)  # centre (325, 152.5), half the width of LEFT
    # 1.8 widths from LEFT (180 px) but 4.4 widths from the face behind (220 px).
    assert find_owner((150, 345), [LEFT, behind]) == 0


def test_faces_with_zero_width_are_skipped():
    assert find_owner((150, 345), [Box(150, 165, 0, 0), LEFT]) == 1


@pytest.mark.parametrize(
    "kwargs", [{"max_reach": 0}, {"ambiguity_ratio": 0}, {"ambiguity_ratio": 1.5}]
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        find_owner((0, 0), [LEFT], **kwargs)
