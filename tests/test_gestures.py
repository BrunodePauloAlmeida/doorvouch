from __future__ import annotations

import pytest

from doorvouch.core.gestures import GestureConfirmer

ADMIN = 7


def feed(confirmer: GestureConfirmer, pattern: str, *, start: float = 0.0, step: float = 0.1):
    """Feed one frame per character ('x' = gesture shown) and collect confirmations."""
    confirmed = []
    for index, mark in enumerate(pattern):
        active = [ADMIN] if mark == "x" else []
        if confirmer.observe(active, start + index * step):
            confirmed.append(index)
    return confirmed


def test_confirms_on_the_frame_that_completes_the_streak():
    assert feed(GestureConfirmer(required_frames=5), "xxxxxxx") == [4]


def test_a_shorter_streak_never_confirms():
    assert feed(GestureConfirmer(required_frames=5), "xxxx....") == []


def test_one_missed_frame_is_tolerated_by_default():
    assert feed(GestureConfirmer(required_frames=5), "xxx.xx") == [5]


def test_two_missed_frames_reset_the_streak():
    assert feed(GestureConfirmer(required_frames=5), "xxx..xxxx") == []


def test_no_gap_is_tolerated_when_configured_strict():
    confirmer = GestureConfirmer(required_frames=3, max_gap_frames=0)
    assert feed(confirmer, "xx.xx") == []


def test_holding_the_gesture_confirms_only_once():
    assert feed(GestureConfirmer(required_frames=3), "x" * 50) == [2]


def test_a_new_gesture_inside_the_cooldown_does_not_confirm():
    confirmer = GestureConfirmer(required_frames=3, cooldown_seconds=3.0)
    # 0.1 s per frame: the second streak completes at 1.0 s, inside the cooldown.
    assert feed(confirmer, "xxx....xxx") == [2]


def test_a_new_gesture_after_the_cooldown_confirms_again():
    confirmer = GestureConfirmer(required_frames=3, cooldown_seconds=0.5)
    assert feed(confirmer, "xxx....xxxx") == [2, 9]


def test_a_gesture_held_through_the_cooldown_confirms_when_it_ends():
    confirmer = GestureConfirmer(required_frames=2, max_gap_frames=0, cooldown_seconds=0.45)
    # Confirmed at frame 1 (0.1 s); dropped at frame 2; held again from frame 3.
    # The cooldown ends at 0.55 s, so frame 6 (0.6 s) is the first that may confirm.
    assert feed(confirmer, "xx.xxxxx") == [1, 6]


def test_keys_are_tracked_independently():
    confirmer = GestureConfirmer(required_frames=2)
    assert confirmer.observe([1], 0.0) == []
    assert confirmer.observe([1, 2], 0.1) == [1]
    assert confirmer.observe([2], 0.2) == [2]


def test_progress_grows_with_the_streak_and_resets():
    confirmer = GestureConfirmer(required_frames=4, max_gap_frames=0)
    assert confirmer.progress(ADMIN) == 0.0
    confirmer.observe([ADMIN], 0.0)
    confirmer.observe([ADMIN], 0.1)
    assert confirmer.progress(ADMIN) == pytest.approx(0.5)
    confirmer.observe([], 0.2)
    assert confirmer.progress(ADMIN) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"required_frames": 0}, {"max_gap_frames": -1}, {"cooldown_seconds": -1.0}],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GestureConfirmer(**kwargs)
