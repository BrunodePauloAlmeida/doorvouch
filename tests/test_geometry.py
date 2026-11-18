from __future__ import annotations

import pytest

from doorvouch.core.geometry import Box, crop_region, detection_scale, distance


def test_box_center():
    assert Box(10, 20, 100, 50).center == (60.0, 45.0)


def test_distance():
    assert distance((0, 0), (3, 4)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(1280, 720, 0.5), (1920, 1080, 1 / 3), (640, 480, 1.0), (320, 240, 1.0), (480, 640, 1.0)],
)
def test_detection_scale_shrinks_only_frames_larger_than_the_limit(width, height, expected):
    assert detection_scale(width, height, 640) == pytest.approx(expected)


def test_detection_scale_rejects_nonsense_sizes():
    with pytest.raises(ValueError):
        detection_scale(0, 0, 640)
    with pytest.raises(ValueError):
        detection_scale(640, 480, 0)


def test_crop_region_adds_a_margin_around_the_box():
    assert crop_region(Box(100, 100, 100, 200), 1000, 1000, margin=0.25) == (75, 50, 225, 350)


def test_crop_region_is_clamped_to_the_image():
    assert crop_region(Box(-20, -10, 100, 100), 150, 120, margin=0.5) == (0, 0, 130, 120)


def test_crop_region_is_none_when_the_box_is_outside_the_image():
    assert crop_region(Box(2000, 100, 50, 50), 640, 480) is None
    assert crop_region(Box(-500, 100, 50, 50), 640, 480) is None


def test_crop_region_rejects_a_negative_margin():
    with pytest.raises(ValueError):
        crop_region(Box(0, 0, 10, 10), 100, 100, margin=-0.1)
