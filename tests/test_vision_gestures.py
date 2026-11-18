"""The MediaPipe gesture adapter, driven by a stand-in for the recognizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from doorvouch.core.observations import HandObservation
from doorvouch.errors import DoorVouchError
from doorvouch.vision.mediapipe_gestures import MediaPipeGestureDetector
from fakes import ScriptedFrame

FRAME = ScriptedFrame(shape=(720, 1280, 3))


@dataclass
class Point:
    x: float | None
    y: float | None


@dataclass
class Category:
    category_name: str | None
    score: float | None


@dataclass
class Result:
    gestures: list = field(default_factory=list)
    hand_landmarks: list = field(default_factory=list)


class FakeRecognizer:
    def __init__(self, result: Result) -> None:
        self.result = result
        self.calls: list[tuple[object, int]] = []
        self.closed = False

    def recognize_for_video(self, image, timestamp_ms: int) -> Result:
        self.calls.append((image, timestamp_ms))
        return self.result

    def close(self) -> None:
        self.closed = True


def detector(result: Result) -> tuple[MediaPipeGestureDetector, FakeRecognizer]:
    recognizer = FakeRecognizer(result)
    return MediaPipeGestureDetector(recognizer, to_image=lambda frame: ("image", frame)), recognizer


def test_a_hand_is_reported_at_the_centre_of_its_landmarks_in_pixels():
    result = Result(
        gestures=[[Category("Thumb_Up", 0.9), Category("None", 0.1)]],
        hand_landmarks=[[Point(0.25, 0.25), Point(0.75, 0.75)]],
    )
    gestures, recognizer = detector(result)

    assert gestures.detect(FRAME, 1.5) == [HandObservation((640.0, 360.0), "Thumb_Up", 0.9)]
    assert recognizer.calls == [(("image", FRAME), 1500)]


@pytest.mark.parametrize("label", ["None", "", "Unknown", None])
def test_a_hand_without_a_known_gesture_has_no_gesture(label):
    result = Result(
        gestures=[[Category(label, 0.8)]],
        hand_landmarks=[[Point(0.5, 0.5)]],
    )
    gestures, _ = detector(result)

    (hand,) = gestures.detect(FRAME, 0.0)
    assert hand.gesture is None
    assert hand.score == 0.8


def test_hands_are_matched_with_their_gestures_by_position():
    result = Result(
        gestures=[[Category("Victory", 0.7)], [Category("Thumb_Up", 0.95)]],
        hand_landmarks=[[Point(0.1, 0.5)], [Point(0.9, 0.5)]],
    )
    gestures, _ = detector(result)

    left, right = gestures.detect(FRAME, 0.0)
    assert (left.gesture, left.center) == ("Victory", (128.0, 360.0))
    assert (right.gesture, right.center) == ("Thumb_Up", (1152.0, 360.0))


def test_a_hand_that_got_no_classification_is_still_reported():
    result = Result(gestures=[], hand_landmarks=[[Point(0.5, 0.5)]])
    gestures, _ = detector(result)
    assert gestures.detect(FRAME, 0.0) == [HandObservation((640.0, 360.0), None, 0.0)]


def test_missing_values_do_not_crash_the_adapter():
    result = Result(
        gestures=[[Category("Thumb_Up", None)]],
        hand_landmarks=[[Point(None, None)], []],
    )
    gestures, _ = detector(result)
    assert gestures.detect(FRAME, 0.0) == [HandObservation((0.0, 0.0), "Thumb_Up", 0.0)]


def test_no_hands_gives_an_empty_list():
    gestures, _ = detector(Result())
    assert gestures.detect(FRAME, 0.0) == []

    gestures, _ = detector(Result(gestures=None, hand_landmarks=None))
    assert gestures.detect(FRAME, 0.0) == []


def test_timestamps_handed_to_mediapipe_always_increase():
    gestures, recognizer = detector(Result())

    for timestamp in (1.0, 1.0, 0.5, 1.0004, 2.0):
        gestures.detect(FRAME, timestamp)

    assert [milliseconds for _, milliseconds in recognizer.calls] == [1000, 1001, 1002, 1003, 2000]


def test_closing_releases_the_recognizer():
    gestures, recognizer = detector(Result())
    gestures.close()
    assert recognizer.closed is True


def test_a_missing_model_says_how_to_get_it(tmp_path):
    with pytest.raises(DoorVouchError, match="doorvouch models download"):
        MediaPipeGestureDetector.from_model(tmp_path / "absent.task")


# With the real MediaPipe --------------------------------------------------------------


def test_installed_mediapipe_has_everything_the_adapter_uses():
    mp = pytest.importorskip("mediapipe")
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    assert hasattr(mp, "Image")
    assert hasattr(mp.ImageFormat, "SRGB")
    assert hasattr(mp_python, "BaseOptions")
    assert hasattr(vision.RunningMode, "VIDEO")
    assert callable(vision.GestureRecognizer.create_from_options)
    assert hasattr(vision.GestureRecognizer, "recognize_for_video")
    vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=b"PK"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def test_real_mediapipe_rejects_a_file_that_is_not_a_model(tmp_path):
    pytest.importorskip("mediapipe")
    pytest.importorskip("cv2")
    garbage = tmp_path / "garbage.task"
    garbage.write_bytes(b"PK\x03\x04 this is not a task bundle")

    with pytest.raises(DoorVouchError, match=r"MediaPipe could not load garbage\.task"):
        MediaPipeGestureDetector.from_model(garbage)
