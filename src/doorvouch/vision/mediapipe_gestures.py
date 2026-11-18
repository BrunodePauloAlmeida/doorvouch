"""Hand gestures through MediaPipe's Gesture Recognizer task."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from doorvouch.core.observations import HandObservation
from doorvouch.errors import DoorVouchError
from doorvouch.vision.base import Frame, load_cv2

# Labels the built-in classifier uses for "a hand, but no known gesture".
_NO_GESTURE = {"", "None", "Unknown"}


class MediaPipeGestureDetector:
    """Reports each hand in view with the gesture MediaPipe recognises for it.

    The built-in classifier knows seven gestures, including ``Thumb_Up``.
    """

    def __init__(self, recognizer: Any, to_image: Callable[[Frame], Any]) -> None:
        self._recognizer = recognizer
        self._to_image = to_image
        self._last_timestamp_ms = -1

    @classmethod
    def from_model(cls, model: Path, *, max_hands: int = 4, min_confidence: float = 0.5) -> Self:
        if not model.is_file():
            raise DoorVouchError(f"Model file not found: {model}. Run `doorvouch models download`.")
        cv2 = load_cv2()
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise DoorVouchError(
                "MediaPipe is not installed. Run `uv sync` in the project folder."
            ) from error

        options = vision.GestureRecognizerOptions(
            # The bytes are passed instead of the path, which keeps working when
            # the folder name has accented characters on Windows.
            base_options=mp_python.BaseOptions(model_asset_buffer=model.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        try:
            recognizer = vision.GestureRecognizer.create_from_options(options)
        except OSError as error:
            # MediaPipe loads a native library that needs the system's OpenGL
            # libraries, which a bare Linux installation does not have.
            raise DoorVouchError(
                f"MediaPipe could not load its native library: {error}. "
                "On Debian or Ubuntu, run `sudo apt install libegl1 libgles2`."
            ) from error
        except (RuntimeError, ValueError) as error:
            raise DoorVouchError(
                f"MediaPipe could not load {model.name}: {error}. "
                "Run `doorvouch models download --force` and then `doorvouch doctor`."
            ) from error

        def to_image(frame: Frame) -> Any:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        return cls(recognizer, to_image)

    def detect(self, frame: Frame, timestamp: float) -> list[HandObservation]:
        height, width = frame.shape[:2]
        # MediaPipe's video mode needs strictly increasing integer milliseconds.
        timestamp_ms = max(int(timestamp * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._recognizer.recognize_for_video(self._to_image(frame), timestamp_ms)

        gestures = list(result.gestures or [])
        hands: list[HandObservation] = []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            if not landmarks:
                continue
            center_x = sum(float(point.x or 0.0) for point in landmarks) / len(landmarks) * width
            center_y = sum(float(point.y or 0.0) for point in landmarks) / len(landmarks) * height
            categories = gestures[index] if index < len(gestures) else []
            name: str | None = None
            score = 0.0
            if categories:
                top = categories[0]
                score = float(top.score or 0.0)
                label = str(top.category_name or "")
                name = None if label in _NO_GESTURE else label
            hands.append(HandObservation((center_x, center_y), name, score))
        return hands

    def close(self) -> None:
        close = getattr(self._recognizer, "close", None)
        if close is not None:
            close()
