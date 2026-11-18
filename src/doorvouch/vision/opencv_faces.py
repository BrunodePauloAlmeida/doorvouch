"""Face detection (YuNet) and face embeddings (SFace) through OpenCV."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Self

from doorvouch.core.embeddings import normalize
from doorvouch.core.geometry import Box, crop_region, detection_scale
from doorvouch.core.observations import FaceObservation
from doorvouch.errors import DoorVouchError
from doorvouch.vision.base import Frame, load_cv2


class OpenCvFaceAnalyzer:
    """Finds faces with YuNet and describes each one with an SFace embedding.

    Detection runs on a copy shrunk so its longest side is ``max_side`` pixels:
    YuNet was trained for faces up to about 300 pixels, and a face close to a
    high-resolution webcam is larger than that. The embedding is always taken
    from the full-resolution frame.
    """

    def __init__(self, detector: Any, recognizer: Any, *, cv2: Any, max_side: int = 640) -> None:
        self._detector = detector
        self._recognizer = recognizer
        self._cv2 = cv2
        self._max_side = max_side
        self._input_size: tuple[int, int] | None = None

    @classmethod
    def from_models(
        cls,
        detector_model: Path,
        recognizer_model: Path,
        *,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        max_side: int = 640,
    ) -> Self:
        cv2 = load_cv2()
        _quiet(cv2)
        detector = _create(
            cv2,
            cv2.FaceDetectorYN,
            detector_model,
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
        recognizer = _create(cv2, cv2.FaceRecognizerSF, recognizer_model)
        return cls(detector, recognizer, cv2=cv2, max_side=max_side)

    def analyze(self, frame: Frame) -> list[FaceObservation]:
        cv2 = self._cv2
        height, width = frame.shape[:2]
        scale = detection_scale(width, height, self._max_side)
        small = frame
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        small_height, small_width = small.shape[:2]
        if self._input_size != (small_width, small_height):
            self._detector.setInputSize((small_width, small_height))
            self._input_size = (small_width, small_height)

        try:
            _, rows = self._detector.detect(small)
        except cv2.error as error:
            raise DoorVouchError(
                f"The face detector failed: {_brief(error)}. "
                "Run `doorvouch doctor` for a diagnosis."
            ) from error
        if rows is None:
            return []

        faces: list[FaceObservation] = []
        for row in rows:
            # A row is x, y, w, h, five landmark (x, y) pairs and the score.
            full = row.astype("float32", copy=True)
            full[0:14:2] *= width / small_width
            full[1:14:2] *= height / small_height
            aligned = self._recognizer.alignCrop(frame, full)
            feature = self._recognizer.feature(aligned)
            try:
                embedding = normalize(float(value) for value in feature.flatten())
            except ValueError:
                continue
            x, y, w, h = (float(value) for value in full[:4])
            faces.append(FaceObservation(Box(x, y, w, h), embedding, float(full[14])))
        return faces

    def self_test(self) -> int:
        """Run both networks once on blank images and return the embedding size.

        Used by ``doorvouch doctor``: a model that OpenCV cannot execute fails here,
        before a camera is involved.
        """
        import numpy

        self.analyze(numpy.zeros((480, 640, 3), dtype=numpy.uint8))
        try:
            feature = self._recognizer.feature(numpy.zeros((112, 112, 3), dtype=numpy.uint8))
        except self._cv2.error as error:
            raise DoorVouchError(f"The face recognizer failed: {_brief(error)}") from error
        size = int(feature.size)
        if size == 0:
            raise DoorVouchError("The face recognizer returned an empty embedding.")
        return size


def face_jpeg(
    frame: Frame,
    box: Box,
    *,
    cv2: Any | None = None,
    margin: float = 0.25,
    max_side: int = 256,
    quality: int = 85,
) -> bytes | None:
    """A small JPEG of the face in ``box``, or ``None`` if it lies outside the frame."""
    cv2 = cv2 or load_cv2()
    height, width = frame.shape[:2]
    region = crop_region(box, width, height, margin=margin)
    if region is None:
        return None
    left, top, right, bottom = region
    crop = frame[top:bottom, left:right]
    scale = detection_scale(right - left, bottom - top, max_side)
    if scale < 1.0:
        size = (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(encoded.tobytes()) if encoded_ok else None


def _create(cv2: Any, factory: Any, model: Path, *arguments: Any) -> Any:
    if not model.is_file():
        raise DoorVouchError(f"Model file not found: {model}. Run `doorvouch models download`.")
    try:
        if _needs_buffer(model):
            import numpy

            data: Any = numpy.frombuffer(model.read_bytes(), dtype=numpy.uint8)
            return factory.create("onnx", data, numpy.zeros(0, dtype=numpy.uint8), *arguments)
        return factory.create(str(model), "", *arguments)
    except cv2.error as error:
        raise DoorVouchError(
            f"OpenCV could not load {model.name}: {_brief(error)}. "
            "Run `doorvouch models download --force` and then `doorvouch doctor`."
        ) from error


def _needs_buffer(model: Path) -> bool:
    """Whether to hand OpenCV the model's bytes instead of its path.

    On Windows, OpenCV cannot open a path that has accented characters.
    """
    return os.name == "nt" and not str(model).isascii()


def _quiet(cv2: Any) -> None:
    """Hide OpenCV's informational warnings; real failures still raise."""
    with contextlib.suppress(AttributeError):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


def _brief(error: Exception) -> str:
    lines = str(error).strip().splitlines()
    return lines[-1].strip() if lines else type(error).__name__
