"""The OpenCV face adapter, driven by stand-ins for the two networks.

NumPy is needed for the frames, so these tests are skipped where it is not
installed. The last tests use the real OpenCV when it is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doorvouch.core.geometry import Box
from doorvouch.errors import DoorVouchError
from doorvouch.vision import opencv_faces
from doorvouch.vision.opencv_faces import OpenCvFaceAnalyzer, face_jpeg

np = pytest.importorskip("numpy")

LANDMARKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def row(x: float, y: float, w: float, h: float, score: float):
    """One detection as YuNet reports it: box, five landmark points, score."""
    return [x, y, w, h, *LANDMARKS, score]


def detections(*rows):
    return np.array(rows, dtype=np.float32)


def image(width: int, height: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


class FakeCv2:
    INTER_AREA = 3

    class error(Exception):
        pass

    def __init__(self) -> None:
        self.resizes: list[tuple[tuple[int, int], int | None]] = []

    def resize(self, frame, size, interpolation=None):
        self.resizes.append((size, interpolation))
        width, height = size
        return image(width, height)


class FakeDetector:
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows = rows
        self.error = error
        self.input_sizes: list[tuple[int, int]] = []
        self.seen: list[tuple[int, ...]] = []

    def setInputSize(self, size) -> None:
        self.input_sizes.append(size)

    def detect(self, small):
        self.seen.append(small.shape)
        if self.error is not None:
            raise self.error
        return 1, self.rows


class FakeRecognizer:
    def __init__(self, *features, error: Exception | None = None) -> None:
        self.features = list(features)
        self.error = error
        self.aligned: list[tuple[tuple[int, ...], list[float]]] = []

    def alignCrop(self, frame, detection):
        self.aligned.append((frame.shape, [float(value) for value in detection]))
        return "aligned"

    def feature(self, aligned):
        if self.error is not None:
            raise self.error
        return self.features.pop(0)


def analyzer(detector, recognizer, cv2=None, **options) -> OpenCvFaceAnalyzer:
    return OpenCvFaceAnalyzer(detector, recognizer, cv2=cv2 or FakeCv2(), **options)


def test_detection_runs_on_a_smaller_copy_and_results_are_mapped_back():
    cv2 = FakeCv2()
    detector = FakeDetector(detections(row(100, 50, 80, 100, 0.93)))
    recognizer = FakeRecognizer(np.array([[3.0, 4.0]], dtype=np.float32))

    (found,) = analyzer(detector, recognizer, cv2).analyze(image(1280, 720))

    assert cv2.resizes == [((640, 360), FakeCv2.INTER_AREA)]
    assert detector.seen == [(360, 640, 3)]
    assert found.box == Box(200.0, 100.0, 160.0, 200.0)
    assert found.score == pytest.approx(0.93)
    assert found.embedding == pytest.approx((0.6, 0.8))

    # The embedding is computed from the full frame, with the landmarks scaled too.
    ((shape, detection),) = recognizer.aligned
    assert shape == (720, 1280, 3)
    assert detection[:4] == [200.0, 100.0, 160.0, 200.0]
    assert detection[4:14] == [2.0 * value for value in LANDMARKS]
    assert detection[14] == pytest.approx(0.93)


def test_width_and_height_are_scaled_separately():
    # 1000x563 shrinks to 640x360, which is not exactly the same ratio on both axes.
    detector = FakeDetector(detections(row(64, 36, 64, 36, 0.9)))
    recognizer = FakeRecognizer(np.array([[1.0, 0.0]]))

    (found,) = analyzer(detector, recognizer).analyze(image(1000, 563))

    assert found.box.x == pytest.approx(100.0)
    assert found.box.w == pytest.approx(100.0)
    assert found.box.y == pytest.approx(36 * 563 / 360)
    assert found.box.h == pytest.approx(36 * 563 / 360)


def test_small_frames_are_used_as_they_are():
    cv2 = FakeCv2()
    detector = FakeDetector(detections(row(10, 20, 30, 40, 0.9)))
    recognizer = FakeRecognizer(np.array([[1.0, 0.0]]))

    (found,) = analyzer(detector, recognizer, cv2).analyze(image(320, 240))

    assert cv2.resizes == []
    assert detector.seen == [(240, 320, 3)]
    assert found.box == Box(10.0, 20.0, 30.0, 40.0)


def test_the_detector_is_told_the_input_size_only_when_it_changes():
    detector = FakeDetector()
    faces = analyzer(detector, FakeRecognizer())

    faces.analyze(image(1280, 720))
    faces.analyze(image(1280, 720))
    faces.analyze(image(640, 480))

    assert detector.input_sizes == [(640, 360), (640, 480)]


def test_a_frame_without_faces_gives_an_empty_list():
    assert analyzer(FakeDetector(rows=None), FakeRecognizer()).analyze(image(640, 480)) == []


def test_a_face_with_an_empty_embedding_is_dropped():
    detector = FakeDetector(detections(row(10, 10, 80, 100, 0.9), row(300, 10, 80, 100, 0.8)))
    recognizer = FakeRecognizer(np.zeros((1, 4)), np.array([[0.0, 2.0]]))

    (found,) = analyzer(detector, recognizer).analyze(image(640, 480))

    assert found.box.x == 300.0
    assert found.embedding == (0.0, 1.0)


def test_a_failing_detector_points_to_the_doctor():
    failure = FakeCv2.error("OpenCV(5.0.0) error:\n  the real reason  ")
    faces = analyzer(FakeDetector(error=failure), FakeRecognizer())

    with pytest.raises(DoorVouchError) as raised:
        faces.analyze(image(640, 480))
    assert "the real reason" in str(raised.value)
    assert "doorvouch doctor" in str(raised.value)


def test_self_test_returns_the_embedding_size():
    faces = analyzer(FakeDetector(), FakeRecognizer(np.ones((1, 128), dtype=np.float32)))
    assert faces.self_test() == 128


def test_self_test_rejects_an_empty_embedding():
    faces = analyzer(FakeDetector(), FakeRecognizer(np.zeros((1, 0))))
    with pytest.raises(DoorVouchError, match="empty embedding"):
        faces.self_test()


def test_self_test_reports_a_recognizer_that_cannot_run():
    faces = analyzer(FakeDetector(), FakeRecognizer(error=FakeCv2.error("unsupported layer")))
    with pytest.raises(DoorVouchError, match="The face recognizer failed: unsupported layer"):
        faces.self_test()


# Loading the networks --------------------------------------------------------------


class FakeFactory:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error

    def create(self, *arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return "network"


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model bytes")
    return path


def test_a_model_is_loaded_from_its_path(model):
    factory = FakeFactory()
    assert opencv_faces._create(FakeCv2(), factory, model, (320, 320), 0.7) == "network"
    assert factory.calls == [(str(model), "", (320, 320), 0.7)]


def test_a_model_can_be_loaded_from_its_bytes(model, monkeypatch):
    monkeypatch.setattr(opencv_faces, "_needs_buffer", lambda path: True)
    factory = FakeFactory()

    opencv_faces._create(FakeCv2(), factory, model, (320, 320), 0.7)

    ((framework, data, config, *rest),) = factory.calls
    assert framework == "onnx"
    assert bytes(data) == b"model bytes"
    assert len(config) == 0
    assert rest == [(320, 320), 0.7]


def test_bytes_are_used_only_for_accented_paths_on_windows(monkeypatch):
    plain = Path("models") / "model.onnx"
    accented = Path("café") / "model.onnx"

    with monkeypatch.context() as patched:
        patched.setattr(opencv_faces.os, "name", "nt")
        on_windows = (opencv_faces._needs_buffer(plain), opencv_faces._needs_buffer(accented))

    assert on_windows == (False, True)
    assert opencv_faces._needs_buffer(accented) is (os.name == "nt")


def test_a_missing_model_says_how_to_get_it(tmp_path):
    with pytest.raises(DoorVouchError, match="doorvouch models download"):
        opencv_faces._create(FakeCv2(), FakeFactory(), tmp_path / "absent.onnx")


def test_a_model_opencv_rejects_says_how_to_replace_it(model):
    factory = FakeFactory(error=FakeCv2.error("Failed to parse ONNX model"))
    with pytest.raises(DoorVouchError) as raised:
        opencv_faces._create(FakeCv2(), factory, model)
    assert "OpenCV could not load model.onnx: Failed to parse ONNX model" in str(raised.value)
    assert "doorvouch models download --force" in str(raised.value)


# With the real OpenCV ---------------------------------------------------------------


def test_installed_opencv_has_everything_the_adapters_use():
    cv2 = pytest.importorskip("cv2")
    names = [
        "CAP_ANY",
        "CAP_DSHOW",
        "CAP_PROP_BUFFERSIZE",
        "CAP_PROP_FOURCC",
        "CAP_PROP_FRAME_HEIGHT",
        "CAP_PROP_FRAME_WIDTH",
        "COLOR_BGR2RGB",
        "FONT_HERSHEY_SIMPLEX",
        "FaceDetectorYN",
        "FaceRecognizerSF",
        "IMWRITE_JPEG_QUALITY",
        "INTER_AREA",
        "LINE_AA",
        "VideoCapture",
        "WND_PROP_VISIBLE",
        "error",
    ]
    assert [name for name in names if not hasattr(cv2, name)] == []
    assert callable(cv2.FaceDetectorYN.create)
    assert callable(cv2.FaceRecognizerSF.create)
    assert isinstance(cv2.VideoWriter.fourcc(*"MJPG"), int)


def test_real_opencv_rejects_a_file_that_is_not_a_model(tmp_path):
    pytest.importorskip("cv2")
    garbage = tmp_path / "garbage.onnx"
    garbage.write_bytes(b"this is not a model")

    with pytest.raises(DoorVouchError, match=r"OpenCV could not load garbage\.onnx"):
        OpenCvFaceAnalyzer.from_models(garbage, garbage)


def test_face_snapshot_is_a_small_jpeg():
    cv2 = pytest.importorskip("cv2")
    frame = np.full((1080, 1920, 3), 127, dtype=np.uint8)

    data = face_jpeg(frame, Box(400.0, 100.0, 800.0, 900.0))

    assert data is not None
    assert data.startswith(b"\xff\xd8")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == 256


def test_face_snapshot_keeps_small_faces_at_their_own_size():
    cv2 = pytest.importorskip("cv2")
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)

    data = face_jpeg(frame, Box(200.0, 100.0, 100.0, 130.0), margin=0.0)

    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (130, 100)


def test_no_snapshot_for_a_box_outside_the_frame():
    pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert face_jpeg(frame, Box(700.0, 100.0, 50.0, 50.0)) is None
