"""The webcam wrapper, driven by a stand-in for OpenCV's capture API."""

from __future__ import annotations

import itertools

import pytest

from doorvouch.errors import DoorVouchError
from doorvouch.vision import camera as camera_module
from doorvouch.vision.camera import Camera

WIDTH, HEIGHT, FOURCC, BUFFER = 3, 4, 6, 38
ANY, DSHOW = 0, 700


class FakeCapture:
    def __init__(self, *, opened: bool = True, reads=(), actual=(640.0, 480.0)) -> None:
        self.opened = opened
        self.settings: dict[int, object] = {}
        self.released = False
        self._reads = itertools.chain(reads, itertools.repeat((False, None)))
        self._actual = {WIDTH: actual[0], HEIGHT: actual[1]}

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: object) -> bool:
        self.settings[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self._actual.get(prop, 0.0)

    def read(self):
        return next(self._reads)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_ANY = ANY
    CAP_DSHOW = DSHOW
    CAP_PROP_FRAME_WIDTH = WIDTH
    CAP_PROP_FRAME_HEIGHT = HEIGHT
    CAP_PROP_FOURCC = FOURCC
    CAP_PROP_BUFFERSIZE = BUFFER

    class VideoWriter:
        @staticmethod
        def fourcc(*letters: str) -> str:
            return "".join(letters)

    def __init__(self, captures: dict[int, FakeCapture]) -> None:
        self.captures = captures
        self.requests: list[tuple[int, int]] = []

    def VideoCapture(self, index: int, backend: int) -> FakeCapture:
        self.requests.append((index, backend))
        return self.captures[backend]


class FakeTime:
    def __init__(self) -> None:
        self.now = 50.0
        self.naps = 0

    def monotonic(self) -> float:
        self.now += 0.25
        return self.now

    def sleep(self, seconds: float) -> None:
        self.naps += 1


@pytest.fixture
def clock(monkeypatch) -> FakeTime:
    fake = FakeTime()
    monkeypatch.setattr(camera_module, "time", fake)
    return fake


def install(monkeypatch, captures: dict[int, FakeCapture], backends=(ANY,)) -> FakeCv2:
    cv2 = FakeCv2(captures)
    monkeypatch.setattr(camera_module, "load_cv2", lambda: cv2)
    monkeypatch.setattr(camera_module, "_backends", lambda _cv2: list(backends))
    return cv2


def test_opening_asks_for_mjpg_at_the_requested_size(monkeypatch):
    capture = FakeCapture()
    cv2 = install(monkeypatch, {ANY: capture})

    with Camera(1, width=1280, height=720) as camera:
        assert cv2.requests == [(1, ANY)]
        assert capture.settings == {FOURCC: "MJPG", WIDTH: 1280, HEIGHT: 720, BUFFER: 1}
        # The camera may deliver something else than what was asked for.
        assert camera.resolution == (640, 480)
    assert capture.released is True


def test_a_camera_that_does_not_open_is_explained(monkeypatch):
    capture = FakeCapture(opened=False)
    install(monkeypatch, {ANY: capture})

    with pytest.raises(DoorVouchError, match="Could not open camera 3"):
        Camera(3).open()
    assert capture.released is True


def test_the_next_backend_is_tried_when_the_first_fails(monkeypatch):
    direct_show, default = FakeCapture(opened=False), FakeCapture()
    cv2 = install(monkeypatch, {DSHOW: direct_show, ANY: default}, backends=(DSHOW, ANY))

    with Camera(0):
        pass

    assert cv2.requests == [(0, DSHOW), (0, ANY)]
    assert direct_show.released is True
    assert default.settings[FOURCC] == "MJPG"


def test_directshow_is_preferred_on_windows_only(monkeypatch):
    with monkeypatch.context() as patched:
        patched.setattr(camera_module.os, "name", "nt")
        on_windows = camera_module._backends(FakeCv2)
    with monkeypatch.context() as patched:
        patched.setattr(camera_module.os, "name", "posix")
        elsewhere = camera_module._backends(FakeCv2)

    assert on_windows == [DSHOW, ANY]
    assert elsewhere == [ANY]


def test_frames_come_with_increasing_timestamps_and_failed_reads_are_skipped(monkeypatch, clock):
    capture = FakeCapture(reads=[(True, "first"), (False, None), (True, None), (True, "second")])
    install(monkeypatch, {ANY: capture})

    with Camera() as camera:
        frames = camera.frames()
        assert next(frames) == (50.25, "first")
        assert next(frames) == (50.5, "second")
    assert clock.naps == 2


def test_a_camera_that_goes_silent_raises_instead_of_hanging(monkeypatch, clock):
    capture = FakeCapture(reads=[(True, "only")])
    install(monkeypatch, {ANY: capture})

    with Camera() as camera:
        frames = camera.frames()
        next(frames)
        with pytest.raises(DoorVouchError, match="stopped delivering frames"):
            next(frames)
    assert clock.naps == 29


def test_a_closed_camera_cannot_be_used(monkeypatch):
    install(monkeypatch, {ANY: FakeCapture()})
    camera = Camera()

    with pytest.raises(DoorVouchError, match="not open"):
        _ = camera.resolution
    with pytest.raises(DoorVouchError, match="not open"):
        next(camera.frames())

    camera.open()
    camera.close()
    camera.close()  # closing twice is harmless
    with pytest.raises(DoorVouchError, match="not open"):
        _ = camera.resolution
