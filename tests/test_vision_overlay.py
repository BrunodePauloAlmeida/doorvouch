"""The preview window, driven by a stand-in that records what would be drawn."""

from __future__ import annotations

import pytest

from doorvouch.core.engine import FrameResult, IdentifiedFace
from doorvouch.core.people import Person
from doorvouch.vision import overlay
from doorvouch.vision.overlay import OpenCvDisplay, describe, draw_faces, draw_lines
from helpers import axis, face

NO_KEY = -1
ESCAPE = 27


class Canvas:
    shape = (720, 1280, 3)

    def copy(self) -> Canvas:
        return Canvas()


class Recorder:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    WND_PROP_VISIBLE = 4

    def __init__(self, keys=()) -> None:
        self.rectangles: list[tuple] = []
        self.texts: list[tuple] = []
        self.windows: list[str] = []
        self.visible = 1.0
        self.destroyed = False
        self._keys = list(keys)

    def rectangle(self, canvas, top_left, bottom_right, color, thickness) -> None:
        self.rectangles.append((top_left, bottom_right, color))

    def putText(self, canvas, text, origin, font, scale, color, thickness, line) -> None:
        self.texts.append((text, origin, color))

    def imshow(self, title, canvas) -> None:
        self.windows.append(title)

    def waitKey(self, delay) -> int:
        return self._keys.pop(0) if self._keys else NO_KEY

    def getWindowProperty(self, title, prop) -> float:
        return self.visible

    def destroyAllWindows(self) -> None:
        self.destroyed = True

    def written(self) -> list[str]:
        """Each text once, in drawing order (every text is drawn as outline plus fill)."""
        return list(dict.fromkeys(text for text, _, _ in self.texts))


class FakeTime:
    def __init__(self) -> None:
        self.now = 10.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def cv2(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(overlay, "load_cv2", lambda: recorder)
    return recorder


@pytest.fixture
def clock(monkeypatch) -> FakeTime:
    fake = FakeTime()
    monkeypatch.setattr(overlay, "time", fake)
    return fake


def identified(status: str, similarity: float, **extra) -> IdentifiedFace:
    return IdentifiedFace(face(axis(0), x=200, y=150), status, similarity, **extra)


@pytest.mark.parametrize(
    ("shown", "caption"),
    [
        (identified("admin", 0.71, person=Person(1, "Renée", is_admin=True)), "Renee (admin) 0.71"),
        (identified("known", 0.5, person=Person(2, "Ruth")), "Ruth 0.50"),
        (identified("pending", 0.12, pending_id=3), "pending #3 0.12"),
        (identified("uncertain", 0.33), "uncertain 0.33"),
        (identified("unknown", 0.05), "unknown 0.05"),
    ],
)
def test_caption_names_the_face_and_shows_the_similarity(shown, caption):
    assert describe(shown) == caption


def test_each_face_gets_a_box_and_a_caption_in_the_colour_of_its_status():
    recorder = Recorder()
    admin = identified("admin", 0.9, person=Person(1, "Alice", is_admin=True))
    stranger = IdentifiedFace(face(axis(1), x=600.4, y=5.0), "unknown", 0.1)

    draw_faces(recorder, Canvas(), [admin, stranger])

    green, red = overlay._COLORS["admin"], overlay._COLORS["unknown"]
    assert recorder.rectangles == [
        ((200, 150), (300, 280), green),
        ((600, 5), (700, 135), red),
    ]
    # The caption sits above the box, but never above the top edge of the frame.
    assert recorder.texts[1] == ("Alice (admin) 0.90", (200, 142), green)
    assert recorder.texts[3] == ("unknown 0.10", (600, 20), red)
    assert len(recorder.texts) == 4


def test_lines_stack_from_the_top_or_end_at_the_bottom():
    recorder = Recorder()

    draw_lines(recorder, Canvas(), ["first", "second"])
    draw_lines(recorder, Canvas(), ["third", "fourth"], bottom=True)

    positions = {text: origin for text, origin, _ in recorder.texts}
    assert positions == {
        "first": (12, 28),
        "second": (12, 56),
        "third": (12, 676),
        "fourth": (12, 704),
    }


def test_the_live_view_shows_frame_rate_and_gesture_progress(cv2):
    display = OpenCvDisplay("Preview")
    result = FrameResult(faces=(identified("unknown", 0.05),), gesture_progress={1: 0.6})

    assert display.show(Canvas(), result, 12.34) is True

    assert cv2.windows == ["Preview"]
    assert cv2.written() == ["unknown 0.05", "12.3 FPS", "Thumbs-up 60%"]


@pytest.mark.parametrize("key", [ord("q"), ESCAPE])
def test_q_or_escape_stops(cv2, key):
    cv2._keys = [NO_KEY, key]
    display = OpenCvDisplay()

    assert display.show_faces(Canvas(), [], []) is True
    assert display.show_faces(Canvas(), [], []) is False


def test_closing_the_window_stops(cv2):
    display = OpenCvDisplay()
    assert display.show_faces(Canvas(), [], []) is True

    cv2.visible = 0.0
    assert display.show_faces(Canvas(), [], []) is False


def test_a_window_that_is_not_visible_yet_does_not_stop_the_first_frame(cv2):
    cv2.visible = 0.0
    assert OpenCvDisplay().show_faces(Canvas(), [], []) is True


def test_announcements_stay_on_screen_for_a_few_seconds(cv2, clock):
    display = OpenCvDisplay()
    display.announce("Renée is in view")

    display.show_faces(Canvas(), [], ["caption"])
    assert cv2.written() == ["caption", "Renee is in view"]

    cv2.texts.clear()
    clock.now += 5.0
    display.show_faces(Canvas(), [], ["caption"])
    assert cv2.written() == ["caption"]


def test_closing_the_display_destroys_the_window(cv2):
    OpenCvDisplay().close()
    assert cv2.destroyed is True


def test_real_opencv_draws_on_a_real_image():
    real_cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    canvas = numpy.zeros((480, 640, 3), dtype=numpy.uint8)

    draw_faces(real_cv2, canvas, [identified("unknown", 0.05)])
    draw_lines(real_cv2, canvas, ["15.0 FPS"], bottom=True)

    assert canvas.any()
