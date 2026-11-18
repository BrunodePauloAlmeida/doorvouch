"""The real networks on real photos.

Everything else in this folder uses stand-ins for the camera and the networks.
These tests run the actual pipeline, and need three things that are not part of
the repository. Without them they are skipped:

* the packages OpenCV, MediaPipe and NumPy;
* the model files (``doorvouch models download``);
* MediaPipe's public sample photos for its gesture recognizer, in the folder
  named by the ``DOORVOUCH_SAMPLES`` environment variable.

Continuous integration provides all three, on Linux and on Windows. To run the
tests on your own computer::

    mkdir -p data/samples
    for name in thumbs_up pointing_up victory; do
        curl --fail --location --output data/samples/$name.jpg \
            https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/$name.jpg
    done
    DOORVOUCH_SAMPLES=data/samples uv run pytest tests/test_real_models.py -v

The photos show two people. One is in ``thumbs_up.jpg`` and ``pointing_up.jpg``,
the other in ``victory.jpg``, each making the gesture the file is named after.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, NoReturn

import pytest

from doorvouch import models
from doorvouch.apps.live import run_live
from doorvouch.config import Settings
from doorvouch.core.association import find_owner
from doorvouch.core.embeddings import cosine_similarity
from doorvouch.core.engine import EngineConfig, VouchingEngine
from doorvouch.core.gallery import SFACE_COSINE_THRESHOLD
from doorvouch.vision.mediapipe_gestures import MediaPipeGestureDetector
from doorvouch.vision.opencv_faces import OpenCvFaceAnalyzer, face_jpeg

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

PHOTOS = ("thumbs_up", "pointing_up", "victory")
CONFIG = EngineConfig()


def unavailable(reason: str) -> NoReturn:
    """Skip the test, or fail it where the real models are required to be present."""
    if os.environ.get("DOORVOUCH_REQUIRE_REAL_MODELS"):
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def model_paths() -> dict[str, Path]:
    directory = Settings.load().models_dir
    if not all(models.inspect(spec, directory).ok for spec in models.ALL_MODELS):
        unavailable(f"the model files are not in {directory}: run `doorvouch models download`")
    return models.require(directory)


@pytest.fixture(scope="module")
def photos() -> dict[str, Any]:
    folder = os.environ.get("DOORVOUCH_SAMPLES")
    if not folder:
        unavailable("DOORVOUCH_SAMPLES does not name a folder with the sample photos")
    loaded = {}
    for name in PHOTOS:
        image = cv2.imread(str(Path(folder) / f"{name}.jpg"))
        if image is None:
            unavailable(f"{name}.jpg is not in {folder}")
        loaded[name] = image
    return loaded


@pytest.fixture(scope="module")
def faces(model_paths) -> OpenCvFaceAnalyzer:
    return OpenCvFaceAnalyzer.from_models(
        model_paths["face_detector"], model_paths["face_recognizer"]
    )


@pytest.fixture
def gestures(model_paths):
    detector = MediaPipeGestureDetector.from_model(model_paths["gesture_recognizer"])
    yield detector
    detector.close()


def the_face(faces: OpenCvFaceAnalyzer, image):
    (found,) = faces.analyze(image)
    return found


def still(frame, *, start: float, frames: int = 8):
    """One picture as a short video at ten frames a second."""
    return [(start + index / 10, frame) for index in range(frames)]


def side_by_side(left, right, *, gap: int = 120):
    """Two photos on one canvas, far enough apart that no hand is near the other face."""
    height = max(left.shape[0], right.shape[0])
    canvas = np.full((height, left.shape[1] + gap + right.shape[1], 3), 127, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    return canvas


def enroll(store, name: str, embedding, *, is_admin: bool = False):
    person = store.add_person(name, is_admin=is_admin)
    store.add_face_sample(person.id, embedding)
    return person


def test_a_face_is_found_and_turned_into_a_unit_vector(faces, photos):
    found = the_face(faces, photos["thumbs_up"])

    assert found.score > 0.8
    assert found.box.w > 64
    assert len(found.embedding) == 128
    assert math.sqrt(sum(value * value for value in found.embedding)) == pytest.approx(1.0)


def test_the_same_person_matches_across_photos_and_a_stranger_does_not(faces, photos):
    thumbs_up = the_face(faces, photos["thumbs_up"]).embedding
    pointing_up = the_face(faces, photos["pointing_up"]).embedding
    victory = the_face(faces, photos["victory"]).embedding

    assert cosine_similarity(thumbs_up, pointing_up) > SFACE_COSINE_THRESHOLD
    assert cosine_similarity(thumbs_up, victory) < CONFIG.visitor_max_similarity
    assert cosine_similarity(pointing_up, victory) < CONFIG.visitor_max_similarity


@pytest.mark.parametrize(
    ("photo", "gesture"),
    [("thumbs_up", "Thumb_Up"), ("pointing_up", "Pointing_Up"), ("victory", "Victory")],
)
def test_a_gesture_is_recognised_and_belongs_to_the_face_in_the_photo(
    faces, gestures, photos, photo, gesture
):
    image = photos[photo]
    hands = []
    for moment in (1.0, 1.1, 1.2):
        hands = gestures.detect(image, moment)

    (hand,) = hands
    assert hand.gesture == gesture
    assert hand.score >= CONFIG.min_gesture_score
    assert find_owner(hand.center, [the_face(faces, image).box]) == 0


def test_an_admins_thumbs_up_registers_the_visitor_next_to_them(faces, gestures, photos, store):
    # The admin is enrolled from one photo and recognised in another one.
    admin = enroll(store, "Alice", the_face(faces, photos["pointing_up"]).embedding, is_admin=True)
    scene = side_by_side(photos["thumbs_up"], photos["victory"])

    stats = run_live(
        still(scene, start=10.0),
        faces=faces,
        gestures=gestures,
        engine=VouchingEngine(store),
        snapshot=face_jpeg,
    )

    assert stats.events == {"person_seen": 1, "pending_created": 1}
    (pending,) = store.list_pending()
    assert pending.vouched_by == admin.id
    assert pending.snapshot_path.read_bytes().startswith(b"\xff\xd8")

    # Once the visitor has a name, the same scene holds two known people.
    store.name_pending(pending.id, "Carol")
    stats = run_live(
        still(scene, start=20.0), faces=faces, gestures=gestures, engine=VouchingEngine(store)
    )

    assert stats.events == {"person_seen": 2, "vouch_without_visitor": 1}
    assert store.list_pending() == []


def test_a_thumbs_up_from_somebody_who_is_not_an_admin_is_ignored(faces, gestures, photos, store):
    # Here the admin is the person on the right, who is not the one gesturing.
    enroll(store, "Alice", the_face(faces, photos["victory"]).embedding, is_admin=True)
    scene = side_by_side(photos["thumbs_up"], photos["victory"])

    stats = run_live(
        still(scene, start=10.0), faces=faces, gestures=gestures, engine=VouchingEngine(store)
    )

    assert stats.events == {"person_seen": 1, "gesture_ignored": 1}
    assert store.list_pending() == []


def test_an_admins_other_gestures_do_not_vouch(faces, gestures, photos, store):
    enroll(store, "Alice", the_face(faces, photos["thumbs_up"]).embedding, is_admin=True)
    scene = side_by_side(photos["pointing_up"], photos["victory"])

    stats = run_live(
        still(scene, start=10.0), faces=faces, gestures=gestures, engine=VouchingEngine(store)
    )

    assert stats.events == {"person_seen": 1}
    assert store.list_pending() == []
