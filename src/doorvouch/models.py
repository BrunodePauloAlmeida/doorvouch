"""The model files DoorVouch needs, where to get them and how to check them.

Models are never committed to the repository. ``doorvouch models download`` puts
them in ``<home>/models`` and verifies size and SHA-256 where those are pinned.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from doorvouch import __version__
from doorvouch.errors import DoorVouchError

# Pinned commit of https://github.com/opencv/opencv_zoo (2026-10-20). The sizes and
# hashes below are the Git LFS object ids recorded at that commit.
_ZOO = "https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models"

_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One downloadable model file."""

    key: str
    filename: str
    urls: tuple[str, ...]
    license: str
    about: str
    size: int
    sha256: str


FACE_DETECTOR = ModelSpec(
    key="face_detector",
    filename="face_detection_yunet_2026may.onnx",
    urls=(
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2026may.onnx",
        # An unmodified mirror; the pinned SHA-256 guards against any difference.
        "https://huggingface.co/pollen-robotics/face_detection_yunet_2026may/resolve/main/"
        "face_detection_yunet_2026may.onnx",
    ),
    license="MIT",
    about="YuNet face detector, from the OpenCV Zoo",
    size=229_738,
    sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
)

FACE_RECOGNIZER = ModelSpec(
    key="face_recognizer",
    filename="face_recognition_sface_2021dec.onnx",
    urls=(
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
        "face_recognition_sface_2021dec.onnx",
    ),
    license="Apache-2.0",
    about="SFace face recognizer, from the OpenCV Zoo",
    size=38_696_353,
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)

# Version 1 of Google's bundle. The "latest" address serves the same bytes today
# and is only accepted for as long as it still does.
_GESTURES = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer"
GESTURE_RECOGNIZER = ModelSpec(
    key="gesture_recognizer",
    filename="gesture_recognizer.task",
    urls=(
        f"{_GESTURES}/float16/1/gesture_recognizer.task",
        f"{_GESTURES}/float16/latest/gesture_recognizer.task",
    ),
    license="see Google's model card",
    about="MediaPipe hand gesture recognizer",
    size=8_373_440,
    sha256="97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
)

ALL_MODELS: tuple[ModelSpec, ...] = (FACE_DETECTOR, FACE_RECOGNIZER, GESTURE_RECOGNIZER)

Opener = Callable[[str], AbstractContextManager[BinaryIO]]


@dataclass(frozen=True, slots=True)
class Inspection:
    """What was found on disk for a model. ``problem`` is ``None`` when the file is good."""

    path: Path
    problem: str | None
    size: int = 0
    sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.problem is None


def inspect(spec: ModelSpec, directory: Path) -> Inspection:
    """Check the copy of ``spec`` inside ``directory`` without changing anything."""
    return _inspect_file(spec, directory / spec.filename)


def _inspect_file(spec: ModelSpec, path: Path) -> Inspection:
    if not path.is_file():
        return Inspection(path, "missing")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    checksum = digest.hexdigest()
    return Inspection(path, _problem(spec, size, checksum), size, checksum)


def _problem(spec: ModelSpec, size: int, checksum: str) -> str | None:
    if size != spec.size:
        return f"size is {size} bytes, expected {spec.size}"
    if checksum != spec.sha256:
        return f"SHA-256 is {checksum}, expected {spec.sha256}"
    return None


def _open_url(url: str) -> AbstractContextManager[BinaryIO]:
    if not url.startswith("https://"):
        raise DoorVouchError(f"Refusing to download from a non-HTTPS address: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": f"doorvouch/{__version__}"})
    response: AbstractContextManager[BinaryIO] = urllib.request.urlopen(request, timeout=60)
    return response


def download(
    spec: ModelSpec,
    directory: Path,
    *,
    force: bool = False,
    opener: Opener = _open_url,
) -> Inspection:
    """Make sure a verified copy of ``spec`` is in ``directory``.

    A good existing file is kept unless ``force`` is set. Every address in
    ``spec.urls`` is tried in turn, and a download that fails verification is
    thrown away, never installed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    existing = inspect(spec, directory)
    if existing.ok and not force:
        return existing

    target = directory / spec.filename
    partial = target.with_name(target.name + ".part")
    failures: list[str] = []
    for url in spec.urls:
        try:
            with opener(url) as response, partial.open("wb") as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
            fetched = _inspect_file(spec, partial)
            if not fetched.ok:
                raise DoorVouchError(fetched.problem or "verification failed")
            os.replace(partial, target)
        except (DoorVouchError, OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            failures.append(f"  {url}\n    {error}")
            continue
        return Inspection(target, None, fetched.size, fetched.sha256)

    attempts = "\n".join(failures)
    raise DoorVouchError(
        f"Could not download {spec.filename}:\n{attempts}\n"
        f"You can fetch it by hand from the first address above and save it as {target}"
    )


def require(directory: Path, specs: Iterable[ModelSpec] = ALL_MODELS) -> dict[str, Path]:
    """Paths of the models by key, or an error that says how to get them."""
    paths: dict[str, Path] = {}
    problems: list[str] = []
    for spec in specs:
        found = inspect(spec, directory)
        if found.ok:
            paths[spec.key] = found.path
        else:
            problems.append(f"  {spec.filename}: {found.problem}")
    if problems:
        listing = "\n".join(problems)
        raise DoorVouchError(
            f"Model files are not ready in {directory}:\n{listing}\n"
            "Run `doorvouch models download`."
        )
    return paths
