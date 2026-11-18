from __future__ import annotations

import hashlib
import io
import urllib.error
from pathlib import Path

import pytest

from doorvouch import models
from doorvouch.errors import DoorVouchError
from doorvouch.models import ModelSpec

GOOD = b"model-bytes" * 100
SPEC = ModelSpec(
    key="tiny",
    filename="tiny.onnx",
    urls=("https://first.example/tiny.onnx", "https://second.example/tiny.onnx"),
    license="MIT",
    about="a test model",
    size=len(GOOD),
    sha256=hashlib.sha256(GOOD).hexdigest(),
)


class FakeWeb:
    """Answers each address with fixed bytes, or raises the given error."""

    def __init__(self, pages: dict[str, bytes | Exception]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def __call__(self, url: str):
        self.requested.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return io.BytesIO(page)


def test_registry_entries_are_downloadable_over_https_and_distinct():
    assert len({spec.key for spec in models.ALL_MODELS}) == len(models.ALL_MODELS)
    assert len({spec.filename for spec in models.ALL_MODELS}) == len(models.ALL_MODELS)
    for spec in models.ALL_MODELS:
        assert spec.urls and all(url.startswith("https://") for url in spec.urls)
        assert all(url.endswith(spec.filename) for url in spec.urls)
        assert spec.license


def test_inspect_reports_a_missing_file(tmp_path: Path):
    found = models.inspect(SPEC, tmp_path)
    assert not found.ok
    assert found.problem == "missing"


def test_inspect_accepts_the_pinned_bytes(tmp_path: Path):
    (tmp_path / SPEC.filename).write_bytes(GOOD)
    found = models.inspect(SPEC, tmp_path)
    assert found.ok
    assert found.size == len(GOOD)
    assert found.sha256 == SPEC.sha256


def test_inspect_rejects_wrong_size_and_wrong_content(tmp_path: Path):
    (tmp_path / SPEC.filename).write_bytes(GOOD[:-1])
    assert "size is" in models.inspect(SPEC, tmp_path).problem

    (tmp_path / SPEC.filename).write_bytes(b"X" * len(GOOD))
    assert "SHA-256" in models.inspect(SPEC, tmp_path).problem


def test_every_registered_model_is_pinned_by_size_and_hash():
    # A guess about what a file "looks like" is not a check. The gesture bundle, for
    # one, is a zip archive that does not start with the usual zip signature.
    for spec in models.ALL_MODELS:
        assert spec.size > 0
        assert len(spec.sha256) == 64
        assert set(spec.sha256) <= set("0123456789abcdef")


def test_download_saves_a_verified_file(tmp_path: Path):
    web = FakeWeb({SPEC.urls[0]: GOOD})
    found = models.download(SPEC, tmp_path / "models", opener=web)
    assert found.ok
    assert found.path.read_bytes() == GOOD
    assert web.requested == [SPEC.urls[0]]
    assert not list((tmp_path / "models").glob("*.part"))


def test_download_keeps_a_good_file_unless_forced(tmp_path: Path):
    (tmp_path / SPEC.filename).write_bytes(GOOD)
    web = FakeWeb({SPEC.urls[0]: GOOD})

    models.download(SPEC, tmp_path, opener=web)
    assert web.requested == []

    models.download(SPEC, tmp_path, opener=web, force=True)
    assert web.requested == [SPEC.urls[0]]


def test_download_replaces_a_damaged_file(tmp_path: Path):
    (tmp_path / SPEC.filename).write_bytes(b"truncated")
    models.download(SPEC, tmp_path, opener=FakeWeb({SPEC.urls[0]: GOOD}))
    assert (tmp_path / SPEC.filename).read_bytes() == GOOD


def test_download_falls_back_to_the_next_address(tmp_path: Path):
    web = FakeWeb({SPEC.urls[0]: urllib.error.URLError("offline"), SPEC.urls[1]: GOOD})
    found = models.download(SPEC, tmp_path, opener=web)
    assert found.ok
    assert web.requested == list(SPEC.urls)


def test_download_never_installs_bytes_that_fail_verification(tmp_path: Path):
    web = FakeWeb({SPEC.urls[0]: b"an error page", SPEC.urls[1]: b"X" * len(GOOD)})
    with pytest.raises(DoorVouchError) as failure:
        models.download(SPEC, tmp_path, opener=web)
    message = str(failure.value)
    assert SPEC.urls[0] in message and SPEC.urls[1] in message
    assert "by hand" in message
    assert list(tmp_path.iterdir()) == []


def test_only_https_addresses_are_fetched():
    with pytest.raises(DoorVouchError, match="non-HTTPS"):
        models._open_url("http://example.com/model.onnx")


def test_require_lists_what_is_missing(tmp_path: Path):
    with pytest.raises(DoorVouchError) as failure:
        models.require(tmp_path, [SPEC])
    assert "tiny.onnx: missing" in str(failure.value)
    assert "doorvouch models download" in str(failure.value)

    (tmp_path / SPEC.filename).write_bytes(GOOD)
    assert models.require(tmp_path, [SPEC]) == {"tiny": tmp_path / SPEC.filename}
