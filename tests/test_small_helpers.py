from __future__ import annotations

from pathlib import Path

import pytest

from doorvouch.config import Settings
from doorvouch.fps import FpsMeter
from doorvouch.text import ascii_label


def test_fps_is_zero_until_two_frames_were_seen():
    meter = FpsMeter()
    assert meter.fps == 0.0
    meter.tick(1.0)
    assert meter.fps == 0.0


def test_fps_of_evenly_spaced_frames():
    meter = FpsMeter()
    for index in range(11):
        meter.tick(index * 0.05)
    assert meter.fps == pytest.approx(20.0)


def test_fps_follows_only_the_recent_window():
    meter = FpsMeter(window=5)
    for index in range(10):
        meter.tick(index * 1.0)  # slow start: 1 FPS
    for index in range(1, 6):
        meter.tick(9.0 + index * 0.1)  # then 10 FPS
    assert meter.fps == pytest.approx(10.0)


def test_fps_window_must_hold_two_frames():
    with pytest.raises(ValueError):
        FpsMeter(window=1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Renée", "Renee"), ("Jürgen", "Jurgen"), ("Zoë", "Zoe"), ("Alice", "Alice"), ("李", "?")],
)
def test_ascii_label_strips_accents_for_the_preview_font(text, expected):
    assert ascii_label(text) == expected


def test_home_comes_from_argument_then_environment_then_default(tmp_path: Path):
    environ = {"DOORVOUCH_HOME": str(tmp_path / "from-env")}

    assert Settings.load(home=tmp_path / "explicit", environ=environ).home == tmp_path / "explicit"
    assert Settings.load(environ=environ).home == tmp_path / "from-env"
    assert Settings.load(environ={}, cwd=tmp_path).home == tmp_path / "data"


def test_settings_derive_their_paths_from_home(tmp_path: Path):
    settings = Settings(home=tmp_path)
    assert settings.models_dir == tmp_path / "models"
    assert settings.snapshots_dir == tmp_path / "snapshots"
    assert settings.database_path == tmp_path / "doorvouch.db"
