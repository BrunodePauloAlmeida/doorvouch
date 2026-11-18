from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from doorvouch.storage import SqliteStore


@pytest.fixture
def store(tmp_path: Path):
    """An in-memory store that writes pending snapshots under the test's temp dir."""
    with SqliteStore(snapshots_dir=tmp_path / "snapshots") as opened:
        yield opened
