"""Where DoorVouch keeps its files and which thresholds it uses."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from doorvouch.core.engine import EngineConfig

HOME_VARIABLE = "DOORVOUCH_HOME"
DEFAULT_HOME = "data"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. ``home`` holds the database, the snapshots and the models."""

    home: Path
    engine: EngineConfig = field(default_factory=EngineConfig)

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def snapshots_dir(self) -> Path:
        return self.home / "snapshots"

    @property
    def database_path(self) -> Path:
        return self.home / "doorvouch.db"

    @classmethod
    def load(
        cls,
        *,
        home: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> Settings:
        """Pick the home directory: the argument, then ``DOORVOUCH_HOME``, then ``./data``."""
        variables = os.environ if environ is None else environ
        chosen = home or variables.get(HOME_VARIABLE) or DEFAULT_HOME
        path = Path(chosen).expanduser()
        return cls(home=path if path.is_absolute() else (cwd or Path.cwd()) / path)
