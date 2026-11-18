"""The people DoorVouch knows about."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from doorvouch.core.embeddings import Embedding


@dataclass(frozen=True, slots=True)
class Person:
    """Someone with a name. Admins may vouch for visitors with a thumbs-up."""

    id: int
    name: str
    is_admin: bool = False


@dataclass(frozen=True, slots=True)
class PendingVisitor:
    """A face an admin vouched for that has not been given a name yet."""

    id: int
    embedding: Embedding
    vouched_by: int | None
    seen_count: int
    created_at: datetime
    last_seen_at: datetime
    snapshot_path: Path | None = None
