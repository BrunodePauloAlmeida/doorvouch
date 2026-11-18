"""Everything DoorVouch remembers, in one SQLite file."""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from doorvouch.core.embeddings import Embedding
from doorvouch.core.people import PendingVisitor, Person
from doorvouch.errors import DoorVouchError

SCHEMA_VERSION = 1

_SCHEMA = """
BEGIN;
CREATE TABLE people (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE TABLE face_samples (
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX face_samples_person ON face_samples(person_id);
CREATE TABLE pending_visitors (
    id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    snapshot_file TEXT,
    vouched_by INTEGER REFERENCES people(id) ON DELETE SET NULL,
    seen_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    person_id INTEGER,
    pending_id INTEGER,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX events_person ON events(person_id);
PRAGMA user_version = 1;
COMMIT;
"""


@dataclass(frozen=True, slots=True)
class PersonSummary:
    """A person plus how many face samples they have."""

    person: Person
    samples: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Event:
    """One line of the event log."""

    id: int
    created_at: datetime
    kind: str
    person_id: int | None
    pending_id: int | None
    detail: dict[str, object]


def _iso(moment: datetime | None) -> str:
    moment = moment or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _pack(embedding: Embedding) -> bytes:
    if not embedding:
        raise ValueError("embedding is empty")
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _unpack(blob: bytes) -> Embedding:
    if not blob or len(blob) % 4:
        raise DoorVouchError("The database holds a damaged face embedding.")
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def _clean_name(name: str) -> str:
    clean = " ".join(name.split())
    if not clean:
        raise DoorVouchError("A person needs a name.")
    return clean


class SqliteStore:
    """People, face samples, pending visitors and the event log.

    Embeddings are stored as little-endian float32. Snapshots of pending
    visitors are JPEG files in ``snapshots_dir``; when that is ``None`` no
    picture is kept.
    """

    def __init__(self, path: Path | str = ":memory:", *, snapshots_dir: Path | None = None) -> None:
        self._path = str(path)
        self._snapshots_dir = snapshots_dir
        in_memory = self._path == ":memory:"
        if not in_memory:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not in_memory:
            # Lets `doorvouch run` read while another terminal enrolls or names people.
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    @classmethod
    def open(cls, home: Path) -> Self:
        """Open the store kept inside the DoorVouch home directory."""
        return cls(home / "doorvouch.db", snapshots_dir=home / "snapshots")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _migrate(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise DoorVouchError(
                f"The database at {self._path} was created by a newer DoorVouch "
                f"(schema {version}, this version understands {SCHEMA_VERSION})."
            )
        if version == 0:
            self._conn.executescript(_SCHEMA)

    # People -----------------------------------------------------------------

    def add_person(
        self, name: str, *, is_admin: bool = False, now: datetime | None = None
    ) -> Person:
        clean = _clean_name(name)
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO people (name, is_admin, created_at) VALUES (?, ?, ?)",
                (clean, int(is_admin), _iso(now)),
            )
        return Person(_rowid(cursor), clean, is_admin)

    def get_person(self, person_id: int) -> Person | None:
        row = self._conn.execute(
            "SELECT id, name, is_admin FROM people WHERE id = ?", (person_id,)
        ).fetchone()
        return _person(row) if row is not None else None

    def find_person_by_name(self, name: str) -> Person | None:
        row = self._conn.execute(
            "SELECT id, name, is_admin FROM people WHERE name = ? COLLATE NOCASE ORDER BY id",
            (_clean_name(name),),
        ).fetchone()
        return _person(row) if row is not None else None

    def list_people(self) -> list[PersonSummary]:
        rows = self._conn.execute(
            """
            SELECT p.id, p.name, p.is_admin, p.created_at, COUNT(s.id) AS samples
            FROM people AS p LEFT JOIN face_samples AS s ON s.person_id = p.id
            GROUP BY p.id ORDER BY p.id
            """
        ).fetchall()
        return [
            PersonSummary(_person(row), int(row["samples"]), _moment(row["created_at"]))
            for row in rows
        ]

    def has_admin(self) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM people AS p JOIN face_samples AS s ON s.person_id = p.id
            WHERE p.is_admin = 1 LIMIT 1
            """
        ).fetchone()
        return row is not None

    def remove_person(self, person_id: int) -> bool:
        """Delete a person, their face samples and every event that mentions them."""
        with self._conn:
            self._conn.execute("DELETE FROM events WHERE person_id = ?", (person_id,))
            cursor = self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        return cursor.rowcount > 0

    def add_face_sample(
        self, person_id: int, embedding: Embedding, *, now: datetime | None = None
    ) -> int:
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO face_samples (person_id, embedding, created_at) VALUES (?, ?, ?)",
                    (person_id, _pack(embedding), _iso(now)),
                )
        except sqlite3.IntegrityError as error:
            raise DoorVouchError(f"There is no person #{person_id}.") from error
        return _rowid(cursor)

    def load_gallery(self) -> list[tuple[Person, Embedding]]:
        rows = self._conn.execute(
            """
            SELECT p.id, p.name, p.is_admin, s.embedding
            FROM face_samples AS s JOIN people AS p ON p.id = s.person_id
            ORDER BY s.id
            """
        ).fetchall()
        return [(_person(row), _unpack(row["embedding"])) for row in rows]

    # Pending visitors -------------------------------------------------------

    def create_pending(
        self,
        embedding: Embedding,
        *,
        vouched_by: int | None,
        snapshot: bytes | None = None,
        now: datetime | None = None,
    ) -> PendingVisitor:
        stamp = _iso(now)
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO pending_visitors (embedding, vouched_by, created_at, last_seen_at)
                VALUES (?, ?, ?, ?)
                """,
                (_pack(embedding), vouched_by, stamp, stamp),
            )
            pending_id = _rowid(cursor)
            if snapshot is not None and self._snapshots_dir is not None:
                filename = f"pending-{pending_id:06d}.jpg"
                self._snapshots_dir.mkdir(parents=True, exist_ok=True)
                (self._snapshots_dir / filename).write_bytes(snapshot)
                self._conn.execute(
                    "UPDATE pending_visitors SET snapshot_file = ? WHERE id = ?",
                    (filename, pending_id),
                )
        pending = self.get_pending(pending_id)
        if pending is None:  # pragma: no cover - the row was inserted just above
            raise DoorVouchError("The pending visitor could not be saved.")
        return pending

    def touch_pending(self, pending_id: int, *, now: datetime | None = None) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE pending_visitors SET seen_count = seen_count + 1, last_seen_at = ?
                WHERE id = ?
                """,
                (_iso(now), pending_id),
            )

    def get_pending(self, pending_id: int) -> PendingVisitor | None:
        row = self._conn.execute(
            "SELECT * FROM pending_visitors WHERE id = ?", (pending_id,)
        ).fetchone()
        return self._pending(row) if row is not None else None

    def list_pending(self) -> list[PendingVisitor]:
        rows = self._conn.execute("SELECT * FROM pending_visitors ORDER BY id").fetchall()
        return [self._pending(row) for row in rows]

    def name_pending(
        self,
        pending_id: int,
        name: str,
        *,
        is_admin: bool = False,
        now: datetime | None = None,
    ) -> Person:
        """Turn a pending visitor into a known person and forget the snapshot."""
        pending = self.get_pending(pending_id)
        if pending is None:
            raise DoorVouchError(f"There is no pending visitor #{pending_id}.")
        clean = _clean_name(name)
        stamp = _iso(now)
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO people (name, is_admin, created_at) VALUES (?, ?, ?)",
                (clean, int(is_admin), stamp),
            )
            person = Person(_rowid(cursor), clean, is_admin)
            self._conn.execute(
                "INSERT INTO face_samples (person_id, embedding, created_at) VALUES (?, ?, ?)",
                (person.id, _pack(pending.embedding), stamp),
            )
            self._conn.execute("DELETE FROM pending_visitors WHERE id = ?", (pending_id,))
            self._insert_event("pending_named", person.id, pending_id, {}, stamp)
        _unlink(pending.snapshot_path)
        return person

    def discard_pending(self, pending_id: int, *, now: datetime | None = None) -> bool:
        """Forget a pending visitor: the embedding and the snapshot are deleted."""
        pending = self.get_pending(pending_id)
        if pending is None:
            return False
        with self._conn:
            self._conn.execute("DELETE FROM pending_visitors WHERE id = ?", (pending_id,))
            self._insert_event("pending_discarded", None, pending_id, {}, _iso(now))
        _unlink(pending.snapshot_path)
        return True

    def _pending(self, row: sqlite3.Row) -> PendingVisitor:
        filename = row["snapshot_file"]
        path = None
        if filename is not None and self._snapshots_dir is not None:
            path = self._snapshots_dir / str(filename)
        return PendingVisitor(
            id=int(row["id"]),
            embedding=_unpack(row["embedding"]),
            vouched_by=row["vouched_by"],
            seen_count=int(row["seen_count"]),
            created_at=_moment(row["created_at"]),
            last_seen_at=_moment(row["last_seen_at"]),
            snapshot_path=path,
        )

    # Events -----------------------------------------------------------------

    def log_event(
        self,
        kind: str,
        *,
        person_id: int | None = None,
        pending_id: int | None = None,
        detail: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> int:
        with self._conn:
            return self._insert_event(kind, person_id, pending_id, detail or {}, _iso(now))

    def _insert_event(
        self,
        kind: str,
        person_id: int | None,
        pending_id: int | None,
        detail: Mapping[str, object],
        stamp: str,
    ) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO events (created_at, kind, person_id, pending_id, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            (stamp, kind, person_id, pending_id, json.dumps(dict(detail), sort_keys=True)),
        )
        return _rowid(cursor)

    def list_events(self, *, limit: int = 50) -> list[Event]:
        """The most recent events, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (max(0, limit),)
        ).fetchall()
        return [
            Event(
                id=int(row["id"]),
                created_at=_moment(row["created_at"]),
                kind=str(row["kind"]),
                person_id=row["person_id"],
                pending_id=row["pending_id"],
                detail=json.loads(row["detail"]),
            )
            for row in rows
        ]

    # Housekeeping -----------------------------------------------------------

    def data_version(self) -> int:
        """A number that changes when another connection modifies the database."""
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def backup_to(self, path: Path) -> None:
        """Copy the whole database to ``path`` (snapshots are not copied)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(path))
        try:
            self._conn.backup(target)
        finally:
            target.close()


def _rowid(cursor: sqlite3.Cursor) -> int:
    rowid = cursor.lastrowid
    if rowid is None:  # pragma: no cover - sqlite always sets it after an INSERT
        raise DoorVouchError("The database did not return an id for the new row.")
    return int(rowid)


def _person(row: sqlite3.Row) -> Person:
    return Person(int(row["id"]), str(row["name"]), bool(row["is_admin"]))


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _unlink(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)
