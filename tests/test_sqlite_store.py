from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from doorvouch.errors import DoorVouchError
from doorvouch.storage import SqliteStore
from doorvouch.storage.sqlite_store import SCHEMA_VERSION
from helpers import axis

NOON = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def test_people_are_listed_with_their_sample_counts(store):
    alice = store.add_person("Alice", is_admin=True, now=NOON)
    ruth = store.add_person("Ruth", now=NOON)
    store.add_face_sample(alice.id, axis(0))
    store.add_face_sample(alice.id, axis(1))

    summaries = store.list_people()

    assert [(s.person, s.samples) for s in summaries] == [(alice, 2), (ruth, 0)]
    assert summaries[0].created_at == NOON
    assert store.get_person(alice.id) == alice
    assert store.get_person(999) is None


def test_names_are_trimmed_and_must_not_be_empty(store):
    assert store.add_person("  Alice   Mary ").name == "Alice Mary"
    with pytest.raises(DoorVouchError, match="needs a name"):
        store.add_person("   ")


def test_people_can_be_found_by_name_ignoring_case(store):
    alice = store.add_person("Alice")
    assert store.find_person_by_name("alice") == alice
    assert store.find_person_by_name("Beth") is None


def test_gallery_round_trip_keeps_embeddings_to_float32_precision(store):
    alice = store.add_person("Alice", is_admin=True)
    embedding = (0.1, -0.25, 0.333333333, 0.9)
    store.add_face_sample(alice.id, embedding)

    ((person, loaded),) = store.load_gallery()

    assert person == alice
    assert loaded == pytest.approx(embedding, abs=1e-7)


def test_a_sample_needs_an_existing_person(store):
    with pytest.raises(DoorVouchError, match="no person #42"):
        store.add_face_sample(42, axis(0))


def test_has_admin_needs_an_admin_with_a_face(store):
    assert store.has_admin() is False
    alice = store.add_person("Alice", is_admin=True)
    assert store.has_admin() is False
    store.add_face_sample(alice.id, axis(0))
    assert store.has_admin() is True


def test_removing_a_person_erases_their_samples_and_events(store):
    alice = store.add_person("Alice", is_admin=True)
    ruth = store.add_person("Ruth")
    store.add_face_sample(alice.id, axis(0))
    store.add_face_sample(ruth.id, axis(1))
    store.log_event("person_seen", person_id=alice.id)
    store.log_event("person_seen", person_id=ruth.id)

    assert store.remove_person(alice.id) is True

    assert [person for person, _ in store.load_gallery()] == [ruth]
    assert [event.person_id for event in store.list_events()] == [ruth.id]
    assert store.remove_person(alice.id) is False


def test_removing_the_admin_keeps_the_visitors_they_vouched_for(store):
    alice = store.add_person("Alice", is_admin=True)
    pending = store.create_pending(axis(2), vouched_by=alice.id)
    store.remove_person(alice.id)
    assert store.get_pending(pending.id).vouched_by is None


def test_pending_visitor_keeps_embedding_snapshot_and_counters(store, tmp_path: Path):
    alice = store.add_person("Alice", is_admin=True)

    pending = store.create_pending(axis(2), vouched_by=alice.id, snapshot=b"jpeg", now=NOON)

    assert pending.embedding == pytest.approx(axis(2))
    assert pending.vouched_by == alice.id
    assert pending.seen_count == 1
    assert pending.created_at == pending.last_seen_at == NOON
    assert pending.snapshot_path == tmp_path / "snapshots" / f"pending-{pending.id:06d}.jpg"
    assert pending.snapshot_path.read_bytes() == b"jpeg"

    store.touch_pending(pending.id, now=NOON + timedelta(hours=1))
    again = store.get_pending(pending.id)
    assert again.seen_count == 2
    assert again.last_seen_at == NOON + timedelta(hours=1)
    assert store.list_pending() == [again]


def test_no_snapshot_is_kept_without_a_snapshot_directory():
    with SqliteStore() as store:
        pending = store.create_pending(axis(2), vouched_by=None, snapshot=b"jpeg")
        assert pending.snapshot_path is None


def test_naming_a_pending_visitor_makes_them_a_person_and_forgets_the_snapshot(store):
    pending = store.create_pending(axis(2), vouched_by=None, snapshot=b"jpeg")
    snapshot = pending.snapshot_path

    brian = store.name_pending(pending.id, "Brian", now=NOON)

    assert brian.name == "Brian"
    assert brian.is_admin is False
    assert store.list_pending() == []
    assert not snapshot.exists()
    ((person, embedding),) = store.load_gallery()
    assert person == brian
    assert embedding == pytest.approx(axis(2))
    (event,) = store.list_events()
    assert (event.kind, event.person_id, event.pending_id) == (
        "pending_named",
        brian.id,
        pending.id,
    )


def test_naming_needs_an_existing_pending_visitor_and_a_name(store):
    with pytest.raises(DoorVouchError, match="no pending visitor #5"):
        store.name_pending(5, "Brian")
    pending = store.create_pending(axis(2), vouched_by=None)
    with pytest.raises(DoorVouchError, match="needs a name"):
        store.name_pending(pending.id, " ")
    assert store.get_pending(pending.id) is not None


def test_discarding_a_pending_visitor_deletes_embedding_and_snapshot(store):
    pending = store.create_pending(axis(2), vouched_by=None, snapshot=b"jpeg")

    assert store.discard_pending(pending.id) is True

    assert store.list_pending() == []
    assert not pending.snapshot_path.exists()
    assert store.load_gallery() == []
    assert [event.kind for event in store.list_events()] == ["pending_discarded"]
    assert store.discard_pending(pending.id) is False


def test_events_come_back_newest_first_with_their_details(store):
    store.log_event("first", detail={"reason": "not_admin"}, now=NOON)
    store.log_event("second", pending_id=3, now=NOON + timedelta(seconds=1))
    store.log_event("third", now=NOON + timedelta(seconds=2))

    events = store.list_events(limit=2)

    assert [event.kind for event in events] == ["third", "second"]
    assert events[1].pending_id == 3
    assert store.list_events()[2].detail == {"reason": "not_admin"}
    assert store.list_events()[2].created_at == NOON


def test_timestamps_must_carry_a_timezone(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        store.log_event("x", now=datetime(2026, 11, 10, 12, 0))


def test_data_survives_closing_and_reopening(tmp_path: Path):
    with SqliteStore.open(tmp_path / "home") as first:
        alice = first.add_person("Alice", is_admin=True)
        first.add_face_sample(alice.id, axis(0))
        first.create_pending(axis(2), vouched_by=alice.id, snapshot=b"jpeg")

    with SqliteStore.open(tmp_path / "home") as second:
        assert [person for person, _ in second.load_gallery()] == [alice]
        (pending,) = second.list_pending()
        assert pending.snapshot_path.read_bytes() == b"jpeg"
    assert (tmp_path / "home" / "doorvouch.db").is_file()


def test_database_from_a_newer_version_is_refused(tmp_path: Path):
    path = tmp_path / "doorvouch.db"
    SqliteStore(path).close()
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()

    with pytest.raises(DoorVouchError, match="newer DoorVouch"):
        SqliteStore(path)


def test_backup_copies_people_and_pending_visitors(store, tmp_path: Path):
    alice = store.add_person("Alice", is_admin=True)
    store.add_face_sample(alice.id, axis(0))
    store.create_pending(axis(2), vouched_by=alice.id)

    copy_path = tmp_path / "copies" / "trial.db"
    store.backup_to(copy_path)

    with SqliteStore(copy_path) as copy:
        assert [person for person, _ in copy.load_gallery()] == [alice]
        assert len(copy.list_pending()) == 1
        copy.discard_pending(copy.list_pending()[0].id)
    assert len(store.list_pending()) == 1


def test_data_version_changes_only_for_other_connections(tmp_path: Path):
    path = tmp_path / "doorvouch.db"
    with SqliteStore(path) as mine, SqliteStore(path) as theirs:
        start = mine.data_version()
        mine.add_person("Alice")
        assert mine.data_version() == start
        theirs.add_person("Ruth")
        assert mine.data_version() != start
