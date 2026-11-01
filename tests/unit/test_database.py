"""Unit tests for the Storage_Database (Requirements 18.13, 18.14, 18.24, 18.30).

The fault-injection tests use `FaultFactory` from `tests/storage_faults.py`. It counts the
statements that `Database.transaction` sends, so a test can fail or crash at each one in turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import stat
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from conftest import posix_only
from local_agent.storage import StaleState, StorageError
from local_agent.storage.database import THREAD_NAME_PREFIX, Database
from storage_faults import (
    FaultFactory,
    SimulatedCrash,
    crash,
    disk_full_error,
    disk_io_error,
)

SCHEMA = "CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "agent.db"


@pytest.fixture
def db(db_path: Path) -> Iterator[Database]:
    database = Database.open(db_path, create=True)
    try:
        yield database
    finally:
        database.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listing(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def _create_schema(path: Path) -> None:
    """Create the `item` table in a new database at `path`, then close it."""
    database = Database.open(path, create=True)
    try:
        database.run_sync(lambda conn: conn.execute(SCHEMA))
    finally:
        database.close()


def _names(database: Database) -> list[str]:
    return database.run_sync(
        lambda conn: [row[0] for row in conn.execute("SELECT name FROM item ORDER BY id")]
    )


def _reopened_names(path: Path) -> list[str]:
    database = Database.open(path, create=False)
    try:
        return _names(database)
    finally:
        database.close()


def _insert_three(conn: sqlite3.Connection) -> str:
    for name in ("a", "b", "c"):
        conn.execute("INSERT INTO item (name) VALUES (?)", (name,))
    return "done"


def _db_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(THREAD_NAME_PREFIX)]


# --- Pragmas (18.13) -------------------------------------------------------------------------


def test_pragmas_read_back_as_configured(db: Database) -> None:
    def pragmas(conn: sqlite3.Connection) -> dict[str, object]:
        names = ("journal_mode", "synchronous", "foreign_keys", "busy_timeout")
        return {name: conn.execute(f"PRAGMA {name}").fetchone()[0] for name in names}

    values = db.run_sync(pragmas)

    assert str(values["journal_mode"]).lower() == "wal"
    assert values["synchronous"] == 1
    assert values["foreign_keys"] == 1
    assert values["busy_timeout"] == 5000


def test_foreign_keys_are_enforced(db: Database) -> None:
    def violate(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (parent_id INTEGER REFERENCES parent (id))")
        conn.execute("INSERT INTO child (parent_id) VALUES (99)")

    with pytest.raises(StorageError) as caught:
        db.run_sync(violate)

    assert "FOREIGN KEY" in caught.value.reason


def test_open_reports_the_database_path_in_errors(db_path: Path) -> None:
    with pytest.raises(StorageError) as caught:
        Database.open(db_path, create=False)

    assert caught.value.path == str(db_path)
    assert str(db_path) in str(caught.value)


# --- Writer thread (14.1, 18.14) -------------------------------------------------------------


def test_connection_is_created_on_the_writer_thread(db_path: Path) -> None:
    created_on: list[str] = []

    def factory(conn: sqlite3.Connection) -> sqlite3.Connection:
        created_on.append(threading.current_thread().name)
        return conn

    database = Database.open(db_path, create=True, connection_factory=factory)
    database.close()

    assert len(created_on) == 1
    assert created_on[0].startswith(THREAD_NAME_PREFIX)
    assert created_on[0] != threading.current_thread().name


async def test_every_sql_call_runs_on_the_writer_thread(db_path: Path) -> None:
    loop_thread = threading.current_thread().name
    database = Database.open(db_path, create=True)
    seen: list[str] = []

    def trace(conn: sqlite3.Connection) -> None:
        conn.set_trace_callback(lambda _sql: seen.append(threading.current_thread().name))

    try:
        database.run_sync(trace)
        database.run_sync(lambda conn: conn.execute(SCHEMA))
        await database.read(lambda conn: conn.execute("SELECT count(*) FROM item").fetchone())
        await database.transaction(_insert_three)
        await database.transaction(lambda conn: conn.execute("DELETE FROM item"))
    finally:
        database.close()

    assert len(seen) >= 6  # CREATE, SELECT, BEGIN, 3 INSERTs, COMMIT, BEGIN, DELETE, COMMIT, ...
    assert {name for name in seen if not name.startswith(THREAD_NAME_PREFIX)} == set()
    assert loop_thread not in seen


async def test_callbacks_never_run_on_the_event_loop_thread(db: Database) -> None:
    loop_ident = threading.get_ident()
    idents: list[int] = []

    def record(conn: sqlite3.Connection) -> None:
        idents.append(threading.get_ident())

    db.run_sync(record)
    await db.read(record)
    await db.transaction(record)

    assert len(idents) == 3
    assert loop_ident not in idents
    assert len(set(idents)) == 1


def test_connection_cannot_be_used_from_another_thread(db: Database) -> None:
    conn = db.run_sync(lambda c: c)

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


async def test_transactions_run_in_submission_order(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))

    def insert(name: str) -> Callable[[sqlite3.Connection], object]:
        def run(conn: sqlite3.Connection) -> object:
            return conn.execute("INSERT INTO item (name) VALUES (?)", (name,))

        return run

    names = [f"n{i}" for i in range(20)]
    tasks = [asyncio.ensure_future(db.transaction(insert(name))) for name in names]
    await asyncio.gather(*tasks)

    assert _names(db) == names


async def test_read_runs_without_a_transaction(db: Database) -> None:
    in_transaction = await db.read(lambda conn: conn.in_transaction)

    assert in_transaction is False


async def test_transaction_runs_inside_a_transaction(db: Database) -> None:
    in_transaction = await db.transaction(lambda conn: conn.in_transaction)

    assert in_transaction is True


# --- Exceptions roll back and map to StorageError (18.30, 16.11) -----------------------------


async def test_exception_rolls_back_and_raises_storage_error(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))

    def fail(conn: sqlite3.Connection) -> None:
        _insert_three(conn)
        raise RuntimeError("secret user text")

    with pytest.raises(StorageError) as caught:
        await db.transaction(fail)

    assert _names(db) == []
    assert caught.value.path == str(db.path)
    assert caught.value.reason == "RuntimeError"
    assert "secret" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


async def test_sqlite_error_keeps_the_sqlite_message(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))

    def fail(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO item (name) VALUES ('kept out')")
        conn.execute("INSERT INTO item (id, name) VALUES (1, 'duplicate id')")

    with pytest.raises(StorageError) as caught:
        await db.transaction(fail)

    assert "UNIQUE" in caught.value.reason
    assert "kept out" not in caught.value.reason
    assert _names(db) == []


async def test_stale_state_rolls_back_and_passes_through(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))

    def lose_the_race(conn: sqlite3.Connection) -> None:
        _insert_three(conn)
        raise StaleState

    with pytest.raises(StaleState):
        await db.transaction(lose_the_race)

    assert _names(db) == []


async def test_storage_error_from_the_callback_passes_through(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))
    original = StorageError("elsewhere", "own reason")

    def fail(conn: sqlite3.Connection) -> None:
        _insert_three(conn)
        raise original

    with pytest.raises(StorageError) as caught:
        await db.transaction(fail)

    assert caught.value is original
    assert _names(db) == []


async def test_connection_is_usable_after_a_failed_transaction(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))

    def fail(conn: sqlite3.Connection) -> None:
        raise ValueError

    with pytest.raises(StorageError):
        await db.transaction(fail)
    result = await db.transaction(_insert_three)

    assert result == "done"
    assert _names(db) == ["a", "b", "c"]


async def test_committed_transaction_returns_the_callback_result(db: Database) -> None:
    db.run_sync(lambda conn: conn.execute(SCHEMA))

    result = await db.transaction(_insert_three)

    assert result == "done"
    assert _names(db) == ["a", "b", "c"]


def test_run_sync_maps_sqlite_errors_to_storage_error(db: Database) -> None:
    with pytest.raises(StorageError) as caught:
        db.run_sync(lambda conn: conn.execute("SELECT * FROM no_such_table"))

    assert "no such table" in caught.value.reason
    assert caught.value.path == str(db.path)


async def test_read_maps_sqlite_errors_to_storage_error(db: Database) -> None:
    with pytest.raises(StorageError) as caught:
        await db.read(lambda conn: conn.execute("SELECT * FROM no_such_table"))

    assert "no such table" in caught.value.reason


# --- Injected faults (18.9, 18.30) -----------------------------------------------------------


async def _clean_statements(path: Path) -> tuple[str, ...]:
    """The statements `transaction(_insert_three)` sends, counted on a clean run."""
    counter = FaultFactory()
    database = Database.open(path, create=False, connection_factory=counter)
    try:
        await database.transaction(_insert_three)
    finally:
        database.close()
    return counter.connection.statements


async def test_transaction_sends_only_begin_the_callback_and_commit(tmp_path: Path) -> None:
    path = tmp_path / "count.db"
    _create_schema(path)

    statements = await _clean_statements(path)

    assert statements[0] == "BEGIN IMMEDIATE"
    assert statements[-1] == "COMMIT"
    assert len(statements) == 5  # BEGIN IMMEDIATE, three INSERTs, COMMIT


@pytest.mark.parametrize("error", [disk_io_error, disk_full_error], ids=["disk_io", "disk_full"])
async def test_error_at_every_statement_rolls_back(
    tmp_path: Path, error: Callable[[], sqlite3.OperationalError]
) -> None:
    path = tmp_path / "faults.db"
    _create_schema(path)
    total = len(await _clean_statements(path))
    assert _reopened_names(path) == ["a", "b", "c"]  # the clean run committed

    for k in range(1, total + 1):
        path_k = tmp_path / f"faults-{k}.db"
        _create_schema(path_k)
        factory = FaultFactory(fail_at=k, error=error)
        database = Database.open(path_k, create=False, connection_factory=factory)
        try:
            with pytest.raises(StorageError) as caught:
                await database.transaction(_insert_three)
            assert factory.connection.fired == (k,)
        finally:
            database.close()

        expected = "disk I/O error" if error is disk_io_error else "database or disk is full"
        assert caught.value.reason == expected
        assert _reopened_names(path_k) == []


async def test_failed_commit_is_followed_by_a_rollback(tmp_path: Path) -> None:
    path = tmp_path / "commit.db"
    _create_schema(path)
    commit_at = len(await _clean_statements(path))
    path_2 = tmp_path / "commit-2.db"
    _create_schema(path_2)
    factory = FaultFactory(fail_at=commit_at)
    database = Database.open(path_2, create=False, connection_factory=factory)

    try:
        with pytest.raises(StorageError):
            await database.transaction(_insert_three)
        statements = factory.connection.statements
        assert statements[-2:] == ("COMMIT", "ROLLBACK")
        assert await database.transaction(_insert_three) == "done"
    finally:
        database.close()

    assert _reopened_names(path_2) == ["a", "b", "c"]


async def test_failed_begin_raises_storage_error_without_rollback(tmp_path: Path) -> None:
    path = tmp_path / "begin.db"
    _create_schema(path)
    factory = FaultFactory(fail_at=1)
    database = Database.open(path, create=False, connection_factory=factory)

    try:
        with pytest.raises(StorageError) as caught:
            await database.transaction(_insert_three)
    finally:
        database.close()

    assert caught.value.reason == "disk I/O error"
    assert factory.connection.statements == ("BEGIN IMMEDIATE",)


async def test_failed_rollback_does_not_hide_the_original_error(tmp_path: Path) -> None:
    path = tmp_path / "rollback.db"
    _create_schema(path)
    commit_at = len(await _clean_statements(path))
    path_2 = tmp_path / "rollback-2.db"
    _create_schema(path_2)
    factory = FaultFactory(fail_at={commit_at, commit_at + 1}, error=disk_full_error)
    database = Database.open(path_2, create=False, connection_factory=factory)

    try:
        with pytest.raises(StorageError) as caught:
            await database.transaction(_insert_three)
    finally:
        database.close()

    assert caught.value.reason == "database or disk is full"
    assert factory.connection.fired == (commit_at, commit_at + 1)
    assert _reopened_names(path_2) == []


async def test_simulated_crash_leaves_no_commit_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "base.db"
    _create_schema(path)
    total = len(await _clean_statements(path))

    for k in range(1, total + 1):
        path_k = tmp_path / f"crash-{k}.db"
        _create_schema(path_k)
        factory = FaultFactory(fail_at=k, error=crash)
        database = Database.open(path_k, create=False, connection_factory=factory)
        try:
            with pytest.raises(SimulatedCrash):
                await database.transaction(_insert_three)
            assert factory.connection.crashed
        finally:
            database.close()

        assert _reopened_names(path_k) == [], f"crash at statement {k} left a visible commit"


async def test_simulated_crash_keeps_earlier_commits(tmp_path: Path) -> None:
    path = tmp_path / "earlier.db"
    _create_schema(path)
    counter = FaultFactory()
    database = Database.open(path, create=False, connection_factory=counter)
    try:
        await database.transaction(_insert_three)
        await database.transaction(_insert_three)
        commits = counter.connection.commit_points
    finally:
        database.close()
    assert len(commits) == 2
    path_2 = tmp_path / "earlier-2.db"
    _create_schema(path_2)

    factory = FaultFactory(fail_at=commits[1], error=crash)
    database = Database.open(path_2, create=False, connection_factory=factory)
    try:
        await database.transaction(_insert_three)
        with pytest.raises(SimulatedCrash):
            await database.transaction(_insert_three)
    finally:
        database.close()

    assert _reopened_names(path_2) == ["a", "b", "c"]


async def test_crashed_transaction_does_not_roll_back_by_itself(tmp_path: Path) -> None:
    path = tmp_path / "norollback.db"
    _create_schema(path)
    factory = FaultFactory(fail_at=3, error=crash)
    database = Database.open(path, create=False, connection_factory=factory)

    try:
        with pytest.raises(SimulatedCrash):
            await database.transaction(_insert_three)
        assert "ROLLBACK" not in factory.connection.statements
        assert "COMMIT" not in factory.connection.statements
    finally:
        database.close()


# --- Open refuses without touching the file (18.24) ------------------------------------------


def test_garbage_file_raises_storage_error_and_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "garbage.db"
    path.write_bytes(b"this is not a sqlite database\n" * 200)
    before = _sha256(path)
    listing = _listing(tmp_path)

    with pytest.raises(StorageError) as caught:
        Database.open(path, create=True)

    assert caught.value.path == str(path)
    assert "not a database" in caught.value.reason
    assert _sha256(path) == before
    assert _listing(tmp_path) == listing
    assert _db_threads() == []


def test_corrupt_database_fails_the_integrity_check_and_is_untouched(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.db"
    database = Database.open(path, create=True)
    try:
        database.run_sync(lambda conn: conn.execute(SCHEMA))
        database.run_sync(
            lambda conn: conn.executemany(
                "INSERT INTO item (name) VALUES (?)", [(f"row {i} " * 40,) for i in range(300)]
            )
        )
    finally:
        database.close()
    page_size = 4096
    data = bytearray(path.read_bytes())
    assert len(data) > 3 * page_size
    data[page_size : 3 * page_size] = b"\xff" * (2 * page_size)
    path.write_bytes(bytes(data))
    before = _sha256(path)
    listing = _listing(tmp_path)

    with pytest.raises(StorageError) as caught:
        Database.open(path, create=False)

    assert caught.value.path == str(path)
    assert _sha256(path) == before
    assert _listing(tmp_path) == listing
    assert _db_threads() == []


@posix_only
def test_read_only_file_raises_storage_error_and_is_untouched(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root can write to a read-only file")
    path = tmp_path / "readonly.db"
    _create_schema(path)
    path.chmod(0o400)
    before = _sha256(path)
    listing = _listing(tmp_path)

    with pytest.raises(StorageError) as caught:
        Database.open(path, create=False)

    assert caught.value.path == str(path)
    assert _sha256(path) == before
    assert _listing(tmp_path) == listing
    assert _db_threads() == []


def test_old_sqlite_raises_storage_error_and_leaves_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "old.db"
    _create_schema(path)
    before = _sha256(path)
    listing = _listing(tmp_path)
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 23, 0))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.23.0")

    with pytest.raises(StorageError) as caught:
        Database.open(path, create=False)

    assert "3.23.0" in caught.value.reason
    assert "3.24" in caught.value.reason
    assert _sha256(path) == before
    assert _listing(tmp_path) == listing
    assert _db_threads() == []


def test_old_sqlite_does_not_create_a_missing_file(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 23, 0))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.23.0")

    with pytest.raises(StorageError):
        Database.open(db_path, create=True)

    assert not db_path.exists()


def test_missing_file_without_create_raises_and_creates_nothing(
    db_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(StorageError) as caught:
        Database.open(db_path, create=False)

    assert "does not exist" in caught.value.reason
    assert _listing(tmp_path) == []
    assert _db_threads() == []


def test_failed_open_stops_the_writer_thread(tmp_path: Path) -> None:
    path = tmp_path / "garbage.db"
    path.write_bytes(b"x" * 5000)

    with pytest.raises(StorageError):
        Database.open(path, create=False)

    assert _db_threads() == []


@posix_only
def test_new_database_file_mode_is_0600(db_path: Path) -> None:
    database = Database.open(db_path, create=True)
    try:
        database.run_sync(lambda conn: conn.execute(SCHEMA))
        mode = stat.S_IMODE(db_path.stat().st_mode)
    finally:
        database.close()

    assert mode == 0o600


@posix_only
def test_new_database_file_mode_is_0600_under_a_permissive_umask(db_path: Path) -> None:
    old = os.umask(0)
    try:
        database = Database.open(db_path, create=True)
        database.close()
    finally:
        os.umask(old)

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


@posix_only
def test_reopen_does_not_change_the_file_mode(db_path: Path) -> None:
    _create_schema(db_path)
    db_path.chmod(0o640)

    database = Database.open(db_path, create=False)
    database.close()

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o640


# --- Lifecycle -------------------------------------------------------------------------------


def test_close_stops_the_writer_thread_and_is_idempotent(db_path: Path) -> None:
    database = Database.open(db_path, create=True)
    assert len(_db_threads()) == 1

    database.close()
    database.close()

    assert _db_threads() == []


def test_run_sync_after_close_raises_storage_error(db_path: Path) -> None:
    database = Database.open(db_path, create=True)
    database.close()

    with pytest.raises(StorageError) as caught:
        database.run_sync(lambda conn: conn.execute("SELECT 1"))

    assert caught.value.reason == "database is closed"


async def test_read_and_transaction_after_close_raise_storage_error(db_path: Path) -> None:
    database = Database.open(db_path, create=True)
    database.close()

    with pytest.raises(StorageError):
        await database.read(lambda conn: conn.execute("SELECT 1"))
    with pytest.raises(StorageError):
        await database.transaction(lambda conn: conn.execute("SELECT 1"))


def test_data_survives_close_and_reopen(db_path: Path) -> None:
    database = Database.open(db_path, create=True)
    try:
        database.run_sync(lambda conn: conn.execute(SCHEMA))
        database.run_sync(_insert_three)
    finally:
        database.close()

    assert _reopened_names(db_path) == ["a", "b", "c"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_wal_sidecars_are_created_private(db_path: Path) -> None:
    database = Database.open(db_path, create=True)
    try:
        database.run_sync(lambda conn: conn.execute(SCHEMA))
        sidecars = [p for p in (Path(f"{db_path}-wal"), Path(f"{db_path}-shm")) if p.exists()]
        modes = [stat.S_IMODE(p.stat().st_mode) for p in sidecars]
    finally:
        database.close()

    assert sidecars  # WAL mode creates at least one sidecar while the database is open
    assert all(mode & 0o077 == 0 for mode in modes)
