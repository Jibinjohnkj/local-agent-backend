"""Storage_Database: one SQLite connection on one dedicated writer thread (14.1, 18.13, 18.14).

`Database` owns a single `sqlite3` connection. The connection is created inside a
`ThreadPoolExecutor(max_workers=1)` thread and never leaves it, so `check_same_thread` keeps its
default and every statement runs on that thread, never on the event loop thread. One thread also
serializes all transactions in submission order.

Writes use `autocommit=True` with an explicit `BEGIN IMMEDIATE ... COMMIT`, so each transaction
takes the write lock up front. Timestamps are stored as integers, so no datetime adapters exist.

Open never deletes, truncates, or recreates the database file or its `-wal`/`-shm` sidecars. A file
that is not a database, fails `quick_check`, cannot be written, or belongs to a SQLite older than
3.24 raises `StorageError` and is left as it was (18.24). When the file is already corrupt, the
integrity check runs before the first pragma that writes, so the file's bytes stay unchanged.

An existing file that holds tables and is not yet in WAL mode is switched to WAL by
`migrations.apply` (through `ensure_wal`) once the schema is accepted, not by `Database.open`. A
refused schema therefore leaves the file byte-identical (18.22). A new or empty file, and a file
already in WAL mode, is in WAL mode when `open` returns.

`StorageError.reason` carries the SQLite or OS message, or only the exception class name for other
exceptions, so statement parameters and stored text never reach a log (16.11).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Self
from urllib.parse import quote

from local_agent.storage import StaleState, StorageError

__all__ = [
    "MIN_SQLITE_VERSION",
    "ConnectionFactory",
    "Database",
    "StaleState",
    "StorageError",
    "ensure_wal",
    "run_in_transaction",
]

MIN_SQLITE_VERSION: Final = (3, 24)
"""UPSERT needs 3.24. Partial indexes need 3.8."""

THREAD_NAME_PREFIX: Final = "local-agent-db"

_FILE_MODE: Final = 0o600
_BUSY_TIMEOUT_MS: Final = 5000
_SYNCHRONOUS_NORMAL: Final = 1
_QUICK_CHECK_REPORT_LINES: Final = 3

type ConnectionFactory = Callable[[sqlite3.Connection], sqlite3.Connection]
"""Wraps the fully configured connection, for fault injection in tests.

The factory runs on the writer thread after the pragmas and `quick_check` passed, so its statement
counter sees only the statements that `read`, `transaction`, and `run_sync` execute. Open, close,
and `PRAGMA optimize` use the real connection and bypass the wrapper.
"""


def _reason(exc: BaseException) -> str:
    """The text that may be logged for `exc`.

    SQLite and OS messages name the failure, never the stored values. Any other exception is
    reduced to its class name, because its message could hold user text or a secret.
    """
    if isinstance(exc, sqlite3.Error):
        return str(exc) or type(exc).__name__
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return type(exc).__name__


def _storage_error(path: str, exc: BaseException) -> StorageError:
    return StorageError(path, _reason(exc))


def _scalar(conn: sqlite3.Connection, sql: str) -> object:
    row = conn.execute(sql).fetchone()
    return None if row is None else row[0]


def _database_uri(path: Path, mode: str) -> str:
    """A SQLite URI for `path` with `mode`. Windows drive paths get a leading slash."""
    posix = path.absolute().as_posix()
    if not posix.startswith("/"):
        posix = f"/{posix}"
    return f"file:{quote(posix)}?mode={mode}"


def _check_sqlite_version(path: str) -> None:
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        required = ".".join(str(part) for part in MIN_SQLITE_VERSION)
        raise StorageError(
            path,
            f"SQLite {sqlite3.sqlite_version} is too old; version {required} or newer is required",
        )


def _precreate(path: Path) -> None:
    """Create the file as `0600` so SQLite never creates it with a broader mode (17.36)."""
    if sys.platform == "win32":
        pass  # A new file inherits the protected DACL of the Storage_Directory (17.37).
    else:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, _FILE_MODE)
        except FileExistsError:
            return  # Another process won the race; SQLite opens whatever is there.
        os.close(fd)


def _is_wal(conn: sqlite3.Connection) -> bool:
    mode = _scalar(conn, "PRAGMA journal_mode")
    return isinstance(mode, str) and mode.lower() == "wal"


def _has_user_tables(conn: sqlite3.Connection) -> bool:
    sql = (
        "SELECT count(*) FROM sqlite_schema "
        "WHERE type = 'table' AND substr(name, 1, 7) != 'sqlite_'"
    )
    return _scalar(conn, sql) != 0


def ensure_wal(conn: sqlite3.Connection, path: str) -> None:
    """Put the database in WAL mode and read the mode back. Does nothing when it already is."""
    if _is_wal(conn):
        return
    mode = _scalar(conn, "PRAGMA journal_mode=WAL")
    if not (isinstance(mode, str) and mode.lower() == "wal"):
        raise StorageError(path, f"journal_mode is {mode!r}, expected 'wal'")


def _configure(conn: sqlite3.Connection, path: str) -> None:
    """Apply the pragmas, reading each back, and verify the file's integrity (18.13, 18.24)."""
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    if _scalar(conn, "PRAGMA busy_timeout") != _BUSY_TIMEOUT_MS:
        raise StorageError(path, "busy_timeout could not be set")

    # Run the integrity check before any pragma that writes, so a damaged file is not touched.
    report = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
    if report != ["ok"]:
        detail = "; ".join(report[:_QUICK_CHECK_REPORT_LINES]) or "no result"
        raise StorageError(path, f"integrity check failed: {detail}")

    # Switching an existing rollback-journal file to WAL rewrites its header. A file that already
    # holds tables stays as it is until `migrations.apply` has accepted its schema and calls
    # `ensure_wal`, so a refused schema leaves the file's bytes unchanged (18.22).
    if _is_wal(conn) or not _has_user_tables(conn):
        ensure_wal(conn, path)

    conn.execute("PRAGMA synchronous=NORMAL")
    if _scalar(conn, "PRAGMA synchronous") != _SYNCHRONOUS_NORMAL:
        raise StorageError(path, "synchronous could not be set to NORMAL")

    conn.execute("PRAGMA foreign_keys=ON")
    if _scalar(conn, "PRAGMA foreign_keys") != 1:
        raise StorageError(path, "foreign_keys could not be enabled")


def _connect(
    path: Path, *, create: bool, factory: ConnectionFactory | None
) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """Open and configure the connection. Runs on the writer thread.

    Returns the real connection and the one handed to callers (the same object without a factory).
    """
    text = str(path)
    existed = path.exists()
    if not existed and not create:
        raise StorageError(text, "database file does not exist")
    # `mode=rw` silently opens a file read-only when it is not writable, and a database already in
    # WAL mode still reads that way. Check first, so no sidecar is created next to such a file.
    if existed and not os.access(path, os.R_OK | os.W_OK):
        raise StorageError(text, "database file is not readable and writable")

    try:
        if not existed:
            _precreate(path)
        conn = sqlite3.connect(
            _database_uri(path, "rw" if existed else "rwc"),
            uri=True,
            autocommit=True,
            timeout=_BUSY_TIMEOUT_MS / 1000,
        )
    except (sqlite3.Error, OSError) as exc:
        raise _storage_error(text, exc) from exc

    try:
        _configure(conn, text)
    except StorageError:
        conn.close()
        raise
    except sqlite3.Error as exc:
        conn.close()
        raise _storage_error(text, exc) from exc
    except BaseException:
        conn.close()
        raise
    return conn, conn if factory is None else factory(conn)


def _rollback(conn: sqlite3.Connection) -> None:
    """Roll back, ignoring a failure: the original error is the one worth reporting."""
    with contextlib.suppress(sqlite3.Error):
        conn.execute("ROLLBACK")


def run_in_transaction[T](
    conn: sqlite3.Connection, fn: Callable[[sqlite3.Connection], T], path: str = ""
) -> T:
    """Run `fn(conn)` inside `BEGIN IMMEDIATE ... COMMIT` on the calling thread (18.30).

    Any `Exception` rolls back and is re-raised as `StorageError` naming `path`, except
    `StaleState` and `StorageError`, which are re-raised as they are. A `BaseException` (a
    simulated kill in tests) skips both the commit and the rollback, like a real kill. The caller
    must be on the thread that owns `conn`; `Database.transaction` and `Database.run_sync`
    guarantee that. `run_in_transaction` issues no statements besides `BEGIN IMMEDIATE`, `fn`'s
    own, `COMMIT`, and `ROLLBACK`, so a statement counter sees exactly those.
    """
    try:
        conn.execute("BEGIN IMMEDIATE")
    except Exception as exc:
        raise _storage_error(path, exc) from exc
    try:
        result = fn(conn)
        conn.execute("COMMIT")
    except StaleState, StorageError:
        _rollback(conn)
        raise
    except Exception as exc:
        _rollback(conn)
        raise _storage_error(path, exc) from exc
    return result


class Database:
    """The Storage_Database: one connection, one writer thread. Build one with `Database.open`."""

    __slots__ = ("_closed", "_conn", "_executor", "_optimize_on_close", "_raw", "path")

    path: Path

    def __init__(
        self,
        path: Path,
        executor: ThreadPoolExecutor,
        raw: sqlite3.Connection,
        conn: sqlite3.Connection,
    ) -> None:
        self.path = path
        self._executor = executor
        self._raw = raw
        self._conn = conn
        self._closed = False
        self._optimize_on_close = True

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        create: bool,
        connection_factory: ConnectionFactory | None = None,
    ) -> Self:
        """Open `path` on a new writer thread. Blocks the caller; call it before the loop exists.

        An existing file is opened `mode=rw`. An absent file is created only when `create` is true
        (`mode=rwc`, pre-created as `0600` on POSIX); otherwise it is a `StorageError`. Raises
        `StorageError` for an old SQLite library, a file that is not a database, a failed
        `quick_check`, a file that cannot be written, or a pragma that does not read back as set.
        The file is never deleted, truncated, or replaced.
        """
        _check_sqlite_version(str(path))
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=THREAD_NAME_PREFIX)
        try:
            raw, conn = executor.submit(
                _connect, path, create=create, factory=connection_factory
            ).result()
        except BaseException:
            executor.shutdown(wait=True)
            raise
        return cls(path, executor, raw, conn)

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError(str(self.path), "database is closed")

    def _guarded[T](self, fn: Callable[[sqlite3.Connection], T]) -> Callable[[], T]:
        """Bind `fn` to the connection and map SQLite and OS failures to `StorageError`."""
        conn = self._conn

        def call() -> T:
            try:
                return fn(conn)
            except (sqlite3.Error, OSError) as exc:
                raise _storage_error(str(self.path), exc) from exc

        return call

    def run_sync[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run `fn(conn)` on the writer thread and block until it returns. Startup only (14.1).

        Calling this from the event loop thread would block the loop, so it is for the steps that
        run before the loop exists. Use `run_in_transaction` inside `fn` for an atomic step.
        """
        self._ensure_open()
        return self._executor.submit(self._guarded(fn)).result()

    async def read[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run `fn(conn)` on the writer thread without a transaction and await the result."""
        self._ensure_open()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._guarded(fn))

    async def transaction[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run `fn(conn)` in `BEGIN IMMEDIATE ... COMMIT` on the writer thread (18.30).

        Any `Exception` rolls back and re-raises as `StorageError`, or as `StaleState` when `fn`
        raised it. A `BaseException` skips both the commit and the rollback.
        """
        self._ensure_open()
        conn = self._conn
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, run_in_transaction, conn, fn, str(self.path)
        )

    def refuse(self) -> None:
        """Record that the schema was refused, so `close` leaves the file unmodified (18.22).

        `PRAGMA optimize` can create `sqlite_stat1`, which would write to a refused database.
        """
        self._optimize_on_close = False

    def close(self) -> None:
        """Run `PRAGMA optimize`, close the connection on its thread, and stop the thread.

        Safe to call twice. A WAL checkpoint is not required: the next open replays the WAL.
        `PRAGMA optimize` is skipped after `refuse`.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.submit(self._close_connection).result()
        finally:
            self._executor.shutdown(wait=True)

    def _close_connection(self) -> None:
        try:
            if self._optimize_on_close:
                with contextlib.suppress(sqlite3.Error):
                    self._raw.execute("PRAGMA optimize")
        finally:
            self._raw.close()
