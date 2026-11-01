"""Fault injection for the Storage_Database, shared by the storage tests (18.9, 18.30).

`FaultInjectingConnection` wraps the real `sqlite3.Connection` and counts every statement that
reaches it: `execute`, `executemany`, `executescript`, `commit`, and `rollback`, including the same
calls on cursors made by `cursor()`. At a chosen 1-based index it raises a chosen exception
instead of running the statement, so the statement has no effect.

Pass a `FaultFactory` as `Database.open(..., connection_factory=...)`. The factory runs on the
writer thread after the pragmas and `quick_check`, so the counter sees only what `read`,
`transaction`, and `run_sync` execute. For `Database.transaction` that is `BEGIN IMMEDIATE`, the
statements of `fn`, `COMMIT`, and, after an `Exception`, `ROLLBACK`.

Enumerate crash points with a clean run first:

    counter = FaultFactory()                         # no fault, counts only
    db = Database.open(path, create=True, connection_factory=counter)
    ...run the scenario...
    for k in range(1, counter.connection.calls + 1):  # or counter.connection.commit_points
        ...run it again with FaultFactory(fail_at=k, error=crash)...

Three errors cover the design's cases: `disk_io_error`, `disk_full_error`, and `crash`.

A `SimulatedCrash` is a `BaseException`, so `run_in_transaction` skips both the commit and the
rollback, as a kill does. The connection then counts as dead: later calls raise `SimulatedCrash`
and never reach SQLite. Closing the `Database` still works, because `close` uses the real
connection, and SQLite discards the uncommitted work like a WAL replay after a kill.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Final, Self, cast

__all__ = [
    "ErrorSource",
    "FaultFactory",
    "FaultInjectingConnection",
    "SimulatedCrash",
    "crash",
    "disk_full_error",
    "disk_io_error",
]

type ErrorSource = BaseException | Callable[[], BaseException]
"""An exception to raise, or a function that builds a fresh one for each raise."""

type _Parameters = Sequence[object] | Mapping[str, object]

COMMIT_SQL: Final = "COMMIT"


class SimulatedCrash(BaseException):
    """A process kill. Derives from `BaseException` so no `except Exception` handler sees it."""


def disk_io_error() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("disk I/O error")


def disk_full_error() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database or disk is full")


def crash() -> SimulatedCrash:
    return SimulatedCrash("simulated crash")


def _normalize(sql: str) -> str:
    return " ".join(sql.split())


def _is_commit(sql: str) -> bool:
    words = _normalize(sql).rstrip(";").upper().split(" ")
    return words[0] in {"COMMIT", "END"}


class _Counter:
    """The shared statement counter and fault schedule for one connection and its cursors."""

    def __init__(self, fail_at: Collection[int], error: ErrorSource) -> None:
        self.fail_at = frozenset(fail_at)
        self._error = error
        self.statements: list[str] = []
        self.fired: list[int] = []
        self.dead = False

    def _build(self) -> BaseException:
        return self._error() if callable(self._error) else self._error

    def hit(self, sql: str) -> None:
        """Count one statement and raise the planned error if its index is a fault point."""
        if self.dead:
            raise SimulatedCrash("connection abandoned after a simulated crash")
        self.statements.append(sql)
        index = len(self.statements)
        if index in self.fail_at:
            self.fired.append(index)
            error = self._build()
            if not isinstance(error, Exception):
                self.dead = True
            raise error


class _FaultInjectingCursor:
    """A cursor of the wrapped connection. Its `execute*` calls count like the connection's."""

    def __init__(self, cursor: sqlite3.Cursor, counter: _Counter) -> None:
        self._cursor = cursor
        self._counter = counter

    def execute(self, sql: str, parameters: _Parameters = (), /) -> Self:
        self._counter.hit(sql)
        self._cursor.execute(sql, parameters)
        return self

    def executemany(self, sql: str, seq_of_parameters: Iterable[_Parameters], /) -> Self:
        self._counter.hit(sql)
        self._cursor.executemany(sql, seq_of_parameters)
        return self

    def executescript(self, sql_script: str, /) -> Self:
        self._counter.hit(sql_script)
        self._cursor.executescript(sql_script)
        return self

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Any:
        return next(self._cursor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class FaultInjectingConnection:
    """Wraps `conn`; counts statements and raises an error at the 1-based indexes in `fail_at`.

    `fail_at` is one index or a collection of indexes (for example `{k, k + 1}` for two faults in
    a row). With none, the wrapper only counts. `error` is an exception, or a function that
    builds one per raise (`disk_io_error`, `disk_full_error`, `crash`). A raised error replaces
    the statement, which then does not run.

    Attributes other than the methods below, such as `in_transaction`, `close`, and
    `total_changes`, pass through to the real connection.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        fail_at: int | Collection[int] | None = None,
        error: ErrorSource = disk_io_error,
    ) -> None:
        points: Collection[int] = (
            () if fail_at is None else (fail_at,) if isinstance(fail_at, int) else fail_at
        )
        self._conn = conn
        self._counter = _Counter(points, error)

    @property
    def real(self) -> sqlite3.Connection:
        """The wrapped connection, for assertions that must not be counted or faulted."""
        return self._conn

    @property
    def statements(self) -> tuple[str, ...]:
        """The SQL text of every counted statement, in order. Index 0 is statement 1."""
        return tuple(self._counter.statements)

    @property
    def calls(self) -> int:
        """The number of statements counted so far, including faulted ones."""
        return len(self._counter.statements)

    @property
    def commit_points(self) -> tuple[int, ...]:
        """The 1-based indexes of the `COMMIT` statements counted so far."""
        return tuple(
            index for index, sql in enumerate(self._counter.statements, start=1) if _is_commit(sql)
        )

    @property
    def fired(self) -> tuple[int, ...]:
        """The indexes at which an error was raised."""
        return tuple(self._counter.fired)

    @property
    def crashed(self) -> bool:
        """Whether a `SimulatedCrash` (or other `BaseException`) ended this connection."""
        return self._counter.dead

    def execute(self, sql: str, parameters: _Parameters = (), /) -> sqlite3.Cursor:
        self._counter.hit(sql)
        return self._conn.execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Iterable[_Parameters], /) -> sqlite3.Cursor:
        self._counter.hit(sql)
        return self._conn.executemany(sql, seq_of_parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        self._counter.hit(sql_script)
        return self._conn.executescript(sql_script)

    def commit(self) -> None:
        self._counter.hit(COMMIT_SQL)
        self._conn.commit()

    def rollback(self) -> None:
        self._counter.hit("ROLLBACK")
        self._conn.rollback()

    def cursor(self) -> _FaultInjectingCursor:
        return _FaultInjectingCursor(self._conn.cursor(), self._counter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def as_connection(self) -> sqlite3.Connection:
        """This wrapper typed as `sqlite3.Connection`, as `ConnectionFactory` requires."""
        return cast("sqlite3.Connection", self)


class FaultFactory:
    """A `ConnectionFactory` that wraps the connection in a `FaultInjectingConnection`.

    Keeps the wrapper in `connection` so a test can read the statement counter after the run.
    Each call replaces it, so use a new `FaultFactory` for each `Database`.
    """

    def __init__(
        self,
        *,
        fail_at: int | Collection[int] | None = None,
        error: ErrorSource = disk_io_error,
    ) -> None:
        self._fail_at = fail_at
        self._error = error
        self._connection: FaultInjectingConnection | None = None

    def __call__(self, conn: sqlite3.Connection) -> sqlite3.Connection:
        wrapper = FaultInjectingConnection(conn, fail_at=self._fail_at, error=self._error)
        self._connection = wrapper
        return wrapper.as_connection()

    @property
    def connection(self) -> FaultInjectingConnection:
        """The wrapper made by the last call. Raises if `Database.open` has not run the factory."""
        if self._connection is None:
            raise RuntimeError("the factory has not wrapped a connection yet")
        return self._connection
