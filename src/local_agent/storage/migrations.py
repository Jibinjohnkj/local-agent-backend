"""Schema_Version handling and schema v1 (13.12, 18.19-18.22).

The `schema_version` table holds exactly one integer row. `apply` inspects the database read-only,
decides with two pure functions, and only then writes:

- `classify_schema(tables, version_rows)` names the state: `Empty`, `AtVersion`, `NoVersionTable`,
  or `BadVersionValue`.
- `plan(state, supported, migrations)` returns the migrations to run, in ascending order, or a
  `Refusal` for a version above the supported one, a missing `schema_version` table next to other
  tables, or a `schema_version` table that does not hold exactly one integer (18.22).

A `Refusal` becomes `SchemaError` before the first write, and `Database.open` leaves an existing
non-WAL file with tables in its rollback-journal mode, so a refused file stays byte-identical. After
a plan is accepted, `apply` switches the file to WAL mode. Each migration and its `schema_version`
write run in one `BEGIN IMMEDIATE ... COMMIT` (18.20), so a failure rolls back that migration alone
and raises `MigrationError` with the version before it and the version it targeted (18.21).

Migrations are forward-only and are never edited after release. DDL is transactional in SQLite, so a
migration that adds a column or rebuilds a table is atomic. `PRAGMA foreign_keys=ON` stays in force
while a migration runs, so a migration that drops and recreates a parent table (`contexts`, `tasks`)
would cascade-delete its children. Rebuild a parent only after copying its children out of reach.

`apply` blocks on the writer thread through `Database.run_sync`, so it runs at startup, before the
event loop exists.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from local_agent.logging_setup import get_logger
from local_agent.storage import StorageError
from local_agent.storage.database import Database, ensure_wal, run_in_transaction

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MIGRATIONS",
    "NO_READABLE_VERSION",
    "SCHEMA_V1",
    "AtVersion",
    "BadVersionValue",
    "Empty",
    "Migration",
    "MigrationError",
    "NoVersionTable",
    "Refusal",
    "SchemaError",
    "SchemaState",
    "apply",
    "classify_schema",
    "plan",
]

CURRENT_SCHEMA_VERSION: Final = 1

NO_READABLE_VERSION: Final = "no readable schema version"

_VERSION_TABLE: Final = "schema_version"

_log = get_logger("storage.migrations")


@dataclass(frozen=True, slots=True)
class Migration:
    """One forward step to `version`. `statements` run in order inside a single transaction."""

    version: int
    description: str
    statements: tuple[str, ...]


SCHEMA_V1: Final[tuple[str, ...]] = (
    """CREATE TABLE schema_version (
    version INTEGER NOT NULL CHECK (typeof(version) = 'integer')
)""",
    """CREATE TABLE contexts (
    context_id        TEXT    PRIMARY KEY,
    system_prompt     TEXT,
    created_at_ms     INTEGER NOT NULL,
    last_activity_ms  INTEGER NOT NULL
)""",
    "CREATE INDEX contexts_by_activity ON contexts (last_activity_ms)",
    """CREATE TABLE messages (
    context_id    TEXT    NOT NULL REFERENCES contexts (context_id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    task_id       TEXT,
    role          TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content       TEXT,
    tool_calls    TEXT,
    tool_call_id  TEXT,
    PRIMARY KEY (context_id, seq),
    CHECK ((role = 'tool') = (tool_call_id IS NOT NULL)),
    CHECK (role = 'assistant' OR tool_calls IS NULL),
    CHECK (role = 'assistant' OR content IS NOT NULL)
)""",
    "CREATE INDEX messages_by_task ON messages (task_id) WHERE task_id IS NOT NULL",
    """CREATE TABLE tasks (
    task_id         TEXT    PRIMARY KEY,
    context_id      TEXT    NOT NULL REFERENCES contexts (context_id) ON DELETE CASCADE,
    state           TEXT    NOT NULL CHECK (state IN ('submitted', 'working', 'input-required',
                                                      'completed', 'failed', 'canceled')),
    status_message  TEXT,
    metadata        TEXT    NOT NULL DEFAULT '{}',
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL,
    a2a_snapshot    TEXT
)""",
    "CREATE INDEX tasks_by_context ON tasks (context_id, state)",
    (
        "CREATE INDEX tasks_live ON tasks (state) "
        "WHERE state IN ('submitted', 'working', 'input-required')"
    ),
    "CREATE INDEX tasks_by_update ON tasks (updated_at_ms DESC, task_id DESC)",
    """CREATE TABLE artifacts (
    task_id      TEXT    NOT NULL REFERENCES tasks (task_id) ON DELETE CASCADE,
    artifact_id  TEXT    NOT NULL,
    ordinal      INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    closed       INTEGER NOT NULL CHECK (closed IN (0, 1)),
    PRIMARY KEY (task_id, artifact_id),
    UNIQUE (task_id, ordinal)
)""",
)
"""Schema_Version 1. Column meanings are in the design's "Storage_Database schema" section."""

MIGRATIONS: Final[tuple[Migration, ...]] = (Migration(1, "initial schema", SCHEMA_V1),)
"""Every migration, versions 1..n without gaps. Tests inject their own list through `apply`."""


class SchemaError(Exception):
    """The Storage_Database has a Schema_Version this Backend refuses to use (18.22).

    `stored` is the integer found, or `None` when no readable version exists. Nothing was written.
    """

    path: str
    stored: int | None
    supported: int

    def __init__(self, path: str, stored: int | None, supported: int) -> None:
        super().__init__(path, stored, supported)
        self.path = path
        self.stored = stored
        self.supported = supported

    @property
    def reason(self) -> str:
        found = NO_READABLE_VERSION if self.stored is None else f"schema version {self.stored}"
        return f"{found}; highest supported schema version is {self.supported}"

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


class MigrationError(Exception):
    """A migration failed and was rolled back (18.21).

    `from_version` is the Schema_Version before the failed migration, which is also the version the
    database is left at. `target_version` is the version the failed migration was creating.
    `reason` is the SQLite or OS message, already free of stored values (16.11).
    """

    path: str
    from_version: int
    target_version: int
    reason: str

    def __init__(self, path: str, from_version: int, target_version: int, reason: str = "") -> None:
        super().__init__(path, from_version, target_version, reason)
        self.path = path
        self.from_version = from_version
        self.target_version = target_version
        self.reason = reason

    def __str__(self) -> str:
        text = (
            f"{self.path}: migration from schema version {self.from_version} "
            f"to {self.target_version} failed and was rolled back"
        )
        return f"{text}: {self.reason}" if self.reason else text


@dataclass(frozen=True, slots=True)
class Empty:
    """No user tables, including a zero-length file (18.19)."""


@dataclass(frozen=True, slots=True)
class AtVersion:
    """`schema_version` holds exactly one integer."""

    version: int


@dataclass(frozen=True, slots=True)
class NoVersionTable:
    """User tables exist but `schema_version` does not."""


@dataclass(frozen=True, slots=True)
class BadVersionValue:
    """`schema_version` exists but does not hold exactly one integer."""

    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class Refusal:
    """The database cannot be used. `stored` is `None` when no version is readable (18.22)."""

    stored: int | None
    supported: int

    def error(self, path: str) -> SchemaError:
        return SchemaError(path, self.stored, self.supported)


type SchemaState = Empty | AtVersion | NoVersionTable | BadVersionValue


def classify_schema(
    tables: frozenset[str], version_rows: Sequence[tuple[Any, ...]] | None
) -> SchemaState:
    """Name the schema state from the user table names and the `schema_version` rows (pure).

    `version_rows` is `None` when the `schema_version` table is absent. A value counts as an
    integer only when its Python type is `int`, so text such as `'1'` and a real such as `1.0`
    are bad values.
    """
    if version_rows is None:
        return NoVersionTable() if tables else Empty()
    if len(version_rows) == 1 and len(version_rows[0]) == 1:
        value = version_rows[0][0]
        if type(value) is int:
            return AtVersion(value)
    return BadVersionValue(tuple(tuple(row) for row in version_rows))


def plan(
    state: SchemaState, supported: int, migrations: Sequence[Migration]
) -> list[Migration] | Refusal:
    """The migrations to run in ascending order, or a `Refusal` (pure).

    `Empty` runs every migration up to `supported`. `AtVersion(v)` runs those above `v`, which is
    none when `v == supported`. A version above `supported`, a missing version table, and a bad
    version value are refused.
    """
    match state:
        case Empty():
            current = 0
        case AtVersion(version=version) if version <= supported:
            current = version
        case AtVersion(version=version):
            return Refusal(version, supported)
        case NoVersionTable() | BadVersionValue():
            return Refusal(None, supported)
    return sorted(
        (m for m in migrations if current < m.version <= supported), key=lambda m: m.version
    )


def _inspect(conn: sqlite3.Connection) -> SchemaState:
    """Read the schema state without writing."""
    names = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_schema "
            "WHERE type = 'table' AND substr(name, 1, 7) != 'sqlite_'"
        )
    ]
    tables = frozenset(names)
    version_table = next((name for name in names if name.lower() == _VERSION_TABLE), None)
    if version_table is None:
        return classify_schema(tables, None)
    quoted = '"' + version_table.replace('"', '""') + '"'
    columns = [str(row[1]).lower() for row in conn.execute(f"PRAGMA table_info({quoted})")]
    if "version" not in columns:
        return classify_schema(tables, [])
    rows = [tuple(row) for row in conn.execute(f"SELECT version FROM {quoted} LIMIT 2")]
    return classify_schema(tables, rows)


def _step(migration: Migration, from_version: int) -> Callable[[sqlite3.Connection], None]:
    """One migration's statements plus its `schema_version` write, for a single transaction."""

    def run(conn: sqlite3.Connection) -> None:
        for statement in migration.statements:
            conn.execute(statement)
        if from_version == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (migration.version,))
        else:
            conn.execute("UPDATE schema_version SET version = ?", (migration.version,))

    return run


def _in_transaction(
    step: Callable[[sqlite3.Connection], None], path: str
) -> Callable[[sqlite3.Connection], None]:
    def run(conn: sqlite3.Connection) -> None:
        run_in_transaction(conn, step, path)

    return run


def _check_chain(migrations: Sequence[Migration]) -> None:
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise ValueError(f"migration versions must be 1..n in order, got {versions}")


def apply(db: Database, path: str, *, migrations: Sequence[Migration] | None = None) -> int:
    """Bring the database to the highest migration version and return that version.

    `path` names the database in errors. `migrations` defaults to `MIGRATIONS`, read at call time,
    and the supported version is the highest version in it. Raises `SchemaError` before any write
    when the schema is refused (18.22) and `MigrationError` when a migration fails (18.21).
    """
    available = tuple(MIGRATIONS if migrations is None else migrations)
    _check_chain(available)
    supported = len(available)

    state = db.run_sync(_inspect)
    outcome = plan(state, supported, available)
    if isinstance(outcome, Refusal):
        db.refuse()
        raise outcome.error(path)
    db.run_sync(lambda conn: ensure_wal(conn, path))

    current = state.version if isinstance(state, AtVersion) else 0
    for migration in outcome:
        started = time.monotonic()
        try:
            db.run_sync(_in_transaction(_step(migration, current), path))
        except StorageError as exc:
            raise MigrationError(path, current, migration.version, exc.reason) from exc
        _log.info(
            "schema_migrated",
            from_version=current,
            to_version=migration.version,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        current = migration.version
    return current
