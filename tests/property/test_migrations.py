"""Property 54: migrations preserve data and roll back on failure (Requirements 18.19-18.21).

Only schema version 1 is released, so the property also runs over a synthetic, test-only migration
list injected through `apply(..., migrations=...)`: v1 is the real `SCHEMA_V1`, v2 adds columns, and
v3 rebuilds the `messages` table. Every example:

1. builds a database at version `v` by applying migrations `1..v` to an empty or absent file and
   writes a generated set of Contexts, Messages, Tasks and Artifacts;
2. applies the remaining migrations on a copy and checks the final version, that every row reads
   back equal through the version 1 columns, and that a second `apply` changes nothing;
3. replays the same run on another copy with a fault injected at one statement inside one
   migration `m` (an I/O error, a full disk, or a simulated kill), then checks that the file is at
   version `m - 1` with schema and rows identical to a database migrated only up to `m - 1`, that
   `MigrationError` names the path and both versions, and that a later clean `apply` finishes.

The statement indexes of each migration come from a clean, counting run, so every statement inside
every remaining migration (`BEGIN IMMEDIATE` through `COMMIT`) is a possible fault point.
"""

from __future__ import annotations

import contextlib
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.storage.database import Database, run_in_transaction
from local_agent.storage.migrations import (
    MIGRATIONS,
    SCHEMA_V1,
    Migration,
    MigrationError,
    apply,
)
from storage_faults import (
    ErrorSource,
    FaultFactory,
    SimulatedCrash,
    crash,
    disk_full_error,
    disk_io_error,
)

type Row = tuple[Any, ...]

V1_TABLES = frozenset({"schema_version", "contexts", "messages", "tasks", "artifacts"})
V1_INDEXES = frozenset(
    {
        "contexts_by_activity",
        "messages_by_task",
        "tasks_by_context",
        "tasks_live",
        "tasks_by_update",
    }
)

SYNTHETIC: tuple[Migration, ...] = (
    Migration(1, "initial schema", SCHEMA_V1),
    Migration(
        2,
        "add columns",
        (
            "ALTER TABLE contexts ADD COLUMN label TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE artifacts ADD COLUMN media_type TEXT",
        ),
    ),
    Migration(
        3,
        "rebuild messages with a token_count column",
        (
            """CREATE TABLE messages_new (
    context_id    TEXT    NOT NULL REFERENCES contexts (context_id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    task_id       TEXT,
    role          TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content       TEXT,
    tool_calls    TEXT,
    tool_call_id  TEXT,
    token_count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (context_id, seq),
    CHECK ((role = 'tool') = (tool_call_id IS NOT NULL)),
    CHECK (role = 'assistant' OR tool_calls IS NULL),
    CHECK (role = 'assistant' OR content IS NOT NULL)
)""",
            "INSERT INTO messages_new (context_id, seq, task_id, role, content, tool_calls, "
            "tool_call_id) SELECT context_id, seq, task_id, role, content, tool_calls, "
            "tool_call_id FROM messages",
            "DROP TABLE messages",
            "ALTER TABLE messages_new RENAME TO messages",
            "CREATE INDEX messages_by_task ON messages (task_id) WHERE task_id IS NOT NULL",
        ),
    ),
)

ERRORS: dict[str, ErrorSource] = {
    "io": disk_io_error,
    "full": disk_full_error,
    "crash": crash,
}

# --- generated data -----------------------------------------------------------------------------

text_values = st.one_of(
    st.text(alphabet=st.characters(blacklist_categories=["Cs"]), max_size=12),
    st.sampled_from(
        ["", " ", "a\r\nb", "\r", "\n", "\t", "\x00", "x\x00y", "\U0001f389", "{not json"]
    ),
)
key_values = st.text(alphabet="abcdefghij0123456789-_", min_size=1, max_size=6)
millis = st.integers(min_value=0, max_value=2**62)
states = st.sampled_from(
    ["submitted", "working", "input-required", "completed", "failed", "canceled"]
)


@dataclass(frozen=True)
class Dataset:
    contexts: list[Row]
    messages: list[Row]
    tasks: list[Row]
    artifacts: list[Row]


@st.composite
def datasets(draw: st.DrawFn) -> Dataset:
    context_ids = draw(st.lists(key_values, max_size=3, unique=True))
    contexts = [
        (cid, draw(st.none() | text_values), draw(millis), draw(millis)) for cid in context_ids
    ]

    messages: list[Row] = []
    for cid in context_ids:
        for seq in draw(st.lists(st.integers(min_value=0, max_value=50), max_size=3, unique=True)):
            role = draw(st.sampled_from(["user", "assistant", "tool"]))
            task_id = draw(st.none() | key_values)
            row: Row
            if role == "assistant":
                row = (
                    cid,
                    seq,
                    task_id,
                    role,
                    draw(st.none() | text_values),
                    draw(st.none() | text_values),
                    None,
                )
            elif role == "tool":
                row = (cid, seq, task_id, role, draw(text_values), None, draw(key_values))
            else:
                row = (cid, seq, task_id, role, draw(text_values), None, None)
            messages.append(row)

    tasks: list[Row] = []
    artifacts: list[Row] = []
    if context_ids:
        for task_id in draw(st.lists(key_values, max_size=3, unique=True)):
            tasks.append(
                (
                    task_id,
                    draw(st.sampled_from(context_ids)),
                    draw(states),
                    draw(st.none() | text_values),
                    draw(text_values),
                    draw(millis),
                    draw(millis),
                    draw(st.none() | text_values),
                )
            )
            count = draw(st.integers(min_value=0, max_value=3))
            artifact_ids = draw(st.lists(key_values, min_size=count, max_size=count, unique=True))
            ordinals = draw(
                st.lists(
                    st.integers(min_value=0, max_value=20),
                    min_size=count,
                    max_size=count,
                    unique=True,
                )
            )
            for artifact_id, ordinal in zip(artifact_ids, ordinals, strict=True):
                artifacts.append(
                    (
                        task_id,
                        artifact_id,
                        ordinal,
                        draw(text_values),
                        draw(st.sampled_from([0, 1])),
                    )
                )
    return Dataset(contexts, messages, tasks, artifacts)


# --- database helpers ---------------------------------------------------------------------------


@contextlib.contextmanager
def opened(path: Path, factory: FaultFactory | None = None) -> Iterator[Database]:
    database = Database.open(path, create=True, connection_factory=factory)
    try:
        yield database
    finally:
        database.close()


def _insert(conn: sqlite3.Connection, data: Dataset) -> None:
    conn.executemany(
        "INSERT INTO contexts (context_id, system_prompt, created_at_ms, last_activity_ms) "
        "VALUES (?, ?, ?, ?)",
        data.contexts,
    )
    conn.executemany(
        "INSERT INTO messages (context_id, seq, task_id, role, content, tool_calls, "
        "tool_call_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        data.messages,
    )
    conn.executemany(
        "INSERT INTO tasks (task_id, context_id, state, status_message, metadata, "
        "created_at_ms, updated_at_ms, a2a_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        data.tasks,
    )
    conn.executemany(
        "INSERT INTO artifacts (task_id, artifact_id, ordinal, text, closed) "
        "VALUES (?, ?, ?, ?, ?)",
        data.artifacts,
    )


V1_QUERIES: dict[str, str] = {
    "contexts": "SELECT context_id, system_prompt, created_at_ms, last_activity_ms "
    "FROM contexts ORDER BY context_id",
    "messages": "SELECT context_id, seq, task_id, role, content, tool_calls, tool_call_id "
    "FROM messages ORDER BY context_id, seq",
    "tasks": "SELECT task_id, context_id, state, status_message, metadata, created_at_ms, "
    "updated_at_ms, a2a_snapshot FROM tasks ORDER BY task_id",
    "artifacts": "SELECT task_id, artifact_id, ordinal, text, closed FROM artifacts "
    "ORDER BY task_id, artifact_id",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _v1_view(conn: sqlite3.Connection) -> dict[str, list[Row]]:
    """Every row of the four data tables through the version 1 columns. Absent tables are empty."""
    present = _table_names(conn)
    return {
        name: [tuple(row) for row in conn.execute(sql)] if name in present else []
        for name, sql in V1_QUERIES.items()
    }


def _versions(conn: sqlite3.Connection) -> list[Row] | None:
    if "schema_version" not in _table_names(conn):
        return None
    return [tuple(row) for row in conn.execute("SELECT version FROM schema_version")]


def _full_snapshot(conn: sqlite3.Connection) -> dict[str, list[Row]]:
    """The schema text and every row of every column, for exact before/after comparison."""
    snapshot: dict[str, list[Row]] = {
        "schema": sorted(
            (
                tuple(row)
                for row in conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_schema")
            ),
            key=repr,
        )
    }
    for name in sorted(_table_names(conn)):
        quoted = '"' + name.replace('"', '""') + '"'
        snapshot[name] = sorted(
            (tuple(row) for row in conn.execute(f"SELECT * FROM {quoted}")), key=repr
        )
    return snapshot


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _copy(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copyfile(source, target)


def _build_pre(
    path: Path, migrations: Sequence[Migration], version: int, absent: bool, data: Dataset
) -> None:
    """Create the database at `version` with `data`. Version 0 is a zero-length or absent file."""
    if version == 0:
        if not absent:
            path.write_bytes(b"")
        return
    with opened(path) as database:
        assert apply(database, str(path), migrations=migrations[:version]) == version
        database.run_sync(
            lambda conn: run_in_transaction(conn, lambda c: _insert(c, data), str(path))
        )


def _check_final_schema(conn: sqlite3.Connection, final: int) -> None:
    assert _table_names(conn) >= V1_TABLES
    indexes = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_schema WHERE type = 'index'")
    }
    assert indexes >= V1_INDEXES
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    if final >= 2:
        assert {"label"} <= _columns(conn, "contexts")
        assert {"priority"} <= _columns(conn, "tasks")
        assert {"media_type"} <= _columns(conn, "artifacts")
        assert {row[0] for row in conn.execute("SELECT label FROM contexts")} <= {""}
        assert {row[0] for row in conn.execute("SELECT priority FROM tasks")} <= {0}
    if final >= 3:
        assert {"token_count"} <= _columns(conn, "messages")
        assert {row[0] for row in conn.execute("SELECT token_count FROM messages")} <= {0}


# Feature: local-agent-backend, Property 54: Migrations preserve data and roll back on failure
@settings(max_examples=100)
@given(
    synthetic=st.booleans(),
    start=st.integers(min_value=0, max_value=3),
    absent=st.booleans(),
    data=datasets(),
    fault_pick=st.integers(min_value=0, max_value=10_000),
    error_kind=st.sampled_from(sorted(ERRORS)),
)
def test_migrations_preserve_data_and_roll_back_on_failure(
    synthetic: bool,
    start: int,
    absent: bool,
    data: Dataset,
    fault_pick: int,
    error_kind: str,
) -> None:
    migrations = SYNTHETIC if synthetic else tuple(MIGRATIONS)
    target = len(migrations)
    version = start % (target + 1)
    remaining = migrations[version:]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pre, clean, faulty, reference = (
            root / f"{name}.db" for name in ("pre", "clean", "faulty", "reference")
        )

        _build_pre(pre, migrations, version, absent, data)
        before: dict[str, list[Row]] = {name: [] for name in V1_QUERIES}
        if pre.exists():
            with opened(pre) as database:
                before = database.run_sync(_v1_view)
        assert {name: len(rows) for name, rows in before.items()} == (
            {
                "contexts": len(data.contexts),
                "messages": len(data.messages),
                "tasks": len(data.tasks),
                "artifacts": len(data.artifacts),
            }
            if version > 0
            else {"contexts": 0, "messages": 0, "tasks": 0, "artifacts": 0}
        )

        # Clean run: every remaining migration applies, nothing is lost, and each statement inside
        # a migration is counted so it can become a fault point below.
        _copy(pre, clean)
        counter = FaultFactory()
        with opened(clean, counter) as database:
            assert apply(database, str(clean), migrations=migrations) == target
            statements = counter.connection.statements
            commits = counter.connection.commit_points

            assert database.run_sync(_versions) == [(target,)]
            assert database.run_sync(_v1_view) == before
            database.run_sync(lambda conn: _check_final_schema(conn, target))
            finished = database.run_sync(_full_snapshot)

            assert apply(database, str(clean), migrations=migrations) == target
            assert database.run_sync(_full_snapshot) == finished

        begins = [
            index for index, sql in enumerate(statements, start=1) if sql == "BEGIN IMMEDIATE"
        ]
        assert len(begins) == len(commits) == len(remaining)
        if not remaining:
            return

        windows = list(zip(begins, commits, strict=True))
        points = [k for begin, commit in windows for k in range(begin, commit + 1)]
        fault_at = points[fault_pick % len(points)]
        failed = next(i for i, (begin, commit) in enumerate(windows) if begin <= fault_at <= commit)
        m = remaining[failed].version

        # Failure run: the same scenario with one injected fault inside migration `m`.
        _copy(pre, faulty)
        fault = FaultFactory(fail_at=fault_at, error=ERRORS[error_kind])
        caught: MigrationError | None = None
        with opened(faulty, fault) as database:
            if error_kind == "crash":
                with pytest.raises(SimulatedCrash):
                    apply(database, str(faulty), migrations=migrations)
            else:
                with pytest.raises(MigrationError) as excinfo:
                    apply(database, str(faulty), migrations=migrations)
                caught = excinfo.value
        assert fault.connection.fired == (fault_at,)

        if caught is not None:
            assert caught.path == str(faulty)
            assert caught.from_version == m - 1
            assert caught.target_version == m
            message = str(caught)
            assert str(faulty) in message
            assert str(m - 1) in message
            assert str(m) in message

        # The reference is the same data migrated only as far as `m - 1`.
        _copy(pre, reference)
        with opened(reference) as database:
            assert apply(database, str(reference), migrations=migrations[: m - 1]) == m - 1
            expected = database.run_sync(_full_snapshot)

        with opened(faulty) as database:
            assert database.run_sync(_versions) == (None if m == 1 else [(m - 1,)])
            assert database.run_sync(_full_snapshot) == expected

            # Nothing is stuck: a later clean start finishes the job without losing a row.
            assert apply(database, str(faulty), migrations=migrations) == target
            assert database.run_sync(_versions) == [(target,)]
            assert database.run_sync(_v1_view) == before
            database.run_sync(lambda conn: _check_final_schema(conn, target))
