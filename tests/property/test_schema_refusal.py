"""Property 55: an unsupported schema is refused and the database file is left unchanged.

`classify_schema` and `plan` are checked against an independent statement of the refusal rule. For
every refused case the test also builds a real SQLite file, runs the startup sequence
(`Database.open` then `apply`), and compares the file's SHA-256 and the directory listing before and
after. The file is generated both in WAL mode and in rollback-journal mode, because switching an
existing file to WAL rewrites its header.
"""

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.storage.database import Database
from local_agent.storage.migrations import (
    NO_READABLE_VERSION,
    Migration,
    Refusal,
    SchemaError,
    apply,
    classify_schema,
    plan,
)

VERSION_TABLE_NAMES = ("schema_version", "SCHEMA_VERSION", "Schema_Version")

# Hypothesis runs the body many times, so keep each example small.
table_names = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True).filter(
    lambda name: not name.startswith("sqlite_") and name != "schema_version"
)

# A SQLite column without a declared type keeps text as text and a real as a real.
version_values = st.one_of(
    st.integers(min_value=-2, max_value=7),
    st.text(max_size=5),
    st.sampled_from(["0", "1", "2", "3"]),
    st.floats(allow_nan=False, allow_infinity=False),
    st.sampled_from([1.0, 2.0]),
    st.none(),
)

# `None` is an absent table. A list is the rows of a present table: none, one, or several.
version_contents = st.one_of(
    st.none(),
    st.lists(version_values, max_size=3),
    st.lists(st.integers(min_value=-2, max_value=7), min_size=1, max_size=1),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listing(directory: Path) -> list[str]:
    return sorted(os.listdir(directory))


def _build(
    path: Path,
    tables: list[str],
    contents: list[object] | None,
    version_table_name: str,
    *,
    wal: bool,
) -> None:
    conn = sqlite3.connect(path, autocommit=True)
    try:
        if wal:
            conn.execute("PRAGMA journal_mode=WAL")
        for name in tables:
            conn.execute(f'CREATE TABLE "{name}" (id INTEGER)')
            conn.execute(f'INSERT INTO "{name}" (id) VALUES (1)')
        if contents is not None:
            conn.execute(f"CREATE TABLE {version_table_name} (version)")
            for value in contents:
                conn.execute(f"INSERT INTO {version_table_name} (version) VALUES (?)", (value,))
    finally:
        conn.close()


def _is_integer(value: object) -> bool:
    return type(value) is int


def _expected_refusal(tables: list[str], contents: list[object] | None, supported: int) -> bool:
    """The rule from Property 55, stated without `classify_schema` or `plan`."""
    if contents is None:
        return bool(tables)
    if len(contents) != 1 or not _is_integer(contents[0]):
        return True
    stored = contents[0]
    assert isinstance(stored, int)
    return stored > supported


def _expected_stored(contents: list[object] | None) -> int | None:
    if contents is not None and len(contents) == 1 and isinstance(contents[0], int):
        return contents[0]
    return None


# Feature: local-agent-backend, Property 55: Unsupported schema refused without modification
@settings(max_examples=100)
@given(
    tables=st.lists(table_names, max_size=4, unique_by=str.lower),
    contents=version_contents,
    version_table_name=st.sampled_from(VERSION_TABLE_NAMES),
    supported=st.integers(min_value=1, max_value=3),
    wal=st.booleans(),
)
def test_unsupported_schema_is_refused_without_modification(
    tables: list[str],
    contents: list[object] | None,
    version_table_name: str,
    supported: int,
    wal: bool,
) -> None:
    migrations = tuple(Migration(v, f"step {v}", ()) for v in range(1, supported + 1))
    refuse = _expected_refusal(tables, contents, supported)

    state = classify_schema(
        frozenset(tables) | (frozenset() if contents is None else frozenset({"schema_version"})),
        None if contents is None else [(value,) for value in contents],
    )
    outcome = plan(state, supported, migrations)

    assert isinstance(outcome, Refusal) == refuse
    if isinstance(outcome, Refusal):
        assert outcome.stored == _expected_stored(contents)
        assert outcome.supported == supported
    else:
        stored = _expected_stored(contents) or 0
        assert [m.version for m in outcome] == [v for v in range(1, supported + 1) if v > stored]

    if not refuse:
        return

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "agent.db"
        _build(path, tables, contents, version_table_name, wal=wal)
        digest = _sha256(path)
        listing = _listing(root)

        database = Database.open(path, create=False)
        try:
            with pytest.raises(SchemaError) as caught:
                apply(database, str(path), migrations=migrations)
            assert _sha256(path) == digest
        finally:
            database.close()

        assert _sha256(path) == digest
        assert _listing(root) == listing

        error = caught.value
        assert error.path == str(path)
        assert error.stored == _expected_stored(contents)
        assert error.supported == supported
        message = str(error)
        assert str(path) in message
        assert (NO_READABLE_VERSION if error.stored is None else str(error.stored)) in message
        assert str(supported) in message
