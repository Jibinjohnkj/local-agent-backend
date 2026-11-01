"""Migration fixture databases under `tests/fixtures/db/` (Requirements 18.19, 18.20, 18.22, 18.24).

The checked-in files are binary and never regenerated. Each test works on a copy in `tmp_path`.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import re
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from local_agent.storage import StorageError
from local_agent.storage.database import Database
from local_agent.storage.migrations import CURRENT_SCHEMA_VERSION, SchemaError, apply

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "db"
SAMPLES = sorted(FIXTURE_DIR.glob("v*_sample.db"))
TABLES = {"schema_version", "contexts", "messages", "tasks", "artifacts"}


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_fixtures", FIXTURE_DIR / "build_fixtures.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listing(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def _golden(sample: Path) -> dict[str, Any]:
    golden = sample.with_suffix(".expected.json")
    return json.loads(golden.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture
def opened() -> Iterator[list[Database]]:
    """Databases opened by a test, closed afterwards."""
    databases: list[Database] = []
    yield databases
    for database in databases:
        database.close()


def _open(opened: list[Database], path: Path, *, create: bool = False) -> Database:
    database = Database.open(path, create=create)
    opened.append(database)
    return database


def _table_names(database: Database) -> set[str]:
    return database.run_sync(
        lambda conn: {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    )


def _versions(database: Database) -> list[int]:
    return database.run_sync(
        lambda conn: [row[0] for row in conn.execute("SELECT version FROM schema_version")]
    )


def test_a_sample_fixture_is_checked_in() -> None:
    assert [sample.name for sample in SAMPLES][:1] == ["v1_sample.db"]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda path: path.name)
def test_sample_migrates_to_current_and_loads_equal_to_golden(
    sample: Path, tmp_path: Path, opened: list[Database]
) -> None:
    match = re.fullmatch(r"v(\d+)_sample\.db", sample.name)
    assert match is not None
    released_version = int(match.group(1))
    copy = tmp_path / sample.name
    shutil.copyfile(sample, copy)

    # The fixture really is at the version in its name before anything migrates it.
    with contextlib.closing(sqlite3.connect(copy)) as raw:
        stored = [row[0] for row in raw.execute("SELECT version FROM schema_version")]
    assert stored == [released_version]

    database = _open(opened, copy)
    assert apply(database, str(copy)) == CURRENT_SCHEMA_VERSION

    assert _versions(database) == [CURRENT_SCHEMA_VERSION]
    assert database.run_sync(lambda conn: builder.dump_database(conn)) == _golden(sample)
    assert database.run_sync(lambda conn: conn.execute("PRAGMA foreign_key_check").fetchall()) == []
    assert database.run_sync(lambda conn: conn.execute("PRAGMA journal_mode").fetchone()) == (
        "wal",
    )

    # A second apply on the migrated database changes nothing.
    assert apply(database, str(copy)) == CURRENT_SCHEMA_VERSION
    assert database.run_sync(lambda conn: builder.dump_database(conn)) == _golden(sample)


def test_v1_sample_covers_the_documented_edge_cases() -> None:
    golden = _golden(FIXTURE_DIR / "v1_sample.db")
    messages = [m for c in golden["contexts"] for m in c["messages"]]
    tasks = [t for c in golden["contexts"] for t in c["tasks"]]
    artifacts = [a for t in tasks for a in t["artifacts"]]

    assert any("\r\n" in (m["content"] or "") for m in messages)
    assert any("\U0001f389" in (m["content"] or "") for m in messages)
    assert any(m["role"] == "assistant" and m["content"] == "" for m in messages)
    assert {t["state"] for t in tasks} == {
        "submitted",
        "working",
        "input-required",
        "completed",
        "failed",
        "canceled",
    }
    assert {a["closed"] for a in artifacts} == {0, 1}

    arguments: list[str] = []
    for message in messages:
        if message["tool_calls"] is not None:
            arguments.extend(call["arguments"] for call in json.loads(message["tool_calls"]))

    def is_json(text: str) -> bool:
        try:
            json.loads(text)
        except ValueError:
            return False
        return True

    assert any(not is_json(text) for text in arguments)


REFUSALS = [
    pytest.param("newer_v999.db", 999, id="newer_v999"),
    pytest.param("no_version_table.db", None, id="no_version_table"),
    pytest.param("bad_version_two_rows.db", None, id="bad_version_two_rows"),
    pytest.param("bad_version_text.db", None, id="bad_version_text"),
]


@pytest.mark.parametrize(("name", "stored"), REFUSALS)
def test_refused_schema_leaves_the_copy_unchanged(
    name: str, stored: int | None, tmp_path: Path, opened: list[Database]
) -> None:
    copy = tmp_path / name
    shutil.copyfile(FIXTURE_DIR / name, copy)
    before = _sha256(copy)

    database = _open(opened, copy)
    with pytest.raises(SchemaError) as excinfo:
        apply(database, str(copy))
    database.close()

    assert excinfo.value.stored == stored
    assert excinfo.value.supported == CURRENT_SCHEMA_VERSION
    assert _sha256(copy) == before
    assert _listing(tmp_path) == [name]


def test_not_sqlite_copy_is_refused_at_open_and_left_unchanged(tmp_path: Path) -> None:
    copy = tmp_path / "not_sqlite.db"
    shutil.copyfile(FIXTURE_DIR / "not_sqlite.db", copy)
    before = _sha256(copy)

    with pytest.raises(StorageError):
        Database.open(copy, create=False)

    assert _sha256(copy) == before
    assert _listing(tmp_path) == ["not_sqlite.db"]


def _assert_fresh_schema(database: Database, path: Path) -> None:
    assert apply(database, str(path)) == 1
    assert _table_names(database) == TABLES
    assert _versions(database) == [1]


def test_zero_length_file_ends_at_version_1_with_every_table(
    tmp_path: Path, opened: list[Database]
) -> None:
    path = tmp_path / "empty.db"
    path.write_bytes(b"")

    _assert_fresh_schema(_open(opened, path), path)


def test_absent_file_ends_at_version_1_with_every_table(
    tmp_path: Path, opened: list[Database]
) -> None:
    path = tmp_path / "absent.db"
    assert not path.exists()

    _assert_fresh_schema(_open(opened, path, create=True), path)


def test_builder_writes_the_golden_it_checks_in(tmp_path: Path) -> None:
    written = builder.build_all(tmp_path)

    assert sorted(path.name for path in written) == sorted(builder.FIXTURES)
    assert _listing(tmp_path) == sorted(builder.FIXTURES)
    assert (tmp_path / "v1_sample.expected.json").read_bytes() == (
        FIXTURE_DIR / "v1_sample.expected.json"
    ).read_bytes()


def test_builder_refuses_to_overwrite_an_existing_fixture(tmp_path: Path) -> None:
    existing = tmp_path / "newer_v999.db"
    existing.write_bytes(b"released")

    with pytest.raises(FileExistsError, match=r"newer_v999\.db"):
        builder.build_all(tmp_path)

    assert existing.read_bytes() == b"released"
    assert _listing(tmp_path) == ["newer_v999.db"]
    assert builder.main(["--out-dir", str(tmp_path)]) == 1
