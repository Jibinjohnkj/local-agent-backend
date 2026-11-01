"""Build the checked-in migration fixture databases (Requirements 18.19, 18.20, 18.22, 18.24).

Run once per released Schema_Version, then commit the output. The files are never regenerated:

    uv run python tests/fixtures/db/build_fixtures.py

The script refuses to overwrite an existing fixture. It checks every target first and writes
nothing when any of them exists, so a fixture that was already released cannot change by accident.
`--out-dir` writes somewhere else, which is how the tests exercise this script.

Files, all written in SQLite's default rollback-journal mode, as a database that has not yet been
opened by a Backend with a WAL-capable build would be (`migrations.apply` switches it to WAL):

- `v1_sample.db`: a Schema_Version 1 database with realistic data. It holds CRLF and emoji text, an
  assistant message with an empty text and one with NULL text, Tool_Call arguments that are not
  JSON, a Task in each of the six states, and closed and unclosed artifacts.
- `v1_sample.expected.json`: the Contexts and Tasks of `v1_sample.db` as `dump_database` loads
  them. Migrating the sample to any later version must keep this dump equal.
- `newer_v999.db`: Schema_Version 999.
- `no_version_table.db`: the v1 tables without `schema_version`.
- `bad_version_two_rows.db`: `schema_version` with two rows.
- `bad_version_text.db`: `schema_version` holding the text `'1'` instead of the integer 1.
- `not_sqlite.db`: random bytes.

The v1 data uses `SCHEMA_V1` from the package. Migrations are never edited after release, so that
tuple always describes version 1.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from local_agent.storage.migrations import SCHEMA_V1

__all__ = ["FIXTURES", "build_all", "dump_database", "main"]

HERE: Final = Path(__file__).resolve().parent

_TOOL_CALLS_VERBATIM: Final = json.dumps(
    [
        {"id": "call_1", "name": "fs__read_file", "arguments": "{path: 'C:\\temp\\a.txt', "},
        {"id": "call_2", "name": "fs__list_dir", "arguments": '{"path": "/tmp"}'},
        {"id": "call_3", "name": "web__fetch", "arguments": "not json at all \U0001f50d\r\n"},
    ],
    ensure_ascii=False,
)

_NO_VERSION_TABLE_DDL: Final = SCHEMA_V1[1:]
"""Every v1 statement except `CREATE TABLE schema_version`."""

_UNTYPED_VERSION_DDL: Final = "CREATE TABLE schema_version (version)"
"""A `schema_version` table without the integer CHECK or INTEGER affinity, so it can hold text."""


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, autocommit=True)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _create(path: Path, ddl: Sequence[str], fill: Callable[[sqlite3.Connection], None]) -> None:
    conn = _connect(path)
    try:
        conn.execute("BEGIN")
        for statement in ddl:
            conn.execute(statement)
        fill(conn)
        conn.execute("COMMIT")
    finally:
        conn.close()


def _set_version(conn: sqlite3.Connection, *versions: Any) -> None:
    for version in versions:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def _fill_v1_sample(conn: sqlite3.Connection) -> None:
    _set_version(conn, 1)
    contexts = [
        (
            "ctx-alpha",
            "You are helpful.\r\nAnswer in plain text. \U0001f600",
            1_700_000_000_000,
            1_700_000_090_000,
        ),
        ("ctx-beta", None, 1_700_000_100_000, 1_700_000_130_000),
        ("ctx-gamma", "", 1_700_000_200_000, 1_700_000_200_000),
    ]
    conn.executemany("INSERT INTO contexts VALUES (?, ?, ?, ?)", contexts)

    messages = [
        # context_id, seq, task_id, role, content, tool_calls, tool_call_id
        (
            "ctx-alpha",
            0,
            "task-completed",
            "user",
            "Read my file.\r\nThen summarize it \U0001f389 caf\u00e9",
            None,
            None,
        ),
        ("ctx-alpha", 1, "task-completed", "assistant", "", _TOOL_CALLS_VERBATIM, None),
        ("ctx-alpha", 2, "task-completed", "tool", "Error: bad arguments\r\n", None, "call_1"),
        ("ctx-alpha", 3, "task-completed", "tool", "a.txt\r\nb.txt", None, "call_2"),
        ("ctx-alpha", 4, "task-completed", "tool", "\U0001f937 nothing found", None, "call_3"),
        (
            "ctx-alpha",
            5,
            "task-completed",
            "assistant",
            "Done \U0001f44d\r\nTwo lines.",
            None,
            None,
        ),
        ("ctx-alpha", 6, "task-failed", "user", "Try again", None, None),
        ("ctx-alpha", 7, "task-failed", "assistant", None, None, None),
        ("ctx-beta", 0, "task-working", "user", "Line one\rLine two\nLine three", None, None),
        ("ctx-beta", 1, None, "assistant", "No producing task recorded", None, None),
    ]
    conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)", messages)

    snapshot = json.dumps({"history": [], "metadata": {"source": "fixture"}})
    tasks = [
        # task_id, context_id, state, status_message, metadata, created, updated, a2a_snapshot
        (
            "task-completed",
            "ctx-alpha",
            "completed",
            None,
            "{}",
            1_700_000_010_000,
            1_700_000_050_000,
            snapshot,
        ),
        (
            "task-failed",
            "ctx-alpha",
            "failed",
            "The model request failed \U0001f4a5",
            '{"attempt": 2}',
            1_700_000_060_000,
            1_700_000_070_000,
            None,
        ),
        (
            "task-canceled",
            "ctx-alpha",
            "canceled",
            "Canceled by the client",
            "{}",
            1_700_000_080_000,
            1_700_000_090_000,
            None,
        ),
        (
            "task-submitted",
            "ctx-beta",
            "submitted",
            None,
            "{}",
            1_700_000_110_000,
            1_700_000_110_000,
            None,
        ),
        (
            "task-working",
            "ctx-beta",
            "working",
            "Discovering local tools...",
            "{}",
            1_700_000_115_000,
            1_700_000_120_000,
            snapshot,
        ),
        (
            "task-input",
            "ctx-beta",
            "input-required",
            "Approve fs__write_file?",
            '{"tool_call_id": "call_9"}',
            1_700_000_125_000,
            1_700_000_130_000,
            None,
        ),
    ]
    conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tasks)

    artifacts = [
        # task_id, artifact_id, ordinal, text, closed
        ("task-completed", "art-1", 0, "Done \U0001f44d\r\nTwo lines.", 1),
        ("task-failed", "art-1", 0, "Partial answer \U0001f6a7", 0),
        ("task-canceled", "art-1", 0, "", 1),
        ("task-working", "art-1", 0, "First response, closed.\r\n", 1),
        ("task-working", "art-2", 1, "Second response, still streaming", 0),
        ("task-input", "art-1", 0, "", 0),
    ]
    conn.executemany("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)", artifacts)


def dump_database(conn: sqlite3.Connection) -> dict[str, Any]:
    """Load every Context with its messages, Tasks, and artifacts as plain JSON-ready values.

    Columns are named explicitly, so a later migration that adds a column does not change the
    dump. Text and JSON columns are returned as stored.
    """
    contexts: list[dict[str, Any]] = []
    for context_id, system_prompt, created, activity in conn.execute(
        "SELECT context_id, system_prompt, created_at_ms, last_activity_ms "
        "FROM contexts ORDER BY context_id"
    ).fetchall():
        messages = [
            {
                "seq": seq,
                "task_id": task_id,
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "tool_call_id": tool_call_id,
            }
            for seq, task_id, role, content, tool_calls, tool_call_id in conn.execute(
                "SELECT seq, task_id, role, content, tool_calls, tool_call_id "
                "FROM messages WHERE context_id = ? ORDER BY seq",
                (context_id,),
            )
        ]
        tasks: list[dict[str, Any]] = []
        for row in conn.execute(
            "SELECT task_id, state, status_message, metadata, created_at_ms, updated_at_ms, "
            "a2a_snapshot FROM tasks WHERE context_id = ? ORDER BY task_id",
            (context_id,),
        ).fetchall():
            task_id = row[0]
            artifacts = [
                {"artifact_id": artifact_id, "ordinal": ordinal, "text": text, "closed": closed}
                for artifact_id, ordinal, text, closed in conn.execute(
                    "SELECT artifact_id, ordinal, text, closed FROM artifacts "
                    "WHERE task_id = ? ORDER BY ordinal",
                    (task_id,),
                )
            ]
            tasks.append(
                {
                    "task_id": task_id,
                    "state": row[1],
                    "status_message": row[2],
                    "metadata": row[3],
                    "created_at_ms": row[4],
                    "updated_at_ms": row[5],
                    "a2a_snapshot": row[6],
                    "artifacts": artifacts,
                }
            )
        contexts.append(
            {
                "context_id": context_id,
                "system_prompt": system_prompt,
                "created_at_ms": created,
                "last_activity_ms": activity,
                "messages": messages,
                "tasks": tasks,
            }
        )
    return {"contexts": contexts}


def _build_v1_sample(path: Path) -> None:
    _create(path, SCHEMA_V1, _fill_v1_sample)


def _build_v1_golden(path: Path) -> None:
    source = path.resolve().with_name("v1_sample.db")
    conn = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    try:
        text = json.dumps(dump_database(conn), ensure_ascii=False, indent=2) + "\n"
    finally:
        conn.close()
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _build_newer(path: Path) -> None:
    _create(path, SCHEMA_V1, lambda conn: _set_version(conn, 999))


def _build_no_version_table(path: Path) -> None:
    _create(path, _NO_VERSION_TABLE_DDL, lambda _conn: None)


def _build_two_rows(path: Path) -> None:
    _create(path, SCHEMA_V1, lambda conn: _set_version(conn, 1, 1))


def _build_text_version(path: Path) -> None:
    _create(
        path, (_UNTYPED_VERSION_DDL, *_NO_VERSION_TABLE_DDL), lambda conn: _set_version(conn, "1")
    )


def _build_not_sqlite(path: Path) -> None:
    with path.open("xb") as handle:
        handle.write(random.Random(20240601).randbytes(4096))


FIXTURES: Final[dict[str, Callable[[Path], None]]] = {
    "v1_sample.db": _build_v1_sample,
    "v1_sample.expected.json": _build_v1_golden,
    "newer_v999.db": _build_newer,
    "no_version_table.db": _build_no_version_table,
    "bad_version_two_rows.db": _build_two_rows,
    "bad_version_text.db": _build_text_version,
    "not_sqlite.db": _build_not_sqlite,
}
"""File name to builder. Order matters: the golden JSON is read back from `v1_sample.db`."""


def build_all(out_dir: Path) -> list[Path]:
    """Write every fixture into `out_dir`, or raise `FileExistsError` and write nothing."""
    targets = [out_dir / name for name in FIXTURES]
    existing = [target.name for target in targets if target.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing fixtures: {', '.join(existing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        FIXTURES[target.name](target)
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--out-dir", type=Path, default=HERE, help="where to write the fixtures")
    args = parser.parse_args(argv)
    try:
        written = build_all(args.out_dir)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
