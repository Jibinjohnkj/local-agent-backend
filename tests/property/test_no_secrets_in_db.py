"""Property 61: No Backend_Config secrets in the Storage_Database (task 17.19).

Scenarios run on `SQLiteSessionStore` with tool exceptions, LLM rejection reasons, and tool
results that embed configured secrets. Two checks run on each example:

* **Sanitized run.** The text of MCP content results (success and `isError` replies) is replaced by
  a secret-free text, so no legitimate carrier of secret text remains. Every byte of the database
  file and its `-wal` and `-shm` files, and every TEXT value of every table, must then be free of
  every length-4 window of every secret.
* **Full run.** The script is run as drawn. Every TEXT value must be free of the windows except
  user and assistant message content and tool messages converted from MCP content items, which
  may legitimately carry such text (16.10, 16.11).

The generated secrets are built so that each of their 4-character windows holds one of the
symbols `@#%^~`. Fixed texts and generated text therefore cannot contain a window by chance.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sqlite3
import string
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    DB_NAME,
    CallScript,
    ScenarioRun,
    ScenarioScript,
    open_store,
    run_scenario,
    scenario_scripts,
)

from fakes import ToolOutcome

WINDOW = 4
MCP_CONTENT_OUTCOMES = (ToolOutcome.SUCCESS, ToolOutcome.ERROR)
SYMBOLS = "@#%^~"
_PIECE_CHARS = string.ascii_letters + string.digits


@st.composite
def _secret(draw: st.DrawFn) -> str:
    """Two to four pieces of one to three letters or digits, each followed by a symbol.

    A run of letters and digits is at most three characters long, so every 4-character window of
    the secret contains a symbol.
    """
    pieces = draw(
        st.lists(
            st.tuples(
                st.text(alphabet=_PIECE_CHARS, min_size=1, max_size=3), st.sampled_from(SYMBOLS)
            ),
            min_size=2,
            max_size=4,
        )
    )
    return "".join(body + symbol for body, symbol in pieces)


secret_sets = st.lists(_secret(), min_size=1, max_size=3, unique=True)


def windows(secrets: list[str]) -> set[str]:
    """Every substring the Redactor must mask: the 4-character windows (short secrets whole)."""
    found: set[str] = set()
    for secret in secrets:
        if len(secret) < WINDOW:
            found.add(secret)
        found.update(secret[i : i + WINDOW] for i in range(len(secret) - WINDOW + 1))
    return found


def strip_mcp_secrets(script: ScenarioScript) -> ScenarioScript:
    """The script with the text of MCP content results replaced by a secret-free text."""

    def clean_call(call: CallScript) -> CallScript:
        if call.behavior.outcome not in MCP_CONTENT_OUTCOMES:
            return call
        return dataclasses.replace(call, behavior=dataclasses.replace(call.behavior, text="ok"))

    turns = tuple(
        dataclasses.replace(turn, calls=tuple(clean_call(c) for c in turn.calls))
        for turn in script.turns
    )
    return dataclasses.replace(script, turns=turns)


def stored_files(directory: Path) -> list[Path]:
    """The database file and its sidecar files (`-wal`, `-shm`, `-journal`)."""
    return sorted(p for p in directory.iterdir() if p.name.startswith(DB_NAME))


def leaks_in_files(directory: Path, needles: set[str]) -> list[str]:
    """Raw bytes of every database file, so freed pages and WAL frames count too."""
    found: list[str] = []
    for path in stored_files(directory):
        data = path.read_bytes()
        found += [
            f"{path.name}: {needle!r}" for needle in sorted(needles) if needle.encode() in data
        ]
    return found


def mcp_tool_call_ids(conn: sqlite3.Connection, run: ScenarioRun) -> set[str]:
    """IDs of the Tool_Calls whose Tool_Result is the converted content of an MCP reply."""
    behaviors = {call.tag: call.behavior.outcome for call in run.script.calls}
    answered_by_mcp = {
        tag
        for call in run.mcp.calls
        if not call.cancelled
        and behaviors.get(tag := str(call.args.get("tag"))) in MCP_CONTENT_OUTCOMES
    }
    ids: set[str] = set()
    rows = conn.execute("SELECT tool_calls FROM messages WHERE tool_calls IS NOT NULL").fetchall()
    for (encoded,) in rows:
        for call in json.loads(encoded):
            try:
                arguments: Any = json.loads(call["arguments"])
            except ValueError:
                # Arguments are stored as the model sent them, so a call with invalid JSON is in
                # the database too. It was never sent to an MCP server, so it has no MCP result.
                continue
            if isinstance(arguments, dict) and arguments.get("tag") in answered_by_mcp:
                ids.add(call["id"])
    return ids


def text_values(
    db_path: str, run: ScenarioRun | None, *, allow_carriers: bool
) -> list[tuple[str, str, str]]:
    """`(table, column, text)` for every TEXT value that must be free of secrets.

    With `allow_carriers`, the content of user and assistant messages (and the Tool_Calls of
    assistant messages) and of tool messages converted from MCP content items are left out.
    """
    values: list[tuple[str, str, str]] = []
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        mcp_ids = set() if run is None else mcp_tool_call_ids(conn, run)
        tables = [
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                record = dict(zip(columns, row, strict=True))
                for column, value in record.items():
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", "replace")
                    if not isinstance(value, str):
                        continue
                    if (
                        allow_carriers
                        and table == "messages"
                        and column in ("content", "tool_calls")
                    ):
                        role = record["role"]
                        if role in ("user", "assistant"):
                            continue
                        if role == "tool" and record["tool_call_id"] in mcp_ids:
                            continue
                    values.append((table, column, value))
    return values


def leaks_in_values(values: list[tuple[str, str, str]], needles: set[str]) -> list[str]:
    return [
        f"{table}.{column}: {needle!r} in {text!r}"
        for table, column, text in values
        for needle in sorted(needles)
        if needle in text
    ]


# Feature: local-agent-backend, Property 61: No Backend_Config secrets in the Storage_Database
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_no_secrets_in_the_storage_database(data: st.DataObject) -> None:
    secrets = data.draw(secret_sets, label="secrets")
    script = data.draw(scenario_scripts(secrets=secrets, max_turns=3), label="script")
    needles = windows(secrets)

    # Sanitized run: nothing may carry a secret, so the whole file can be scanned.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        with open_store("sqlite", secrets=secrets, directory=directory) as env:
            run = run_scenario(strip_mcp_secrets(script), env)
            assert run.terminal
            assert not leaks_in_files(directory, needles), "secret bytes in the database files"
            assert not leaks_in_values(text_values(env.db_path, run, allow_carriers=False), needles)
        # After close the WAL is checkpointed into the main file; scan what is left on disk.
        assert not leaks_in_files(directory, needles), "secret bytes after close"

    # Full run: MCP content and user/assistant text may carry them; everything else may not.
    with open_store("sqlite", secrets=secrets) as env:
        run = run_scenario(script, env)
        assert run.terminal
        leaked = leaks_in_values(text_values(env.db_path, run, allow_carriers=True), needles)
        assert not leaked, leaked
