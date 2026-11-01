"""Property 53: Restart_Recovery integrity and idempotence (Requirements 18.15-18.18, 2.9).

Each example writes Contexts and Tasks through the real `SQLiteSessionStore` on a real
`Database` (migrations applied), in all six Task_States, with assistant messages whose Tool_Calls
are partly answered and several Tasks per Context. The writer issues the same separate commits the
Orchestrator does. A `FaultInjectingConnection` can end that run with `SimulatedCrash` at the k-th
statement, so the database is also one that a killed process left behind. The database is then
reopened, as at startup, and `recovery.run` is called.

The checks use an oracle that is derived from the raw rows read before recovery, not from
`plan_recovery`: every non-terminal Task becomes `failed` with the restart status, each of its
unanswered Tool_Calls gets exactly one synthetic result appended in history and Tool_Call order,
Last_Activity of every recovered Context equals the recovery time, and every other row stays as it
was. `plan_recovery` is fed the same rows and must predict the applied changes. A second run, and a
run on a database with no non-terminal Task, must leave the file contents unchanged.

`recovery.run(db, clock)` takes no LLM, MCP_Server, or client object, so the "no call" clause is
checked at the OS boundary instead: while `run` executes, opening a socket connection, starting a
subprocess, or starting a thread is recorded (and refused), and the record must stay empty.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sqlite3
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from hypothesis import event, given, settings
from hypothesis import strategies as st
from strategies import T0, ContextSpec, build_database, database_specs

from fakes import FixedClock
from local_agent.a2a_server.a2a_compat import ServerCallContext
from local_agent.a2a_server.events import TaskState, from_sdk
from local_agent.logging_setup import Redactor
from local_agent.models import ToolCall
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.storage import recovery
from local_agent.storage.database import Database
from local_agent.storage.migrations import apply
from local_agent.storage.recovery import RecoveryAction, StoredMessage, TaskRow, plan_recovery
from local_agent.storage.task_store import ProjectedTaskStore

type Row = tuple[Any, ...]

# The exact texts the requirement fixes (18.16). Written out here instead of imported, so a change
# to the constants in `recovery.py` fails this test.
RESTART_STATUS = "The backend restarted while this task was running"
RESTART_TOOL_RESULT = "Tool call not completed: the backend restarted while this task was running"

LIVE_NAMES = frozenset({"submitted", "working", "input-required"})

RECOVERY_MS = T0 + 10_000_000  # later than every setup time
RESTART_ELAPSED_MS = 5_000_000  # a clock that differs from the recovery time, for the 2nd run

SERVER_CONTEXT = ServerCallContext()


# --- reading the database -----------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """Every row of the four data tables, in a fixed order."""

    contexts: list[Row]
    messages: list[Row]
    tasks: list[Row]
    artifacts: list[Row]


def snapshot(db: Database) -> Snapshot:
    def read(conn: sqlite3.Connection) -> Snapshot:
        def rows(sql: str) -> list[Row]:
            return [tuple(row) for row in conn.execute(sql)]

        return Snapshot(
            contexts=rows(
                "SELECT context_id, system_prompt, created_at_ms, last_activity_ms"
                " FROM contexts ORDER BY context_id"
            ),
            messages=rows(
                "SELECT context_id, seq, task_id, role, content, tool_calls, tool_call_id"
                " FROM messages ORDER BY context_id, seq"
            ),
            tasks=rows(
                "SELECT task_id, context_id, state, status_message, metadata, created_at_ms,"
                " updated_at_ms, a2a_snapshot FROM tasks ORDER BY task_id"
            ),
            artifacts=rows(
                "SELECT task_id, artifact_id, ordinal, text, closed FROM artifacts"
                " ORDER BY task_id, artifact_id"
            ),
        )

    return db.run_sync(read)


def file_bytes(db: Database) -> bytes:
    """The database file after a full checkpoint, so the WAL holds nothing and the file is whole."""
    db.run_sync(lambda conn: conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall())
    return db.path.read_bytes()


@dataclass(frozen=True)
class View:
    """What the two store views say about one Task."""

    session_state: TaskState
    session_status: str | None
    projected_state: TaskState
    projected_status: str | None
    projected_proto: bytes


async def read_views(db: Database, task_ids: Sequence[str]) -> dict[str, View]:
    clock = FixedClock(RECOVERY_MS)
    sessions = SQLiteSessionStore(db, Redactor(()), clock)
    projected = ProjectedTaskStore(sessions, clock=clock)
    views: dict[str, View] = {}
    for task_id in task_ids:
        record = await sessions.get_task(task_id)
        task = await projected.get(task_id, SERVER_CONTEXT)
        assert record is not None
        assert task is not None
        message = task.status.message
        views[task_id] = View(
            session_state=from_sdk(record.state),
            session_status=record.status_message,
            projected_state=from_sdk(task.status.state),
            projected_status=message.parts[0].text if message.parts else None,
            projected_proto=task.SerializeToString(deterministic=True),
        )
    return views


# --- the oracle and the checks ------------------------------------------------------------------


def assert_history_integrity(messages: Sequence[Row], *, terminal: bool) -> None:
    """Part (a) always; part (b) as well when every Task of the history is terminal."""
    contexts = {row[0] for row in messages}
    for context_id in contexts:
        history = sorted((row for row in messages if row[0] == context_id), key=lambda r: r[1])
        issued: list[str] = []
        answered: Counter[str] = Counter()
        for _, _, _, role, _, tool_calls, tool_call_id in history:
            if role == "assistant" and tool_calls is not None:
                issued += [call["id"] for call in json.loads(tool_calls)]
            elif role == "tool":
                assert tool_call_id in issued, (context_id, tool_call_id)  # part (a)
                answered[tool_call_id] += 1
        if terminal:
            assert all(answered[i] == 1 for i in issued), (context_id, issued, answered)  # (b)


def expected_after(before: Snapshot) -> Snapshot:
    """The database recovery must leave, worked out from the rows read before it ran."""
    live = sorted((row for row in before.tasks if row[2] in LIVE_NAMES), key=lambda r: (r[5], r[0]))
    by_context: dict[str, list[Row]] = {}
    for row in before.messages:
        by_context.setdefault(row[0], []).append(row)
    next_seq = {c: max(r[1] for r in rows) + 1 for c, rows in by_context.items()}
    answered = {c: {r[6] for r in rows if r[3] == "tool"} for c, rows in by_context.items()}

    added: list[Row] = []
    for task in live:
        context_id = task[1]
        for row in by_context.get(context_id, []):
            if row[3] != "assistant" or row[2] != task[0] or row[5] is None:
                continue
            for call in json.loads(row[5]):
                if call["id"] in answered[context_id]:
                    continue
                answered[context_id].add(call["id"])
                seq = next_seq[context_id]
                next_seq[context_id] += 1
                added.append(
                    (context_id, seq, task[0], "tool", RESTART_TOOL_RESULT, None, call["id"])
                )

    live_ids = {row[0] for row in live}
    recovered_contexts = {row[1] for row in live}
    return Snapshot(
        contexts=[
            (c, p, created, RECOVERY_MS if c in recovered_contexts else last)
            for c, p, created, last in before.contexts
        ],
        messages=sorted(before.messages + added, key=lambda r: (r[0], r[1])),
        tasks=[
            (t, c, "failed", RESTART_STATUS, meta, created, RECOVERY_MS, snap)
            if t in live_ids
            else (t, c, state, status, meta, created, updated, snap)
            for t, c, state, status, meta, created, updated, snap in before.tasks
        ],
        artifacts=before.artifacts,
    )


def plan_inputs(
    before: Snapshot,
) -> tuple[list[TaskRow], dict[str, list[StoredMessage]]]:
    tasks = [TaskRow(r[0], r[1], TaskState(r[2]), r[5]) for r in before.tasks]
    histories: dict[str, list[StoredMessage]] = {row[0]: [] for row in before.contexts}
    for r in before.messages:
        calls = (
            ()
            if r[5] is None
            else tuple(ToolCall(c["id"], c["name"], c["arguments"]) for c in json.loads(r[5]))
        )
        histories[r[0]].append(StoredMessage(r[1], r[2], r[3], calls, r[6]))
    return tasks, histories


def assert_plan_matches(plan: Sequence[RecoveryAction], before: Snapshot, after: Snapshot) -> None:
    """`plan_recovery` on the pre-recovery rows predicts exactly the changes that were applied."""
    states_before = {row[0]: row[2] for row in before.tasks}
    applied = {
        row[0]: (row[1], TaskState(states_before[row[0]]))
        for row in after.tasks
        if row[2] != states_before[row[0]]
    }
    assert {a.task_id: (a.context_id, a.from_state) for a in plan} == applied

    known = {(row[0], row[1]) for row in before.messages}
    new_rows = [row for row in after.messages if (row[0], row[1]) not in known]
    for action in plan:
        mine = tuple(r[6] for r in new_rows if r[2] == action.task_id)
        assert mine == action.synthetic_results
    for context_id in {a.context_id for a in plan}:
        appended = [r[2] for r in new_rows if r[0] == context_id]
        deduped = [t for i, t in enumerate(appended) if i == 0 or appended[i - 1] != t]
        planned = [a.task_id for a in plan if a.context_id == context_id and a.synthetic_results]
        assert deduped == planned


@contextlib.contextmanager
def no_outbound_calls() -> Iterator[list[str]]:
    """Record, and refuse, anything that could reach an LLM, an MCP_Server, or a client."""
    calls: list[str] = []

    def forbid(name: str) -> Any:
        def refuse(*args: object, **kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"{name} during Restart_Recovery")

        return refuse

    threads_before = {t.ident for t in threading.enumerate()}
    with contextlib.ExitStack() as stack:
        for target, attribute, name in (
            (socket.socket, "connect", "socket.connect"),
            (socket.socket, "connect_ex", "socket.connect_ex"),
            (subprocess.Popen, "__init__", "subprocess.Popen"),
            (asyncio, "create_subprocess_exec", "asyncio.create_subprocess_exec"),
            (asyncio, "create_subprocess_shell", "asyncio.create_subprocess_shell"),
            (threading.Thread, "start", "threading.Thread.start"),
        ):
            stack.enter_context(mock.patch.object(target, attribute, forbid(name)))
        yield calls
    if {t.ident for t in threading.enumerate()} != threads_before:
        calls.append("thread set changed")


# Feature: local-agent-backend, Property 53: Restart_Recovery integrity and idempotence
@settings(max_examples=100)
@given(contexts=database_specs(), crash_pick=st.none() | st.integers(min_value=0, max_value=10**6))
def test_restart_recovery_integrity_and_idempotence(
    contexts: tuple[ContextSpec, ...], crash_pick: int | None
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clean" / "local-agent.db"
        path.parent.mkdir()
        statements = build_database(path, contexts, None)
        if crash_pick is not None:
            path = Path(tmp) / "killed" / "local-agent.db"
            path.parent.mkdir()
            build_database(path, contexts, crash_pick % statements + 1)

        db = Database.open(path, create=False)
        try:
            apply(db, str(path))
            before = snapshot(db)
            bytes_before = file_bytes(db)
            has_live = any(row[2] in LIVE_NAMES for row in before.tasks)
            event(f"killed={crash_pick is not None}")
            event(f"live tasks={'some' if has_live else 'none'}")
            task_ids = [row[0] for row in before.tasks]
            views_before = asyncio.run(read_views(db, task_ids))
            # The generator and the kill must not produce a history that is already invalid.
            assert_history_integrity(before.messages, terminal=False)

            tasks, histories = plan_inputs(before)
            plan = plan_recovery(tasks, histories)

            with no_outbound_calls() as outbound:
                report = recovery.run(db, FixedClock(RECOVERY_MS))
            assert outbound == []

            after = snapshot(db)
            bytes_after = file_bytes(db)
            views_after = asyncio.run(read_views(db, task_ids))

            # Results 1-3 and 5: the exact database, and the plan that predicts it.
            expected = expected_after(before)
            assert after.contexts == expected.contexts
            assert after.messages == expected.messages
            assert after.tasks == expected.tasks
            assert after.artifacts == expected.artifacts
            assert report.actions == tuple(plan)
            assert_plan_matches(plan, before, after)

            # Result 1 and 2 in both store views.
            for row in before.tasks:
                view = views_after[row[0]]
                if row[2] in LIVE_NAMES:
                    assert view.session_state is TaskState.FAILED
                    assert view.projected_state is TaskState.FAILED
                    assert view.session_status == RESTART_STATUS
                    assert view.projected_status == RESTART_STATUS
                else:
                    assert view == views_before[row[0]]

            # Result 4: every Message_History satisfies history integrity.
            assert_history_integrity(after.messages, terminal=True)
            event(f"synthetic results={'some' if report.tool_results_added else 'none'}")

            # Result 6: no live Task leaves the file as it was, and a second run changes nothing.
            if not has_live:
                assert bytes_after == bytes_before
                assert report.actions == ()
            again = recovery.run(db, FixedClock(RECOVERY_MS + RESTART_ELAPSED_MS))
            assert again.actions == ()
            assert file_bytes(db) == bytes_after
            assert snapshot(db) == after
        finally:
            db.close()
