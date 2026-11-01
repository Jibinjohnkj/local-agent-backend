"""Property 50: Store agreement and history integrity under kill injection (task 17.17).

Each example draws one or two Contexts with one or two scenario scripts each. The scripts run on
the real `TaskRunner` over `SQLiteSessionStore`, once cleanly (this gives the statement count) and
then again with `SimulatedCrash` at drawn statements k. The crash abandons the connection without
commit or rollback, as a kill does. The database is reopened on a new `Database`, as at startup,
and checked for every `task_id`:

* `SessionStore.get_task` and `ProjectedTaskStore.get` agree on state, status message, and
  artifacts, and both agree with the raw `tasks` row;
* the last Status_Event state the sink recorded before the crash is the committed state, or the
  committed state is one allowed transition ahead (the commit returned and the crash came before
  the event was enqueued). A Task whose creation never committed has no row and no events;
* every effect observable before the crash is committed: each message an LLM request carried is in
  the stored history, each tool event names a Tool_Call of a stored assistant message, and no event
  saw more messages than are stored;
* the raw `messages` rows satisfy history integrity part (a) (every tool-role message answers an
  earlier unanswered Tool_Call of its Context), and part (b) (exactly one tool-role message per
  Tool_Call) holds for every Task committed as terminal, with the synthetic results that were
  committed with the terminal state.

No Restart_Recovery runs here: the property reads the database as the killed process left it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypothesis import event, given, settings
from hypothesis import strategies as st
from scenarios import (
    ContextPlan,
    KillResult,
    ScenarioRun,
    ScenarioScript,
    kill_run,
    scenario_scripts,
)

from fakes import FixedClock, StatusEvent
from local_agent.a2a_server.a2a_compat import ServerCallContext
from local_agent.a2a_server.events import TERMINAL_STATES, TaskState, from_sdk, is_allowed
from local_agent.logging_setup import Redactor
from local_agent.models import ChatMessage, TaskRecord
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.storage.database import Database
from local_agent.storage.task_store import ProjectedTaskStore

type Row = tuple[Any, ...]

READ_CLOCK_MS = 10_000_000_000_000
TOOL_KINDS = frozenset({"tool_call_started", "tool_call_completed", "approval_required"})


# --- reading the killed database ----------------------------------------------------------------


@dataclass(frozen=True)
class Rows:
    """The raw rows of the reopened database."""

    tasks: dict[str, tuple[str, TaskState]]  # task_id -> (context_id, state)
    messages: list[Row]  # (context_id, seq, task_id, role, tool_calls, tool_call_id), by seq


def read_rows(db: Database) -> Rows:
    def read(conn: sqlite3.Connection) -> Rows:
        tasks = {
            row[0]: (row[1], TaskState(row[2]))
            for row in conn.execute("SELECT task_id, context_id, state FROM tasks")
        }
        messages = [
            tuple(row)
            for row in conn.execute(
                "SELECT context_id, seq, task_id, role, tool_calls, tool_call_id"
                " FROM messages ORDER BY context_id, seq"
            )
        ]
        return Rows(tasks, messages)

    return db.run_sync(read)


def issued_ids(row: Row) -> list[str]:
    return [] if row[4] is None else [call["id"] for call in json.loads(row[4])]


# --- history integrity (a) and (b) --------------------------------------------------------------


def assert_integrity(rows: Rows) -> None:
    """Part (a) for every Context, part (b) for every Task committed as terminal."""
    by_context: dict[str, list[Row]] = {}
    for row in rows.messages:
        by_context.setdefault(row[0], []).append(row)

    for context_id, history in by_context.items():
        open_calls: Counter[str] = Counter()  # issued and not yet answered
        for row in history:
            if row[3] == "assistant":
                open_calls.update(issued_ids(row))
            elif row[3] == "tool":
                # (a) the Tool_Call was issued by an earlier assistant message, and not answered.
                assert open_calls[row[5]] > 0, (context_id, row[1], row[5])
                open_calls[row[5]] -= 1

    for task_id, (_, state) in rows.tasks.items():
        if state not in TERMINAL_STATES:
            continue
        issued = Counter(i for r in rows.messages if r[2] == task_id for i in issued_ids(r))
        answered = Counter(r[5] for r in rows.messages if r[2] == task_id and r[3] == "tool")
        # (b) exactly one tool-role message per Tool_Call of this Task.
        assert answered == issued, (task_id, state.value, dict(issued), dict(answered))


# --- the two store views ------------------------------------------------------------------------


def assert_views_agree(record: TaskRecord, projected: Any, row_state: TaskState) -> None:
    session_state = from_sdk(record.state)
    assert session_state is row_state
    assert from_sdk(projected.status.state) is session_state
    message = projected.status.message
    text = message.parts[0].text if message.parts else None
    assert text == record.status_message
    assert projected.id == record.task_id
    assert projected.context_id == record.context_id
    assert [(a.artifact_id, a.parts[0].text) for a in projected.artifacts] == [
        (a.artifact_id, a.text) for a in sorted(record.artifacts, key=lambda a: a.ordinal)
    ]


def assert_recorded_state(run: ScenarioRun, committed: TaskState) -> None:
    """The sink's last state is the committed one, or the committed one is one move ahead."""
    recorded = run.sink.last_state or TaskState.SUBMITTED
    assert committed is recorded or is_allowed(recorded, committed), (
        f"{run.task_id}: the sink last saw {recorded.value}, the database holds {committed.value}"
    )


def contiguous(part: list[dict[str, Any]], whole: list[dict[str, Any]]) -> bool:
    if not part:
        return True
    return any(whole[i : i + len(part)] == part for i in range(len(whole) - len(part) + 1))


def assert_observable_effects(
    run: ScenarioRun, history: list[ChatMessage], task_id: str, rows: Rows
) -> None:
    stored = [message.to_openai() for message in history]
    # The next LLM request was sent: every message it carried is stored, in order.
    for request in run.llm.requests:
        carried = [m for m in request.messages if m.get("role") != "system"]
        assert contiguous(carried, stored), (task_id, request.index, carried)

    # The next Status_Event was emitted: it saw no more messages than are stored, and a tool event
    # names a Tool_Call of an assistant message that is stored.
    own_calls = {i for r in rows.messages if r[2] == task_id for i in issued_ids(r)}
    for sink_event in run.sink.events:
        if sink_event.history_len is not None:
            assert sink_event.history_len <= len(stored), (task_id, sink_event.index)
        if isinstance(sink_event, StatusEvent) and sink_event.kind in TOOL_KINDS:
            call_id = (sink_event.metadata or {}).get("tool_call_id")
            if call_id is not None:
                assert call_id in own_calls, (task_id, sink_event.kind, call_id)


async def verify(db: Database, result: KillResult, rows: Rows) -> None:
    clock = FixedClock(READ_CLOCK_MS)
    sessions = SQLiteSessionStore(db, Redactor(()), clock)
    projected = ProjectedTaskStore(sessions, clock=clock)
    context = ServerCallContext()

    run_ids = {run.task_id for run in result.runs}
    assert set(rows.tasks) <= run_ids, "a Task row exists for a Task that never started"

    for run in result.runs:
        record = await sessions.get_task(run.task_id)
        view = await projected.get(run.task_id, context)
        if record is None:
            # The creation commit never happened: nothing was created and nothing was emitted.
            assert view is None
            assert run.task_id not in rows.tasks
            assert run.sink.events == []
            continue
        assert view is not None
        assert_views_agree(record, view, rows.tasks[run.task_id][1])
        assert_recorded_state(run, from_sdk(record.state))
        _, history = await sessions.get_history(run.context_id)
        assert_observable_effects(run, history, run.task_id, rows)


# --- the property -------------------------------------------------------------------------------

_plans = st.lists(
    st.lists(scenario_scripts(max_turns=3), min_size=1, max_size=2), min_size=1, max_size=2
)


def to_plans(scripts: list[list[ScenarioScript]]) -> list[ContextPlan]:
    return [ContextPlan(f"ctx-{n}", tuple(group)) for n, group in enumerate(scripts, start=1)]


def check(result: KillResult) -> Rows:
    db = Database.open(result.db_path, create=False)  # reopen, as at startup
    try:
        rows = read_rows(db)
        assert_integrity(rows)
        asyncio.run(verify(db, result, rows))
        return rows
    finally:
        db.close()


# Feature: local-agent-backend, Property 50: Store agreement and history integrity under kill injection  # noqa: E501
@settings(max_examples=100, deadline=None)
@given(
    plans=_plans,
    picks=st.lists(st.integers(min_value=0, max_value=10**6), min_size=1, max_size=2),
)
def test_store_agreement_under_kill_injection(
    plans: list[list[ScenarioScript]], picks: list[int]
) -> None:
    """**Validates: Requirements 18.8, 18.9, 18.10, 18.11, 11.2, 11.5**"""
    contexts = to_plans(plans)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clean = kill_run(contexts, root / "clean", None, probe_history=True)
        assert not clean.crashed
        clean_rows = check(clean)
        assert all(run.terminal for run in clean.runs)
        assert all(state in TERMINAL_STATES for _, state in clean_rows.tasks.values())

        for number, pick in enumerate(picks):
            k = pick % clean.statements + 1
            killed = kill_run(contexts, root / f"killed-{number}", k, probe_history=True)
            event(f"crashed={killed.crashed}")
            rows = check(killed)
            live = [s for _, s in rows.tasks.values() if s not in TERMINAL_STATES]
            event(f"live tasks={'some' if live else 'none'}")
