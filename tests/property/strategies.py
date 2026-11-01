"""Shared Hypothesis strategies and the generated-database writer.

* `stored_texts()` draws text that ingress accepts: any Unicode without lone surrogates (ingress
  replaces those with U+FFFD), plus CR, LF, CRLF, tabs, NUL, an astral character, and the empty
  string.
* `db_states()` (alias `database_specs()`) draws Contexts whose Tasks are in all six Task_States,
  with assistant messages whose Tool_Calls are partly answered, artifacts, and several Tasks per
  Context. `build_database()` writes such a draw through the real `SQLiteSessionStore` with the
  commits the Orchestrator makes, and can end the run with `SimulatedCrash` at the k-th statement,
  so the file is also one that a killed process left behind. Restart_Recovery (P53) reads it.

The scenario-script strategy lives in `scenarios.py` next to the scripts it builds; the scenario
kill-injection helper (`kill_run`) is there too.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hypothesis import strategies as st

from fakes import FixedClock
from local_agent.a2a_server.events import TaskState
from local_agent.logging_setup import Redactor
from local_agent.models import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import ArtifactWrite, StateChange, StorageTxn
from local_agent.storage.database import Database
from local_agent.storage.migrations import apply
from storage_faults import FaultFactory, SimulatedCrash, crash

__all__ = [
    "EARLIER_TOOL_RESULT",
    "T0",
    "ContextSpec",
    "TaskSpec",
    "TurnSpec",
    "build_database",
    "call_id",
    "database_specs",
    "db_states",
    "stored_texts",
    "text_chars",
    "write_script",
    "write_task",
]

TERMINAL = (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED)

EARLIER_TOOL_RESULT = "Tool call not completed: the task ended"
"""The synthetic result a terminal Task carries for a call that never got an answer (11.5)."""

T0 = 1_700_000_000_000
"""Wall-clock time of the first generated write; later writes add small offsets."""

text_chars = st.characters(blacklist_categories=["Cs"])


def stored_texts(*, max_size: int = 8) -> st.SearchStrategy[str]:
    """Text as the stores must hold it: no lone surrogates; CR/LF/CRLF, NUL, astral, and empty."""
    return st.one_of(
        st.text(alphabet=text_chars, max_size=max_size),
        st.sampled_from(["", " ", "a\r\nb", "\r", "\n", "\t", "\x00", "\U0001f389"]),
    )


_ids = st.text(alphabet=text_chars, min_size=1, max_size=6)


# --- generated database specs -------------------------------------------------------------------


@dataclass(frozen=True)
class TurnSpec:
    """One assistant message with `calls` Tool_Calls. `answered` lists the call indexes that got
    a Tool_Result, in the order the results were appended."""

    content: str | None
    calls: int
    answered: tuple[int, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    state: TaskState
    created_offset_ms: int
    user_text: str
    turns: tuple[TurnSpec, ...]
    artifact_texts: tuple[str, ...]
    status_message: str | None
    metadata: dict[str, str] | None


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    system_prompt: str | None
    tasks: tuple[TaskSpec, ...]


@st.composite
def _turn_specs(draw: st.DrawFn) -> TurnSpec:
    calls = draw(st.integers(min_value=0, max_value=3))
    answered: tuple[int, ...] = ()
    if calls:
        answered = tuple(draw(st.lists(st.integers(min_value=0, max_value=calls - 1), unique=True)))
    return TurnSpec(draw(st.none() | stored_texts()), calls, answered)


@st.composite
def _task_specs(draw: st.DrawFn, task_id: str) -> TaskSpec:
    state = draw(st.sampled_from(list(TaskState)))
    turns: tuple[TurnSpec, ...] = ()
    artifacts: tuple[str, ...] = ()
    if state is not TaskState.SUBMITTED:
        turns = tuple(draw(st.lists(_turn_specs(), max_size=3)))
        artifacts = tuple(draw(st.lists(stored_texts(), max_size=2)))
    return TaskSpec(
        task_id=task_id,
        state=state,
        created_offset_ms=draw(st.integers(min_value=0, max_value=3)) * 1000,  # ties are likely
        user_text=draw(stored_texts()),
        turns=turns,
        artifact_texts=artifacts,
        status_message=draw(st.none() | stored_texts()),
        metadata=draw(st.none() | st.fixed_dictionaries({"note": stored_texts()})),
    )


@st.composite
def database_specs(draw: st.DrawFn) -> tuple[ContextSpec, ...]:
    """One to three Contexts of one to four Tasks each, in all six Task_States."""
    # A drawn prefix, "#", and a running number is unique: the text after the last "#" differs.
    prefixes = draw(st.lists(_ids, min_size=1, max_size=3))
    context_ids = [f"{p}#{n}" for n, p in enumerate(prefixes)]
    counts = [draw(st.integers(min_value=1, max_value=4)) for _ in context_ids]
    task_prefixes = draw(st.lists(_ids, min_size=sum(counts), max_size=sum(counts)))
    task_ids = [f"{p}#{n}" for n, p in enumerate(task_prefixes)]
    contexts: list[ContextSpec] = []
    taken = 0
    for context_id, count in zip(context_ids, counts, strict=True):
        tasks = tuple(draw(_task_specs(task_id)) for task_id in task_ids[taken : taken + count])
        taken += count
        contexts.append(ContextSpec(context_id, draw(st.none() | stored_texts()), tasks))
    return tuple(contexts)


db_states = database_specs
"""The name the design uses for the generated-rows strategy (`db_states()`)."""


# --- writing the database through the real store ------------------------------------------------


def call_id(task_id: str, turn: int, index: int) -> str:
    """Unique per Task, Turn, and call: the text after the last `#` is always `turn.index`."""
    return f"{task_id}#{turn}.{index}"


async def write_task(store: SQLiteSessionStore, context_id: str, spec: TaskSpec) -> None:
    """Write one Task with the commits the Orchestrator makes, ending in `spec.state`."""
    now = T0 + spec.created_offset_ms
    task = spec.task_id
    await store.create_task(task, context_id, now, None)
    await store.append(context_id, [UserMessage(spec.user_text)], task_id=task, now_ms=now + 1)
    if spec.state is TaskState.SUBMITTED:
        return

    change = StateChange(task, frozenset({TaskState.SUBMITTED}), TaskState.WORKING, None, None)
    await store.set_state(context_id, change, now_ms=now + 2)
    unanswered: list[str] = []
    for turn_index, turn in enumerate(spec.turns):
        calls = tuple(
            ToolCall(id=call_id(task, turn_index, i), name="tool", arguments=f"{{{i}")
            for i in range(turn.calls)
        )
        await store.append(
            context_id, [AssistantMessage(turn.content, calls)], task_id=task, now_ms=now + 3
        )
        for index in turn.answered:
            result = ToolMessage(calls[index].id, f"result {index}")
            await store.append(context_id, [result], task_id=task, now_ms=now + 4)
        unanswered += [c.id for i, c in enumerate(calls) if i not in turn.answered]

    writes = tuple(
        ArtifactWrite(task, f"response-{i}", i, text, closed=bool(i % 2))
        for i, text in enumerate(spec.artifact_texts)
    )
    if writes:
        await store.commit(StorageTxn(context_id, now + 5, artifacts=writes))

    if spec.state is TaskState.INPUT_REQUIRED:
        change = StateChange(
            task, frozenset({TaskState.WORKING}), TaskState.INPUT_REQUIRED, None, None
        )
        await store.set_state(context_id, change, now_ms=now + 6)
    elif spec.state in TERMINAL:
        # The terminal commit carries the results still missing (18.11), so a terminal Task is
        # always fully answered.
        await store.commit(
            StorageTxn(
                context_id,
                now + 6,
                appends=tuple((task, ToolMessage(c, EARLIER_TOOL_RESULT)) for c in unanswered),
                state=StateChange(
                    task,
                    frozenset({TaskState.WORKING}),
                    spec.state,
                    spec.status_message,
                    spec.metadata,
                ),
            )
        )


async def write_script(store: SQLiteSessionStore, contexts: Sequence[ContextSpec]) -> None:
    for context in contexts:
        for index, spec in enumerate(context.tasks):
            await write_task(store, context.context_id, spec)
            if index == 0 and context.system_prompt is not None:
                await store.set_system_prompt(
                    context.context_id, SystemMessage(context.system_prompt), now_ms=T0 + 7
                )


def build_database(path: Path, contexts: Sequence[ContextSpec], crash_at: int | None) -> int:
    """Create `path` and write `contexts`. Return the number of statements the writer ran.

    The schema is applied on a plain connection first, so the fault counter only sees the writer.
    With `crash_at=k`, the k-th statement raises `SimulatedCrash`, which ends the run like a kill.
    """
    migrated = Database.open(path, create=True)
    try:
        apply(migrated, str(path))
    finally:
        migrated.close()
    factory = FaultFactory(fail_at=crash_at, error=crash)
    db = Database.open(path, create=False, connection_factory=factory)
    try:
        store = SQLiteSessionStore(db, Redactor(()), FixedClock(T0))
        with contextlib.suppress(SimulatedCrash):
            asyncio.run(write_script(store, contexts))
        return factory.connection.calls
    finally:
        db.close()
