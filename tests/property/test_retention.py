"""Property test for Session_Idle_Timeout retention (task 11.8).

Each example builds random Contexts (random created-at and Last_Activity times, several Tasks in
any states with artifacts, a history and an optional System_Prompt) in the in-memory store and in
a real `SQLiteSessionStore`, then runs one `sweep` on each. It checks that:

* both stores delete exactly the IDs `select_expired` returns, in the same order;
* on SQLite, the same pure selector fed from the rows actually stored in the database (the
  model-based check of the SQL predicate) gives that set too;
* deleted Contexts are gone from `SessionStore.get_task`, `ProjectedTaskStore.get`, `list_tasks`,
  and `get_history`, and a new Task with a deleted `context_id` creates an empty Context;
* every surviving Context keeps its history and Tasks unchanged;
* a failure injected at any statement of the SQLite sweep transaction (an I/O error, a full disk,
  or a kill) leaves every Context present, and the runners are not told to forget anything.
"""

import asyncio
import sqlite3
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from local_agent.a2a_server.a2a_compat import ServerCallContext
from local_agent.a2a_server.events import LIVE_STATES, TaskState
from local_agent.logging_setup import Redactor
from local_agent.models import ChatMessage, SystemMessage, TaskRecord, UserMessage
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import (
    UNSET,
    ArtifactWrite,
    SessionStore,
    StateChange,
    StorageError,
    StorageTxn,
    TaskQuery,
)
from local_agent.storage.database import Database
from local_agent.storage.migrations import apply
from local_agent.storage.retention import ContextActivity, select_expired, sweep
from local_agent.storage.task_store import ProjectedTaskStore
from storage_faults import (
    ErrorSource,
    FaultFactory,
    SimulatedCrash,
    crash,
    disk_full_error,
    disk_io_error,
)

SERVER_CONTEXT = ServerCallContext()
ERRORS: dict[str, ErrorSource] = {
    "io": disk_io_error,
    "full": disk_full_error,
    "crash": crash,
}


@dataclass(frozen=True)
class TaskSpec:
    state: TaskState
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class ContextSpec:
    created_at_ms: int
    last_activity_ms: int
    tasks: tuple[TaskSpec, ...]
    messages: tuple[str, ...]
    system_prompt: str | None


@dataclass(frozen=True)
class Scenario:
    now_ms: int
    timeout_s: int
    contexts: tuple[ContextSpec, ...]


@dataclass(frozen=True)
class ContextView:
    """What a reader sees of one Context: its slot and history, and each of its Tasks."""

    history: tuple[SystemMessage | None, tuple[ChatMessage, ...]]
    tasks: tuple[TaskRecord | None, ...]


class FixedClock:
    def __init__(self, now_ms: int) -> None:
        self._now_ms = now_ms

    def now_ms(self) -> int:
        return self._now_ms

    def monotonic(self) -> float:
        return float(self._now_ms) / 1000


class RecordingRunners:
    """Records what `sweep` tells the runner registry to forget."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def forget(self, context_ids: Sequence[str]) -> None:
        self.calls.append(list(context_ids))


def context_id(index: int) -> str:
    return f"ctx-{index}"


def task_id(context_index: int, task_index: int) -> str:
    return f"task-{context_index}-{task_index}"


def all_task_ids(scenario: Scenario) -> list[tuple[str, str]]:
    """Every `(context_id, task_id)` of the scenario."""
    return [
        (context_id(i), task_id(i, j))
        for i, context in enumerate(scenario.contexts)
        for j in range(len(context.tasks))
    ]


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    timeout_s = draw(st.one_of(st.sampled_from([60, 604_800]), st.integers(60, 604_800)))
    timeout_ms = timeout_s * 1000
    now_ms = draw(st.integers(10**12, 2 * 10**12))
    age = st.one_of(
        st.sampled_from([timeout_ms - 1, timeout_ms, timeout_ms + 1]),  # the boundary
        st.integers(0, 3 * timeout_ms),
        st.integers(-(10**6), -1),  # Last_Activity in the future
        st.integers(3 * timeout_ms, 5 * 10**11),  # long before this process started
    )
    task = st.builds(
        TaskSpec,
        state=st.sampled_from(list(TaskState)),
        artifacts=st.lists(st.text(max_size=8), max_size=2).map(tuple),
    )

    contexts: list[ContextSpec] = []
    for _ in range(draw(st.integers(0, 6))):
        last_activity_ms = now_ms - draw(age)
        contexts.append(
            ContextSpec(
                created_at_ms=last_activity_ms - draw(st.integers(0, 10**9)),
                last_activity_ms=last_activity_ms,
                tasks=tuple(draw(st.lists(task, min_size=1, max_size=3))),
                messages=tuple(draw(st.lists(st.text(max_size=8), min_size=1, max_size=3))),
                system_prompt=draw(st.none() | st.text(max_size=8)),
            )
        )
    return Scenario(now_ms=now_ms, timeout_s=timeout_s, contexts=tuple(contexts))


def expected_ids(scenario: Scenario) -> list[str]:
    """The Contexts the model says are expired, from `select_expired` over the scenario."""
    activity = [
        ContextActivity(
            context_id=context_id(i),
            last_activity_ms=context.last_activity_ms,
            has_live_task=any(task.state in LIVE_STATES for task in context.tasks),
        )
        for i, context in enumerate(scenario.contexts)
    ]
    chosen = select_expired(activity, scenario.now_ms, scenario.timeout_s)
    # An inline oracle of 11.11 and 11.12, so a bug in `select_expired` itself shows up here.
    live_names = {"submitted", "working", "input-required"}
    literal = [
        context_id(i)
        for i, context in enumerate(scenario.contexts)
        if scenario.now_ms - context.last_activity_ms > scenario.timeout_s * 1000
        and not any(task.state.value in live_names for task in context.tasks)
    ]
    assert chosen == literal
    return chosen


async def populate(store: SessionStore, scenario: Scenario) -> None:
    """Write the scenario so that each Context ends with exactly its `last_activity_ms`."""
    for i, context in enumerate(scenario.contexts):
        cid = context_id(i)
        for j in range(len(context.tasks)):
            created_at = context.created_at_ms if j == 0 else context.last_activity_ms
            assert await store.create_task(task_id(i, j), cid, created_at, None) is (j == 0)
        for j, task in enumerate(context.tasks):
            tid = task_id(i, j)
            change = (
                None
                if task.state is TaskState.SUBMITTED
                else StateChange(tid, frozenset({TaskState.SUBMITTED}), task.state, None, None)
            )
            writes = tuple(
                ArtifactWrite(tid, f"response-{k}", k, text, closed=True)
                for k, text in enumerate(task.artifacts)
            )
            if change is not None or writes:
                await store.commit(
                    StorageTxn(cid, context.last_activity_ms, state=change, artifacts=writes)
                )
        # The appends are the last activity, so Last_Activity is exactly `last_activity_ms`.
        await store.commit(
            StorageTxn(
                cid,
                context.last_activity_ms,
                appends=tuple((task_id(i, 0), UserMessage(text)) for text in context.messages),
                system_prompt=(
                    UNSET if context.system_prompt is None else SystemMessage(context.system_prompt)
                ),
            )
        )


async def snapshot(store: SessionStore, scenario: Scenario) -> dict[str, ContextView]:
    views: dict[str, ContextView] = {}
    for i, context in enumerate(scenario.contexts):
        cid = context_id(i)
        prompt, history = await store.get_history(cid)
        views[cid] = ContextView(
            history=(prompt, tuple(history)),
            tasks=tuple([await store.get_task(task_id(i, j)) for j in range(len(context.tasks))]),
        )
    return views


def read_activity(conn: sqlite3.Connection) -> list[ContextActivity]:
    """The `ContextActivity` of every Context, built from the rows stored in the database."""
    live = {
        str(context)
        for context, state in conn.execute("SELECT context_id, state FROM tasks")
        if TaskState(state) in LIVE_STATES
    }
    return [
        ContextActivity(str(context), int(last_activity_ms), str(context) in live)
        for context, last_activity_ms in conn.execute(
            "SELECT context_id, last_activity_ms FROM contexts ORDER BY rowid"
        )
    ]


async def verify_swept(
    store: SessionStore,
    scenario: Scenario,
    before: dict[str, ContextView],
    deleted: list[str],
    runners: RecordingRunners,
) -> None:
    """Check the state after a successful sweep, then recreate each deleted Context."""
    gone = set(deleted)
    assert runners.calls == ([deleted] if deleted else [])

    empty = {
        cid: ContextView(history=(None, ()), tasks=(None,) * len(view.tasks))
        for cid, view in before.items()
    }
    after = await snapshot(store, scenario)
    assert after == {cid: empty[cid] if cid in gone else before[cid] for cid in before}

    projected = ProjectedTaskStore(store, clock=FixedClock(scenario.now_ms))
    for cid, tid in all_task_ids(scenario):
        assert (await store.get_task(tid) is None) == (cid in gone)
        assert (await projected.get(tid, SERVER_CONTEXT) is None) == (cid in gone)
    listed = {task.task_id for task in (await store.list_tasks(TaskQuery(page_size=500))).tasks}
    assert listed == {tid for cid, tid in all_task_ids(scenario) if cid not in gone}
    for cid in gone:
        assert await store.active_task(cid) is None

    # A message with a deleted `context_id` starts a new, empty Context (2.13).
    for cid in deleted:
        assert await store.create_task(f"again-{cid}", cid, scenario.now_ms, None) is True
        assert await store.get_history(cid) == (None, [])


async def run_memory(scenario: Scenario) -> list[str]:
    clock = FixedClock(scenario.now_ms)
    store = InMemorySessionStore(clock)
    await populate(store, scenario)
    before = await snapshot(store, scenario)
    runners = RecordingRunners()
    deleted = await sweep(store, runners, clock, scenario.timeout_s)
    await verify_swept(store, scenario, before, deleted, runners)
    return deleted


@dataclass(frozen=True)
class SqliteRun:
    deleted: list[str]
    stored_selection: list[str]  # `select_expired` over the rows read back from the database
    sweep_base: int  # statements counted before the sweep
    sweep_statements: int  # statements counted during the sweep


async def run_sqlite(path: Path, scenario: Scenario) -> SqliteRun:
    clock = FixedClock(scenario.now_ms)
    factory = FaultFactory()
    db = Database.open(path, create=True, connection_factory=factory)
    try:
        apply(db, str(path))
        store = SQLiteSessionStore(db, Redactor([]), clock)
        await populate(store, scenario)
        before = await snapshot(store, scenario)
        activity = await db.read(read_activity)
        stored_selection = select_expired(activity, scenario.now_ms, scenario.timeout_s)

        runners = RecordingRunners()
        base = factory.connection.calls
        deleted = await sweep(store, runners, clock, scenario.timeout_s)
        statements = factory.connection.calls - base
        await verify_swept(store, scenario, before, deleted, runners)
    finally:
        db.close()
    return SqliteRun(deleted, stored_selection, base, statements)


async def run_sqlite_fault(
    path: Path, scenario: Scenario, fail_at: int, error: ErrorSource
) -> None:
    """Fail statement `fail_at` of the sweep transaction and check nothing was deleted."""
    clock = FixedClock(scenario.now_ms)
    factory = FaultFactory(fail_at=fail_at, error=error)
    db = Database.open(path, create=True, connection_factory=factory)
    runners = RecordingRunners()
    try:
        apply(db, str(path))
        store = SQLiteSessionStore(db, Redactor([]), clock)
        await populate(store, scenario)
        before = await snapshot(store, scenario)
        await db.read(read_activity)  # the same reads as the clean run, so the counts line up

        raised = False
        try:
            await sweep(store, runners, clock, scenario.timeout_s)
        except StorageError, SimulatedCrash:
            raised = True
        assert raised
        assert factory.connection.fired == (fail_at,)
        assert runners.calls == []
    finally:
        db.close()

    reopened = Database.open(path, create=False)
    try:
        after_store = SQLiteSessionStore(reopened, Redactor([]), clock)
        assert await snapshot(after_store, scenario) == before
        projected = ProjectedTaskStore(after_store, clock=clock)
        for _, tid in all_task_ids(scenario):
            assert await projected.get(tid, SERVER_CONTEXT) is not None
    finally:
        reopened.close()


# Feature: local-agent-backend, Property 21: Retention
@given(
    scenario=scenarios(),
    fault_pick=st.integers(min_value=0, max_value=10_000),
    error_kind=st.sampled_from(sorted(ERRORS)),
)
def test_retention(scenario: Scenario, fault_pick: int, error_kind: str) -> None:
    """**Validates: Requirements 11.11, 11.12, 11.14**"""
    expected = expected_ids(scenario)

    assert asyncio.run(run_memory(scenario)) == expected

    with tempfile.TemporaryDirectory() as tmp:
        clean = asyncio.run(run_sqlite(Path(tmp) / "clean.db", scenario))
        assert clean.stored_selection == expected
        assert clean.deleted == expected

        # BEGIN IMMEDIATE, the SELECT, the DELETE batches, and COMMIT: fail one of them.
        assert clean.sweep_statements >= 3
        fail_at = clean.sweep_base + 1 + fault_pick % clean.sweep_statements
        asyncio.run(run_sqlite_fault(Path(tmp) / "fault.db", scenario, fail_at, ERRORS[error_kind]))
