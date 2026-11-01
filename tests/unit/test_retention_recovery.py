"""Unit tests for the retention and recovery examples (Requirements 11.12, 11.13, 18.18, 18.24).

Retention runs against both Session_Stores through `sweep`, with the clock moved a year ahead.
Recovery runs against a real Storage_Database in `tmp_path`. The failure test injects a disk error
into the second recovery transaction with `storage_faults.FaultFactory`, after a clean run on an
identical database has shown where that transaction's statements sit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from conftest import FakeClock
from local_agent.a2a_server.events import TaskState, to_sdk
from local_agent.config.models import OrchestrationConfig
from local_agent.logging_setup import Redactor
from local_agent.models import AssistantMessage, ToolCall, ToolMessage, UserMessage
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import SessionStore, StateChange
from local_agent.storage import StorageError, migrations, recovery, retention
from local_agent.storage.database import Database
from storage_faults import FaultFactory, disk_io_error

T0 = 1_700_000_000_000
YEAR_S = 365 * 24 * 60 * 60
ALL_STATES = frozenset(TaskState)


# --- Helpers ----------------------------------------------------------------------------------


class _Runners:
    """A `ForgetContexts` fake that records each call."""

    def __init__(self) -> None:
        self.forgotten: list[list[str]] = []

    def forget(self, context_ids: Sequence[str]) -> None:
        self.forgotten.append(list(context_ids))


def _change(task_id: str, new: TaskState) -> StateChange:
    return StateChange(
        task_id=task_id, expected_from=ALL_STATES, new=new, status_message=None, metadata=None
    )


async def _add_task(
    store: SessionStore, context_id: str, task_id: str, state: TaskState, created_ms: int = T0
) -> None:
    """Create `task_id` in `context_id` and move it to `state`, all stamped `created_ms`."""
    await store.create_task(task_id, context_id, created_ms, None)
    if state is not TaskState.SUBMITTED:
        await store.set_state(context_id, _change(task_id, state), now_ms=created_ms)
    await store.append(
        context_id, [UserMessage(f"hello from {task_id}")], task_id=task_id, now_ms=created_ms
    )


def _open(path: Path, factory: FaultFactory | None = None, *, create: bool = False) -> Database:
    database = Database.open(path, create=create, connection_factory=factory)
    if create:
        try:
            migrations.apply(database, str(path))
        except BaseException:
            database.close()
            raise
    return database


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Retention (11.12, 11.13) -----------------------------------------------------------------


@pytest.fixture(params=["sqlite", "memory"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, clock: FakeClock
) -> Iterator[SessionStore]:
    if request.param == "memory":
        yield InMemorySessionStore(clock)
        return
    database = _open(tmp_path / "agent.db", create=True)
    try:
        yield SQLiteSessionStore(database, Redactor([]), clock)
    finally:
        database.close()


async def test_unset_idle_timeout_deletes_nothing_after_a_year(
    store: SessionStore, clock: FakeClock
) -> None:
    """11.13: with no Session_Idle_Timeout, no Context goes away however long it sits idle."""
    assert OrchestrationConfig().session_idle_timeout_s is None
    await _add_task(store, "c-done", "t-done", TaskState.COMPLETED)
    await _add_task(store, "c-failed", "t-failed", TaskState.FAILED)
    await _add_task(store, "c-live", "t-live", TaskState.SUBMITTED)

    clock.advance(YEAR_S)

    for context_id, task_id in [
        ("c-done", "t-done"),
        ("c-failed", "t-failed"),
        ("c-live", "t-live"),
    ]:
        record = await store.get_task(task_id)
        assert record is not None
        assert record.context_id == context_id
        _, history = await store.get_history(context_id)
        assert history == [UserMessage(f"hello from {task_id}")]

    # The year was long enough: with a timeout set, the two finished Contexts are expired.
    runners = _Runners()
    deleted = await retention.sweep(store, runners, clock, 60)
    assert sorted(deleted) == ["c-done", "c-failed"]


@pytest.mark.parametrize(
    "state", [TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED]
)
async def test_context_with_a_live_task_is_never_deleted(
    store: SessionStore, clock: FakeClock, state: TaskState
) -> None:
    """11.12: a `submitted`, `working`, or `input-required` Task keeps its Context."""
    await _add_task(store, "c-live", "t-live", state)
    await _add_task(store, "c-idle", "t-idle", TaskState.COMPLETED)
    clock.advance(YEAR_S)

    runners = _Runners()
    deleted = await retention.sweep(store, runners, clock, 60)

    assert deleted == ["c-idle"]
    assert runners.forgotten == [["c-idle"]]
    assert await store.get_task("t-idle") is None
    assert await store.get_history("c-idle") == (None, [])
    live = await store.get_task("t-live")
    assert live is not None
    assert live.state == to_sdk(state)
    assert await store.active_task("c-live") == "t-live"
    _, history = await store.get_history("c-live")
    assert history == [UserMessage("hello from t-live")]


async def test_context_is_deleted_once_its_live_task_finishes(
    store: SessionStore, clock: FakeClock
) -> None:
    await _add_task(store, "c1", "t1", TaskState.SUBMITTED)
    clock.advance(YEAR_S)
    assert await retention.sweep(store, _Runners(), clock, 60) == []

    await store.set_state("c1", _change("t1", TaskState.FAILED), now_ms=clock.now_ms())
    clock.advance(61)

    assert await retention.sweep(store, _Runners(), clock, 60) == ["c1"]


# --- Recovery (18.18, 18.24) ------------------------------------------------------------------

CALL_1 = ToolCall(id="call_1", name="echo", arguments="{}")
CALL_2A = ToolCall(id="call_2a", name="echo", arguments="{}")
CALL_2B = ToolCall(id="call_2b", name="echo", arguments="{}")
CALL_3 = ToolCall(id="call_3", name="echo", arguments="{}")


async def _build_recovery_database(path: Path, clock: FakeClock) -> None:
    """Three live Tasks created in order t1, t2, t3, each with unanswered Tool_Calls, plus t4."""
    database = _open(path, create=True)
    try:
        store = SQLiteSessionStore(database, Redactor([]), clock)
        await _add_task(store, "c1", "t1", TaskState.WORKING, T0)
        await store.append("c1", [AssistantMessage(None, (CALL_1,))], task_id="t1", now_ms=T0)
        await _add_task(store, "c2", "t2", TaskState.INPUT_REQUIRED, T0 + 1)
        await store.append(
            "c2",
            [AssistantMessage("two calls", (CALL_2A, CALL_2B)), ToolMessage("call_2a", "done")],
            task_id="t2",
            now_ms=T0 + 1,
        )
        await _add_task(store, "c3", "t3", TaskState.WORKING, T0 + 2)
        await store.append("c3", [AssistantMessage(None, (CALL_3,))], task_id="t3", now_ms=T0 + 2)
        await _add_task(store, "c4", "t4", TaskState.COMPLETED, T0 + 3)
    finally:
        database.close()


@pytest.mark.parametrize("with_terminal_tasks", [False, True], ids=["empty", "terminal-only"])
async def test_recovery_without_a_live_task_leaves_the_file_unchanged(
    tmp_path: Path, clock: FakeClock, with_terminal_tasks: bool
) -> None:
    """18.18: no `submitted`, `working`, or `input-required` Task means no write."""
    path = tmp_path / "agent.db"
    database = _open(path, create=True)
    try:
        if with_terminal_tasks:
            store = SQLiteSessionStore(database, Redactor([]), clock)
            await _add_task(store, "c1", "t1", TaskState.COMPLETED)
            await _add_task(store, "c2", "t2", TaskState.FAILED)
            await _add_task(store, "c3", "t3", TaskState.CANCELED)
    finally:
        database.close()
    before = _sha256(path)

    database = _open(path)
    try:
        clock.advance(60)
        report = recovery.run(database, clock)
        during = _sha256(path)
    finally:
        database.close()

    assert report.actions == ()
    assert report.tasks_recovered == 0
    assert during == before
    assert _sha256(path) == before


@pytest.mark.parametrize("fault", ["commit", "first-statement"])
async def test_recovery_failure_rolls_back_that_task_and_keeps_earlier_ones(
    tmp_path: Path, clock: FakeClock, fault: str
) -> None:
    """18.24: the failing Task rolls back, earlier ones stay recovered, `StorageError` is raised."""
    counting_path = tmp_path / "counting.db"
    path = tmp_path / "agent.db"
    await _build_recovery_database(counting_path, clock)
    await _build_recovery_database(path, clock)

    # A clean run on the twin database shows where the second transaction's statements are.
    counter = FaultFactory()
    twin = _open(counting_path, counter)
    try:
        assert recovery.run(twin, clock).tasks_recovered == 3
    finally:
        twin.close()
    statements = counter.connection.statements
    begins = [i for i, sql in enumerate(statements, start=1) if sql == "BEGIN IMMEDIATE"]
    assert len(begins) == 3
    pick: dict[str, Callable[[], int]] = {
        "commit": lambda: counter.connection.commit_points[1],
        "first-statement": lambda: begins[1] + 1,
    }
    fail_at = pick[fault]()

    faulty = FaultFactory(fail_at=fail_at, error=disk_io_error)
    database = _open(path, faulty)
    try:
        with pytest.raises(StorageError):
            recovery.run(database, clock)
    finally:
        database.close()
    assert faulty.connection.fired == (fail_at,)

    database = _open(path)
    try:
        store = SQLiteSessionStore(database, Redactor([]), clock)

        first = await store.get_task("t1")
        assert first is not None
        assert first.state == to_sdk(TaskState.FAILED)
        assert first.status_message == recovery.RECOVERY_STATUS
        _, history_1 = await store.get_history("c1")
        assert history_1[-1] == ToolMessage("call_1", recovery.RECOVERY_TOOL_RESULT)

        second = await store.get_task("t2")
        assert second is not None
        assert second.state == to_sdk(TaskState.INPUT_REQUIRED)
        assert second.status_message is None
        _, history_2 = await store.get_history("c2")
        assert history_2[-1] == ToolMessage("call_2a", "done")
        assert len(history_2) == 3

        third = await store.get_task("t3")
        assert third is not None
        assert third.state == to_sdk(TaskState.WORKING)
        _, history_3 = await store.get_history("c3")
        assert history_3[-1] == AssistantMessage(None, (CALL_3,))

        # The next start recovers the rest (18.24).
        report = recovery.run(database, clock)
        assert [action.task_id for action in report.actions] == ["t2", "t3"]
        for task_id in ("t2", "t3"):
            recovered = await store.get_task(task_id)
            assert recovered is not None
            assert recovered.state == to_sdk(TaskState.FAILED)
    finally:
        database.close()
