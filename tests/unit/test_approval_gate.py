"""Unit tests for `ApprovalGate` (task 17.5): status contents, decisions, timeout, and discard.

The gate takes an injectable `sleep` for the approval timeout. A sleep that returns at once fires
the timeout immediately, and a sleep that never returns models a timeout that has not expired yet.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from local_agent.a2a_server.events import (
    StorageWriteFailed,
    TaskEventEmitter,
    TaskState,
    from_sdk,
)
from local_agent.logging_setup import Redactor
from local_agent.models import Clock, ToolCall
from local_agent.orchestrator.approval_gate import ApprovalGate, Decision
from local_agent.orchestrator.tool_dispatch import Prepared
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.store import StorageError, StorageTxn

TASK = "task-1"
CONTEXT = "ctx-1"
TIMEOUT_S = 30.0

Event = tuple[TaskState, str | None, dict[str, Any], bool]


def a_prepared(call_id: str = "c1") -> Prepared:
    return Prepared(
        call=ToolCall(call_id, "fs__write_file", '{"path": "a.txt"}'),
        server="fs",
        tool="write_file",
        args={"path": "a.txt", "content": "hi"},
        protected=True,
    )


class RecordingSink:
    """Records every status event the emitter delivers."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._arrived = asyncio.Event()

    async def status(
        self, state: TaskState, text: str | None, metadata: Mapping[str, Any] | None, *, final: bool
    ) -> None:
        self.events.append((state, text, dict(metadata or {}), final))
        self._arrived.set()

    async def wait_for(self, state: TaskState, count: int = 1) -> None:
        """Block until the sink has seen `state` `count` times."""
        async with asyncio.timeout(5):
            while self.states.count(state) < count:
                self._arrived.clear()
                await self._arrived.wait()

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        return None

    @property
    def states(self) -> list[TaskState]:
        return [event[0] for event in self.events]


class FlakyStore(InMemorySessionStore):
    """An in-memory store whose commits fail while `failures` is positive."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(clock)
        self.failures = 0

    async def commit(self, txn: StorageTxn) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise StorageError("/tmp/db", "disk full")
        await super().commit(txn)


class Sleeper:
    """A `sleep` that records its delays, and either returns at once or blocks until cancelled."""

    def __init__(self, *, blocks: bool) -> None:
        self._blocks = blocks
        self.calls: list[float] = []
        self.cancelled = False

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if not self._blocks:
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class Harness:
    def __init__(self, store: FlakyStore, sink: RecordingSink, emitter: TaskEventEmitter) -> None:
        self.store = store
        self.sink = sink
        self.emitter = emitter

    async def stored_state(self) -> TaskState:
        record = await self.store.get_task(TASK)
        assert record is not None
        return from_sdk(record.state)

    async def wait_for_state(self, state: TaskState, count: int = 1) -> None:
        await self.sink.wait_for(state, count)


async def make(clock: Clock) -> Harness:
    """A Task that is already `working`, as it is when the dispatcher asks for approval."""
    store = FlakyStore(clock)
    await store.create_task(TASK, CONTEXT, clock.now_ms(), None)
    sink = RecordingSink()
    emitter = TaskEventEmitter(
        store, sink, Redactor([]), clock, "/tmp/db", task_id=TASK, context_id=CONTEXT
    )
    assert await emitter.transition(TaskState.WORKING)
    sink.events.clear()
    return Harness(store, sink, emitter)


def start(gate: ApprovalGate, prepared: Prepared | None = None) -> asyncio.Task[Decision]:
    return asyncio.ensure_future(gate.request(prepared or a_prepared()))


async def test_request_moves_task_to_input_required_with_full_prompt(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)

    (state, text, metadata, final) = h.sink.events[0]
    assert state is TaskState.INPUT_REQUIRED
    assert final is True
    assert text is not None
    for expected in ("write_file", "fs__write_file", "fs", "a.txt", "approve", "deny"):
        assert expected in text
    assert metadata["kind"] == "tool_approval_required"
    assert metadata["tool_name"] == "write_file"
    assert metadata["llm_tool_name"] == "fs__write_file"
    assert metadata["server"] == "fs"
    assert metadata["tool_call_id"] == "c1"
    assert metadata["arguments"] == {"path": "a.txt", "content": "hi"}
    assert await h.stored_state() is TaskState.INPUT_REQUIRED

    gate.resolve(Decision.APPROVE)
    await task


async def test_pending_call_is_held_in_memory_until_decided(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)

    assert gate.pending is not None
    assert gate.pending.call_id == "c1"
    assert gate.pending.llm_name == "fs__write_file"
    assert gate.pending.args == {"path": "a.txt", "content": "hi"}

    gate.resolve(Decision.DENY)
    await task
    assert gate.pending is None


@pytest.mark.parametrize("decision", [Decision.APPROVE, Decision.DENY])
async def test_resolve_returns_the_users_decision(clock: Clock, decision: Decision) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    gate.resolve(decision)

    assert await task is decision
    assert gate.pending is None
    # The executor moves the Task back to `working` on a continuation message (5.8), not the gate.
    assert h.sink.states == [TaskState.INPUT_REQUIRED]
    assert h.emitter.state is TaskState.INPUT_REQUIRED


async def test_second_resolve_is_ignored(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    gate.resolve(Decision.DENY)
    gate.resolve(Decision.APPROVE)

    assert await task is Decision.DENY


async def test_resolve_without_pending_request_does_nothing(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    gate.resolve(Decision.APPROVE)
    gate.discard()

    assert gate.pending is None
    assert h.sink.events == []


async def test_only_one_approval_may_be_pending(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    first = start(gate, a_prepared("c1"))
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    with pytest.raises(RuntimeError, match="already pending"):
        await gate.request(a_prepared("c2"))

    assert gate.pending is not None
    assert gate.pending.call_id == "c1"
    gate.resolve(Decision.APPROVE)
    assert await first is Decision.APPROVE


async def test_timeout_denies_and_returns_task_to_working(clock: Clock) -> None:
    h = await make(clock)
    sleeper = Sleeper(blocks=False)  # returns at once: the timeout fires immediately
    gate = ApprovalGate(h.emitter, approval_timeout_s=TIMEOUT_S, sleep=sleeper)

    decision = await gate.request(a_prepared())

    assert decision is Decision.TIMEOUT
    assert sleeper.calls == [TIMEOUT_S]
    assert h.sink.states == [TaskState.INPUT_REQUIRED, TaskState.WORKING]
    assert h.sink.events[1][3] is False  # `working` does not close the stream
    assert h.emitter.state is TaskState.WORKING
    assert await h.stored_state() is TaskState.WORKING
    assert gate.pending is None


async def test_late_resolve_after_timeout_is_a_noop(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter, approval_timeout_s=TIMEOUT_S, sleep=Sleeper(blocks=False))

    assert await gate.request(a_prepared()) is Decision.TIMEOUT
    gate.resolve(Decision.APPROVE)

    assert gate.pending is None
    assert h.sink.states == [TaskState.INPUT_REQUIRED, TaskState.WORKING]


async def test_answer_before_timeout_wins_and_stops_the_timer(clock: Clock) -> None:
    h = await make(clock)
    sleeper = Sleeper(blocks=True)  # the timeout has not expired
    gate = ApprovalGate(h.emitter, approval_timeout_s=TIMEOUT_S, sleep=sleeper)

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    gate.resolve(Decision.APPROVE)

    assert await task is Decision.APPROVE
    assert sleeper.calls == [TIMEOUT_S]
    await asyncio.sleep(0)
    assert sleeper.cancelled
    assert h.sink.states == [TaskState.INPUT_REQUIRED]


async def test_decision_arriving_with_the_timer_wins_over_the_timeout(clock: Clock) -> None:
    h = await make(clock)
    holder: list[ApprovalGate] = []

    async def answering_sleep(seconds: float) -> None:
        holder[0].resolve(Decision.DENY)

    gate = ApprovalGate(h.emitter, approval_timeout_s=TIMEOUT_S, sleep=answering_sleep)
    holder.append(gate)

    assert await gate.request(a_prepared()) is Decision.DENY
    assert h.sink.states == [TaskState.INPUT_REQUIRED]


async def test_without_timeout_the_task_waits_indefinitely(clock: Clock) -> None:
    h = await make(clock)
    sleeper = Sleeper(blocks=False)  # would fire at once, so any call to it shows up as a TIMEOUT
    gate = ApprovalGate(h.emitter, sleep=sleeper)

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    for _ in range(50):
        await asyncio.sleep(0)

    assert not task.done()
    assert sleeper.calls == []
    assert h.emitter.state is TaskState.INPUT_REQUIRED
    assert await h.stored_state() is TaskState.INPUT_REQUIRED
    assert h.sink.states == [TaskState.INPUT_REQUIRED]

    gate.resolve(Decision.APPROVE)
    assert await task is Decision.APPROVE


@pytest.mark.parametrize("timeout_s", [None, TIMEOUT_S])
async def test_discard_cancels_the_waiting_request(clock: Clock, timeout_s: float | None) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter, approval_timeout_s=timeout_s, sleep=Sleeper(blocks=True))

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    gate.discard()

    assert gate.pending is None
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.resolve(Decision.APPROVE)  # a later resolve is a no-op
    assert gate.pending is None
    # The gate does not change the Task_State on discard; the cancel path does.
    assert h.sink.states == [TaskState.INPUT_REQUIRED]


async def test_cancelling_the_request_clears_the_pending_entry(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter, approval_timeout_s=TIMEOUT_S, sleep=Sleeper(blocks=True))

    task = start(gate)
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.pending is None


async def test_request_after_a_terminal_state_denies_without_waiting(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter, sleep=Sleeper(blocks=True))
    assert await h.emitter.transition(TaskState.CANCELED)
    h.sink.events.clear()

    async with asyncio.timeout(5):
        decision = await gate.request(a_prepared())

    assert decision is Decision.DENY
    assert gate.pending is None
    assert h.sink.events == []


async def test_failed_commit_raises_and_clears_the_pending_entry(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)
    h.store.failures = 1

    with pytest.raises(StorageWriteFailed):
        await gate.request(a_prepared())

    assert gate.pending is None
    assert h.sink.events == []
    assert await h.stored_state() is TaskState.WORKING


async def test_gate_serves_consecutive_requests_for_one_task(clock: Clock) -> None:
    h = await make(clock)
    gate = ApprovalGate(h.emitter)

    first = start(gate, a_prepared("c1"))
    await h.wait_for_state(TaskState.INPUT_REQUIRED)
    gate.resolve(Decision.APPROVE)
    assert await first is Decision.APPROVE

    # The executor's continuation path: back to `working` before the next call is considered.
    assert await h.emitter.transition(TaskState.WORKING)
    second = start(gate, a_prepared("c2"))
    await h.wait_for_state(TaskState.INPUT_REQUIRED, count=2)

    assert gate.pending is not None
    assert gate.pending.call_id == "c2"
    assert h.sink.events[-1][2]["tool_call_id"] == "c2"
    gate.resolve(Decision.DENY)
    assert await second is Decision.DENY
