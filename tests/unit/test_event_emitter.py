"""Unit tests for `TaskEventEmitter` (task 17.1): commit before emit, seal, and failure paths."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from local_agent.a2a_server.events import (
    DetachedSink,
    StorageWriteFailed,
    TaskEventEmitter,
    TaskState,
    to_sdk,
)
from local_agent.logging_setup import Redactor
from local_agent.models import Clock, ToolCall
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.store import StorageError, StorageTxn

TASK = "task-1"
CONTEXT = "ctx-1"


@dataclass(frozen=True)
class Call:
    """A `ResolvedCall` for tests."""

    call: ToolCall
    server: str
    tool: str
    args: Mapping[str, Any]


def a_call(call_id: str = "c1") -> Call:
    return Call(ToolCall(call_id, "fs__read_file", "{}"), "fs", "read_file", {"path": "a.txt"})


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


class RecordingSink:
    """Records events and what the store held at the moment each status event arrived."""

    def __init__(self, store: InMemorySessionStore) -> None:
        self._store = store
        self.events: list[tuple[Any, ...]] = []

    async def status(
        self, state: TaskState, text: str | None, metadata: Mapping[str, Any] | None, *, final: bool
    ) -> None:
        record = await self._store.get_task(TASK)
        assert record is not None
        stored = (record.status_message, record.state)
        self.events.append(("status", state, text, dict(metadata or {}), final, stored))

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        self.events.append(("artifact", artifact_id, text, append, last_chunk))


async def make(clock: Clock) -> tuple[FlakyStore, RecordingSink, TaskEventEmitter]:
    store = FlakyStore(clock)
    await store.create_task(TASK, CONTEXT, clock.now_ms(), None)
    sink = RecordingSink(store)
    emitter = TaskEventEmitter(
        store, sink, Redactor(["s3cret-value"]), clock, "/tmp/db", task_id=TASK, context_id=CONTEXT
    )
    return store, sink, emitter


async def test_status_event_follows_its_commit(clock: Clock) -> None:
    store, sink, emitter = await make(clock)

    assert await emitter.transition(TaskState.WORKING, "Discovering local tools...")

    (event,) = sink.events
    assert event[1] is TaskState.WORKING
    assert event[2] == "Discovering local tools..."
    assert event[4] is False  # not final
    assert event[5][0] == "Discovering local tools..."  # already stored when the event arrived
    record = await store.get_task(TASK)
    assert record is not None
    assert record.status_message == "Discovering local tools..."


async def test_disallowed_transition_changes_nothing(clock: Clock) -> None:
    store, sink, emitter = await make(clock)

    assert not await emitter.transition(TaskState.FAILED, "nope")  # submitted -> failed

    assert emitter.state is TaskState.SUBMITTED
    assert sink.events == []
    record = await store.get_task(TASK)
    assert record is not None
    assert record.status_message is None


async def test_status_text_is_redacted_before_commit_and_emit(clock: Clock) -> None:
    store, sink, emitter = await make(clock)

    await emitter.transition(TaskState.WORKING, "using s3cret-value now")

    record = await store.get_task(TASK)
    assert record is not None
    assert record.status_message is not None
    assert "s3cret" not in record.status_message
    assert "s3cret" not in sink.events[0][2]


async def test_terminal_transition_flushes_open_artifact_and_seals(clock: Clock) -> None:
    store, sink, emitter = await make(clock)
    await emitter.transition(TaskState.WORKING)
    await emitter.token("response-0", "Hel", first=True)
    await emitter.token("response-0", "lo", first=False)

    assert await emitter.transition(TaskState.FAILED, "interrupted")

    record = await store.get_task(TASK)
    assert record is not None
    assert [(a.text, a.closed) for a in record.artifacts] == [("Hello", False)]
    assert emitter.sealed
    count = len(sink.events)
    await emitter.token("response-0", "late", first=False)
    await emitter.tool_completed(a_call(), ok=True)
    assert not await emitter.transition(TaskState.CANCELED)
    assert len(sink.events) == count
    assert sink.events[-1][4] is True  # the failed event is final


async def test_tokens_are_not_committed_until_the_artifact_closes(clock: Clock) -> None:
    store, sink, emitter = await make(clock)
    await emitter.transition(TaskState.WORKING)

    await emitter.token("response-0", "ab", first=True)
    await emitter.token("response-0", "", first=False)  # empty chunks emit nothing
    record = await store.get_task(TASK)
    assert record is not None
    assert record.artifacts == ()
    assert [(a.text, a.closed) for a in emitter.live_artifacts()] == [("ab", False)]

    await emitter.close_artifact("response-0")

    record = await store.get_task(TASK)
    assert record is not None
    assert [(a.text, a.closed) for a in record.artifacts] == [("ab", True)]
    assert sink.events[1:] == [
        ("artifact", "response-0", "ab", False, False),
        ("artifact", "response-0", "", True, True),
    ]


async def test_commit_failure_emits_nothing_and_raises(clock: Clock) -> None:
    store, sink, emitter = await make(clock)
    await emitter.transition(TaskState.WORKING)
    store.failures = 1

    with pytest.raises(StorageWriteFailed):
        await emitter.transition(TaskState.COMPLETED, "done")

    assert len(sink.events) == 1
    assert emitter.state is TaskState.WORKING
    record = await store.get_task(TASK)
    assert record is not None
    assert record.state == to_sdk(TaskState.WORKING)


async def test_failed_submitted_to_working_counts_as_working(clock: Clock) -> None:
    store, sink, emitter = await make(clock)
    store.failures = 1

    with pytest.raises(StorageWriteFailed):
        await emitter.transition(TaskState.WORKING, "Discovering local tools...")
    assert sink.events == []
    assert emitter.state is TaskState.WORKING

    # The row is still `submitted`; the follow-up commit accepts that (18.30).
    assert await emitter.transition(TaskState.FAILED, "The conversation could not be saved")
    record = await store.get_task(TASK)
    assert record is not None
    assert record.state == to_sdk(TaskState.FAILED)
    assert emitter.sealed


async def test_second_failed_commit_emits_final_failed_unpersisted(clock: Clock) -> None:
    store, sink, emitter = await make(clock)
    await emitter.transition(TaskState.WORKING)
    store.failures = 2

    with pytest.raises(StorageWriteFailed):
        await emitter.transition(TaskState.COMPLETED)
    assert await emitter.transition(TaskState.FAILED, "The conversation could not be saved")

    assert emitter.sealed
    status = sink.events[-1]
    assert status[1] is TaskState.FAILED
    assert status[4] is True
    assert status[5][1] == to_sdk(TaskState.WORKING)  # the row kept its last committed state


async def test_tool_events_carry_metadata_and_complete_once(clock: Clock) -> None:
    _store, sink, emitter = await make(clock)
    await emitter.transition(TaskState.WORKING)
    call = a_call()

    await emitter.tool_started(call, "Calling read_file (a.txt)...")
    await emitter.tool_completed(call, ok=False)

    started, completed = sink.events[1][3], sink.events[2][3]
    assert started == {
        "kind": "tool_call_started",
        "tool_name": "read_file",
        "server": "fs",
        "tool_call_id": "c1",
    }
    assert completed["kind"] == "tool_call_completed"
    assert completed["outcome"] == "failed"
    with pytest.raises(AssertionError):
        await emitter.tool_completed(call, ok=True)


async def test_approval_flow_and_reprompt(clock: Clock) -> None:
    _store, sink, emitter = await make(clock)
    await emitter.transition(TaskState.WORKING)

    assert await emitter.approval_required(a_call())
    await emitter.reprompt_approval()
    # Back to working without emitting, then emit the committed state (the continuation path).
    assert await emitter.transition(TaskState.WORKING, emit=False)
    count = len(sink.events)
    await emitter.emit_current_status()

    required, reprompt = sink.events[1], sink.events[2]
    assert required[1] is TaskState.INPUT_REQUIRED
    assert required[3]["kind"] == "tool_approval_required"
    assert required[3]["arguments"] == {"path": "a.txt"}
    assert required[4] is True
    assert reprompt[1] is TaskState.INPUT_REQUIRED
    assert reprompt[3]["tool_call_id"] == "c1"
    assert reprompt[4] is True
    assert len(sink.events) == count + 1
    assert sink.events[-1][1] is TaskState.WORKING


async def test_detached_sink_drops_events_but_commits(clock: Clock) -> None:
    store, _sink, emitter = await make(clock)
    emitter.attach(DetachedSink())

    assert await emitter.transition(TaskState.WORKING, "x")

    record = await store.get_task(TASK)
    assert record is not None
    assert record.status_message == "x"
