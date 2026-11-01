"""Property test for the Task_State transition table and the emitter seal (task 17.2)."""

import asyncio
import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from hypothesis import given
from hypothesis import strategies as st

from local_agent.a2a_server import events
from local_agent.a2a_server.events import (
    ALLOWED,
    RECOVERY_ONLY,
    TERMINAL_STATES,
    TaskEventEmitter,
    TaskState,
    is_allowed,
    to_sdk,
)
from local_agent.logging_setup import Redactor
from local_agent.models import TaskRecord, ToolCall
from local_agent.session.memory_store import InMemorySessionStore

TASK = "task-1"
CONTEXT = "ctx-1"

STATES = list(TaskState)


@dataclass(frozen=True)
class Resolved:
    """A `ResolvedCall` for the emitter."""

    call: ToolCall
    server: str
    tool: str
    args: Mapping[str, Any]


def resolved(index: int) -> Resolved:
    return Resolved(ToolCall(f"c{index}", "fs__read_file", "{}"), "fs", "read_file", {"n": index})


class TickingClock:
    """Every `now_ms()` call returns a later time, so a changed row always shows a new timestamp."""

    def __init__(self) -> None:
        self._now = 1_700_000_000_000

    def now_ms(self) -> int:
        self._now += 1
        return self._now

    def monotonic(self) -> float:
        return float(self._now)


class RecordingSink:
    """Records every event the emitter delivers."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    async def status(
        self, state: TaskState, text: str | None, metadata: Mapping[str, Any] | None, *, final: bool
    ) -> None:
        self.events.append(("status", state, text, final))

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        self.events.append(("artifact", artifact_id, text, append, last_chunk))


class RecordingLog:
    """Stands in for the module logger and keeps the event names it receives."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def _record(self, event: str, **_fields: Any) -> None:
        self.names.append(event)

    info = warning = error = _record


Op = tuple[Any, ...]

ops_strategy = st.lists(
    st.one_of(
        st.tuples(st.just("transition"), st.sampled_from(STATES), st.none() | st.text(max_size=8)),
        st.tuples(st.just("status"), st.text(max_size=8)),
        st.tuples(st.just("token"), st.integers(0, 2), st.text(max_size=6), st.booleans()),
        st.tuples(st.just("close"), st.integers(0, 2)),
        st.tuples(st.just("tool_started"), st.integers(0, 3)),
        st.tuples(st.just("tool_completed"), st.integers(0, 3), st.booleans()),
        st.tuples(st.just("approval"), st.integers(0, 3)),
        st.tuples(st.just("reprompt")),
        st.tuples(st.just("emit_current")),
    ),
    max_size=30,
)


async def stored(store: InMemorySessionStore) -> TaskRecord:
    record = await store.get_task(TASK)
    assert record is not None
    return record


async def run_ops(ops: list[Op]) -> None:
    """Apply `ops` to a fresh emitter, checking the invariants after each one."""
    clock = TickingClock()
    store = InMemorySessionStore(clock)
    await store.create_task(TASK, CONTEXT, clock.now_ms(), None)
    sink = RecordingSink()
    emitter = TaskEventEmitter(
        store, sink, Redactor([]), clock, "/tmp/db", task_id=TASK, context_id=CONTEXT
    )

    model = TaskState.SUBMITTED
    entered = [model]
    closed: set[int] = set()
    started: set[int] = set()
    completed: set[int] = set()

    for op in ops:
        kind = op[0]
        before_record = await stored(store)
        before_events = list(sink.events)
        before_state = emitter.state
        was_sealed = emitter.sealed
        assert was_sealed == (model in TERMINAL_STATES)

        target: TaskState | None = None
        if kind == "transition":
            target = op[1]
            accepted = await emitter.transition(target, op[2])
            assert accepted == is_allowed(model, target)
        elif kind == "approval":
            target = TaskState.INPUT_REQUIRED
            accepted = await emitter.approval_required(resolved(op[1]))
            assert accepted == is_allowed(model, target)
        elif kind == "status":
            await emitter.status(op[1])
        elif kind == "token":
            if op[1] in closed:
                continue
            await emitter.token(f"response-{op[1]}", op[2], op[3])
        elif kind == "close":
            await emitter.close_artifact(f"response-{op[1]}")
            if not was_sealed:
                closed.add(op[1])
        elif kind == "tool_started":
            if op[1] in started:
                continue
            started.add(op[1])
            await emitter.tool_started(resolved(op[1]))
        elif kind == "tool_completed":
            if op[1] in completed:
                continue
            completed.add(op[1])
            await emitter.tool_completed(resolved(op[1]), ok=op[2])
        elif kind == "reprompt":
            await emitter.reprompt_approval()
        else:
            await emitter.emit_current_status()

        record = await stored(store)
        if target is not None and not is_allowed(model, target):
            # A disallowed request changes nothing, in memory or in the Session_Store (2.9).
            assert emitter.state is before_state
            assert record == before_record
            assert sink.events == before_events
        elif target is not None:
            model = target
            entered.append(model)
            assert sink.events[-1][:2] == ("status", target)

        if was_sealed:
            # Nothing is emitted or stored after a terminal state (3.9, 4.9, 14.6).
            assert emitter.state is before_state
            assert record == before_record
            assert sink.events == before_events

        assert emitter.state is model
        assert record.state == to_sdk(model)
        for event in sink.events:
            if event[0] == "status":
                assert event[3] == (
                    event[1] in TERMINAL_STATES or event[1] is TaskState.INPUT_REQUIRED
                )

    # The states actually entered use only the allowed pairs.
    for pair in itertools.pairwise(entered):
        assert pair in ALLOWED
        assert pair not in RECOVERY_ONLY
    # The first terminal Status_Event is the last event of any kind.
    terminal_index = next(
        (i for i, e in enumerate(sink.events) if e[0] == "status" and e[1] in TERMINAL_STATES),
        None,
    )
    if terminal_index is not None:
        assert terminal_index == len(sink.events) - 1


# Feature: local-agent-backend, Property 4: Transition table and emitter seal
@given(ops=ops_strategy, current=st.sampled_from(STATES), new=st.sampled_from(STATES))
def test_transition_table_and_emitter_seal(
    ops: list[Op], current: TaskState, new: TaskState
) -> None:
    """**Validates: Requirements 2.9, 3.9, 4.9, 14.6**"""
    # `submitted -> failed` is admitted only with `recovery=True`.
    assert not is_allowed(TaskState.SUBMITTED, TaskState.FAILED)
    assert is_allowed(TaskState.SUBMITTED, TaskState.FAILED, recovery=True)
    # `recovery=True` adds nothing else.
    assert is_allowed(current, new, recovery=True) == (
        is_allowed(current, new) or (current, new) == (TaskState.SUBMITTED, TaskState.FAILED)
    )

    log = RecordingLog()
    with patch.object(events, "_log", log):
        asyncio.run(run_ops(ops))
    rejected = log.names.count("transition_rejected")
    disallowed = _count_disallowed(ops)
    assert rejected == disallowed


def _count_disallowed(ops: list[Op]) -> int:
    """Replay only the state model to count the requests the table must reject."""
    model = TaskState.SUBMITTED
    count = 0
    for op in ops:
        if op[0] not in ("transition", "approval"):
            continue
        target = op[1] if op[0] == "transition" else TaskState.INPUT_REQUIRED
        if is_allowed(model, target):
            model = target
        else:
            count += 1
    return count
