"""Admission, cancel guard, and continuation of `HostRequestHandler` / `HostAgentExecutor` (20.10).

The handler and executor are built directly, over a real Session_Store (both kinds), the real
`ProjectedTaskStore` and `TaskRunnerRegistry`, the real `TaskRunner`, and the scripted LLM and MCP
fakes of `tests/fakes.py`. Requests go through `on_message_send_stream` and `on_cancel_task`, so
the `a2a-sdk` request handler drives the executor as it does in production.

Covered: new Task and Context IDs (2.1), Context association (2.2), an unknown, retention-deleted,
or earlier-process `context_id` (2.3, 2.13, 11.14, 18.6), a busy Context including a `submitted`
Task and an unclaimed reservation (1.14, 11.7), a re-prompt on an unrecognized approval reply
(5.11), messages to `submitted`/`working` Tasks (1.15, 2.14), cancel of an unknown, terminal, or
recovered Task (2.11, 2.12, 18.17), and a failed creation commit (18.30).

Requirements: 2.1, 2.2, 2.3, 2.13, 2.14, 5.11, 11.7, 11.14, 18.17, 18.30.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from a2a.server.events import Event
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    CancelTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskStatusUpdateEvent,
)
from scenarios import (
    WRITE_FILE,
    CallKind,
    CallScript,
    ScenarioScript,
    StoreEnv,
    StoreKind,
    TurnScript,
    build_fakes,
    char_estimate,
    open_store,
    store_kind,
    tagged_arguments,
)
from structlog.testing import capture_logs

from fakes import ClockSleep, FakeLlmGateway, FakeMcpManager, ManualTimers
from local_agent.a2a_server.a2a_compat import (
    A2AError,
    ServerCallContext,
    TaskNotCancelableError,
    TaskNotFoundError,
)
from local_agent.a2a_server.dispatch import HOST_CONTRACT_ERROR_KEY
from local_agent.a2a_server.errors import (
    ContextBusyError,
    StorageWriteError,
    TaskNotAwaitingInputError,
)
from local_agent.a2a_server.events import REPROMPT_TEXT, TaskState, from_sdk, to_sdk
from local_agent.a2a_server.executor import HostAgentExecutor, HostRequestHandler
from local_agent.config.models import ApprovalKeywords, OrchestrationConfig
from local_agent.models import AssistantMessage, TaskRecord, UserMessage
from local_agent.orchestrator.loop import RunnerDeps, TaskRunner, TaskRunnerRegistry
from local_agent.session.store import SessionStore, StateChange, TaskQuery
from local_agent.storage import StorageError
from local_agent.storage.task_store import ProjectedTaskStore

__all__ = ["store_kind"]  # the fixture is used by name below

_WAIT_ROUNDS = 1_000
_SEND_TIMEOUT_S = 10.0

# A protected `write_file` call: the Task reaches `input-required` and waits for the user.
APPROVAL_SCRIPT = ScenarioScript(
    "write it",
    (
        TurnScript(
            calls=(
                CallScript(
                    "w1",
                    CallKind.PROTECTED,
                    WRITE_FILE,
                    tagged_arguments({"path": "a.txt", "content": "x"}, "w1"),
                ),
            )
        ),
    ),
)
PLAIN_SCRIPT = ScenarioScript("hello", ())  # every turn answers "ok"


# --- The rig ------------------------------------------------------------------------------------


@dataclass
class Rig:
    """A handler wired as in production, with the fakes and spies a test inspects."""

    env: StoreEnv
    sessions: SessionStore
    registry: TaskRunnerRegistry
    handler: HostRequestHandler
    llm: FakeLlmGateway
    mcp: FakeMcpManager
    timers: ManualTimers
    created: list[TaskRunner] = field(default_factory=list)
    started: list[str] = field(default_factory=list)


def _agent_card() -> AgentCard:
    return AgentCard(
        name="test-agent",
        description="test agent",
        version="1",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


@contextlib.asynccontextmanager
async def running_rig(
    env: StoreEnv, script: ScenarioScript, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Rig]:
    clock = env.clock
    timers = ManualTimers(clock)
    llm, mcp = build_fakes(script, clock, ClockSleep(clock))
    deps = RunnerDeps(
        sessions=env.store,
        llm=llm,
        registry=mcp.registry,
        mcp=mcp,
        orchestration=OrchestrationConfig(per_session_discovery=True),
        base_instructions=None,
        model="fake/model",
        redactor=env.redactor,
        clock=clock,
        db_path=env.db_path,
        sleep=timers,
        estimate_tokens=char_estimate,
    )
    registry = TaskRunnerRegistry(deps)
    tasks = ProjectedTaskStore(env.store, registry, clock)
    executor = HostAgentExecutor(
        sessions=env.store,
        runners=registry,
        task_store=tasks,
        approval_keywords=ApprovalKeywords(),
        clock=clock,
    )
    handler = HostRequestHandler(executor, tasks, _agent_card())
    rig = Rig(env, env.store, registry, handler, llm, mcp, timers)

    real_create = registry.create

    def spy_create(task_id: str, context_id: str, message: Message) -> TaskRunner:
        runner = real_create(task_id, context_id, message)
        rig.created.append(runner)
        return runner

    real_start = TaskRunner.start

    def spy_start(self: TaskRunner) -> None:
        rig.started.append(self.task_id)
        real_start(self)

    monkeypatch.setattr(registry, "create", spy_create)
    monkeypatch.setattr(TaskRunner, "start", spy_start)
    try:
        yield rig
    finally:
        await registry.cancel_all()
        await handler.aclose()


@pytest.fixture
def env(store_kind: StoreKind) -> Iterator[StoreEnv]:
    with open_store(store_kind) as opened:
        yield opened


@pytest.fixture
async def rig(env: StoreEnv, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rig]:
    async with running_rig(env, PLAIN_SCRIPT, monkeypatch) as built:
        yield built


@pytest.fixture
async def approval_rig(env: StoreEnv, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Rig]:
    async with running_rig(env, APPROVAL_SCRIPT, monkeypatch) as built:
        yield built


# --- Requests and events ------------------------------------------------------------------------


def user_message(text: str, *, context_id: str = "", task_id: str = "") -> Message:
    return Message(
        message_id=uuid.uuid4().hex,
        role=Role.ROLE_USER,
        parts=[Part(text=text)],
        context_id=context_id,
        task_id=task_id,
    )


def call_context() -> ServerCallContext:
    return ServerCallContext()


async def send(rig: Rig, message: Message, context: ServerCallContext | None = None) -> list[Event]:
    """Run one `SendStreamingMessage` request to the end of its stream and collect the events."""
    ctx = context if context is not None else call_context()
    events: list[Event] = []
    async with asyncio.timeout(_SEND_TIMEOUT_S):
        request = SendMessageRequest(message=message)
        async for event in rig.handler.on_message_send_stream(request, ctx):
            events.append(event)
    return events


def states(events: list[Event]) -> list[TaskState]:
    """The Task_State of every Task snapshot and Status_Event, in order."""
    return [
        from_sdk(event.status.state)
        for event in events
        if isinstance(event, Task | TaskStatusUpdateEvent)
    ]


def first_task(events: list[Event]) -> Task:
    assert events, "the stream carried no events"
    first = events[0]
    assert isinstance(first, Task), f"the first event is {type(first).__name__}, not a Task"
    return first


async def until(predicate: Callable[[], bool], what: str) -> None:
    for _ in range(_WAIT_ROUNDS):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"timed out waiting for {what}")


async def require_task(sessions: SessionStore, task_id: str) -> TaskRecord:
    record = await sessions.get_task(task_id)
    assert record is not None, f"Task {task_id!r} is not in the Session_Store"
    return record


async def task_ids(sessions: SessionStore, context_id: str | None = None) -> list[str]:
    page = await sessions.list_tasks(TaskQuery(context_id=context_id, page_size=100))
    return sorted(record.task_id for record in page.tasks)


# --- Seeding a Session_Store as an earlier process or a test setup left it ------------------------

_PATH = {
    TaskState.SUBMITTED: (),
    TaskState.WORKING: (TaskState.WORKING,),
    TaskState.INPUT_REQUIRED: (TaskState.WORKING, TaskState.INPUT_REQUIRED),
    TaskState.COMPLETED: (TaskState.WORKING, TaskState.COMPLETED),
    TaskState.CANCELED: (TaskState.WORKING, TaskState.CANCELED),
    TaskState.FAILED: (TaskState.WORKING, TaskState.FAILED),
}


async def seed_task(
    sessions: SessionStore,
    task_id: str,
    context_id: str,
    state: TaskState,
    *,
    now_ms: int,
    recovered: bool = False,
    history: bool = False,
) -> None:
    """Create a Task and move it to `state` through committed transitions.

    `recovered` leaves the Task `failed` straight from `submitted`, the way Restart_Recovery does.
    `history` gives the Context an earlier question and answer.
    """
    await sessions.create_task(task_id, context_id, now_ms, None)
    if history:
        await sessions.append(
            context_id, [UserMessage("earlier question"), AssistantMessage("earlier answer")]
        )
    path = (TaskState.FAILED,) if recovered else _PATH[state]
    current = TaskState.SUBMITTED
    for step, new in enumerate(path, start=1):
        message = "Task failed during Backend restart" if recovered else None
        await sessions.set_state(
            context_id,
            StateChange(task_id, frozenset({current}), new, message, None),
            now_ms=now_ms + step,
        )
        current = new


# --- New Task and Context IDs (2.1) ---------------------------------------------------------------


async def test_new_message_creates_a_task_and_a_context_with_generated_ids(rig: Rig) -> None:
    events = await send(rig, user_message("hello"))

    snapshot = first_task(events)
    assert snapshot.id
    assert snapshot.context_id
    assert snapshot.id != snapshot.context_id
    assert from_sdk(snapshot.status.state) is TaskState.SUBMITTED
    assert states(events)[-1] is TaskState.COMPLETED

    record = await require_task(rig.sessions, snapshot.id)
    assert record.context_id == snapshot.context_id
    assert record.state == to_sdk(TaskState.COMPLETED)
    _, history = await rig.sessions.get_history(snapshot.context_id)
    assert history == [UserMessage("hello"), AssistantMessage("ok")]


async def test_two_messages_without_ids_get_distinct_task_and_context_ids(rig: Rig) -> None:
    first = first_task(await send(rig, user_message("one")))
    second = first_task(await send(rig, user_message("two")))

    assert first.id != second.id
    assert first.context_id != second.context_id
    assert await task_ids(rig.sessions) == sorted([first.id, second.id])


# --- Context association (2.2, 2.3, 2.13, 11.14, 18.6) -------------------------------------------


async def test_unknown_context_id_creates_the_context_and_associates_the_task(rig: Rig) -> None:
    assert await rig.sessions.get_history("ctx-new") == (None, [])

    snapshot = first_task(await send(rig, user_message("hello", context_id="ctx-new")))

    assert snapshot.context_id == "ctx-new"
    assert (await require_task(rig.sessions, snapshot.id)).context_id == "ctx-new"
    _, history = await rig.sessions.get_history("ctx-new")
    assert history == [UserMessage("hello"), AssistantMessage("ok")]


async def test_a_second_task_in_a_context_shares_its_history(rig: Rig) -> None:
    first = first_task(await send(rig, user_message("first", context_id="ctx-a")))
    second = first_task(await send(rig, user_message("second", context_id="ctx-a")))

    assert first.id != second.id
    assert first.context_id == second.context_id == "ctx-a"
    assert await task_ids(rig.sessions, "ctx-a") == sorted([first.id, second.id])
    _, history = await rig.sessions.get_history("ctx-a")
    assert history == [
        UserMessage("first"),
        AssistantMessage("ok"),
        UserMessage("second"),
        AssistantMessage("ok"),
    ]
    sent = [m["content"] for m in rig.llm.requests[1].messages if m["role"] == "user"]
    assert sent == ["first", "second"]


async def test_retention_deleted_context_id_starts_an_empty_context(rig: Rig) -> None:
    old = first_task(await send(rig, user_message("old question", context_id="ctx-old")))
    deleted = await rig.sessions.delete_expired(rig.env.clock.now_ms() + 1_000_000)
    assert deleted == ["ctx-old"]
    rig.registry.forget(deleted)
    assert await rig.sessions.get_history("ctx-old") == (None, [])

    fresh = first_task(await send(rig, user_message("new question", context_id="ctx-old")))

    assert fresh.id != old.id
    assert fresh.context_id == "ctx-old"
    assert await task_ids(rig.sessions, "ctx-old") == [fresh.id]
    _, history = await rig.sessions.get_history("ctx-old")
    assert history == [UserMessage("new question"), AssistantMessage("ok")]
    sent = [m["content"] for m in rig.llm.requests[-1].messages if m["role"] == "user"]
    assert sent == ["new question"]


async def test_context_persisted_by_an_earlier_process_is_resumed(rig: Rig) -> None:
    now = rig.env.clock.now_ms()
    await seed_task(
        rig.sessions, "t-before", "ctx-disk", TaskState.COMPLETED, now_ms=now, history=True
    )

    snapshot = first_task(await send(rig, user_message("and now?", context_id="ctx-disk")))

    assert snapshot.context_id == "ctx-disk"
    assert snapshot.id != "t-before"
    assert await task_ids(rig.sessions, "ctx-disk") == sorted(["t-before", snapshot.id])
    sent = [m["content"] for m in rig.llm.requests[0].messages if m["role"] == "user"]
    assert sent == ["earlier question", "and now?"]
    _, history = await rig.sessions.get_history("ctx-disk")
    assert history[:2] == [UserMessage("earlier question"), AssistantMessage("earlier answer")]
    assert history[2:] == [UserMessage("and now?"), AssistantMessage("ok")]


async def test_context_with_a_recovered_task_is_not_busy(rig: Rig) -> None:
    now = rig.env.clock.now_ms()
    await seed_task(
        rig.sessions, "t-recovered", "ctx-rec", TaskState.FAILED, now_ms=now, recovered=True
    )

    snapshot = first_task(await send(rig, user_message("again", context_id="ctx-rec")))

    assert snapshot.context_id == "ctx-rec"
    assert (await require_task(rig.sessions, snapshot.id)).state == to_sdk(TaskState.COMPLETED)


# --- Busy Context (1.14, 11.7) ------------------------------------------------------------------


@pytest.mark.parametrize(
    "live_state", [TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED]
)
async def test_message_to_a_context_with_an_active_task_is_rejected_as_busy(
    rig: Rig, live_state: TaskState
) -> None:
    now = rig.env.clock.now_ms()
    await seed_task(rig.sessions, "t-live", "ctx-busy", live_state, now_ms=now, history=True)
    before = await require_task(rig.sessions, "t-live")
    history_before = await rig.sessions.get_history("ctx-busy")
    ctx = call_context()

    with pytest.raises(ContextBusyError) as raised:
        await send(rig, user_message("another", context_id="ctx-busy"), ctx)

    error = raised.value
    assert error.data == {"context_id": "ctx-busy", "active_task_id": "t-live"}
    assert error.jsonrpc_error() == {
        "code": -32050,
        "message": "Context is busy",
        "data": {"context_id": "ctx-busy", "active_task_id": "t-live"},
    }
    assert ctx.state[HOST_CONTRACT_ERROR_KEY] is error
    assert await require_task(rig.sessions, "t-live") == before
    assert await rig.sessions.get_history("ctx-busy") == history_before
    assert await task_ids(rig.sessions) == ["t-live"]
    assert rig.created == []
    assert rig.llm.requests == []
    assert rig.registry.reservation("ctx-busy") is None


async def test_unclaimed_reservation_makes_the_context_busy_after_the_wait(rig: Rig) -> None:
    held = rig.registry.reserve("ctx-held")
    ctx = call_context()
    pending = asyncio.ensure_future(send(rig, user_message("hi", context_id="ctx-held"), ctx))

    await asyncio.wait_for(rig.timers.wait_armed(), _SEND_TIMEOUT_S)  # waiting for a claim
    assert not pending.done()
    rig.timers.fire()  # the 1 s wait passes with nobody claiming
    with pytest.raises(ContextBusyError) as raised:
        await asyncio.wait_for(pending, _SEND_TIMEOUT_S)

    assert raised.value.data == {"context_id": "ctx-held", "active_task_id": ""}
    assert ctx.state[HOST_CONTRACT_ERROR_KEY] is raised.value
    assert rig.registry.reservation("ctx-held") is held  # the other request still owns it
    assert await task_ids(rig.sessions) == []
    assert rig.created == []
    held.release()
    assert rig.registry.reservation("ctx-held") is None


async def test_claimed_reservation_names_its_task_in_the_busy_error(rig: Rig) -> None:
    held = rig.registry.reserve("ctx-claimed")
    held.claim("t-claimed")

    with pytest.raises(ContextBusyError) as raised:
        await send(rig, user_message("hi", context_id="ctx-claimed"))

    assert raised.value.data == {"context_id": "ctx-claimed", "active_task_id": "t-claimed"}
    assert rig.created == []
    held.release()


async def test_second_message_during_the_creation_commit_is_busy_and_the_first_completes(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    real_create_task = rig.sessions.create_task

    async def gated_create_task(
        task_id: str, context_id: str, now_ms: int, a2a_snapshot: str | None
    ) -> bool:
        await gate.wait()
        return await real_create_task(task_id, context_id, now_ms, a2a_snapshot)

    monkeypatch.setattr(rig.sessions, "create_task", gated_create_task)
    first = asyncio.ensure_future(send(rig, user_message("one", context_id="ctx-race")))

    def claimed() -> bool:
        held = rig.registry.reservation("ctx-race")
        return held is not None and held.task_id is not None

    await until(claimed, "the first request to claim the reservation")
    reservation = rig.registry.reservation("ctx-race")
    assert reservation is not None
    first_task_id = reservation.task_id

    with pytest.raises(ContextBusyError) as raised:
        await send(rig, user_message("two", context_id="ctx-race"))
    assert raised.value.data == {"context_id": "ctx-race", "active_task_id": first_task_id}

    gate.set()
    events = await asyncio.wait_for(first, _SEND_TIMEOUT_S)
    assert first_task(events).id == first_task_id
    assert states(events)[-1] is TaskState.COMPLETED
    assert len(rig.created) == 1
    assert len(rig.llm.requests) == 1
    assert rig.registry.reservation("ctx-race") is None


# --- Messages to Tasks that are not awaiting input (1.15, 2.14) ---------------------------------


@pytest.mark.parametrize("live_state", [TaskState.SUBMITTED, TaskState.WORKING])
async def test_message_to_a_submitted_or_working_task_is_rejected(
    rig: Rig, live_state: TaskState
) -> None:
    now = rig.env.clock.now_ms()
    await seed_task(rig.sessions, "t-live", "ctx-live", live_state, now_ms=now, history=True)
    before = await require_task(rig.sessions, "t-live")
    history_before = await rig.sessions.get_history("ctx-live")
    ctx = call_context()

    with pytest.raises(TaskNotAwaitingInputError) as raised:
        await send(rig, user_message("approve", task_id="t-live"), ctx)

    error = raised.value
    assert error.jsonrpc_error() == {
        "code": -32602,
        "message": "Task is not awaiting input",
        "data": {"task_id": "t-live", "state": f"TASK_STATE_{live_state.name}"},
    }
    assert ctx.state[HOST_CONTRACT_ERROR_KEY] is error
    assert await require_task(rig.sessions, "t-live") == before
    assert await rig.sessions.get_history("ctx-live") == history_before
    assert rig.created == []
    assert rig.llm.requests == []


async def test_message_to_an_unknown_task_is_left_to_the_sdk_and_creates_nothing(rig: Rig) -> None:
    ctx = call_context()

    with pytest.raises(TaskNotFoundError):
        await send(rig, user_message("hi", task_id="t-nope", context_id="ctx-nope"), ctx)

    assert HOST_CONTRACT_ERROR_KEY not in ctx.state
    assert await task_ids(rig.sessions) == []
    assert await rig.sessions.get_history("ctx-nope") == (None, [])
    assert rig.created == []


@pytest.mark.parametrize("final_state", [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED])
async def test_message_to_a_terminal_task_is_rejected_and_changes_nothing(
    rig: Rig, final_state: TaskState
) -> None:
    now = rig.env.clock.now_ms()
    await seed_task(rig.sessions, "t-done", "ctx-done", final_state, now_ms=now, history=True)
    before = await require_task(rig.sessions, "t-done")
    history_before = await rig.sessions.get_history("ctx-done")
    ctx = call_context()

    with pytest.raises(A2AError) as raised:
        await send(rig, user_message("more", task_id="t-done"), ctx)

    assert not isinstance(raised.value, TaskNotAwaitingInputError | ContextBusyError)
    assert HOST_CONTRACT_ERROR_KEY not in ctx.state
    assert await require_task(rig.sessions, "t-done") == before
    assert await rig.sessions.get_history("ctx-done") == history_before
    assert rig.created == []
    assert rig.llm.requests == []


# --- Re-prompt on an unrecognized approval reply (5.11) ----------------------------------------


async def test_unrecognized_reply_reprompts_without_appending_to_the_history(
    approval_rig: Rig,
) -> None:
    rig = approval_rig
    first = await send(rig, user_message("write it", context_id="ctx-ask"))
    snapshot = first_task(first)
    assert states(first)[-1] is TaskState.INPUT_REQUIRED
    runner = rig.registry.get(snapshot.id)
    assert runner is not None
    assert runner.approval_gate.pending is not None
    history_before = await rig.sessions.get_history("ctx-ask")
    requests_before = len(rig.llm.requests)

    reply = await send(rig, user_message("hmm, maybe", context_id="ctx-ask", task_id=snapshot.id))

    assert states(reply) == [TaskState.INPUT_REQUIRED, TaskState.INPUT_REQUIRED]
    assert isinstance(reply[0], Task)
    assert isinstance(reply[1], TaskStatusUpdateEvent)
    assert reply[1].status.message.parts[0].text == REPROMPT_TEXT
    record = await require_task(rig.sessions, snapshot.id)
    assert record.state == to_sdk(TaskState.INPUT_REQUIRED)
    assert record.status_message == REPROMPT_TEXT
    assert await rig.sessions.get_history("ctx-ask") == history_before
    assert len(rig.llm.requests) == requests_before
    assert rig.mcp.calls == []
    assert runner.approval_gate.pending is not None  # the pending Tool_Call is untouched
    assert rig.registry.get(snapshot.id) is runner


async def test_approval_after_a_reprompt_still_resumes_the_task(approval_rig: Rig) -> None:
    rig = approval_rig
    snapshot = first_task(await send(rig, user_message("write it", context_id="ctx-ask")))
    await send(rig, user_message("hmm", context_id="ctx-ask", task_id=snapshot.id))

    reply = await send(rig, user_message(" Approve ", context_id="ctx-ask", task_id=snapshot.id))

    assert from_sdk(first_task(reply).status.state) is TaskState.WORKING
    assert states(reply)[-1] is TaskState.COMPLETED
    assert [(call.server, call.tool) for call in rig.mcp.calls] == [("files", "write_file")]
    assert (await require_task(rig.sessions, snapshot.id)).state == to_sdk(TaskState.COMPLETED)


# --- Cancel (2.11, 2.12, 18.17) -----------------------------------------------------------------


async def cancel(rig: Rig, task_id: str) -> Task:
    async with asyncio.timeout(_SEND_TIMEOUT_S):
        return await rig.handler.on_cancel_task(CancelTaskRequest(id=task_id), call_context())


async def test_cancel_of_an_unknown_task_is_not_found(rig: Rig) -> None:
    with pytest.raises(TaskNotFoundError):
        await cancel(rig, "t-nope")

    assert await task_ids(rig.sessions) == []


@pytest.mark.parametrize("final_state", [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED])
async def test_cancel_of_a_terminal_task_is_not_cancelable(
    rig: Rig, final_state: TaskState
) -> None:
    await seed_task(rig.sessions, "t-done", "ctx-done", final_state, now_ms=rig.env.clock.now_ms())
    before = await require_task(rig.sessions, "t-done")

    with pytest.raises(TaskNotCancelableError):
        await cancel(rig, "t-done")

    assert await require_task(rig.sessions, "t-done") == before


async def test_cancel_of_a_recovered_task_is_not_cancelable(rig: Rig) -> None:
    await seed_task(
        rig.sessions,
        "t-recovered",
        "ctx-rec",
        TaskState.FAILED,
        now_ms=rig.env.clock.now_ms(),
        recovered=True,
    )
    before = await require_task(rig.sessions, "t-recovered")
    assert rig.registry.get("t-recovered") is None

    with pytest.raises(TaskNotCancelableError):
        await cancel(rig, "t-recovered")

    assert await require_task(rig.sessions, "t-recovered") == before
    assert before.state == to_sdk(TaskState.FAILED)


@pytest.mark.parametrize(
    "live_state", [TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED]
)
async def test_cancel_of_a_live_looking_task_without_a_runner_is_logged_and_not_cancelable(
    rig: Rig, live_state: TaskState
) -> None:
    await seed_task(
        rig.sessions, "t-orphan", "ctx-orphan", live_state, now_ms=rig.env.clock.now_ms()
    )
    before = await require_task(rig.sessions, "t-orphan")

    with capture_logs() as logs, pytest.raises(TaskNotCancelableError):
        await cancel(rig, "t-orphan")

    # The SDK must not write `canceled` into the store behind the runner's back.
    assert await require_task(rig.sessions, "t-orphan") == before
    errors = [entry for entry in logs if entry["event"] == "cancel_without_runner"]
    assert [(e["log_level"], e["task_id"], e["context_id"]) for e in errors] == [
        ("error", "t-orphan", "ctx-orphan")
    ]


async def test_cancel_of_a_task_waiting_for_approval_goes_through_its_runner(
    approval_rig: Rig,
) -> None:
    rig = approval_rig
    snapshot = first_task(await send(rig, user_message("write it", context_id="ctx-ask")))
    runner = rig.registry.get(snapshot.id)
    assert runner is not None

    canceled = await cancel(rig, snapshot.id)

    assert from_sdk(canceled.status.state) is TaskState.CANCELED
    assert (await require_task(rig.sessions, snapshot.id)).state == to_sdk(TaskState.CANCELED)
    assert runner.approval_gate.pending is None
    assert rig.mcp.calls == []


# --- Creation commit failure (18.30) ------------------------------------------------------------


@pytest.mark.parametrize("context_id", ["", "ctx-fail"])
async def test_creation_commit_failure_rejects_the_message_and_starts_no_runner(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, context_id: str
) -> None:
    async def failing_create_task(*_args: Any) -> bool:
        raise StorageError(rig.env.db_path, "disk I/O error")

    with monkeypatch.context() as patch:
        patch.setattr(rig.sessions, "create_task", failing_create_task)
        with pytest.raises(StorageWriteError) as raised:
            await send(rig, user_message("hello", context_id=context_id))
        # Let the SDK's background bookkeeping for the failed request finish while storage is
        # still failing, so the checks below see the settled state.
        for _ in range(200):
            await asyncio.sleep(0)

        assert raised.value.message == "Storage could not be written"
        assert len(rig.created) == 1  # the runner was created, then discarded, and never started
        assert rig.started == []
        assert rig.registry.get(rig.created[0].task_id) is None
        assert rig.llm.requests == []
        assert await task_ids(rig.sessions) == []

    if context_id:
        assert await rig.sessions.get_history(context_id) == (None, [])
        assert rig.registry.reservation(context_id) is None

        # The Context is not stuck: the same request now succeeds.
        retry = await send(rig, user_message("hello again", context_id=context_id))
        assert states(retry)[-1] is TaskState.COMPLETED
        assert rig.started == [rig.created[-1].task_id]
