"""Property 57: the A2A error contract (task 20.13).

A Context with an Active_Task (random `context_id` and `task_id`) is seeded into a real
Session_Store, and the real Starlette app from `build_app` is called through `httpx.ASGITransport`:

* a new message to the Context (no `taskId`) must answer `Context is busy`;
* a continuation to the Task (`taskId`, Task in `submitted` or `working`) must answer
  `Task is not awaiting input`.

Each is sent over JSON-RPC (`SendMessage`, `SendStreamingMessage`) and REST (`message:send`,
`message:stream`). The test checks the exact bodies, the echoed JSON-RPC `id`, and that the
committed state (Task, artifacts, pending Tool_Calls, Message_History, Task list) is untouched, no
Task was created, and no runner or admission reservation is left behind.

The lifespan is not run: the rejected requests never reach MCP, the LLM, or a runner.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Final

import httpx
from hypothesis import given
from hypothesis import strategies as st
from scenarios import StoreEnv, StoreKind, open_store, store_kind
from strategies import text_chars

from local_agent.a2a_server.app import Deps, build_app, build_deps
from local_agent.a2a_server.events import TaskState
from local_agent.config.loader import load_config_from_text
from local_agent.models import AssistantMessage, SystemMessage, TaskRecord, ToolCall, UserMessage
from local_agent.orchestrator.loop import RunnerDeps, TaskRunnerRegistry
from local_agent.session.store import ArtifactWrite, StateChange, StorageTxn, TaskQuery
from local_agent.storage.task_store import ProjectedTaskStore

__all__ = ["store_kind"]  # the fixture is used by name below

# Feature: local-agent-backend, Property 57: A2A error contract

CONFIG: Final = load_config_from_text(
    "llm:\n  model: ollama_chat/llama3.1\nstorage:\n  backend: memory\n", {}, "test-config"
)

JSONRPC_METHODS: Final = ("SendMessage", "SendStreamingMessage")
REST_PATHS: Final = ("/message:send", "/message:stream")
TRANSPORTS: Final = (*JSONRPC_METHODS, *REST_PATHS)

ERROR_INFO_TYPE: Final = "type.googleapis.com/google.rpc.ErrorInfo"
A2A_DOMAIN: Final = "a2a-protocol.org"

type JsonId = int | str


# --- Generated cases ----------------------------------------------------------------------------

_ids = st.text(alphabet=text_chars, min_size=1, max_size=12)
_jsonrpc_ids = st.one_of(st.integers(min_value=-(2**31), max_value=2**31), _ids)


@dataclass(frozen=True)
class Case:
    """One seeded Context with an Active_Task, and the requests made against it."""

    context_id: str
    task_id: str
    state: TaskState  # submitted or working
    user_text: str
    system_prompt: str | None
    calls: int  # pending Tool_Calls of the assistant message (working Tasks only)
    artifact_text: str | None  # a response artifact (working Tasks only)
    message_text: str
    continuation_names_context: bool
    request_ids: tuple[JsonId, JsonId]  # one per JSON-RPC method


@st.composite
def cases(draw: st.DrawFn) -> Case:
    state = draw(st.sampled_from([TaskState.SUBMITTED, TaskState.WORKING]))
    working = state is TaskState.WORKING
    return Case(
        context_id=draw(_ids),
        task_id=draw(_ids),
        state=state,
        user_text=draw(st.text(alphabet=text_chars, max_size=8)),
        system_prompt=draw(st.none() | st.text(alphabet=text_chars, min_size=1, max_size=8)),
        calls=draw(st.integers(min_value=0, max_value=3)) if working else 0,
        artifact_text=draw(st.none() | st.text(alphabet=text_chars, max_size=8))
        if working
        else None,
        message_text=draw(st.text(alphabet=text_chars, min_size=1, max_size=12)),
        continuation_names_context=draw(st.booleans()),
        request_ids=(draw(_jsonrpc_ids), draw(_jsonrpc_ids)),
    )


# --- The app and the seeded store ---------------------------------------------------------------


def make_app_deps(env: StoreEnv) -> Deps:
    """`Deps` over the harness's store, with a runner registry built on the same store."""
    base = build_deps(CONFIG, None, clock=env.clock, redactor=env.redactor)
    runners = TaskRunnerRegistry(
        RunnerDeps(
            sessions=env.store,
            llm=base.llm_gateway,
            registry=base.registry,
            mcp=base.mcp_manager,
            orchestration=CONFIG.orchestration,
            base_instructions=CONFIG.agent.base_instructions,
            model=CONFIG.llm.model,
            redactor=env.redactor,
            clock=env.clock,
            db_path=env.db_path,
        )
    )
    return dataclasses.replace(
        base,
        session_store=env.store,
        task_store=ProjectedTaskStore(env.store, runners, env.clock),
        runners=runners,
    )


async def seed(env: StoreEnv, case: Case) -> None:
    """Commit the Context the way the Orchestrator would have, ending in `case.state`."""
    store, clock = env.store, env.clock
    context, task = case.context_id, case.task_id
    await store.create_task(task, context, clock.now_ms(), None)
    await store.append(context, [UserMessage(case.user_text)], task_id=task)
    if case.system_prompt is not None:
        await store.set_system_prompt(context, SystemMessage(case.system_prompt))
    if case.state is not TaskState.WORKING:
        return
    await store.set_state(
        context, StateChange(task, frozenset({TaskState.SUBMITTED}), TaskState.WORKING, None, None)
    )
    calls = tuple(
        ToolCall(f"call-{i}", "files__read_file", '{"path": "a"}') for i in range(case.calls)
    )
    await store.append(context, [AssistantMessage("looking", calls)], task_id=task)
    if case.artifact_text is not None:
        write = ArtifactWrite(task, "response-1", 1, case.artifact_text, closed=False)
        await store.commit(StorageTxn(context, clock.now_ms(), artifacts=(write,)))


@dataclass
class Committed:
    """Everything a rejected request must leave alone."""

    task: TaskRecord | None
    history: tuple[SystemMessage | None, list[Any]]
    active_task: str | None
    all_tasks: tuple[TaskRecord, ...]
    a2a_snapshot: str | None


async def capture(env: StoreEnv, case: Case) -> Committed:
    store = env.reader
    page = await store.list_tasks(TaskQuery(page_size=100))
    return Committed(
        task=await store.get_task(case.task_id),
        history=await store.get_history(case.context_id),
        active_task=await store.active_task(case.context_id),
        all_tasks=page.tasks,
        a2a_snapshot=await store.get_a2a_snapshot(case.task_id),
    )


# --- Requests and expected bodies ---------------------------------------------------------------


@dataclass(frozen=True)
class Expected:
    """What the contract says a rejection looks like."""

    message: str
    jsonrpc_code: int
    http_status: int
    grpc_status: str
    reason: str
    data: dict[str, str]


def busy(case: Case) -> Expected:
    return Expected(
        "Context is busy",
        -32050,
        409,
        "ABORTED",
        "CONTEXT_BUSY",
        {"context_id": case.context_id, "active_task_id": case.task_id},
    )


def not_awaiting(case: Case) -> Expected:
    state = "TASK_STATE_SUBMITTED" if case.state is TaskState.SUBMITTED else "TASK_STATE_WORKING"
    return Expected(
        "Task is not awaiting input",
        -32602,
        400,
        "INVALID_ARGUMENT",
        "INVALID_PARAMS",
        {"task_id": case.task_id, "state": state},
    )


def message_json(case: Case, *, continuation: bool, label: str) -> dict[str, Any]:
    message: dict[str, Any] = {
        "messageId": f"message-{label}",
        "role": "ROLE_USER",
        "parts": [{"text": case.message_text}],
    }
    if continuation:
        message["taskId"] = case.task_id
        if case.continuation_names_context:
            message["contextId"] = case.context_id
    else:
        message["contextId"] = case.context_id
    return message


async def post(
    client: httpx.AsyncClient, transport: str, message: dict[str, Any], request_id: JsonId
) -> httpx.Response:
    headers = {"A2A-Version": "1.0"}
    if transport in JSONRPC_METHODS:
        envelope = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": transport,
            "params": {"message": message},
        }
        return await client.post("/", json=envelope, headers=headers)
    return await client.post(transport, json={"message": message}, headers=headers)


def check_response(
    response: httpx.Response, transport: str, expected: Expected, request_id: JsonId
) -> None:
    where = f"{transport}: {response.status_code} {response.text}"
    if transport in JSONRPC_METHODS:
        assert response.status_code == 200, where
        assert response.json() == {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": expected.jsonrpc_code,
                "message": expected.message,
                "data": expected.data,
            },
        }, where
        return
    assert response.status_code == expected.http_status, where
    assert response.json() == {
        "error": {
            "code": expected.http_status,
            "status": expected.grpc_status,
            "message": expected.message,
            "details": [
                {
                    "@type": ERROR_INFO_TYPE,
                    "reason": expected.reason,
                    "domain": A2A_DOMAIN,
                    "metadata": expected.data,
                }
            ],
        }
    }, where


async def exercise(env: StoreEnv, case: Case) -> None:
    deps = make_app_deps(env)
    app = build_app(CONFIG, deps)
    await seed(env, case)
    before = await capture(env, case)
    assert before.active_task == case.task_id
    assert before.task is not None

    transport_ = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport_, base_url="http://127.0.0.1") as client:
        for continuation, expected in ((False, busy(case)), (True, not_awaiting(case))):
            for number, transport in enumerate(TRANSPORTS):
                request_id = case.request_ids[number % 2]
                label = f"{'continue' if continuation else 'new'}-{number}"
                message = message_json(case, continuation=continuation, label=label)
                response = await post(client, transport, message, request_id)

                check_response(response, transport, expected, request_id)
                assert await capture(env, case) == before, f"{transport} changed the store"
                assert deps.runners.get(case.task_id) is None, transport
                assert deps.runners.reservation(case.context_id) is None, transport


# Feature: local-agent-backend, Property 57: A2A error contract
@given(case=cases())
def test_rejections_use_the_exact_error_contract(store_kind: StoreKind, case: Case) -> None:
    """**Validates: Requirements 1.14, 1.15, 2.14, 11.7**"""
    with open_store(store_kind) as env:
        asyncio.run(exercise(env, case))
