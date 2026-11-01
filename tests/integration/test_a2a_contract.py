"""A2A contract tests against the pinned `a2a-sdk` (task 20.17).

The real Starlette app from `build_app` is called through `httpx.ASGITransport`, over a real
Session_Store of both kinds, the real SDK request handler, dispatchers, and serializers, the real
`TaskRunner`, and the scripted LLM and MCP fakes of `tests/fakes.py`. The tests read the raw bytes
of responses and SSE events, so an SDK upgrade that changes a wire form fails here:

* `test_error_contract_jsonrpc` / `test_error_contract_rest`: both Backend errors on
  `SendMessage` and `SendStreamingMessage` with the exact body and the echoed request ID. The
  JSON-RPC test also guards the two private dispatcher hooks (1.14, 1.15, 2.14).
* `test_protojson_contract`: every `state` and `role` in the raw bytes of a Task in all six states,
  a Status_Event, a Token_Event, get-task, cancel, and both contract errors is a ProtoJSON enum
  name, never an integer or a lower-case word (1.10).
* `test_first_event_snapshot`: the first event of a stream for a new Task, a recognized
  continuation, and an unrecognized continuation (3.10-3.13).
* `test_no_divergent_saves`: no `task_store_divergent_save` record appears in any test of this
  module (an autouse fixture), and the log capture is shown to see such a record (18.9).
* `test_storage_failure_on_create`: a failed creation commit gives `-32603` / 500 "Storage could
  not be written" and leaves no Task or Context row (18.30).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import json
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import httpx
import pytest
from a2a.types import Message, Part, Role, Task
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

from fakes import ClockSleep, FakeLlmGateway, FakeMcpManager, LlmFailure, ManualTimers
from local_agent.a2a_server.a2a_compat import JsonRpcDispatcher, ServerCallContext
from local_agent.a2a_server.app import build_app, build_deps
from local_agent.a2a_server.dispatch import HostJsonRpcDispatcher
from local_agent.a2a_server.events import (
    REPROMPT_TEXT,
    STREAM_CLOSING_STATES,
    TaskState,
    sdk_name,
    to_sdk,
)
from local_agent.config.models import (
    BackendConfig,
    LLMConfig,
    OrchestrationConfig,
    StorageConfig,
)
from local_agent.models import UserMessage
from local_agent.orchestrator.loop import RunnerDeps, TaskRunnerRegistry
from local_agent.session.store import StateChange, TaskQuery
from local_agent.storage import StorageError
from local_agent.storage.task_store import ProjectedTaskStore, snapshot_json
from storage_faults import disk_io_error

__all__ = ["store_kind"]  # the fixture is used by name below

pytestmark = pytest.mark.integration

HEADERS: Final = {"A2A-Version": "1.0"}
TIMEOUT_S: Final = 20.0
SETTLE_ROUNDS: Final = 200

ERROR_INFO_TYPE: Final = "type.googleapis.com/google.rpc.ErrorInfo"
A2A_DOMAIN: Final = "a2a-protocol.org"
STORAGE_MESSAGE: Final = "Storage could not be written"

STATE_NAMES: Final = frozenset(sdk_name(state) for state in TaskState)
ROLE_NAMES: Final = frozenset({"ROLE_USER", "ROLE_AGENT"})
DIVERGENT_EVENT: Final = "task_store_divergent_save"

CONFIG: Final = BackendConfig(
    llm=LLMConfig(model="fake/model"),
    orchestration=OrchestrationConfig(per_session_discovery=True),
    storage=StorageConfig(backend="memory"),
)

PLAIN: Final = ScenarioScript("hello", (TurnScript(chunks=("hel", "lo")),))
"""A Task that streams two chunks and completes."""

REJECTED: Final = ScenarioScript("boom", (TurnScript(failure=LlmFailure.REJECTED, reason="no"),))
"""A Task whose LLM request is refused, so the Task fails (10.9)."""

APPROVAL: Final = ScenarioScript(
    "write it",
    (
        TurnScript(
            chunks=("checking",),
            calls=(
                CallScript(
                    "w1",
                    CallKind.PROTECTED,
                    WRITE_FILE,
                    tagged_arguments({"path": "a.txt", "content": "x"}, "w1"),
                    call_id="call-w1",
                ),
            ),
        ),
    ),
)
"""A Task that streams a chunk, then waits in `input-required` for approval of a protected call."""

SEED_PATHS: Final[dict[TaskState, tuple[TaskState, ...]]] = {
    TaskState.SUBMITTED: (),
    TaskState.WORKING: (TaskState.WORKING,),
    TaskState.INPUT_REQUIRED: (TaskState.WORKING, TaskState.INPUT_REQUIRED),
    TaskState.COMPLETED: (TaskState.WORKING, TaskState.COMPLETED),
    TaskState.FAILED: (TaskState.WORKING, TaskState.FAILED),
    TaskState.CANCELED: (TaskState.WORKING, TaskState.CANCELED),
}

type JsonId = int | str


class Binding(StrEnum):
    JSONRPC = "jsonrpc"
    REST = "rest"


BINDINGS: Final = (Binding.JSONRPC, Binding.REST)
_request_numbers = itertools.count(1)


# --- Canary for 18.9 ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def captured_logs() -> Iterator[list[MutableMapping[str, Any]]]:
    """Capture log records for the test, and fail it on any divergent save (18.9)."""
    with capture_logs() as logs:
        yield logs
    divergent = [record for record in logs if record.get("event") == DIVERGENT_EVENT]
    assert divergent == [], f"the SDK saved a state the Backend had not committed: {divergent}"


# --- The app under test -------------------------------------------------------------------------


@dataclass
class Rig:
    """The production app over `env`'s store with the scripted fakes, and a client for it."""

    env: StoreEnv
    client: httpx.AsyncClient
    runners: TaskRunnerRegistry
    llm: FakeLlmGateway
    mcp: FakeMcpManager


@contextlib.asynccontextmanager
async def running_app(env: StoreEnv, script: ScenarioScript) -> AsyncIterator[Rig]:
    clock = env.clock
    llm, mcp = build_fakes(script, clock, ClockSleep(clock))
    runners = TaskRunnerRegistry(
        RunnerDeps(
            sessions=env.store,
            llm=llm,
            registry=mcp.registry,
            mcp=mcp,
            orchestration=CONFIG.orchestration,
            base_instructions=None,
            model="fake/model",
            redactor=env.redactor,
            clock=clock,
            db_path=env.db_path,
            sleep=ManualTimers(clock),
            estimate_tokens=char_estimate,
        )
    )
    deps = dataclasses.replace(
        build_deps(CONFIG, None, clock=clock),
        session_store=env.store,
        task_store=ProjectedTaskStore(env.store, runners, clock),
        registry=mcp.registry,
        runners=runners,
    )
    transport = httpx.ASGITransport(app=build_app(CONFIG, deps), client=("127.0.0.1", 50000))
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            yield Rig(env, client, runners, llm, mcp)
    finally:
        await runners.cancel_all()


async def drive[T](
    env: StoreEnv, script: ScenarioScript, scenario: Callable[[Rig], Awaitable[T]]
) -> T:
    async with running_app(env, script) as rig, asyncio.timeout(TIMEOUT_S):
        return await scenario(rig)


def run_app[T](
    kind: StoreKind, script: ScenarioScript, scenario: Callable[[Rig], Awaitable[T]]
) -> T:
    """Open a fresh store, run `scenario` against the app in a fresh event loop."""
    with open_store(kind) as env:
        return asyncio.run(drive(env, script, scenario))


async def settle() -> None:
    """Let the SDK's background bookkeeping for a finished request run to the end."""
    for _ in range(SETTLE_ROUNDS):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)


# --- Requests -----------------------------------------------------------------------------------


def message_json(
    text: str, *, context_id: str = "", task_id: str = "", label: str = ""
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "messageId": f"m{next(_request_numbers)}{label}",
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    return message


async def post_message(
    client: httpx.AsyncClient,
    binding: Binding,
    message: dict[str, Any],
    *,
    streaming: bool,
    request_id: JsonId | None = None,
) -> httpx.Response:
    if binding is Binding.JSONRPC:
        envelope = {
            "jsonrpc": "2.0",
            "id": next(_request_numbers) if request_id is None else request_id,
            "method": "SendStreamingMessage" if streaming else "SendMessage",
            "params": {"message": message},
        }
        return await client.post("/", json=envelope, headers=HEADERS)
    path = "/message:stream" if streaming else "/message:send"
    return await client.post(path, json={"message": message}, headers=HEADERS)


async def call_task(
    client: httpx.AsyncClient, binding: Binding, task_id: str, *, cancel: bool
) -> httpx.Response:
    """Get-task or cancel over `binding`."""
    if binding is Binding.JSONRPC:
        envelope = {
            "jsonrpc": "2.0",
            "id": next(_request_numbers),
            "method": "CancelTask" if cancel else "GetTask",
            "params": {"id": task_id},
        }
        return await client.post("/", json=envelope, headers=HEADERS)
    if cancel:
        return await client.post(f"/tasks/{task_id}:cancel", json={}, headers=HEADERS)
    return await client.get(f"/tasks/{task_id}", headers=HEADERS)


def result_of(binding: Binding, response: httpx.Response) -> dict[str, Any]:
    """The result object of a successful response (the Task of get-task and cancel)."""
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    document: dict[str, Any] = response.json()
    if binding is Binding.JSONRPC:
        assert "error" not in document, response.text
        result: dict[str, Any] = document["result"]
        return result
    return document


def sse_data(body: bytes) -> list[str]:
    """The raw JSON text of every SSE event in `body`, in order.

    Lines end with CRLF, LF, or CR. An event ends at a blank line, and several `data` lines of one
    event are joined with a newline.
    """
    texts: list[str] = []
    data: list[str] = []
    for line in body.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line:
            if data:
                texts.append("\n".join(data))
                data.clear()
        elif line.startswith("data:"):
            value = line[len("data:") :]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        texts.append("\n".join(data))
    return texts


@dataclass
class Stream:
    """One parsed SendStreamingMessage response: the raw event texts and the unwrapped events."""

    raw: list[str]
    events: list[dict[str, Any]]

    @property
    def kinds(self) -> list[str]:
        return [next(iter(event)) for event in self.events]

    @property
    def last_state(self) -> str:
        for event in reversed(self.events):
            if "task" in event:
                return str(event["task"]["status"]["state"])
            if "statusUpdate" in event:
                return str(event["statusUpdate"]["status"]["state"])
        raise AssertionError("the stream carried no Task or Status_Event")

    def artifact_texts(self) -> dict[str, str]:
        """The text of every artifact the Token_Events of this stream built, by artifact ID."""
        texts: dict[str, str] = {}
        for event in self.events:
            update = event.get("artifactUpdate")
            if update is None:
                continue
            artifact = update["artifact"]
            text = "".join(part.get("text", "") for part in artifact["parts"])
            key = artifact["artifactId"]
            texts[key] = texts.get(key, "") + text if update.get("append") else text
        return texts


async def stream_message(
    client: httpx.AsyncClient, binding: Binding, message: dict[str, Any]
) -> Stream:
    request_id = next(_request_numbers)
    response = await post_message(client, binding, message, streaming=True, request_id=request_id)
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    assert response.headers["content-type"].startswith("text/event-stream"), response.text
    raw = sse_data(response.content)
    assert raw, "the stream carried no events"
    events: list[dict[str, Any]] = []
    for text in raw:
        document = json.loads(text)
        if binding is Binding.JSONRPC:
            assert "error" not in document, text
            assert document["jsonrpc"] == "2.0"
            assert document["id"] == request_id  # echoed on every event
            events.append(document["result"])
        else:
            events.append(document)
    return Stream(raw, events)


# --- ProtoJSON ----------------------------------------------------------------------------------


def enum_fields(raw: str | bytes) -> list[tuple[str, Any]]:
    """Every (`state` | `role`, value) pair in the JSON text, read before any re-encoding."""
    found: list[tuple[str, Any]] = []

    def collect(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        found.extend((key, value) for key, value in pairs if key in ("state", "role"))
        return dict(pairs)

    json.loads(raw, object_pairs_hook=collect)
    return found


@dataclass
class Seen:
    """The enum names found in raw bytes so far."""

    states: set[str] = dataclasses.field(default_factory=set)
    roles: set[str] = dataclasses.field(default_factory=set)

    def check(self, raw: str | bytes, where: str) -> list[tuple[str, Any]]:
        """Assert every `state` and `role` in `raw` is a ProtoJSON enum name."""
        fields = enum_fields(raw)
        for key, value in fields:
            assert isinstance(value, str), f"{where}: {key} is not a name: {value!r}"
            if key == "state":
                assert value in STATE_NAMES, f"{where}: unknown state spelling {value!r}"
                self.states.add(value)
            else:
                assert value in ROLE_NAMES, f"{where}: unknown role spelling {value!r}"
                self.roles.add(value)
        return fields


# --- Seeding ------------------------------------------------------------------------------------


async def seed_task(
    env: StoreEnv, context_id: str, task_id: str, state: TaskState, *, text: str = "seeded"
) -> None:
    """Commit a Task in `state` the way the Orchestrator would have (user message, transitions)."""
    store = env.store
    # The SDK's private history copy: the request message of a user, and a reply from the agent.
    sdk_history = Task(
        history=[
            Message(message_id=f"{task_id}-user", role=Role.ROLE_USER, parts=[Part(text="hi")]),
            Message(message_id=f"{task_id}-agent", role=Role.ROLE_AGENT, parts=[Part(text="ok")]),
        ]
    )
    await store.create_task(task_id, context_id, env.clock.now_ms(), snapshot_json(sdk_history))
    await store.append(context_id, [UserMessage("hi")], task_id=task_id)
    current = TaskState.SUBMITTED
    for new in SEED_PATHS[state]:
        change = StateChange(task_id, frozenset({current}), new, f"{text} {new.value}", None)
        await store.set_state(context_id, change)
        current = new


async def committed(env: StoreEnv, context_id: str, task_id: str) -> tuple[Any, ...]:
    """Everything stored for a Context and its Task, and the Task list."""
    reader = env.reader
    page = await reader.list_tasks(TaskQuery(page_size=100))
    return (
        await reader.get_task(task_id),
        await reader.get_history(context_id),
        await reader.active_task(context_id),
        page.tasks,
        await reader.get_a2a_snapshot(task_id),
    )


# --- Test 1: the error contract -----------------------------------------------------------------

CONTEXT: Final = "ctx-seeded"
TASK: Final = "task-seeded"


@dataclass(frozen=True)
class Rejection:
    """One Backend error and the seeded Task state that causes it."""

    kind: str  # "busy" or "not-awaiting"
    state: TaskState  # submitted or working

    @property
    def id(self) -> str:
        return f"{self.kind}-{self.state.value}"

    @property
    def continuation(self) -> bool:
        return self.kind == "not-awaiting"

    @property
    def message(self) -> str:
        return "Context is busy" if self.kind == "busy" else "Task is not awaiting input"

    @property
    def jsonrpc_code(self) -> int:
        return -32050 if self.kind == "busy" else -32602

    @property
    def http_status(self) -> int:
        return 409 if self.kind == "busy" else 400

    @property
    def grpc_status(self) -> str:
        return "ABORTED" if self.kind == "busy" else "INVALID_ARGUMENT"

    @property
    def reason(self) -> str:
        return "CONTEXT_BUSY" if self.kind == "busy" else "INVALID_PARAMS"

    @property
    def data(self) -> dict[str, str]:
        if self.kind == "busy":
            return {"context_id": CONTEXT, "active_task_id": TASK}
        return {"task_id": TASK, "state": sdk_name(self.state)}

    def request(self) -> dict[str, Any]:
        if self.continuation:
            return message_json("approve", context_id=CONTEXT, task_id=TASK, label="-continue")
        return message_json("another one", context_id=CONTEXT, label="-new")


REJECTIONS: Final = tuple(
    Rejection(kind, state)
    for kind in ("busy", "not-awaiting")
    for state in (TaskState.SUBMITTED, TaskState.WORKING)
)


async def exercise_rejection(
    rig: Rig, binding: Binding, rejection: Rejection, streaming: bool, request_id: JsonId
) -> httpx.Response:
    """Seed the Task, send the rejected message, and check nothing changed. Return the response."""
    await seed_task(rig.env, CONTEXT, TASK, rejection.state)
    before = await committed(rig.env, CONTEXT, TASK)
    assert before[2] == TASK  # the Task is the Context's Active_Task

    response = await post_message(
        rig.client, binding, rejection.request(), streaming=streaming, request_id=request_id
    )
    await settle()

    assert await committed(rig.env, CONTEXT, TASK) == before, "a rejected message changed the store"
    assert rig.runners.get(TASK) is None
    assert rig.runners.reservation(CONTEXT) is None
    assert rig.llm.requests == []
    return response


def test_dispatcher_hooks_exist_on_the_pinned_sdk() -> None:
    """The two private SDK methods `HostJsonRpcDispatcher` overrides are still there."""
    for name in ("_generate_error_response", "_create_response"):
        assert callable(getattr(JsonRpcDispatcher, name, None)), f"the SDK dropped {name}"
        assert name in vars(HostJsonRpcDispatcher), (
            f"HostJsonRpcDispatcher does not override {name}"
        )


@pytest.mark.parametrize("method", ["SendMessage", "SendStreamingMessage"])
@pytest.mark.parametrize("rejection", REJECTIONS, ids=lambda rejection: rejection.id)
def test_error_contract_jsonrpc(store_kind: StoreKind, method: str, rejection: Rejection) -> None:
    # Feature: local-agent-backend, Property 57: A2A error contract
    # Validates: Requirements 1.14, 1.15, 2.14
    streaming = method == "SendStreamingMessage"
    request_id: JsonId = "req-\u00e9-7" if streaming else 7

    async def scenario(rig: Rig) -> httpx.Response:
        return await exercise_rejection(rig, Binding.JSONRPC, rejection, streaming, request_id)

    response = run_app(store_kind, PLAIN, scenario)

    where = f"{method}: {response.status_code} {response.text}"
    assert response.status_code == 200, where
    assert response.headers["content-type"].startswith("application/json"), where
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": rejection.jsonrpc_code,
            "message": rejection.message,
            "data": rejection.data,
        },
    }, where


@pytest.mark.parametrize("path", ["message:send", "message:stream"])
@pytest.mark.parametrize("rejection", REJECTIONS, ids=lambda rejection: rejection.id)
def test_error_contract_rest(store_kind: StoreKind, path: str, rejection: Rejection) -> None:
    # Feature: local-agent-backend, Property 57: A2A error contract
    # Validates: Requirements 1.14, 1.15, 2.14
    async def scenario(rig: Rig) -> httpx.Response:
        return await exercise_rejection(rig, Binding.REST, rejection, path == "message:stream", 0)

    response = run_app(store_kind, PLAIN, scenario)

    where = f"{path}: {response.status_code} {response.text}"
    assert response.status_code == rejection.http_status, where
    assert response.headers["content-type"].startswith("application/json"), where
    assert response.json() == {
        "error": {
            "code": rejection.http_status,
            "status": rejection.grpc_status,
            "message": rejection.message,
            "details": [
                {
                    "@type": ERROR_INFO_TYPE,
                    "reason": rejection.reason,
                    "domain": A2A_DOMAIN,
                    "metadata": rejection.data,
                }
            ],
        }
    }, where


# --- Test 2: ProtoJSON on the raw bytes ----------------------------------------------------------


@pytest.mark.parametrize("binding", BINDINGS)
def test_protojson_contract(store_kind: StoreKind, binding: Binding) -> None:
    # Feature: local-agent-backend, Property 60: ProtoJSON enum form
    # Validates: Requirements 1.10
    seen = Seen()

    async def plain(rig: Rig) -> None:
        client = rig.client
        # A Task in each of the six states, through get-task.
        for state in TaskState:
            context_id, task_id = f"ctx-{state.value}", f"task-{state.value}"
            await seed_task(rig.env, context_id, task_id, state)
            response = await call_task(client, binding, task_id, cancel=False)
            seen.check(response.content, f"get-task {state.value}")
            task = result_of(binding, response)
            assert task["status"]["state"] == sdk_name(state)
            if state is not TaskState.SUBMITTED:
                assert task["status"]["message"]["role"] == "ROLE_AGENT"

        # Both contract errors, streaming and not, against the seeded submitted/working Tasks.
        for state in (TaskState.SUBMITTED, TaskState.WORKING):
            context_id, task_id = f"ctx-{state.value}", f"task-{state.value}"
            requests = (
                message_json("new", context_id=context_id),
                message_json("go on", context_id=context_id, task_id=task_id),
            )
            for message in requests:
                for streaming in (False, True):
                    response = await post_message(client, binding, message, streaming=streaming)
                    assert response.status_code == (200 if binding is Binding.JSONRPC else 409) or (
                        response.status_code == 400
                    ), response.text
                    assert "error" in response.json(), response.text
                    fields = seen.check(response.content, f"contract error {state.value}")
                    if message.get("taskId"):
                        assert ("state", sdk_name(state)) in fields

        # A new Task: Task, Status_Event, and Token_Event on the SSE bytes.
        stream = await stream_message(client, binding, message_json("hello", context_id="ctx-live"))
        for text in stream.raw:
            seen.check(text, "stream event")
        assert {"task", "statusUpdate", "artifactUpdate"} <= set(stream.kinds)
        assert stream.events[0]["task"]["status"]["state"] == sdk_name(TaskState.SUBMITTED)
        assert stream.last_state == sdk_name(TaskState.COMPLETED)
        live_id = stream.events[0]["task"]["id"]

        # The non-streaming send, and get-task for the finished Task.
        response = await post_message(
            client, binding, message_json("again", context_id="ctx-live-2"), streaming=False
        )
        seen.check(response.content, "send")
        assert result_of(binding, response)["task"]["status"]["state"] == sdk_name(
            TaskState.COMPLETED
        )
        response = await call_task(client, binding, live_id, cancel=False)
        seen.check(response.content, "get-task completed")
        assert result_of(binding, response)["artifacts"]

    async def approval(rig: Rig) -> None:
        client = rig.client
        stream = await stream_message(
            client, binding, message_json("write it", context_id="ctx-cancel")
        )
        for text in stream.raw:
            seen.check(text, "approval stream event")
        assert stream.last_state == sdk_name(TaskState.INPUT_REQUIRED)
        task_id = stream.events[0]["task"]["id"]

        response = await call_task(client, binding, task_id, cancel=True)
        seen.check(response.content, "cancel")
        assert result_of(binding, response)["status"]["state"] == sdk_name(TaskState.CANCELED)
        response = await call_task(client, binding, task_id, cancel=False)
        seen.check(response.content, "get-task canceled")
        assert result_of(binding, response)["status"]["state"] == sdk_name(TaskState.CANCELED)

    run_app(store_kind, PLAIN, plain)
    run_app(store_kind, APPROVAL, approval)

    assert seen.states == STATE_NAMES  # all six states were seen in the raw bytes
    assert seen.roles == ROLE_NAMES


# --- Test 3: the first event of a stream ---------------------------------------------------------


def check_first_event(
    stream: Stream,
    expected: TaskState,
    task_id: str | None,
    context_id: str,
    earlier: Stream | None,
) -> str:
    """Assert the first-event rules for one stream. Return the Task ID."""
    first = stream.events[0]
    assert list(first) == ["task"], f"first event is {list(first)}"
    snapshot = first["task"]
    assert snapshot["status"]["state"] == sdk_name(expected)  # ProtoJSON on the wire
    assert snapshot["contextId"] == context_id
    if task_id is not None:
        assert snapshot["id"] == task_id
    for event in stream.events[1:]:
        assert list(event) in (["statusUpdate"], ["artifactUpdate"]), list(event)
        update = next(iter(event.values()))
        assert update["taskId"] == snapshot["id"]
        assert update["contextId"] == context_id
    snapshot_artifacts = {
        artifact["artifactId"]: "".join(part.get("text", "") for part in artifact["parts"])
        for artifact in snapshot.get("artifacts", [])
    }
    assert snapshot_artifacts == (earlier.artifact_texts() if earlier is not None else {})
    return str(snapshot["id"])


@pytest.mark.parametrize("binding", BINDINGS)
def test_first_event_snapshot(store_kind: StoreKind, binding: Binding) -> None:
    # Feature: local-agent-backend, Property 59: First stream event is a Task_Snapshot
    # Validates: Requirements 3.10, 3.11, 3.12, 3.13
    context_id = "ctx-first"

    async def scenario(rig: Rig) -> None:
        client = rig.client
        # A new Task: the snapshot is `submitted` and carries no artifacts yet (3.10, 3.11).
        new = await stream_message(client, binding, message_json("write it", context_id=context_id))
        task_id = check_first_event(new, TaskState.SUBMITTED, None, context_id, None)
        assert new.last_state == sdk_name(TaskState.INPUT_REQUIRED)
        assert new.artifact_texts() == {"response-1": "checking"}

        # An unrecognized reply: the `input-required` snapshot, one re-prompt, nothing else (3.13).
        reply = message_json("hmm, maybe?", context_id=context_id, task_id=task_id)
        unrecognized = await stream_message(client, binding, reply)
        check_first_event(unrecognized, TaskState.INPUT_REQUIRED, task_id, context_id, new)
        assert unrecognized.kinds == ["task", "statusUpdate"]
        status = unrecognized.events[1]["statusUpdate"]["status"]
        assert status["state"] == sdk_name(TaskState.INPUT_REQUIRED)
        assert status["message"]["parts"][0]["text"] == REPROMPT_TEXT

        # A recognized decision: the `working` snapshot, then the `working` Status_Event (3.12).
        reply = message_json("approve", context_id=context_id, task_id=task_id)
        recognized = await stream_message(client, binding, reply)
        check_first_event(recognized, TaskState.WORKING, task_id, context_id, new)
        assert recognized.kinds[:2] == ["task", "statusUpdate"]
        working = recognized.events[1]["statusUpdate"]["status"]["state"]
        assert working == sdk_name(TaskState.WORKING)
        assert recognized.last_state in {sdk_name(state) for state in STREAM_CLOSING_STATES}

    run_app(store_kind, APPROVAL, scenario)


# --- Test 4: no divergent saves ------------------------------------------------------------------


@pytest.mark.parametrize("binding", BINDINGS)
def test_no_divergent_saves(
    store_kind: StoreKind, binding: Binding, captured_logs: list[MutableMapping[str, Any]]
) -> None:
    # Feature: local-agent-backend, 18.9 canary
    # Validates: Requirements 18.9
    async def finished(rig: Rig, text: str, expected: TaskState) -> None:
        client = rig.client
        stream = await stream_message(client, binding, message_json(text))
        assert stream.last_state == sdk_name(expected)
        task_id = stream.events[0]["task"]["id"]
        task = result_of(binding, await call_task(client, binding, task_id, cancel=False))
        assert task["status"]["state"] == sdk_name(expected)

    async def completes(rig: Rig) -> None:
        await finished(rig, "hello", TaskState.COMPLETED)
        response = await post_message(rig.client, binding, message_json("again"), streaming=False)
        assert result_of(binding, response)["task"]["status"]["state"] == sdk_name(
            TaskState.COMPLETED
        )

    async def fails(rig: Rig) -> None:
        await finished(rig, "boom", TaskState.FAILED)

    async def approved(rig: Rig) -> None:
        stream = await stream_message(rig.client, binding, message_json("write it", context_id="c"))
        task_id = stream.events[0]["task"]["id"]
        reply = message_json("approve", context_id="c", task_id=task_id)
        assert (await stream_message(rig.client, binding, reply)).last_state == sdk_name(
            TaskState.COMPLETED
        )

    async def canceled(rig: Rig) -> None:
        stream = await stream_message(rig.client, binding, message_json("write it"))
        task_id = stream.events[0]["task"]["id"]
        response = await call_task(rig.client, binding, task_id, cancel=True)
        assert result_of(binding, response)["status"]["state"] == sdk_name(TaskState.CANCELED)

    run_app(store_kind, PLAIN, completes)
    run_app(store_kind, REJECTED, fails)
    run_app(store_kind, APPROVAL, approved)
    run_app(store_kind, APPROVAL, canceled)

    assert [r for r in captured_logs if r.get("event") == DIVERGENT_EVENT] == []


def test_divergent_save_canary_sees_a_divergent_save(store_kind: StoreKind) -> None:
    """The log capture is not blind: a terminal save that disagrees with the commit is recorded."""

    async def scenario(rig: Rig) -> None:
        await seed_task(rig.env, "c", "t", TaskState.COMPLETED)
        tasks = ProjectedTaskStore(rig.env.store)
        stale = Task(id="t", context_id="c")
        stale.status.state = to_sdk(TaskState.FAILED)
        await tasks.save(stale, ServerCallContext())

    captured: list[MutableMapping[str, Any]] = []
    with capture_logs() as logs:
        run_app(store_kind, PLAIN, scenario)
        captured.extend(logs)
    assert [r["event"] for r in captured if r.get("event") == DIVERGENT_EVENT] == [DIVERGENT_EVENT]


# --- Test 5: storage failure on create ----------------------------------------------------------


@dataclass(frozen=True)
class Transport:
    binding: Binding
    streaming: bool

    @property
    def id(self) -> str:
        return f"{self.binding.value}-{'stream' if self.streaming else 'send'}"


TRANSPORTS: Final = tuple(
    Transport(binding, streaming) for binding in BINDINGS for streaming in (False, True)
)


def count_rows(path: str) -> dict[str, int]:
    """Rows of the tables a Task creation writes, read on a separate read-only connection."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("contexts", "tasks", "messages", "artifacts")
        }
    finally:
        connection.close()


async def assert_nothing_stored(env: StoreEnv, context_id: str) -> None:
    reader = env.reader
    assert (await reader.list_tasks(TaskQuery(page_size=100))).tasks == ()
    assert await reader.active_task(context_id) is None
    assert await reader.get_history(context_id) == (None, [])
    if env.path is not None:
        assert count_rows(str(env.path)) == {
            "contexts": 0,
            "tasks": 0,
            "messages": 0,
            "artifacts": 0,
        }


@pytest.mark.parametrize("transport", TRANSPORTS, ids=lambda transport: transport.id)
def test_storage_failure_on_create(
    store_kind: StoreKind, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Feature: local-agent-backend, 18.30
    # Validates: Requirements 18.30
    context_id = "ctx-storage"
    request_id = 99

    async def send(rig: Rig) -> httpx.Response:
        return await post_message(
            rig.client,
            transport.binding,
            message_json("hello", context_id=context_id),
            streaming=transport.streaming,
            request_id=request_id,
        )

    async def failing(rig: Rig) -> httpx.Response:
        response = await send(rig)
        await settle()  # the SDK's bookkeeping must not create the Task after the failure either
        await assert_nothing_stored(rig.env, context_id)
        assert rig.llm.requests == []
        assert rig.runners.reservation(context_id) is None
        # The Context is not stuck: the same request now succeeds.
        retry = await send(rig)
        assert retry.status_code == 200, retry.text
        if transport.streaming:
            assert retry.headers["content-type"].startswith("text/event-stream"), retry.text
        else:
            body = retry.json()
            key = "result" if transport.binding is Binding.JSONRPC else "task"
            assert key in body, retry.text
            assert "error" not in body, retry.text
        assert len((await rig.env.reader.list_tasks(TaskQuery(page_size=100))).tasks) == 1
        return response

    if store_kind == "sqlite":
        # The creation commit is the first COMMIT of a clean run: admission only reads.
        with open_store("sqlite") as clean:
            asyncio.run(drive(clean, PLAIN, send))
            assert clean.factory is not None
            commit = clean.factory.connection.commit_points[0]
        with open_store("sqlite", fail_at=commit, error=disk_io_error) as env:
            response = asyncio.run(drive(env, PLAIN, failing))
            assert env.factory is not None
            assert env.factory.connection.fired == (commit,)
    else:
        with open_store("memory") as env:
            real_create = env.store.create_task
            attempts: list[str] = []

            async def create_task(
                task_id: str, ctx: str, now_ms: int, snapshot: str | None
            ) -> bool:
                attempts.append(task_id)
                if len(attempts) == 1:
                    raise StorageError(env.db_path, "disk I/O error")
                return await real_create(task_id, ctx, now_ms, snapshot)

            monkeypatch.setattr(env.store, "create_task", create_task)
            response = asyncio.run(drive(env, PLAIN, failing))

    where = f"{transport.id}: {response.status_code} {response.text}"
    document = response.json()
    if transport.binding is Binding.JSONRPC:
        assert response.status_code == 200, where
        assert document["jsonrpc"] == "2.0", where
        assert document["id"] == request_id, where
        assert document["error"]["code"] == -32603, where
        assert document["error"]["message"] == STORAGE_MESSAGE, where
    else:
        assert response.status_code == 500, where
        assert document["error"]["code"] == 500, where
        assert document["error"]["status"] == "INTERNAL", where
        assert document["error"]["message"] == STORAGE_MESSAGE, where
