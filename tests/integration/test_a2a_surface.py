"""A2A surface integration tests (task 22.3).

The production app runs over a real Session_Store of both kinds (`memory` and `sqlite`), with the
real SDK request handler, `TaskRunner`, and `LlmGateway`. The LLM is the in-process
`ScriptedCompletion` or, where the timing of a live stream matters, the HTTP `FakeLlmServer`. The
wire-level details (exact error bodies, ProtoJSON spellings, the first-event snapshot rules) are in
`test_a2a_contract.py`; this module checks that the whole surface works end to end.

* ASGI transport (`httpx.ASGITransport` buffers a response, so streams are read after they end):
  JSON-RPC and REST send, SSE streaming, get-task, cancel, every JSON-RPC error category, and the
  terminal-state and unknown-Task errors (1.9, 2.8, 2.10, 2.11, 2.12, 3.7, 18.29).
* Real uvicorn socket: the endpoints are reachable (1.1); the first SSE event arrives within 1 s
  while the LLM is still silent (3.1); the stream closes within 1 s of the final status (3.5); a
  cancel on a live stream ends it with `canceled` within 2 s and nothing follows (2.7, 3.9).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Self

import httpx
import pytest

from fakes import LlmFailure, LlmTurn, text_delta
from integration_harness import (
    A2A_HEADERS,
    DATABASE_NAME,
    AppHandle,
    FakeLlmServer,
    ScriptedCompletion,
    StubTurn,
    message_params,
)
from local_agent.session.store import TaskQuery

pytestmark = pytest.mark.integration

type StorageBackend = str
type AppOpener = Callable[..., contextlib.AbstractAsyncContextManager[AppHandle]]

COMPLETED: Final = "TASK_STATE_COMPLETED"
FAILED: Final = "TASK_STATE_FAILED"
CANCELED: Final = "TASK_STATE_CANCELED"
WORKING: Final = "TASK_STATE_WORKING"
SUBMITTED: Final = "TASK_STATE_SUBMITTED"
INPUT_REQUIRED: Final = "TASK_STATE_INPUT_REQUIRED"
CLOSING_STATES: Final = frozenset({COMPLETED, FAILED, CANCELED, INPUT_REQUIRED})
TERMINAL_STATES: Final = frozenset({COMPLETED, FAILED, CANCELED})

TASK_NOT_FOUND: Final = -32001
TASK_NOT_CANCELABLE: Final = -32002
UNSUPPORTED_OPERATION: Final = -32004

FIRST_EVENT_LIMIT_S: Final = 1.0
CLOSE_LIMIT_S: Final = 1.0
CANCEL_LIMIT_S: Final = 2.0
WAIT_S: Final = 15.0

_ids = itertools.count(1)


class Binding(StrEnum):
    JSONRPC = "jsonrpc"
    REST = "rest"


BINDINGS: Final = (Binding.JSONRPC, Binding.REST)


# --- Wire helpers -------------------------------------------------------------------------------


def message_of(
    text: str, *, context_id: str | None = None, task_id: str | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = message_params(text, context_id=context_id, task_id=task_id)[
        "message"
    ]
    return message


def envelope(
    method: str, params: Mapping[str, Any], request_id: int | str | None = None
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": next(_ids) if request_id is None else request_id,
        "method": method,
        "params": dict(params),
    }


async def send(
    client: httpx.AsyncClient, binding: Binding, message: dict[str, Any], *, stream: bool = False
) -> httpx.Response:
    """SendMessage (or SendStreamingMessage) over `binding`. A stream is read to its end."""
    if binding is Binding.JSONRPC:
        method = "SendStreamingMessage" if stream else "SendMessage"
        body = envelope(method, {"message": message})
        return await client.post("/", json=body, headers=A2A_HEADERS)
    path = "/message:stream" if stream else "/message:send"
    return await client.post(path, json={"message": message}, headers=A2A_HEADERS)


async def get_task(client: httpx.AsyncClient, binding: Binding, task_id: str) -> httpx.Response:
    if binding is Binding.JSONRPC:
        return await client.post(
            "/", json=envelope("GetTask", {"id": task_id}), headers=A2A_HEADERS
        )
    return await client.get(f"/tasks/{task_id}", headers=A2A_HEADERS)


async def cancel_task(client: httpx.AsyncClient, binding: Binding, task_id: str) -> httpx.Response:
    if binding is Binding.JSONRPC:
        body = envelope("CancelTask", {"id": task_id})
        return await client.post("/", json=body, headers=A2A_HEADERS)
    return await client.post(f"/tasks/{task_id}:cancel", json={}, headers=A2A_HEADERS)


def ok_body(binding: Binding, response: httpx.Response) -> dict[str, Any]:
    """The result object of a successful call: `{"task": ...}` of a send, or the Task itself."""
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    document: dict[str, Any] = response.json()
    if binding is Binding.JSONRPC:
        assert "error" not in document, response.text
        assert document["jsonrpc"] == "2.0"
        result: dict[str, Any] = document["result"]
        return result
    return document


def sent_task(binding: Binding, response: httpx.Response) -> dict[str, Any]:
    task: dict[str, Any] = ok_body(binding, response)["task"]
    return task


def artifact_text(task: Mapping[str, Any]) -> str:
    """The text of all artifacts of a Task, in order."""
    return "".join(
        part.get("text", "") for artifact in task.get("artifacts", []) for part in artifact["parts"]
    )


def error_reason(details: Sequence[Mapping[str, Any]] | None) -> str | None:
    for detail in details or ():
        if detail.get("@type", "").endswith("google.rpc.ErrorInfo"):
            return str(detail["reason"])
    return None


def assert_a2a_error(
    binding: Binding,
    response: httpx.Response,
    *,
    jsonrpc_code: int,
    rest_status: int,
    reason: str,
) -> dict[str, Any]:
    """The response is the A2A error `reason` on `binding`. Returns the `error` object."""
    where = f"{response.status_code} {response.text}"
    document: dict[str, Any] = response.json()
    error: dict[str, Any] = document["error"]
    if binding is Binding.JSONRPC:
        assert response.status_code == 200, where
        assert error["code"] == jsonrpc_code, where
        assert error_reason(error.get("data")) == reason, where
    else:
        assert response.status_code == rest_status, where
        assert error["code"] == rest_status, where
        assert error_reason(error.get("details")) == reason, where
    return error


def sse_events(binding: Binding, body: bytes) -> list[dict[str, Any]]:
    """The events of a finished SSE body, unwrapped from the JSON-RPC envelope when there is one."""
    events: list[dict[str, Any]] = []
    for line in body.decode("utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        document = json.loads(line[len("data:") :].strip())
        if binding is Binding.JSONRPC:
            assert "error" not in document, line
            document = document["result"]
        events.append(document)
    return events


def event_kind(event: Mapping[str, Any]) -> str:
    return str(next(iter(event)))


def event_state(event: Mapping[str, Any]) -> str | None:
    """The Task_State a `task` or `statusUpdate` event reports, `None` for other events."""
    if "task" in event:
        return str(event["task"]["status"]["state"])
    if "statusUpdate" in event:
        return str(event["statusUpdate"]["status"]["state"])
    return None


def streamed_text(events: Sequence[Mapping[str, Any]]) -> str:
    """The text the Token_Events of a stream build (one artifact per response)."""
    texts: dict[str, str] = {}
    for event in events:
        update = event.get("artifactUpdate")
        if update is None:
            continue
        artifact = update["artifact"]
        chunk = "".join(part.get("text", "") for part in artifact["parts"])
        key = artifact["artifactId"]
        texts[key] = texts.get(key, "") + chunk if update.get("append") else chunk
    return "".join(texts.values())


def assert_stream_shape(events: Sequence[Mapping[str, Any]], final_state: str) -> str:
    """First event is the submitted snapshot, one closing status is last. Returns the Task ID."""
    assert events, "the stream carried no events"
    assert event_kind(events[0]) == "task", [event_kind(event) for event in events]
    assert events[0]["task"]["status"]["state"] == SUBMITTED
    task_id = str(events[0]["task"]["id"])
    states = [event_state(event) for event in events]
    assert states[1] == WORKING
    closing = [index for index, state in enumerate(states) if state in CLOSING_STATES]
    assert closing == [len(events) - 1], states  # the only closing status is the last event
    assert states[-1] == final_state
    return task_id


# --- Store helpers ------------------------------------------------------------------------------


async def stored_task_ids(app: AppHandle) -> list[str]:
    page = await app.deps.session_store.list_tasks(TaskQuery(page_size=100))
    return [record.task_id for record in page.tasks]


def stored_context_count(app: AppHandle) -> int | None:
    """Rows in `contexts` for the sqlite backend, `None` for memory."""
    if app.config.storage.backend != "sqlite":
        return None
    path = Path(app.storage_dir) / DATABASE_NAME
    with contextlib.closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as connection:
        row = connection.execute("SELECT COUNT(*) FROM contexts").fetchone()
    return int(row[0])


@dataclass(frozen=True)
class StoreState:
    """Everything a rejected request must leave alone."""

    task_ids: tuple[str, ...]
    contexts: int | None
    llm_calls: int

    @classmethod
    async def of(cls, app: AppHandle, llm: ScriptedCompletion) -> Self:
        return cls(tuple(await stored_task_ids(app)), stored_context_count(app), len(llm.calls))


async def task_id_of_context(app: AppHandle, context_id: str, timeout_s: float = WAIT_S) -> str:
    """The Task ID the Backend gave the first Task of `context_id`, once it exists."""
    async with asyncio.timeout(timeout_s):
        while True:
            page = await app.deps.session_store.list_tasks(TaskQuery(context_id=context_id))
            if page.tasks:
                return page.tasks[0].task_id
            await asyncio.sleep(0.01)


async def wait_for_state(
    client: httpx.AsyncClient,
    task_id: str,
    states: Sequence[str],
    *,
    artifact_contains: str = "",
    timeout_s: float = WAIT_S,
) -> dict[str, Any]:
    """Poll get-task until the Task is in one of `states` and its artifacts hold the text."""
    task: dict[str, Any] = {}
    async with asyncio.timeout(timeout_s):
        while True:
            task = ok_body(Binding.JSONRPC, await get_task(client, Binding.JSONRPC, task_id))
            if task["status"]["state"] in states and artifact_contains in artifact_text(task):
                return task
            await asyncio.sleep(0.02)


# --- A Task in each state -----------------------------------------------------------------------

DONE_TEXT: Final = "all done"
PARTIAL_TEXT: Final = "par"

COMPLETES: Final = LlmTurn(deltas=(text_delta("all "), text_delta("done")))
REFUSED: Final = LlmTurn(failure=LlmFailure.REJECTED, reason="no")
STALLS: Final = LlmTurn(deltas=(text_delta(PARTIAL_TEXT),), failure=LlmFailure.STALL, fail_after=1)
"""A response that streams `par` and then waits until the Task is cancelled."""


@dataclass
class Finished:
    """The Task IDs of three finished Tasks, and the call count of the LLM when they were done."""

    completed: str
    failed: str
    canceled: str
    llm_calls: int


async def finish_three_tasks(app: AppHandle, llm: ScriptedCompletion) -> Finished:
    """Run one Task to `completed`, one to `failed`, and one to `canceled` through the API."""
    llm.script(COMPLETES, REFUSED, STALLS)
    client = app.client
    completed = sent_task(
        Binding.JSONRPC, await send(client, Binding.JSONRPC, message_of("finish"))
    )
    failed = sent_task(Binding.JSONRPC, await send(client, Binding.JSONRPC, message_of("refuse")))
    assert completed["status"]["state"] == COMPLETED
    assert failed["status"]["state"] == FAILED

    context_id = "ctx-canceled"
    pending = asyncio.create_task(
        send(client, Binding.JSONRPC, message_of("stall", context_id=context_id))
    )
    try:
        canceled_id = await task_id_of_context(app, context_id)
        await wait_for_state(client, canceled_id, [WORKING], artifact_contains=PARTIAL_TEXT)
        reply = await cancel_task(client, Binding.JSONRPC, canceled_id)
        assert ok_body(Binding.JSONRPC, reply)["status"]["state"] == CANCELED
        assert sent_task(Binding.JSONRPC, await pending)["status"]["state"] == CANCELED
    finally:
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
    return Finished(completed["id"], failed["id"], canceled_id, len(llm.calls))


# --- JSON-RPC and REST: send, get-task ----------------------------------------------------------


@pytest.mark.parametrize("binding", BINDINGS)
async def test_send_message_returns_the_finished_task(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 1.1, 2.10, 3.7
    fake_llm.script(COMPLETES)

    reply = await send(app.client, binding, message_of("hi", context_id="ctx-send"))

    task = sent_task(binding, reply)
    assert task["status"]["state"] == COMPLETED  # returned only after the final state (3.7)
    assert task["contextId"] == "ctx-send"
    assert artifact_text(task) == DONE_TEXT
    assert await stored_task_ids(app) == [task["id"]]
    assert len(fake_llm.calls) == 1


@pytest.mark.parametrize("binding", BINDINGS)
async def test_get_task_returns_what_the_task_accumulated(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 2.10
    fake_llm.script(COMPLETES)
    task = sent_task(binding, await send(app.client, binding, message_of("hi")))

    fetched = ok_body(binding, await get_task(app.client, binding, task["id"]))

    assert fetched["id"] == task["id"]
    assert fetched["contextId"] == task["contextId"]
    assert fetched["status"]["state"] == COMPLETED
    assert artifact_text(fetched) == DONE_TEXT


@pytest.mark.parametrize("binding", BINDINGS)
async def test_get_task_of_a_running_task_returns_the_artifacts_so_far(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 2.10, 3.7
    fake_llm.script(STALLS)
    context_id = f"ctx-running-{binding}"
    pending = asyncio.create_task(
        send(app.client, binding, message_of("go", context_id=context_id))
    )
    try:
        task_id = await task_id_of_context(app, context_id)
        await wait_for_state(app.client, task_id, [WORKING], artifact_contains=PARTIAL_TEXT)

        fetched = ok_body(binding, await get_task(app.client, binding, task_id))
        assert fetched["status"]["state"] == WORKING
        assert artifact_text(fetched) == PARTIAL_TEXT
        assert not pending.done(), "a non-streaming send answered before the Task was final (3.7)"

        await cancel_task(app.client, binding, task_id)
        assert sent_task(binding, await pending)["status"]["state"] == CANCELED
    finally:
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending


# --- SSE streaming ------------------------------------------------------------------------------


@pytest.mark.parametrize("binding", BINDINGS)
async def test_streaming_message_ends_after_the_final_status(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 1.1, 3.1, 3.5
    fake_llm.script(COMPLETES)

    response = await send(app.client, binding, message_of("stream it"), stream=True)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(binding, response.content)
    task_id = assert_stream_shape(events, COMPLETED)
    assert any(event_kind(event) == "artifactUpdate" for event in events)
    assert streamed_text(events) == DONE_TEXT
    assert ok_body(binding, await get_task(app.client, binding, task_id))["status"]["state"] == (
        COMPLETED
    )


@pytest.mark.parametrize("binding", BINDINGS)
async def test_streaming_a_failed_task_closes_with_failed(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 3.5
    fake_llm.script(REFUSED)

    response = await send(app.client, binding, message_of("refuse"), stream=True)

    events = sse_events(binding, response.content)
    task_id = assert_stream_shape(events, FAILED)
    final = events[-1]["statusUpdate"]["status"]
    assert final["message"]["parts"][0]["text"]  # a human-readable cause
    stored = ok_body(binding, await get_task(app.client, binding, task_id))
    assert stored["status"]["state"] == FAILED


# --- Cancel -------------------------------------------------------------------------------------


@pytest.mark.parametrize("binding", BINDINGS)
async def test_cancel_stops_a_running_task_and_answers_the_pending_send(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 2.7, 2.10, 3.7, 18.29
    fake_llm.script(STALLS)
    context_id = f"ctx-cancel-{binding}"
    pending = asyncio.create_task(
        send(app.client, binding, message_of("go", context_id=context_id))
    )
    try:
        task_id = await task_id_of_context(app, context_id)
        await wait_for_state(app.client, task_id, [WORKING], artifact_contains=PARTIAL_TEXT)

        started = time.monotonic()
        reply = await cancel_task(app.client, binding, task_id)
        elapsed = time.monotonic() - started

        canceled = ok_body(binding, reply)
        assert canceled["status"]["state"] == CANCELED
        assert elapsed < CANCEL_LIMIT_S
        answered = sent_task(binding, await asyncio.wait_for(pending, CANCEL_LIMIT_S))
        assert answered["id"] == task_id
        assert answered["status"]["state"] == CANCELED
    finally:
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending

    stored = ok_body(binding, await get_task(app.client, binding, task_id))
    assert stored["status"]["state"] == CANCELED
    assert artifact_text(stored) == PARTIAL_TEXT  # what was streamed before the cancel is kept
    # The Context is free again: a new message in it is admitted.
    fake_llm.script(COMPLETES)
    again = sent_task(
        binding, await send(app.client, binding, message_of("next", context_id=context_id))
    )
    assert again["status"]["state"] == COMPLETED
    assert again["id"] != task_id


# --- JSON-RPC error categories ------------------------------------------------------------------


@dataclass(frozen=True)
class RpcCase:
    """One malformed JSON-RPC request and the error category it must produce (1.9)."""

    name: str
    body: bytes
    code: int
    request_id: int | str | None
    element: str  # text the error must contain so a reader can find the invalid element
    reason: str | None = None  # ErrorInfo reason, when the SDK gives one


def rpc_body(document: Any) -> bytes:
    return json.dumps(document).encode()


RPC_CASES: Final = (
    RpcCase("parse-error", b"{not json", -32700, None, ""),
    RpcCase("empty-body", b"", -32700, None, ""),
    RpcCase("not-an-object", b"[1, 2]", -32600, None, "object", "INVALID_REQUEST"),
    RpcCase(
        "missing-method",
        rpc_body({"jsonrpc": "2.0", "id": 5, "params": {}}),
        -32600,
        5,
        "method",
        "INVALID_REQUEST",
    ),
    RpcCase(
        "wrong-jsonrpc-version",
        rpc_body({"jsonrpc": "1.0", "id": "abc", "method": "GetTask", "params": {"id": "x"}}),
        -32600,
        "abc",
        "jsonrpc",
        "INVALID_REQUEST",
    ),
    RpcCase(
        "unknown-method",
        rpc_body({"jsonrpc": "2.0", "id": 6, "method": "NoSuchMethod", "params": {}}),
        -32601,
        6,
        "NoSuchMethod",
    ),
    RpcCase(
        "get-task-without-id",
        rpc_body({"jsonrpc": "2.0", "id": 7, "method": "GetTask", "params": {}}),
        -32602,
        7,
        "id",
        "INVALID_PARAMS",
    ),
    RpcCase(
        "get-task-id-wrong-type",
        rpc_body({"jsonrpc": "2.0", "id": 8, "method": "GetTask", "params": {"id": 5}}),
        -32602,
        8,
        "id",
        "INVALID_PARAMS",
    ),
    RpcCase(
        "cancel-without-id",
        rpc_body({"jsonrpc": "2.0", "id": 9, "method": "CancelTask", "params": {}}),
        -32602,
        9,
        "id",
        "INVALID_PARAMS",
    ),
    RpcCase(
        "send-without-message",
        rpc_body({"jsonrpc": "2.0", "id": "s-1", "method": "SendMessage", "params": {}}),
        -32602,
        "s-1",
        "message",
        "INVALID_PARAMS",
    ),
    RpcCase(
        "stream-without-message",
        rpc_body({"jsonrpc": "2.0", "id": 10, "method": "SendStreamingMessage", "params": {}}),
        -32602,
        10,
        "message",
        "INVALID_PARAMS",
    ),
    RpcCase(
        "send-bad-role",
        rpc_body(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "SendMessage",
                "params": {
                    "message": {"messageId": "m-1", "role": "BOGUS", "parts": [{"text": "hi"}]}
                },
            }
        ),
        -32602,
        11,
        "role",
        "INVALID_PARAMS",
    ),
)


def case_id(case: object) -> str:
    return case.name if isinstance(case, RpcCase) else ""


@pytest.mark.parametrize("case", RPC_CASES, ids=case_id)
async def test_jsonrpc_error_categories(
    app: AppHandle, fake_llm: ScriptedCompletion, case: RpcCase
) -> None:
    # Validates: Requirements 1.9
    before = await StoreState.of(app, fake_llm)

    response = await app.client.post(
        "/", content=case.body, headers={**A2A_HEADERS, "Content-Type": "application/json"}
    )

    where = f"{response.status_code} {response.text}"
    assert response.status_code == 200, where
    document = response.json()
    assert document["jsonrpc"] == "2.0", where
    assert document["id"] == case.request_id, where  # echoed when it can be read, else null
    error = document["error"]
    assert error["code"] == case.code, where
    assert isinstance(error["message"], str)
    assert error["message"], where
    if case.reason is not None:
        assert error_reason(error.get("data")) == case.reason, where
    assert case.element in json.dumps(error), f"the error does not name {case.element!r}: {where}"
    assert await StoreState.of(app, fake_llm) == before  # no Task or Context, no LLM request


async def test_rest_error_categories(app: AppHandle, fake_llm: ScriptedCompletion) -> None:
    # Validates: Requirements 1.9
    before = await StoreState.of(app, fake_llm)
    client = app.client

    not_json = await client.post("/message:send", content=b"{bad", headers=A2A_HEADERS)
    no_message = await client.post("/message:send", json={}, headers=A2A_HEADERS)
    no_message_stream = await client.post("/message:stream", json={}, headers=A2A_HEADERS)
    no_role = await client.post(
        "/message:send",
        json={"message": {"messageId": "m-1", "role": "BOGUS", "parts": [{"text": "hi"}]}},
        headers=A2A_HEADERS,
    )

    error = assert_a2a_error(
        Binding.REST, not_json, jsonrpc_code=-32700, rest_status=400, reason="INVALID_REQUEST"
    )
    assert error["status"] == "INVALID_ARGUMENT"
    for response, element in (
        (no_message, "message"),
        (no_message_stream, "message"),
        (no_role, "role"),
    ):
        error = assert_a2a_error(
            Binding.REST, response, jsonrpc_code=-32602, rest_status=400, reason="INVALID_PARAMS"
        )
        assert error["status"] == "INVALID_ARGUMENT"
        assert element in json.dumps(error), response.text
    assert await StoreState.of(app, fake_llm) == before


# --- Terminal-state and unknown-Task errors -----------------------------------------------------


@pytest.mark.parametrize("binding", BINDINGS)
@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_message_to_a_terminal_task_is_rejected(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding, stream: bool
) -> None:
    # Validates: Requirements 2.8, 18.29
    finished = await finish_three_tasks(app, fake_llm)
    client = app.client

    for task_id in (finished.completed, finished.failed, finished.canceled):
        before_task = ok_body(binding, await get_task(client, binding, task_id))
        record = await app.deps.session_store.get_task(task_id)
        assert record is not None
        history_before = await app.deps.session_store.get_history(record.context_id)

        reply = await send(
            client,
            binding,
            message_of("one more", context_id=record.context_id, task_id=task_id),
            stream=stream,
        )

        error = assert_a2a_error(
            binding,
            reply,
            jsonrpc_code=UNSUPPORTED_OPERATION,
            rest_status=400,
            reason="UNSUPPORTED_OPERATION",
        )
        assert error["message"], error  # the wording is the SDK's and differs by path
        assert ok_body(binding, await get_task(client, binding, task_id)) == before_task
        assert await app.deps.session_store.get_history(record.context_id) == history_before

    assert len(fake_llm.calls) == finished.llm_calls  # no LLM request for the rejected messages


@pytest.mark.parametrize("binding", BINDINGS)
async def test_cancel_of_a_terminal_task_is_not_cancelable(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 2.12
    finished = await finish_three_tasks(app, fake_llm)

    for state, task_id in (
        (COMPLETED, finished.completed),
        (FAILED, finished.failed),
        (CANCELED, finished.canceled),
    ):
        reply = await cancel_task(app.client, binding, task_id)

        assert_a2a_error(
            binding,
            reply,
            jsonrpc_code=TASK_NOT_CANCELABLE,
            rest_status=400,
            reason="TASK_NOT_CANCELABLE",
        )
        stored = ok_body(binding, await get_task(app.client, binding, task_id))
        assert stored["status"]["state"] == state  # unchanged


@pytest.mark.parametrize("binding", BINDINGS)
async def test_unknown_task_is_not_found_and_creates_nothing(
    app: AppHandle, fake_llm: ScriptedCompletion, binding: Binding
) -> None:
    # Validates: Requirements 2.11
    before = await StoreState.of(app, fake_llm)
    client = app.client
    calls = {
        "get": await get_task(client, binding, "no-such-task"),
        "cancel": await cancel_task(client, binding, "no-such-task"),
        "send": await send(
            client, binding, message_of("hi", context_id="ctx-new", task_id="no-such-task")
        ),
        "stream": await send(
            client,
            binding,
            message_of("hi", context_id="ctx-new", task_id="no-such-task"),
            stream=True,
        ),
        "send-no-context": await send(client, binding, message_of("hi", task_id="no-such-task")),
    }

    for name, reply in calls.items():
        assert_a2a_error(
            binding, reply, jsonrpc_code=TASK_NOT_FOUND, rest_status=404, reason="TASK_NOT_FOUND"
        )
        assert "application/json" in reply.headers["content-type"], name
    assert await StoreState.of(app, fake_llm) == before  # no Task, no Context, no LLM request
    assert (await app.deps.session_store.get_history("ctx-new")) == (None, [])


# --- A real socket ------------------------------------------------------------------------------


@dataclass
class StreamEvent:
    at: float
    body: dict[str, Any]


@dataclass
class LiveStream:
    """A streaming send read as it arrives, with the arrival time of every event.

    Use it as `async with LiveStream.start(...)`. Leaving the block drops the connection.
    """

    binding: Binding
    started: float = 0.0
    status_code: int | None = None
    events: list[StreamEvent] = field(default_factory=list)
    closed_at: float | None = None
    error: BaseException | None = None
    _changed: asyncio.Event = field(default_factory=asyncio.Event)
    _reader: asyncio.Task[None] | None = None

    @classmethod
    def start(cls, client: httpx.AsyncClient, binding: Binding, message: dict[str, Any]) -> Self:
        stream = cls(binding)
        stream.started = time.monotonic()
        stream._reader = asyncio.create_task(stream._read(client, message))
        return stream

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        assert self._reader is not None
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader

    async def _read(self, client: httpx.AsyncClient, message: dict[str, Any]) -> None:
        if self.binding is Binding.JSONRPC:
            url, payload = "/", envelope("SendStreamingMessage", {"message": message})
        else:
            url, payload = "/message:stream", {"message": message}
        try:
            async with client.stream("POST", url, json=payload, headers=A2A_HEADERS) as response:
                self.status_code = response.status_code
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    document = json.loads(line[len("data:") :].strip())
                    if self.binding is Binding.JSONRPC:
                        document = document["result"]
                    self.events.append(StreamEvent(time.monotonic(), document))
                    self._changed.set()
        except Exception as exc:  # reported to the test through `error`
            self.error = exc
        finally:
            self.closed_at = time.monotonic()
            self._changed.set()

    @property
    def closed(self) -> bool:
        return self.closed_at is not None

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [event.body for event in self.events]

    async def wait_until(
        self, condition: Callable[[Self], bool], *, timeout_s: float = WAIT_S
    ) -> None:
        async with asyncio.timeout(timeout_s):
            while not condition(self):
                self._changed.clear()
                if condition(self):
                    break
                await self._changed.wait()

    async def wait_for_text(self, text: str) -> None:
        await self.wait_until(lambda stream: text in streamed_text(stream.bodies))

    async def wait_closed(self) -> None:
        await self.wait_until(lambda stream: stream.closed)
        assert self.error is None, repr(self.error)


@pytest.fixture
def stub_llm(llm_stub: FakeLlmServer, monkeypatch: pytest.MonkeyPatch) -> FakeLlmServer:
    """The HTTP LLM stub, reachable from this process without a proxy."""
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost,::1")
    return llm_stub


@contextlib.asynccontextmanager
async def live_app(
    app_factory: AppOpener, backend: str, stub: FakeLlmServer
) -> AsyncIterator[AppHandle]:
    async with app_factory(
        storage_backend=backend,
        transport="uvicorn",
        overrides={"llm": {"api_base": stub.base_url}},
    ) as handle:
        yield handle


async def test_endpoints_are_reachable_on_the_socket(
    app_factory: AppOpener, storage_backend: str, fake_llm: ScriptedCompletion
) -> None:
    # Validates: Requirements 1.1, 3.7
    fake_llm.script(COMPLETES, COMPLETES, COMPLETES)
    async with app_factory(storage_backend=storage_backend, transport="uvicorn") as live:
        client = live.client
        assert live.base_url.startswith("http://127.0.0.1:")

        card = await client.get("/.well-known/agent-card.json")  # no A2A-Version, no token
        assert card.status_code == 200, card.text
        assert card.json()["capabilities"]["streaming"] is True
        assert card.json()["skills"]

        rpc = sent_task(Binding.JSONRPC, await send(client, Binding.JSONRPC, message_of("one")))
        rest = sent_task(Binding.REST, await send(client, Binding.REST, message_of("two")))
        assert rpc["status"]["state"] == rest["status"]["state"] == COMPLETED
        sse = await send(client, Binding.JSONRPC, message_of("three"), stream=True)
        assert sse.headers["content-type"].startswith("text/event-stream")
        assert_stream_shape(sse_events(Binding.JSONRPC, sse.content), COMPLETED)
        assert (await client.get("/health")).status_code == 200


@pytest.mark.parametrize("binding", BINDINGS)
async def test_live_stream_timing(
    app_factory: AppOpener, storage_backend: str, stub_llm: FakeLlmServer, binding: Binding
) -> None:
    # Validates: Requirements 3.1, 3.5, 2.10, 18.29
    gate = threading.Event()
    stub_llm.script(StubTurn(chunks=("par", "tial"), hold_after=1, gate=gate))
    try:
        async with (
            live_app(app_factory, storage_backend, stub_llm) as live,
            LiveStream.start(live.client, binding, message_of("stream it")) as stream,
        ):
            # The first event comes within 1 s while the LLM has said nothing yet.
            await stream.wait_until(lambda s: bool(s.events))
            first = stream.events[0]
            assert first.at - stream.started < FIRST_EVENT_LIMIT_S
            assert event_kind(first.body) == "task"
            assert first.body["task"]["status"]["state"] == SUBMITTED
            task_id = str(first.body["task"]["id"])

            # The LLM sent one chunk and waits: the Task is working, and the stream stays open.
            await stream.wait_for_text("par")
            assert not stream.closed
            midway = await wait_for_state(live.client, task_id, [WORKING], artifact_contains="par")
            assert artifact_text(midway) == "par"

            gate.set()
            await stream.wait_closed()
            assert stream.status_code == 200
            events = stream.bodies
            assert assert_stream_shape(events, COMPLETED) == task_id
            assert streamed_text(events) == "partial"
            final_at, closed_at = stream.events[-1].at, stream.closed_at
            assert closed_at is not None
            assert closed_at - final_at < CLOSE_LIMIT_S  # closes within 1 s of the final status

            done = ok_body(binding, await get_task(live.client, binding, task_id))
            assert done["status"]["state"] == COMPLETED
            assert artifact_text(done) == "partial"
    finally:
        gate.set()


@pytest.mark.parametrize("binding", BINDINGS)
async def test_cancel_ends_a_live_stream(
    app_factory: AppOpener, storage_backend: str, stub_llm: FakeLlmServer, binding: Binding
) -> None:
    # Validates: Requirements 2.7, 2.12, 3.9, 18.29
    gate = threading.Event()
    stub_llm.script(StubTurn(chunks=("par", "tial"), hold_after=1, gate=gate))
    try:
        async with (
            live_app(app_factory, storage_backend, stub_llm) as live,
            LiveStream.start(live.client, binding, message_of("stream it")) as stream,
        ):
            await stream.wait_for_text("par")
            task_id = str(stream.events[0].body["task"]["id"])

            started = time.monotonic()
            reply = await cancel_task(live.client, binding, task_id)
            assert ok_body(binding, reply)["status"]["state"] == CANCELED
            await stream.wait_closed()
            assert time.monotonic() - started < CANCEL_LIMIT_S

            events = stream.bodies
            assert event_state(events[-1]) == CANCELED  # the final status is `canceled` ...
            canceled = [i for i, event in enumerate(events) if event_state(event) == CANCELED]
            assert canceled == [len(events) - 1]  # ... and nothing follows it (3.9)
            assert streamed_text(events) == "par"

            gate.set()  # the LLM would now go on; the canceled Task must not
            await asyncio.sleep(0.3)
            assert len(stream.events) == len(events)
            stored = ok_body(binding, await get_task(live.client, binding, task_id))
            assert stored["status"]["state"] == CANCELED
            assert artifact_text(stored) == "par"
            again = await cancel_task(live.client, binding, task_id)
            assert_a2a_error(
                binding,
                again,
                jsonrpc_code=TASK_NOT_CANCELABLE,
                rest_status=400,
                reason="TASK_NOT_CANCELABLE",
            )
    finally:
        gate.set()
