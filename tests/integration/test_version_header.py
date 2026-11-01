"""The A2A-Version header against the real app and the real SDK (task 22.19).

`test_version_header` runs every A2A operation (send, streaming send, get-task, list-tasks, cancel,
subscribe) over both bindings (JSON-RPC and REST) with each header case:

* `1.0` and `1.0.0` are accepted: the request gets past the gate and does what the operation does.
* `1.1`, `1.0.1`, `2.0`, `0.3`, `abc`, and a present but empty header are rejected with
  `VersionNotSupportedError`: JSON-RPC `-32009`, REST 400 `VERSION_NOT_SUPPORTED`. The request ID is
  echoed, the stored data is byte-for-byte what it was, and the LLM is not called (1.11, 1.12).
* A **missing** header is answered by the version negotiation of the pinned `a2a-sdk` with no
  Backend override (1.13). `MISSING_HEADER_BEHAVIOUR` records what the pinned SDK does (observed on
  1.2.x: the header is read as `0.3`, so every operation gets `-32009`). It is the expected value,
  so an SDK upgrade that changes the negotiation fails the test and forces a decision.

The in-process runs use `open_app` over `httpx.ASGITransport` and over a real uvicorn socket, on
both Session_Store kinds. `test_version_header_real_process` repeats the core cases against a real
`python -m local_agent` child. The property test `tests/property/test_version_gate.py` covers the
gate with generated headers; the wire shapes of the other errors are in `test_a2a_contract.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import sqlite3
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import pytest

from fakes import LlmTurn, text_delta
from integration_harness import (
    A2A_HEADERS,
    DATABASE_NAME,
    AppHandle,
    BackendLauncher,
    FakeLlmServer,
    ScriptedCompletion,
    message_params,
)
from local_agent.session.store import TaskQuery

pytestmark = pytest.mark.integration

type AppOpener = Callable[..., contextlib.AbstractAsyncContextManager[AppHandle]]
type Transport = Literal["asgi", "uvicorn"]
type Header = str | None
"""An `A2A-Version` value, or `None` for a request that omits the header."""

VERSION_NOT_SUPPORTED_CODE: Final = -32009
VERSION_NOT_SUPPORTED: Final = "VERSION_NOT_SUPPORTED"
TASK_NOT_CANCELABLE_CODE: Final = -32002
UNSUPPORTED_OPERATION_CODE: Final = -32004

COMPLETED: Final = "TASK_STATE_COMPLETED"
SUBMITTED: Final = "TASK_STATE_SUBMITTED"
SEED_CONTEXT: Final = "ctx-seed"
HEADER_NAME: Final = "A2A-Version"

ACCEPTED_HEADERS: Final = ("1.0", "1.0.0")
REJECTED_HEADERS: Final = ("1.1", "1.0.1", "2.0", "0.3", "abc", "")


@dataclass(frozen=True)
class MissingHeaderBehaviour:
    """What a request without `A2A-Version` gets from the pinned SDK (1.13)."""

    jsonrpc_code: int
    rest_status: int
    reason: str
    message_mentions: str


MISSING_HEADER_BEHAVIOUR: Final = MissingHeaderBehaviour(
    jsonrpc_code=VERSION_NOT_SUPPORTED_CODE,
    rest_status=400,
    reason=VERSION_NOT_SUPPORTED,
    message_mentions="'0.3'",  # the SDK reads a missing header as 0.3 and says so
)


class Binding(StrEnum):
    JSONRPC = "jsonrpc"
    REST = "rest"


class Op(StrEnum):
    SEND = "SendMessage"
    STREAM = "SendStreamingMessage"
    GET = "GetTask"
    LIST = "ListTasks"
    CANCEL = "CancelTask"
    SUBSCRIBE = "SubscribeToTask"


BINDINGS: Final = (Binding.JSONRPC, Binding.REST)
OPS: Final = tuple(Op)

_numbers = itertools.count(1)


# --- Requests and replies -----------------------------------------------------------------------


def _message(text: str, context_id: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = message_params(text, context_id=context_id)["message"]
    return message


def _request(op: Op, binding: Binding, task_id: str, request_id: int) -> dict[str, Any]:
    """The `httpx` arguments (method, url, json) of one operation on one binding."""
    new_message = {"message": _message(f"hello {request_id}")}
    by_id = {"id": task_id}
    if binding is Binding.JSONRPC:
        by_op: dict[Op, dict[str, Any]] = {
            Op.SEND: new_message,
            Op.STREAM: new_message,
            Op.GET: by_id,
            Op.LIST: {},
            Op.CANCEL: by_id,
            Op.SUBSCRIBE: by_id,
        }
        envelope = {"jsonrpc": "2.0", "id": request_id, "method": op.value, "params": by_op[op]}
        return {"method": "POST", "url": "/", "json": envelope}
    rest: dict[Op, tuple[str, str, dict[str, Any] | None]] = {
        Op.SEND: ("POST", "/message:send", new_message),
        Op.STREAM: ("POST", "/message:stream", new_message),
        Op.GET: ("GET", f"/tasks/{task_id}", None),
        Op.LIST: ("GET", "/tasks", None),
        Op.CANCEL: ("POST", f"/tasks/{task_id}:cancel", {}),
        Op.SUBSCRIBE: ("GET", f"/tasks/{task_id}:subscribe", None),
    }
    method, url, body = rest[op]
    return {"method": method, "url": url, "json": body}


@dataclass(frozen=True)
class Reply:
    """A response: the HTTP status, and one JSON document per body (or per SSE event)."""

    status: int
    streamed: bool
    documents: tuple[dict[str, Any], ...]
    text: str


def _read(response: httpx.Response) -> Reply:
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        lines = [line for line in response.text.splitlines() if line.startswith("data:")]
        documents = tuple(json.loads(line[len("data:") :].strip()) for line in lines)
        return Reply(response.status_code, True, documents, response.text)
    return Reply(response.status_code, False, (response.json(),), response.text)


@dataclass(frozen=True)
class Call:
    op: Op
    binding: Binding
    header: Header
    request_id: int
    reply: Reply

    @property
    def where(self) -> str:
        return (
            f"{self.op.value} on {self.binding.value} with {HEADER_NAME}={self.header!r}: "
            f"{self.reply.status} {self.reply.text[:400]}"
        )


async def call(
    client: httpx.AsyncClient, op: Op, binding: Binding, header: Header, task_id: str
) -> Call:
    """One operation. `header=None` omits the header. A stream is read to its end."""
    request_id = next(_numbers)
    headers = {} if header is None else {HEADER_NAME: header}
    response = await client.request(**_request(op, binding, task_id, request_id), headers=headers)
    return Call(op, binding, header, request_id, _read(response))


# --- Assertions ---------------------------------------------------------------------------------


def _reason(error: Mapping[str, Any]) -> str | None:
    """The `reason` of an error's `ErrorInfo` detail (`data` on JSON-RPC, `details` on REST)."""
    for key in ("data", "details"):
        details = error.get(key)
        if isinstance(details, list):
            for detail in details:
                if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("ErrorInfo"):
                    return str(detail["reason"])
    return None


def _error(done: Call) -> dict[str, Any]:
    """The `error` object of the first (and for a rejected call only) document."""
    document = done.reply.documents[0]
    assert "error" in document, done.where
    error: dict[str, Any] = document["error"]
    return error


def assert_rejected(done: Call, *, message_mentions: str | None = None) -> None:
    """The call got `VersionNotSupportedError` and nothing else: no result, no later event."""
    assert len(done.reply.documents) == 1, done.where
    error = _error(done)
    assert _reason(error) == VERSION_NOT_SUPPORTED, done.where
    if done.binding is Binding.JSONRPC:
        document = done.reply.documents[0]
        assert document["jsonrpc"] == "2.0", done.where
        assert document["id"] == done.request_id, done.where
        assert "result" not in document, done.where
        assert error["code"] == VERSION_NOT_SUPPORTED_CODE, done.where
        if not done.reply.streamed:
            assert done.reply.status == 200, done.where
    else:
        assert done.reply.status == 400, done.where
        assert error["code"] == 400, done.where
    if message_mentions:
        assert message_mentions in error["message"], done.where


def _unwrap(done: Call, document: Mapping[str, Any]) -> dict[str, Any]:
    """The result of a successful call: the JSON-RPC `result`, or the REST body itself."""
    assert "error" not in document, done.where
    if done.binding is Binding.JSONRPC:
        assert document["jsonrpc"] == "2.0", done.where
        result: dict[str, Any] = document["result"]
        return result
    return dict(document)


def _assert_a2a_error(done: Call, jsonrpc_code: int, reason: str) -> None:
    error = _error(done)
    assert _reason(error) == reason, done.where
    if done.binding is Binding.JSONRPC:
        assert error["code"] == jsonrpc_code, done.where
    else:
        assert done.reply.status == 400, done.where


def assert_accepted(done: Call, seed_task_id: str) -> None:
    """The call passed the gate and did what its operation does on a finished seed Task."""
    reply = done.reply
    if done.op is Op.SEND:
        assert _unwrap(done, reply.documents[0])["task"]["status"]["state"] == COMPLETED, done.where
    elif done.op is Op.STREAM:
        events = [_unwrap(done, document) for document in reply.documents]
        assert events[0]["task"]["status"]["state"] == SUBMITTED, done.where
        assert events[-1]["statusUpdate"]["status"]["state"] == COMPLETED, done.where
    elif done.op is Op.GET:
        assert _unwrap(done, reply.documents[0])["id"] == seed_task_id, done.where
    elif done.op is Op.LIST:
        tasks = _unwrap(done, reply.documents[0])["tasks"]
        assert seed_task_id in [task["id"] for task in tasks], done.where
    elif done.op is Op.CANCEL:
        _assert_a2a_error(done, TASK_NOT_CANCELABLE_CODE, "TASK_NOT_CANCELABLE")
    else:
        _assert_a2a_error(done, UNSUPPORTED_OPERATION_CODE, "UNSUPPORTED_OPERATION")


# --- The app under test, in process -------------------------------------------------------------


COMPLETES: Final = LlmTurn(deltas=(text_delta("all "), text_delta("done")))


@dataclass
class Served:
    """The app over a real Session_Store, a client for it, and a finished seed Task."""

    app: AppHandle
    llm: ScriptedCompletion
    seed_task_id: str

    @property
    def client(self) -> httpx.AsyncClient:
        return self.app.client

    async def fingerprint(self) -> tuple[Any, ...]:
        """Everything a rejected request must leave alone, and the count of LLM calls."""
        store = self.app.deps.session_store
        page = await store.list_tasks(TaskQuery(page_size=1000))
        history = await store.get_history(SEED_CONTEXT)
        return (page.tasks, history, self._sqlite_dump(), len(self.llm.calls))

    def _sqlite_dump(self) -> list[str] | None:
        if self.app.config.storage.backend != "sqlite":
            return None
        path = Path(self.app.storage_dir) / DATABASE_NAME
        with contextlib.closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as database:
            return list(database.iterdump())


@pytest.fixture(params=["asgi", "uvicorn"])
def transport(request: pytest.FixtureRequest) -> Transport:
    value: Transport = request.param
    return value


@pytest.fixture
async def served(
    app_factory: AppOpener,
    storage_backend: Literal["memory", "sqlite"],
    transport: Transport,
    fake_llm: ScriptedCompletion,
) -> AsyncIterator[Served]:
    """The app on each Session_Store kind and transport, with one finished Task already stored."""
    fake_llm.script(COMPLETES)
    async with app_factory(storage_backend=storage_backend, transport=transport) as handle:
        seed = await handle.client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "seed",
                "method": "SendMessage",
                "params": {"message": _message("seed", SEED_CONTEXT)},
            },
            headers=A2A_HEADERS,
        )
        task = seed.json()["result"]["task"]
        assert task["status"]["state"] == COMPLETED, seed.text
        yield Served(handle, fake_llm, task["id"])


# --- test_version_header ------------------------------------------------------------------------


@pytest.mark.parametrize("header", ACCEPTED_HEADERS)
async def test_version_header_accepts_1_0(served: Served, header: str) -> None:
    # Validates: Requirements 1.11
    for binding in BINDINGS:
        for op in OPS:
            done = await call(served.client, op, binding, header, served.seed_task_id)

            assert_accepted(done, served.seed_task_id)


@pytest.mark.parametrize("header", REJECTED_HEADERS)
async def test_version_header_rejects_other_versions(served: Served, header: str) -> None:
    # Validates: Requirements 1.12
    before = await served.fingerprint()

    for binding in BINDINGS:
        for op in OPS:
            done = await call(served.client, op, binding, header, served.seed_task_id)

            assert_rejected(done, message_mentions=f"'{header}'" if header else None)

    assert await served.fingerprint() == before  # no Task created, changed, or cancelled


async def test_version_header_missing_matches_the_pinned_sdk(served: Served) -> None:
    # Validates: Requirements 1.13
    expected = MISSING_HEADER_BEHAVIOUR
    before = await served.fingerprint()

    for binding in BINDINGS:
        for op in OPS:
            done = await call(served.client, op, binding, None, served.seed_task_id)

            assert_rejected(done, message_mentions=expected.message_mentions)
            error = _error(done)
            assert _reason(error) == expected.reason, done.where
            code = expected.jsonrpc_code if binding is Binding.JSONRPC else expected.rest_status
            assert error["code"] == code, done.where

    assert await served.fingerprint() == before


async def test_version_header_gate_applies_to_each_request(served: Served) -> None:
    # Validates: Requirements 1.11, 1.12 (a rejected request does not poison the next one)
    for header in ("2.0", "1.0", None, "1.0.0", "1.1", "1.0"):
        for binding in BINDINGS:
            done = await call(served.client, Op.GET, binding, header, served.seed_task_id)

            if header in ACCEPTED_HEADERS:
                assert_accepted(done, served.seed_task_id)
            else:
                assert_rejected(done)


# --- A real `python -m local_agent` process -----------------------------------------------------


async def _list_tasks(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    response = await client.post(
        "/",
        json={"jsonrpc": "2.0", "id": "list", "method": "ListTasks", "params": {}},
        headers=A2A_HEADERS,
    )
    tasks: list[dict[str, Any]] = response.json()["result"]["tasks"]
    return tasks


async def test_version_header_real_process(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer
) -> None:
    # Validates: Requirements 1.11, 1.12, 1.13
    backend = await asyncio.to_thread(backend_launcher.launch, llm_base=llm_stub.base_url)
    async with httpx.AsyncClient(base_url=backend.url, timeout=30.0) as client:
        seed = await client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "seed",
                "method": "SendMessage",
                "params": {"message": _message("seed", SEED_CONTEXT)},
            },
            headers=A2A_HEADERS,
        )
        task = seed.json()["result"]["task"]
        assert task["status"]["state"] == COMPLETED, seed.text
        seed_id: str = task["id"]
        stored = await _list_tasks(client)
        llm_requests = len(llm_stub.requests)

        for header in (*REJECTED_HEADERS, None):
            for binding in BINDINGS:
                for op in OPS:
                    done = await call(client, op, binding, header, seed_id)

                    assert_rejected(done)

        assert await _list_tasks(client) == stored
        assert len(llm_stub.requests) == llm_requests

        for header in ACCEPTED_HEADERS:
            for binding in BINDINGS:
                for op in OPS:
                    done = await call(client, op, binding, header, seed_id)

                    assert_accepted(done, seed_id)

    assert backend.running, backend.stderr_text()
