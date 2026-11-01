"""Concurrency and cancellation integration tests (task 22.6).

The production app runs on a real uvicorn socket in the test's event loop (so SSE events reach the
client as they are sent), over both Session_Stores, with the real echo MCP server
(`tests/fixtures/mcp_echo_server.py`) and a scripted LLM that answers by the text of the Task's
user message. Every test runs with asyncio debug mode and `slow_callback_duration = 0.1`:

* `test_ten_concurrent_tasks_first_status_within_500ms`: 10 Tasks in 10 Contexts, 5 of them started
  while the first 5 wait on a slow tool. Each Task's first event and first Status_Event arrive in
  500 ms, and the Tasks run side by side (2.7, 14.1, 14.2, 14.3).
* `test_concurrent_tool_turn_under_slowest_plus_500ms`: six Tool_Calls of one turn take no more
  than the slowest one plus 500 ms, and their Tool_Results keep the Tool_Call order (8.3, 8.5).
* `test_cancel_during_slow_tool_within_2s`, `test_cancel_during_input_required_within_2s`, and
  `test_cancel_during_stalled_llm_stream_within_2s`: cancel ends the Task as `canceled` in 2 s,
  nothing is emitted afterwards, and the Message_History keeps one Tool_Result per Tool_Call
  (2.7, 14.4, 14.5, 14.6).
* `test_failure_in_one_context_leaves_others_intact`: a rejected LLM request, a failing tool, a
  timed-out tool, and a stalled LLM stream in four Contexts do not change the Tasks of the other
  Contexts (14.7).
* `test_ten_streaming_tasks_on_sqlite_log_no_slow_callbacks`: the 10-Task case over the SQLite
  Session_Store logs no slow-callback warning, and the first status still arrives in 500 ms (14.1,
  14.2, 18.14).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Self, cast

import httpx
import litellm
import pytest

from fakes import FakeRejection, text_delta, tool_call_deltas
from integration_harness import (
    A2A_HEADERS,
    PIDS,
    AppHandle,
    echo_server_entry,
    llm_tool_name,
    message_params,
    read_request_log,
    wait_until_gone,
)
from local_agent.a2a_server.events import TaskState, from_sdk
from local_agent.models import AssistantMessage, ChatMessage, ToolMessage

pytestmark = pytest.mark.integration

type StorageBackend = Literal["memory", "sqlite"]
type AppOpener = Callable[..., contextlib.AbstractAsyncContextManager[AppHandle]]

WORKING: Final = "TASK_STATE_WORKING"
INPUT_REQUIRED: Final = "TASK_STATE_INPUT_REQUIRED"
COMPLETED: Final = "TASK_STATE_COMPLETED"
FAILED: Final = "TASK_STATE_FAILED"
CANCELED: Final = "TASK_STATE_CANCELED"

KIND_TOOL_STARTED: Final = "tool_call_started"
KIND_APPROVAL_REQUIRED: Final = "tool_approval_required"

FIRST_STATUS_S: Final = 0.5  # 14.2
CANCEL_S: Final = 2.0  # 2.7, 14.4
TURN_SLACK_S: Final = 0.5  # 8.3
SLOW_CALLBACK_S: Final = 0.1  # 14.1
STREAM_TIMEOUT_S: Final = 30.0
TASKS: Final = 10

_request_numbers = itertools.count(1)


# --- Event loop watch ---------------------------------------------------------------------------


class SlowCallbackLog(logging.Handler):
    """Collects the "Executing ... took N seconds" warnings of asyncio debug mode."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Executing") and " took " in message:
            self.messages.append(message)


@pytest.fixture(autouse=True)
async def slow_callbacks() -> AsyncIterator[SlowCallbackLog]:
    """Run the test with asyncio debug mode and `slow_callback_duration = 0.1` (14.1)."""
    loop = cast("Any", asyncio.get_running_loop())
    previous = (loop.get_debug(), loop.slow_callback_duration)
    loop.set_debug(True)
    loop.slow_callback_duration = SLOW_CALLBACK_S
    watch = SlowCallbackLog()
    logger = logging.getLogger("asyncio")
    logger.addHandler(watch)
    try:
        yield watch
    finally:
        logger.removeHandler(watch)
        loop.set_debug(previous[0])
        loop.slow_callback_duration = previous[1]


# --- The scripted LLM ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One Tool_Call of a scripted reply: a tool of the echo server and its arguments."""

    call_id: str
    tool: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Reply:
    """One scripted LLM response: text chunks, then Tool_Calls, then (optionally) a stall."""

    chunks: tuple[str, ...] = ()
    calls: tuple[Call, ...] = ()
    chunk_delay_s: float = 0.0
    stall: bool = False
    reject: str | None = None


class RoutedLlm:
    """A stand-in for `litellm.acompletion` that answers by the Task's user message.

    The prompt is the text of the last user message. `plans[prompt][n]` answers the n-th LLM
    request of that user turn (n counts the assistant messages after the prompt), and the last
    reply repeats. Because Tasks are matched by their text and not by arrival order, 10 Tasks
    can interleave in any way. `request_times` and `stream_end_times` hold `time.monotonic()` of
    each request and of the end of each fully read response.
    """

    def __init__(self) -> None:
        self.plans: dict[str, tuple[Reply, ...]] = {}
        self.request_times: list[float] = []
        self.stream_end_times: list[float] = []
        self.requests: list[list[dict[str, Any]]] = []

    async def __call__(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        messages: list[dict[str, Any]] = copy.deepcopy(list(kwargs.get("messages") or []))
        self.requests.append(messages)
        last_user = max(i for i, message in enumerate(messages) if message.get("role") == "user")
        prompt = str(messages[last_user]["content"])
        stage = sum(1 for message in messages[last_user:] if message.get("role") == "assistant")
        plan = self.plans[prompt]
        reply = plan[min(stage, len(plan) - 1)]
        self.request_times.append(time.monotonic())
        if reply.reject is not None:
            raise FakeRejection(reply.reject)
        return self._stream(reply)

    async def _stream(self, reply: Reply) -> AsyncIterator[dict[str, Any]]:
        for text in reply.chunks:
            await asyncio.sleep(reply.chunk_delay_s)
            yield {"choices": [{"delta": text_delta(text)}]}
        for index, call in enumerate(reply.calls):
            deltas = tool_call_deltas(
                index,
                llm_tool_name("echo", call.tool),
                json.dumps(call.arguments),
                call_id=call.call_id,
            )
            for delta in deltas:
                yield {"choices": [{"delta": delta}]}
        if reply.stall:
            await asyncio.get_running_loop().create_future()
        self.stream_end_times.append(time.monotonic())


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> RoutedLlm:
    routed = RoutedLlm()
    monkeypatch.setattr(litellm, "acompletion", routed)
    return routed


# --- The client side of a Task ------------------------------------------------------------------


@dataclass(frozen=True)
class Seen:
    """One SSE event of a SendStreamingMessage response, and when the client read it."""

    at: float
    payload: dict[str, Any]

    @property
    def kind(self) -> str:
        """`task`, `statusUpdate`, `artifactUpdate`, or `message`."""
        return next(iter(self.payload))

    @property
    def state(self) -> str | None:
        if self.kind == "task":
            return str(self.payload["task"]["status"]["state"])
        if self.kind == "statusUpdate":
            return str(self.payload["statusUpdate"]["status"]["state"])
        return None

    @property
    def tool_kind(self) -> str | None:
        """The machine-readable `kind` of a tool Status_Event (4.4), or `None`."""
        if self.kind != "statusUpdate":
            return None
        update = self.payload["statusUpdate"]
        message = update.get("status", {}).get("message") or {}
        for holder in (update.get("metadata"), message.get("metadata")):
            if isinstance(holder, dict) and "kind" in holder:
                return str(holder["kind"])
        return None


class TaskStream:
    """One `SendStreamingMessage` call, read in the background.

    `sent_at` is taken when `start` is called, so a latency is the time the Chat_App waits.
    """

    def __init__(self, client: httpx.AsyncClient, prompt: str, *, context_id: str | None = None):
        self.prompt = prompt
        self._client = client
        self._params = message_params(prompt, context_id=context_id)
        self.events: list[Seen] = []
        self.closed = False
        self.error: BaseException | None = None
        self.sent_at = 0.0
        self._changed = asyncio.Condition()
        self._pump_task: asyncio.Task[None] | None = None

    def start(self) -> Self:
        self.sent_at = time.monotonic()
        self._pump_task = asyncio.create_task(self._pump(), name=f"stream:{self.prompt}")
        return self

    async def aclose(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task

    async def _pump(self) -> None:
        envelope = {
            "jsonrpc": "2.0",
            "id": next(_request_numbers),
            "method": "SendStreamingMessage",
            "params": self._params,
        }
        try:
            async with self._client.stream(
                "POST", "/", json=envelope, headers=A2A_HEADERS
            ) as response:
                assert response.status_code == 200, await response.aread()
                data: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[len("data:") :].removeprefix(" "))
                    elif not line and data:
                        await self._add("\n".join(data))
                        data.clear()
                if data:
                    await self._add("\n".join(data))
        except Exception as exc:  # reported to the test by `wait_for`
            self.error = exc
        finally:
            async with self._changed:
                self.closed = True
                self._changed.notify_all()

    async def _add(self, text: str) -> None:
        document = json.loads(text)
        assert "error" not in document, text
        async with self._changed:
            self.events.append(Seen(time.monotonic(), document["result"]))
            self._changed.notify_all()

    async def wait_for(
        self, predicate: Callable[[Self], bool], what: str, *, timeout_s: float = STREAM_TIMEOUT_S
    ) -> None:
        """Wait until `predicate(self)` holds. Fails if the stream ends or time runs out first."""
        try:
            async with asyncio.timeout(timeout_s), self._changed:
                await self._changed.wait_for(lambda: predicate(self) or self.closed)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}; saw {self.summary()}") from None
        if self.error is not None:
            raise self.error
        assert predicate(self), f"the stream ended before {what}; saw {self.summary()}"

    async def wait_closed(self, what: str, *, timeout_s: float = STREAM_TIMEOUT_S) -> None:
        await self.wait_for(lambda stream: stream.closed, what, timeout_s=timeout_s)

    def summary(self) -> list[str]:
        return [
            f"{seen.kind}:{seen.state or ''}:{seen.tool_kind or ''}".rstrip(":")
            for seen in self.events
        ]

    def has_state(self, state: str) -> bool:
        return any(seen.state == state for seen in self.events)

    def has_tool_kind(self, kind: str) -> bool:
        return any(seen.tool_kind == kind for seen in self.events)

    def has_kind(self, kind: str) -> bool:
        return any(seen.kind == kind for seen in self.events)

    @property
    def last_state(self) -> str | None:
        states = [seen.state for seen in self.events if seen.state is not None]
        return states[-1] if states else None

    @property
    def task_id(self) -> str:
        return str(self.snapshot["id"])

    @property
    def context_id(self) -> str:
        return str(self.snapshot["contextId"])

    @property
    def snapshot(self) -> dict[str, Any]:
        for seen in self.events:
            if seen.kind == "task":
                task: dict[str, Any] = seen.payload["task"]
                return task
        raise AssertionError("the stream carried no Task_Snapshot yet")

    @property
    def first_event_s(self) -> float:
        return self.events[0].at - self.sent_at

    @property
    def first_status_s(self) -> float:
        first = next(seen for seen in self.events if seen.kind == "statusUpdate")
        return first.at - self.sent_at


async def rpc(client: httpx.AsyncClient, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    envelope = {"jsonrpc": "2.0", "id": next(_request_numbers), "method": method, "params": params}
    response = await client.post("/", json=envelope, headers=A2A_HEADERS)
    assert response.status_code == 200, response.text
    document: dict[str, Any] = response.json()
    assert "error" not in document, document
    return document


# --- The app under test -------------------------------------------------------------------------


@dataclass
class Rig:
    """The app on a uvicorn socket with the echo MCP server, and the Task streams of a test."""

    app: AppHandle
    llm: RoutedLlm
    log: Path
    streams: list[TaskStream] = field(default_factory=list)

    @property
    def client(self) -> httpx.AsyncClient:
        return self.app.client

    def stream(self, prompt: str, *, context_id: str | None = None) -> TaskStream:
        stream = TaskStream(self.client, prompt, context_id=context_id).start()
        self.streams.append(stream)
        return stream

    async def history(self, context_id: str) -> list[ChatMessage]:
        _, history = await self.app.deps.session_store.get_history(context_id)
        return history

    async def state_of(self, task_id: str) -> TaskState:
        record = await self.app.deps.session_store.get_task(task_id)
        assert record is not None
        return from_sdk(record.state)

    async def artifact_text(self, task_id: str) -> str:
        record = await self.app.deps.session_store.get_task(task_id)
        assert record is not None
        return "".join(artifact.text for artifact in record.artifacts)

    async def cancel(self, task_id: str) -> tuple[float, dict[str, Any]]:
        """Send CancelTask. Returns the seconds it took and the Task it answered with."""
        started = time.monotonic()
        document = await rpc(self.client, "CancelTask", {"id": task_id})
        return time.monotonic() - started, document["result"]

    def tool_calls_seen_by_server(self) -> list[str]:
        return [
            str(entry["name"])
            for entry in read_request_log(self.log)
            if entry["event"] == "call_tool"
        ]

    async def wait_for_server_call(self, tool: str, *, timeout_s: float = 10.0) -> None:
        """Wait until the echo server logged a `tools/call` for `tool` (the call is in flight)."""
        async with asyncio.timeout(timeout_s):
            while tool not in self.tool_calls_seen_by_server():  # noqa: ASYNC110 (no event)
                await asyncio.sleep(0.02)


@contextlib.asynccontextmanager
async def open_rig(
    app_factory: AppOpener,
    llm: RoutedLlm,
    log: Path,
    *,
    backend: StorageBackend,
    permissions: Mapping[str, list[str]] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AsyncIterator[Rig]:
    async with app_factory(
        storage_backend=backend,
        transport="uvicorn",
        mcp_servers=[echo_server_entry(request_log=log)],
        permissions=permissions,
        overrides=overrides,
    ) as app:
        health = await app.client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json().get("status") == "ok", health.text
        rig = Rig(app, llm, log)
        try:
            yield rig
        finally:
            for stream in rig.streams:
                await stream.aclose()
    pids = PIDS.record_from_log(log)
    assert await asyncio.to_thread(wait_until_gone, pids, 5.0) == []


@pytest.fixture
def echo_log(tmp_path: Path, storage_dir: Path) -> Path:
    """The echo server's request log.

    `open_app` anchors the MCP_Server's working directory at the Storage_Directory, which the
    memory backend never creates, so the directory is made here before the server is spawned.
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path / "echo-requests.jsonl"


# --- Plans --------------------------------------------------------------------------------------


def slow_tool_plan(index: int, *, seconds: float) -> tuple[Reply, ...]:
    """Say something, call `slow` once, then stream 20 text chunks as the final answer."""
    return (
        Reply(
            chunks=("checking ",),
            calls=(Call(f"call-{index}", "slow", {"seconds": seconds, "text": f"result-{index}"}),),
        ),
        Reply(chunks=final_chunks(index), chunk_delay_s=0.005),
    )


def final_chunks(index: int) -> tuple[str, ...]:
    return tuple(f"answer-{index}-part-{part} " for part in range(20))


def assert_history_integrity(history: list[ChatMessage]) -> None:
    """Every Tool_Call of the history has exactly one Tool_Result (9.10, 14.5)."""
    call_ids = [
        call.id
        for message in history
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    ]
    result_ids = [message.tool_call_id for message in history if isinstance(message, ToolMessage)]
    assert sorted(call_ids) == sorted(result_ids), history


def tool_results(history: list[ChatMessage]) -> list[str]:
    return [message.content for message in history if isinstance(message, ToolMessage)]


# --- Ten concurrent Tasks -----------------------------------------------------------------------


async def run_ten_tasks(rig: Rig) -> float:
    """Run 10 Tasks with a slow tool and streamed tokens; return the seconds it took.

    The first 5 start together. The second 5 start while the first 5 wait on their slow tool.
    """
    for index in range(TASKS):
        rig.llm.plans[f"ten-{index}"] = slow_tool_plan(index, seconds=1.0)
    started = time.monotonic()

    first = [rig.stream(f"ten-{index}") for index in range(TASKS // 2)]
    for stream in first:
        await stream.wait_for(
            lambda s: s.has_tool_kind(KIND_TOOL_STARTED), f"{s_name(stream)} to start its tool"
        )
    assert not any(stream.has_state(COMPLETED) for stream in first), "the tools ended too early"

    second = [rig.stream(f"ten-{index}") for index in range(TASKS // 2, TASKS)]
    for stream in [*first, *second]:
        await stream.wait_closed(f"{s_name(stream)} to end")
    return time.monotonic() - started


def s_name(stream: TaskStream) -> str:
    return f"Task {stream.prompt!r}"


async def check_ten_tasks(rig: Rig, elapsed_s: float) -> None:
    streams = rig.streams
    assert len(streams) == TASKS
    for stream in streams:
        assert stream.first_event_s < FIRST_STATUS_S, f"{s_name(stream)}: first event"  # 3.1
        assert stream.first_status_s < FIRST_STATUS_S, f"{s_name(stream)}: first status"  # 14.2
        assert stream.last_state == COMPLETED, (s_name(stream), stream.summary())
    assert len({stream.context_id for stream in streams}) == TASKS
    # 10 Tasks with a 1 s tool each would take 10 s one after another.
    assert elapsed_s < 6.0, f"the Tasks did not run side by side: {elapsed_s:.1f} s"
    for index, stream in enumerate(streams):
        text = await rig.artifact_text(stream.task_id)
        assert "".join(final_chunks(index)) in text
        history = await rig.history(stream.context_id)
        assert tool_results(history) == [f"result-{index}"]  # no result of another Context
        assert_history_integrity(history)


async def test_ten_concurrent_tasks_first_status_within_500ms(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path
) -> None:
    # Feature: local-agent-backend, concurrency; Validates: Requirements 2.7, 14.1, 14.2, 14.3
    async with open_rig(app_factory, llm, echo_log, backend="memory") as rig:
        elapsed_s = await run_ten_tasks(rig)
        await check_ten_tasks(rig, elapsed_s)
        # One echo server handled all 10 calls at once (14.3): it saw all of them.
        assert rig.tool_calls_seen_by_server().count("slow") == TASKS


async def test_ten_streaming_tasks_on_sqlite_log_no_slow_callbacks(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path, slow_callbacks: SlowCallbackLog
) -> None:
    # Feature: local-agent-backend, event loop health; Validates: Requirements 14.1, 14.2, 18.14
    async with open_rig(app_factory, llm, echo_log, backend="sqlite") as rig:
        slow_callbacks.messages.clear()  # the app is up; what follows is the load under test
        elapsed_s = await run_ten_tasks(rig)
        await check_ten_tasks(rig, elapsed_s)
        assert slow_callbacks.messages == []


# --- Concurrent tool turn -----------------------------------------------------------------------


async def test_concurrent_tool_turn_under_slowest_plus_500ms(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path, storage_backend: StorageBackend
) -> None:
    # Feature: local-agent-backend, concurrent tools; Validates: Requirements 8.3, 8.5, 14.3
    durations = (1.0, 0.4, 0.8, 1.0, 0.2, 0.6)
    calls = tuple(
        Call(f"call-{n}", "slow", {"seconds": seconds, "text": f"r{n}"})
        for n, seconds in enumerate(durations)
    )
    llm.plans["fan-out"] = (Reply(chunks=("starting",), calls=calls), Reply(chunks=("all done",)))

    async with open_rig(app_factory, llm, echo_log, backend=storage_backend) as rig:
        stream = rig.stream("fan-out")
        await stream.wait_closed("the fan-out Task to end")
        assert stream.last_state == COMPLETED, stream.summary()

        # The tool phase runs from the end of the first LLM response to the second LLM request.
        tool_phase_s = llm.request_times[1] - llm.stream_end_times[0]
        assert tool_phase_s < max(durations) + TURN_SLACK_S, f"the tools took {tool_phase_s:.2f} s"
        assert tool_phase_s < sum(durations) / 2  # not one after another

        history = await rig.history(stream.context_id)
        assert tool_results(history) == [f"r{n}" for n in range(len(durations))]  # call order
        assert_history_integrity(history)


# --- Cancel -------------------------------------------------------------------------------------


async def test_cancel_during_slow_tool_within_2s(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path, storage_backend: StorageBackend
) -> None:
    # Feature: local-agent-backend, cancel; Validates: Requirements 2.7, 14.4, 14.5, 14.6
    llm.plans["cancel-slow"] = (
        Reply(
            chunks=("working on it",),
            calls=(Call("call-s", "slow", {"seconds": 30, "text": "never"}),),
        ),
        Reply(chunks=("done",)),
    )
    async with open_rig(app_factory, llm, echo_log, backend=storage_backend) as rig:
        stream = rig.stream("cancel-slow")
        await stream.wait_for(lambda s: s.has_tool_kind(KIND_TOOL_STARTED), "the tool to start")
        await rig.wait_for_server_call("slow")  # the call is in flight on the MCP_Server

        elapsed_s, task = await rig.cancel(stream.task_id)

        assert elapsed_s < CANCEL_S, f"cancel took {elapsed_s:.2f} s"
        assert task["status"]["state"] == CANCELED
        assert await rig.state_of(stream.task_id) is TaskState.CANCELED
        await stream.wait_closed("the stream to end after cancel", timeout_s=CANCEL_S)
        assert stream.last_state == CANCELED, stream.summary()
        await assert_nothing_after_cancel(stream)

        history = await rig.history(stream.context_id)
        assert_history_integrity(history)
        (result,) = tool_results(history)
        assert "cancel" in result.lower()  # 14.5
        assert "never" not in result  # the slow result did not arrive


async def test_cancel_during_input_required_within_2s(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path, storage_backend: StorageBackend
) -> None:
    # Feature: local-agent-backend, cancel; Validates: Requirements 2.7, 5.7, 14.4, 14.5
    llm.plans["cancel-approval"] = (
        Reply(
            chunks=("need to write",),
            calls=(Call("call-w", "write_file", {"path": "a.txt", "content": "x"}),),
        ),
        Reply(chunks=("done",)),
    )
    async with open_rig(
        app_factory, llm, echo_log, backend=storage_backend, permissions={"echo": ["write_file"]}
    ) as rig:
        stream = rig.stream("cancel-approval")
        await stream.wait_for(lambda s: s.has_state(INPUT_REQUIRED), "input-required")
        await stream.wait_for(
            lambda s: s.has_tool_kind(KIND_APPROVAL_REQUIRED), "the approval prompt"
        )
        assert await rig.state_of(stream.task_id) is TaskState.INPUT_REQUIRED

        elapsed_s, task = await rig.cancel(stream.task_id)

        assert elapsed_s < CANCEL_S, f"cancel took {elapsed_s:.2f} s"
        assert task["status"]["state"] == CANCELED
        assert await rig.state_of(stream.task_id) is TaskState.CANCELED
        # The stream of the first message closed at `input-required` (3.5); cancel adds nothing.
        await stream.wait_closed("the stream to be closed")
        assert stream.last_state == INPUT_REQUIRED, stream.summary()

        assert "write_file" not in rig.tool_calls_seen_by_server()  # the pending call never ran
        history = await rig.history(stream.context_id)
        assert_history_integrity(history)
        (result,) = tool_results(history)
        assert "cancel" in result.lower()


async def test_cancel_during_stalled_llm_stream_within_2s(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path, storage_backend: StorageBackend
) -> None:
    # Feature: local-agent-backend, cancel; Validates: Requirements 2.7, 14.4, 14.6
    llm.plans["cancel-llm"] = (Reply(chunks=("thinking",), stall=True),)
    llm.plans["after-cancel"] = (Reply(chunks=("fine",)),)
    async with open_rig(app_factory, llm, echo_log, backend=storage_backend) as rig:
        stream = rig.stream("cancel-llm")
        await stream.wait_for(lambda s: s.has_kind("artifactUpdate"), "the first token")

        elapsed_s, task = await rig.cancel(stream.task_id)

        assert elapsed_s < CANCEL_S, f"cancel took {elapsed_s:.2f} s"
        assert task["status"]["state"] == CANCELED
        await stream.wait_closed("the stream to end", timeout_s=CANCEL_S)
        await assert_nothing_after_cancel(stream)

        # The Context is free again: a new Task in it runs to completion.
        again = rig.stream("after-cancel", context_id=stream.context_id)
        await again.wait_closed("the next Task of the Context to end")
        assert again.last_state == COMPLETED, again.summary()


async def assert_nothing_after_cancel(stream: TaskStream) -> None:
    """The final `canceled` Status_Event is the last event, and no event follows later (14.6)."""
    assert stream.events[-1].state == CANCELED, stream.summary()
    assert [seen.state for seen in stream.events].count(CANCELED) == 1, stream.summary()
    count = len(stream.events)
    await asyncio.sleep(0.3)
    assert len(stream.events) == count, stream.summary()


# --- Failure isolation --------------------------------------------------------------------------


async def test_failure_in_one_context_leaves_others_intact(
    app_factory: AppOpener, llm: RoutedLlm, echo_log: Path, storage_backend: StorageBackend
) -> None:
    # Feature: local-agent-backend, isolation; Validates: Requirements 14.4, 14.7
    healthy = 4
    for index in range(healthy):
        llm.plans[f"ok-{index}"] = slow_tool_plan(index, seconds=0.3)
    llm.plans["rejected"] = (Reply(reject="the model refuses this"),)
    llm.plans["tool-error"] = (
        Reply(calls=(Call("call-e", "error", {"message": "boom"}),)),
        Reply(chunks=("recovered",)),
    )
    llm.plans["tool-timeout"] = (
        Reply(calls=(Call("call-h", "hang", {}),)),
        Reply(chunks=("after-timeout",)),
    )
    llm.plans["stalled"] = (Reply(chunks=("stuck",), stall=True),)

    async with open_rig(
        app_factory,
        llm,
        echo_log,
        backend=storage_backend,
        overrides={"orchestration": {"tool_timeout_s": 1}},
    ) as rig:
        stalled = rig.stream("stalled")
        await stalled.wait_for(lambda s: s.has_kind("artifactUpdate"), "the first token")
        others = [rig.stream(f"ok-{index}") for index in range(healthy)]
        rejected = rig.stream("rejected")
        tool_error = rig.stream("tool-error")
        tool_timeout = rig.stream("tool-timeout")
        for stream in [*others, rejected, tool_error, tool_timeout]:
            await stream.wait_closed(f"{s_name(stream)} to end")

        # The failed Task failed; the failing tools did not fail their Tasks (9.8).
        assert rejected.last_state == FAILED, rejected.summary()
        assert tool_error.last_state == COMPLETED, tool_error.summary()
        assert tool_timeout.last_state == COMPLETED, tool_timeout.summary()
        assert "boom" in tool_results(await rig.history(tool_error.context_id))[0]
        assert await rig.artifact_text(tool_timeout.task_id) == "after-timeout"

        # The healthy Tasks are exactly what they would be alone.
        for index, stream in enumerate(others):
            assert stream.last_state == COMPLETED, (s_name(stream), stream.summary())
            history = await rig.history(stream.context_id)
            assert tool_results(history) == [f"result-{index}"]
            assert_history_integrity(history)
            assert "".join(final_chunks(index)) in await rig.artifact_text(stream.task_id)

        # The stalled LLM request of the fourth Context is still waiting, untouched.
        assert not stalled.closed
        assert await rig.state_of(stalled.task_id) is TaskState.WORKING

        elapsed_s, task = await rig.cancel(stalled.task_id)
        assert elapsed_s < CANCEL_S
        assert task["status"]["state"] == CANCELED
