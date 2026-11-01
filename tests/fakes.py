"""Fakes for the orchestration tests: a scripted LLM, a scripted MCP backend, and an event recorder.

The scenario properties (P1-P4, P12-P14, P23, P50-P52, P59, P61) run the real `TaskRunner`,
`ToolDispatcher`, `ApprovalGate`, and `TaskEventEmitter` against these fakes. Nothing here knows
about scenario scripts; `tests/property/scenarios.py` turns a generated script into the plain
values these classes take.

* `FakeLlmGateway` is a real `LlmGateway` whose `litellm.acompletion` is replaced by a scripted
  stream of OpenAI-style deltas. The real retry loop, `LlmStream`, and `StreamAssembler` run, so
  an unreachable endpoint, a rejection, and an interrupted stream reach the runner as the
  `LlmUnreachable`, `LlmRejected`, and `LlmInterrupted` it handles in production. Every request is
  recorded for assertions on the System_Prompt, the history, and the offered tools.
* `FakeMcpManager` implements `McpBackend` over a real `ToolRegistry`. Each call's outcome
  (success, error flag, timeout, exception, unavailable, hang) and latency come from a
  `ToolBehavior`, chosen by the `tag` argument of the call. Latency is a number of event loop
  yields, so concurrent calls can finish out of order without any real waiting.
* `RecordingEventSink` records every Status_Event and Token_Event the emitter delivers, with an
  optional history probe and listeners, which the scenario harness uses to inject approvals and
  cancels at exact points.
* `TickingClock`, `FixedClock`, `ClockSleep`, and `ManualTimers` are the time fakes. `ClockSleep`
  serves the LLM retry delays; `ManualTimers` serves the approval timeout, which only fires when a
  test says so.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from local_agent.a2a_server.events import TaskState
from local_agent.config.models import LLMConfig
from local_agent.llm.gateway import LlmGateway, LlmStream
from local_agent.mcp_client.manager import CallOutcome, McpServerUnavailable
from local_agent.mcp_client.registry import ToolRegistry
from local_agent.models import TextItem

__all__ = [
    "DEFAULT_BEHAVIOR",
    "ArtifactEvent",
    "ClockSleep",
    "FakeLlmGateway",
    "FakeMcpManager",
    "FakeRejection",
    "FixedClock",
    "LlmFailure",
    "LlmTurn",
    "ManualTimers",
    "RecordedRequest",
    "RecordedToolCall",
    "RecordingEventSink",
    "SinkEvent",
    "StatusEvent",
    "TickingClock",
    "ToolBehavior",
    "ToolOutcome",
    "text_delta",
    "tool_call_deltas",
]

WALL_START_MS = 1_700_000_000_000


# --- Time ---------------------------------------------------------------------------------------


class TickingClock:
    """A `Clock` where every reading is later than the one before.

    `now_ms()` moves 1 ms per call and `monotonic()` 1 ms per call, so a changed row always shows a
    new timestamp and a measured duration is small, positive, and the same on every run.
    `advance(seconds)` moves both clocks forward by a larger step.
    """

    def __init__(self, *, start_ms: int = WALL_START_MS, start_monotonic: float = 1_000.0) -> None:
        self._wall_ms = start_ms
        self._monotonic = start_monotonic

    def now_ms(self) -> int:
        self._wall_ms += 1
        return self._wall_ms

    def monotonic(self) -> float:
        self._monotonic += 0.001
        return self._monotonic

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("advance() takes a non-negative number of seconds")
        self._monotonic += seconds
        self._wall_ms += round(seconds * 1000)


class FixedClock:
    """A `Clock` that always reads `ms` (and `ms / 1000` seconds on the monotonic side)."""

    def __init__(self, ms: int = WALL_START_MS) -> None:
        self._ms = ms

    def now_ms(self) -> int:
        return self._ms

    def monotonic(self) -> float:
        return float(self._ms)


class ClockSleep:
    """An awaitable `sleep(seconds)` that advances `clock` instead of waiting.

    It yields to the event loop once, so other tasks still run, and records every delay in `calls`.
    """

    def __init__(self, clock: TickingClock) -> None:
        self._clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock.advance(max(seconds, 0.0))
        await asyncio.sleep(0)


class ManualTimers:
    """An awaitable `sleep(seconds)` that returns only when `fire()` is called.

    The approval timeout uses it: `ApprovalGate` races a decision against `sleep(timeout)`, so a
    test fires the timer to make the timeout win, or never fires it to model a user who answers.
    `requested` lists every delay that was asked for. Create it inside the running loop.
    """

    def __init__(self, clock: TickingClock | None = None) -> None:
        self._clock = clock
        self._waiters: list[asyncio.Future[None]] = []
        self._armed = asyncio.Event()
        self.requested: list[float] = []

    @property
    def pending(self) -> int:
        return sum(1 for waiter in self._waiters if not waiter.done())

    async def __call__(self, seconds: float) -> None:
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        self.requested.append(seconds)
        self._armed.set()
        try:
            await waiter
        finally:
            self._waiters.remove(waiter)
            if not self._waiters:
                self._armed.clear()

    def fire(self) -> None:
        """Complete every pending sleep, as if its time had passed."""
        for waiter in list(self._waiters):
            if not waiter.done():
                if self._clock is not None and self.requested:
                    self._clock.advance(self.requested[-1])
                waiter.set_result(None)

    async def wait_armed(self) -> None:
        """Wait until some coroutine is sleeping on this timer."""
        await self._armed.wait()

    async def fire_when_armed(self) -> None:
        """Wait until some coroutine is sleeping on this timer, then fire it."""
        await self.wait_armed()
        self.fire()


# --- Event recording ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusEvent:
    """A Status_Event as the sink received it. `index` counts every sink event from 1."""

    index: int
    state: TaskState
    text: str | None
    metadata: Mapping[str, Any] | None
    final: bool
    history_len: int | None = None

    @property
    def kind(self) -> str | None:
        """The machine-readable `kind` of a tool event, or `None` for a plain status."""
        if self.metadata is None:
            return None
        kind = self.metadata.get("kind")
        return kind if isinstance(kind, str) else None


@dataclass(frozen=True)
class ArtifactEvent:
    """A Token_Event (or the last-chunk marker) as the sink received it."""

    index: int
    artifact_id: str
    text: str
    append: bool
    last_chunk: bool
    history_len: int | None = None


type SinkEvent = StatusEvent | ArtifactEvent
type HistoryProbe = Callable[[], Awaitable[int | None]]
type Listener = Callable[[SinkEvent], None]


class RecordingEventSink:
    """Records every event the emitter delivers, in order, and notifies listeners.

    `probe`, when given, is awaited at each event and its result is stored as `history_len`: the
    number of messages the Session_Store held when the event arrived. A probe that fails stores
    `None`. Listeners run synchronously after the event is recorded; the scenario harness uses them
    to resolve approvals and to cancel after the n-th event.
    """

    def __init__(self, *, probe: HistoryProbe | None = None) -> None:
        self.events: list[SinkEvent] = []
        self._probe = probe
        self._listeners: list[Listener] = []

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    async def status(
        self,
        state: TaskState,
        text: str | None,
        metadata: Mapping[str, Any] | None,
        *,
        final: bool,
    ) -> None:
        history_len = await self._history_len()
        self._record(
            StatusEvent(len(self.events) + 1, state, text, metadata, final, history_len=history_len)
        )

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        history_len = await self._history_len()
        self._record(
            ArtifactEvent(
                len(self.events) + 1,
                artifact_id,
                text,
                append,
                last_chunk,
                history_len=history_len,
            )
        )

    async def _history_len(self) -> int | None:
        if self._probe is None:
            return None
        try:
            return await self._probe()
        except Exception:
            return None

    def _record(self, event: SinkEvent) -> None:
        self.events.append(event)
        for listener in list(self._listeners):
            listener(event)

    # --- Views ------------------------------------------------------------------------------

    @property
    def statuses(self) -> list[StatusEvent]:
        return [event for event in self.events if isinstance(event, StatusEvent)]

    @property
    def artifact_events(self) -> list[ArtifactEvent]:
        return [event for event in self.events if isinstance(event, ArtifactEvent)]

    @property
    def states(self) -> list[TaskState]:
        """The state of every Status_Event, in order (repeats included)."""
        return [event.state for event in self.statuses]

    @property
    def last_state(self) -> TaskState | None:
        statuses = self.statuses
        return statuses[-1].state if statuses else None

    @property
    def emitted(self) -> dict[str, str]:
        """The concatenated text emitted for each artifact: E(a) in the design."""
        texts: dict[str, str] = {}
        for event in self.artifact_events:
            texts[event.artifact_id] = texts.get(event.artifact_id, "") + event.text
        return texts

    def tool_events(self, kind: str) -> list[StatusEvent]:
        """The Status_Events whose metadata `kind` is `kind`."""
        return [event for event in self.statuses if event.kind == kind]


# --- LLM ----------------------------------------------------------------------------------------


class LlmFailure(StrEnum):
    """How a scripted LLM turn fails."""

    UNREACHABLE = "unreachable"
    """Every attempt fails before the first chunk with a retryable error (10.6, 10.7)."""

    REJECTED = "rejected"
    """The endpoint refuses the request with a non-retryable error carrying `reason` (10.9)."""

    INTERRUPTED = "interrupted"
    """The stream raises after `fail_after` deltas (10.10). With 0 deltas it is retried."""

    STALL = "stall"
    """The stream stops after `fail_after` deltas and never continues, until it is cancelled."""


class FakeRejection(Exception):
    """A non-retryable endpoint error. `status_code` and `message` are what the gateway reads."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def text_delta(text: str) -> dict[str, Any]:
    """An OpenAI-style streaming delta that carries text."""
    return {"content": text}


def tool_call_deltas(
    index: int,
    name: str,
    arguments: str,
    *,
    call_id: str | None = None,
    pieces: int = 1,
) -> list[dict[str, Any]]:
    """The deltas that stream one Tool_Call: the id and name once, the arguments in `pieces` parts.

    The pieces are slices of `arguments` of nearly equal length, so joining them in order gives
    the original text. Empty slices are dropped, except that the first delta always exists.
    """
    count = max(1, pieces)
    size = -(-len(arguments) // count) if arguments else 0
    slices = [arguments[i * size : (i + 1) * size] for i in range(count)] if size else [arguments]
    deltas: list[dict[str, Any]] = []
    for position, piece in enumerate(slices):
        if position and not piece:
            continue
        fragment: dict[str, Any] = {"index": index}
        function: dict[str, Any] = {}
        if position == 0:
            if call_id is not None:
                fragment["id"] = call_id
            function["name"] = name
        if piece:
            function["arguments"] = piece
        fragment["function"] = function
        deltas.append({"tool_calls": [fragment]})
    return deltas


@dataclass(frozen=True)
class LlmTurn:
    """One scripted chat-completion response.

    `deltas` are streamed in order (see `text_delta` and `tool_call_deltas`). `failure` ends the
    response early: `INTERRUPTED` and `STALL` act after `fail_after` deltas; `UNREACHABLE` and
    `REJECTED` act before the first one. `latency` is the number of event loop yields before each
    delta. `on_request` runs when the request arrives, before anything is streamed; the scenario
    harness uses it for registry changes "between reasoning steps" (12.5).
    """

    deltas: tuple[Mapping[str, Any], ...] = ()
    failure: LlmFailure | None = None
    fail_after: int = 0
    reason: str = "invalid request"
    latency: int = 0
    on_request: Callable[[], None] | None = None


@dataclass
class RecordedRequest:
    """One request the runner made: what it sent, and how often the gateway tried it."""

    index: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    attempts: int = 0

    @property
    def system_messages(self) -> list[dict[str, Any]]:
        return [message for message in self.messages if message.get("role") == "system"]


DEFAULT_TURN = LlmTurn(deltas=(text_delta("ok"),))
"""What the fake answers once its script has run out."""


class FakeLlmGateway:
    """A scripted `LlmClient`: request i is answered by `turns[i]`, then by `exhausted`.

    It wraps a real `LlmGateway`, so retries, the first-chunk rule, and error mapping are the
    production ones. `sleep` serves the retry delays (use `ClockSleep`, so no test waits).
    With `drop_calls_without_tools`, a request that offers no tools (the final request at
    Max_Tool_Iterations) gets the text of its turn only, like a model that obeys the missing tools.
    By default the scripted Tool_Calls are streamed anyway, to show the runner ignores them (8.12).
    `on_block` runs when a `STALL` turn stops.
    """

    def __init__(
        self,
        turns: Sequence[LlmTurn],
        *,
        clock: TickingClock,
        sleep: Callable[[float], Awaitable[None]],
        exhausted: LlmTurn = DEFAULT_TURN,
        drop_calls_without_tools: bool = False,
        on_block: Callable[[], None] | None = None,
    ) -> None:
        self._turns = list(turns)
        self._exhausted = exhausted
        self._drop_calls = drop_calls_without_tools
        self._on_block = on_block
        self._turn: LlmTurn = exhausted
        self._deltas: tuple[Mapping[str, Any], ...] = ()
        self.requests: list[RecordedRequest] = []
        self._gateway = LlmGateway(
            LLMConfig(model="fake/model"),
            sleep=sleep,
            clock=clock,
            completion=self._completion,
        )

    @property
    def attempts(self) -> int:
        """How many times the gateway called the completion function, over all requests."""
        return sum(request.attempts for request in self.requests)

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LlmStream:
        index = len(self.requests)
        self._turn = self._turns[index] if index < len(self._turns) else self._exhausted
        self._deltas = self._turn.deltas
        if self._drop_calls and not tools:
            self._deltas = _without_tool_calls(self._deltas)
        self.requests.append(
            RecordedRequest(index, copy.deepcopy(messages), copy.deepcopy(tools), attempts=0)
        )
        if self._turn.on_request is not None:
            self._turn.on_request()
        return await self._gateway.stream(messages, tools)

    async def _completion(self, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        turn = self._turn
        self.requests[-1].attempts += 1
        if turn.failure is LlmFailure.UNREACHABLE:
            raise ConnectionError("connection refused")
        if turn.failure is LlmFailure.REJECTED:
            raise FakeRejection(turn.reason)
        return self._chunks(turn, self._deltas)

    async def _chunks(
        self, turn: LlmTurn, deltas: Sequence[Mapping[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        acts = turn.failure in (LlmFailure.INTERRUPTED, LlmFailure.STALL)
        limit = min(turn.fail_after, len(deltas)) if acts else -1
        for position in range(len(deltas) + 1):
            if position == limit:
                if turn.failure is LlmFailure.STALL:
                    if self._on_block is not None:
                        self._on_block()
                    await asyncio.get_running_loop().create_future()
                raise ConnectionError("stream broke")
            if position == len(deltas):
                return
            for _ in range(turn.latency):
                await asyncio.sleep(0)
            yield {"choices": [{"delta": dict(deltas[position])}]}


def _without_tool_calls(deltas: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    kept: list[Mapping[str, Any]] = []
    for delta in deltas:
        stripped = {key: value for key, value in delta.items() if key != "tool_calls"}
        if stripped:
            kept.append(stripped)
    return tuple(kept)


# --- MCP ----------------------------------------------------------------------------------------


class ToolOutcome(StrEnum):
    """What a scripted MCP tool call does."""

    SUCCESS = "success"
    ERROR = "error"  # the server answers with `isError` set
    TIMEOUT = "timeout"  # `TimeoutError`, as `McpManager.call_tool` raises on Tool_Timeout
    EXCEPTION = "exception"  # a transport or protocol error (9.7)
    UNAVAILABLE = "unavailable"  # `McpServerUnavailable` raised mid-call (9.6)
    HANG = "hang"  # never answers, until the Task is cancelled


@dataclass(frozen=True)
class ToolBehavior:
    """How one Tool_Call ends. `text` is the content, or the exception message. `latency` is a
    number of event loop yields before the outcome."""

    outcome: ToolOutcome = ToolOutcome.SUCCESS
    text: str = "ok"
    latency: int = 0


DEFAULT_BEHAVIOR = ToolBehavior()
"""A call that succeeds with the text "ok" and no delay."""


@dataclass
class RecordedToolCall:
    """One `call_tool` the dispatcher made."""

    index: int
    server: str
    tool: str
    args: dict[str, Any]
    timeout: float
    finished: bool = False
    cancelled: bool = False


class FakeMcpManager:
    """A scripted `McpBackend` over a real `ToolRegistry`.

    `registry` is the `SnapshotSource` the runner reads; change it with `registry.replace_server`
    and `registry.remove_server`, and flip a server with `set_available`. A call's behavior is
    `behaviors[args["tag"]]` when the arguments carry a known `tag`, else `default`. A call to a
    server that is not available raises `McpServerUnavailable`, as the real manager does.
    `on_block` runs when a call hangs.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        available: Mapping[str, bool],
        *,
        behaviors: Mapping[str, ToolBehavior] | None = None,
        default: ToolBehavior = DEFAULT_BEHAVIOR,
        on_block: Callable[[], None] | None = None,
    ) -> None:
        self.registry = registry
        self._available = dict(available)
        self._behaviors = dict(behaviors or {})
        self._default = default
        self._on_block = on_block
        self.calls: list[RecordedToolCall] = []
        self.refreshes = 0
        self.refresh_error: Exception | None = None
        self.in_flight = 0
        self.max_in_flight = 0
        self.finish_order: list[int] = []  # `RecordedToolCall.index` in the order calls ended

    def set_available(self, server: str, available: bool) -> None:
        self._available[server] = available

    def availability(self) -> Mapping[str, bool]:
        return dict(self._available)

    async def refresh_available(self) -> None:
        self.refreshes += 1
        if self.refresh_error is not None:
            raise self.refresh_error

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> CallOutcome:
        record = RecordedToolCall(len(self.calls), server, tool, copy.deepcopy(args), timeout)
        self.calls.append(record)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if not self._available.get(server, False):
                raise McpServerUnavailable(server)
            tag = args.get("tag")
            behavior = self._default
            if isinstance(tag, str):
                behavior = self._behaviors.get(tag, self._default)
            for _ in range(behavior.latency):
                await asyncio.sleep(0)
            outcome = await self._finish(behavior)
            record.finished = True
            return outcome
        except asyncio.CancelledError:
            record.cancelled = True
            raise
        finally:
            self.in_flight -= 1
            self.finish_order.append(record.index)

    async def _finish(self, behavior: ToolBehavior) -> CallOutcome:
        match behavior.outcome:
            case ToolOutcome.SUCCESS:
                return CallOutcome(content=(TextItem(behavior.text),), is_error=False)
            case ToolOutcome.ERROR:
                return CallOutcome(content=(TextItem(behavior.text),), is_error=True)
            case ToolOutcome.TIMEOUT:
                raise TimeoutError
            case ToolOutcome.EXCEPTION:
                raise RuntimeError(behavior.text)
            case ToolOutcome.UNAVAILABLE:
                raise McpServerUnavailable(behavior.text)
            case ToolOutcome.HANG:
                if self._on_block is not None:
                    self._on_block()
                await asyncio.get_running_loop().create_future()
                raise AssertionError("unreachable: the future above is never completed")
