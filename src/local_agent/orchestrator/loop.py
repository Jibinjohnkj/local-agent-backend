"""The Orchestrator: one detached `TaskRunner` per Task, and the registry that owns them.

A `TaskRunner` runs a Task's reasoning loop in its own `asyncio.Task`, not inside the HTTP request
(design decision 1). `execute()` only attaches the SDK `EventQueue` and waits for a stream-closing
state, so a Task survives SSE disconnects (3.6), waits in `input-required` without losing its
in-memory state (5.7), and can be canceled precisely (2.7, 14.4).

The loop follows the design's `_run` / `_reasoning_loop` sketch:

* Every state change goes through the Task's `TaskEventEmitter`, which commits before it emits
  (11.2, 18.8). Every history write (user message, assistant message, Tool_Results, System_Prompt)
  is one `StorageTxn` committed through `TaskEventEmitter.commit` and awaited before the next
  chat-completion request or Status_Event (18.10).
* Each message row carries the producing `task_id`, which Restart_Recovery uses to find a Task's
  Tool_Calls.
* `_finalize` answers every Tool_Call that has no Tool_Result with a synthetic one and commits
  those results, the text of unclosed artifacts, and the terminal state in one transaction
  (11.5, 18.11, 18.12).

Nothing here touches an event loop at import time. Time comes from the injected `Clock`, waits come
from the injected `sleep`, and the only loop access is `asyncio.get_running_loop()`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, Self

from a2a.server.events import EventQueue
from a2a.types import Message
from structlog.contextvars import bound_contextvars

from local_agent.a2a_server.events import (
    DetachedSink,
    EventSink,
    QueueSink,
    StorageWriteFailed,
    TaskEventEmitter,
    TaskState,
)
from local_agent.config.models import OrchestrationConfig
from local_agent.llm.gateway import LlmInterrupted, LlmRejected, LlmStream, LlmUnreachable
from local_agent.logging_setup import Redactor, get_logger
from local_agent.mcp_client.registry import RegistrySnapshot
from local_agent.models import (
    ArtifactRecord,
    AssembledResponse,
    AssistantMessage,
    ChatMessage,
    Clock,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
    replace_lone_surrogates,
)
from local_agent.orchestrator.approval_gate import ApprovalGate
from local_agent.orchestrator.context_window import (
    TurnTooLarge,
    estimate_tokens_sync,
    split_turns,
    trim_to_budget,
)
from local_agent.orchestrator.prompt_builder import build_system_prompt
from local_agent.orchestrator.tool_dispatch import SnapshotSource, ToolBackend, ToolDispatcher
from local_agent.session.store import SessionStore, StorageTxn
from local_agent.storage import StaleState, StorageError

if TYPE_CHECKING:
    from types import TracebackType

__all__ = [
    "AdmissionReservation",
    "ChatRequest",
    "LlmClient",
    "McpBackend",
    "RunnerDeps",
    "TaskRunner",
    "TaskRunnerRegistry",
    "TokenCounter",
    "extract_user_text",
    "unanswered_tool_calls",
]

_log = get_logger("orchestrator.loop")

Sleep = Callable[[float], Awaitable[None]]
TokenCounter = Callable[[str, Sequence[ChatMessage], Sequence[Mapping[str, Any]] | None], int]

DISCOVERY_MESSAGE: Final = "Discovering local tools..."
"""The one `working` status that opens every Task (2.4, 4.1)."""

CANCELED_MESSAGE: Final = "Task canceled"
STORAGE_FAILED_MESSAGE: Final = "The conversation could not be saved"
UNREACHABLE_MESSAGE: Final = (
    "The local LLM is unreachable. Check that the model server is running and that the "
    "configured llm.api_base is correct."
)
INTERRUPTED_MESSAGE: Final = "The LLM response was interrupted before it completed."
REJECTED_PREFIX: Final = "The LLM endpoint rejected the request: "
TURN_TOO_LARGE_PREFIX: Final = "This conversation turn exceeds the context window budget"

NOT_COMPLETED_CANCELED: Final = "Tool call was not completed because the Task was canceled."
NOT_COMPLETED_FAILED: Final = "Tool call was not completed because the Task failed."

CANCEL_TIMEOUT_S: Final = 2.0
"""How long `TaskRunner.cancel` waits for the runner to finish (2.7, 14.4)."""

FINALIZE_TIMEOUT_S: Final = 1.5
"""Bound on the cleanup commit, so a cancel completes within CANCEL_TIMEOUT_S."""

RESERVATION_WAIT_S: Final = 1.0
"""How long a second message waits for the reserved `task_id` of an admission reservation."""


# --- Collaborator protocols ---------------------------------------------------------------------


class LlmClient(Protocol):
    """The part of `LlmGateway` the runner uses. Tests can supply a fake."""

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LlmStream: ...


class McpBackend(ToolBackend, Protocol):
    """The part of `McpManager` the runner uses: availability, tool calls, and refresh."""

    async def refresh_available(self) -> None: ...


@dataclass(frozen=True)
class RunnerDeps:
    """Everything a `TaskRunner` shares with the other runners.

    `estimate_tokens` is the synchronous token counter. It runs on a worker thread, so a large
    history never blocks the event loop (14.1). `sleep` is used for the approval timeout and the
    admission wait; tests pass a fake that advances a fake clock.
    """

    sessions: SessionStore
    llm: LlmClient
    registry: SnapshotSource
    mcp: McpBackend
    orchestration: OrchestrationConfig
    base_instructions: str | None
    model: str
    redactor: Redactor
    clock: Clock
    db_path: str
    sleep: Sleep = asyncio.sleep
    estimate_tokens: TokenCounter = estimate_tokens_sync


@dataclass(frozen=True)
class ChatRequest:
    """One chat-completion request: the trimmed messages (System_Prompt first) and the tools."""

    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None

    def wire_messages(self) -> list[dict[str, Any]]:
        return [message.to_openai() for message in self.messages]


# --- Pure helpers -------------------------------------------------------------------------------


def extract_user_text(message: Message) -> str:
    """The user's text: every text part of `message`, joined with a newline.

    Lone surrogates become U+FFFD, so the text can be stored and sent as UTF-8 (18.5).
    """
    texts = [part.text for part in message.parts if part.WhichOneof("content") == "text"]
    return replace_lone_surrogates("\n".join(texts))


def unanswered_tool_calls(history: Sequence[ChatMessage]) -> list[ToolCall]:
    """The Tool_Calls in `history` that no later tool message answers, in history order.

    A repeated call ID is returned once, because one tool message answers every call with that ID.
    """
    last_answer: dict[str, int] = {}
    for index, message in enumerate(history):
        if isinstance(message, ToolMessage):
            last_answer[message.tool_call_id] = index
    pending: list[ToolCall] = []
    seen: set[str] = set()
    for index, message in enumerate(history):
        if not isinstance(message, AssistantMessage):
            continue
        for call in message.tool_calls:
            if last_answer.get(call.id, -1) > index or call.id in seen:
                continue
            seen.add(call.id)
            pending.append(call)
    return pending


def _tool_call_ids(history: Sequence[ChatMessage]) -> set[str]:
    return {
        call.id
        for message in history
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }


# --- Event sink that knows when the stream may close --------------------------------------------


class _ClosingSink:
    """Forwards events to the attached sink and notes when a stream-closing event went out.

    `execute()` waits on `closing` (through `TaskRunner.wait_stream_closing`). After a final event
    the sink detaches itself: the SDK closes that queue, and the next event (an approval timeout,
    for example) belongs to no stream (3.5).
    """

    def __init__(self, closing: asyncio.Event) -> None:
        self.inner: EventSink = DetachedSink()
        self._closing = closing

    async def status(
        self,
        state: TaskState,
        text: str | None,
        metadata: Mapping[str, Any] | None,
        *,
        final: bool,
    ) -> None:
        inner = self.inner
        try:
            await inner.status(state, text, metadata, final=final)
        finally:
            if final:
                if self.inner is inner:
                    self.inner = DetachedSink()
                self._closing.set()

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        await self.inner.artifact(artifact_id, text, append=append, last_chunk=last_chunk)


class _PinnedRegistry:
    """The registry snapshot of the latest request, so the names the LLM saw resolve its calls.

    Until a request has been built it answers with the live snapshot.
    """

    def __init__(self, live: SnapshotSource) -> None:
        self._live = live
        self.pinned: RegistrySnapshot | None = None

    def snapshot(self) -> RegistrySnapshot:
        return self.pinned if self.pinned is not None else self._live.snapshot()


# --- TaskRunner ---------------------------------------------------------------------------------


class TaskRunner:
    """Runs one Task's reasoning loop in its own `asyncio.Task` (design decision 1).

    The executor creates a runner through `TaskRunnerRegistry.create`, calls `attach`, and then
    `start`. It waits for a stream-closing state with `wait_stream_closing`. `cancel` stops the
    runner within 2 seconds and discards a pending approval (2.7, 14.4).
    """

    def __init__(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        deps: RunnerDeps,
        is_new_context: bool | None = None,
        on_finished: Callable[[TaskRunner], None] | None = None,
    ) -> None:
        """`is_new_context` says whether the Task opened its Context (6.12). `None` means: decide
        from the stored history, which is empty for a new Context."""
        self.task_id = task_id
        self.context_id = context_id
        self._user_text = user_text
        self._deps = deps
        self._is_new_context = is_new_context
        self._on_finished = on_finished
        self._closing = asyncio.Event()
        self._sink = _ClosingSink(self._closing)
        self.emitter = TaskEventEmitter(
            deps.sessions,
            self._sink,
            deps.redactor,
            deps.clock,
            deps.db_path,
            task_id=task_id,
            context_id=context_id,
        )
        cfg = deps.orchestration
        self.approval_gate = ApprovalGate(
            self.emitter, approval_timeout_s=cfg.approval_timeout_s, sleep=deps.sleep
        )
        self._tools = _PinnedRegistry(deps.registry)
        self._dispatcher = ToolDispatcher(
            self._tools,
            deps.mcp,
            self.approval_gate,
            self.emitter,
            deps.redactor,
            deps.clock,
            tool_timeout_s=cfg.tool_timeout_s,
            max_output_chars=cfg.max_tool_output_chars,
        )
        self._task: asyncio.Task[None] | None = None
        self._finalizing = False
        self._responses = 0
        self._known_call_ids: set[str] = set()
        # Tool_Calls of the last committed assistant message that have no committed result yet.
        # Only a fallback for `missing_results` when the history cannot be read.
        self._outstanding: tuple[ToolCall, ...] = ()

    # --- Control surface used by the executor ------------------------------------------------

    def attach(self, event_queue: EventQueue) -> None:
        """Send later events to `event_queue` (an `execute()` or `cancel()` call attached).

        Re-arms `wait_stream_closing` unless the Task is already over.
        """
        self.attach_sink(QueueSink(event_queue, self.task_id, self.context_id))

    def attach_sink(self, sink: EventSink) -> None:
        """Like `attach`, for any `EventSink`. Tests use this with a recording sink."""
        self._sink.inner = sink
        if not self.emitter.sealed:
            self._closing.clear()

    def detach(self) -> None:
        """Drop later events until the next `attach`."""
        self._sink.inner = DetachedSink()

    def start(self) -> None:
        """Start the reasoning loop as a detached `asyncio.Task`. A second call does nothing."""
        if self._task is not None:
            return
        task = asyncio.get_running_loop().create_task(
            self._run(), name=f"task-runner-{self.task_id}"
        )
        task.add_done_callback(self._task_done)
        self._task = task

    async def wait_stream_closing(self) -> None:
        """Wait until the Task reaches `completed`, `failed`, `canceled`, or `input-required`.

        Returns once that Status_Event has been handed to the sink (3.5). It also returns when the
        runner has ended without one, so `execute()` can never hang on a dead runner.
        """
        await self._closing.wait()

    async def cancel(self) -> None:
        """Stop the Task within 2 seconds: discard a pending approval and cancel the runner (2.7).

        The runner's cancellation handler commits and emits `canceled`. In-flight LLM requests and
        Tool_Calls are cancelled with it (14.4), and whatever they produce later is dropped
        because the coroutine awaiting it no longer exists (14.5). A runner that is already
        finalizing for another reason is left to finish; it is about to be terminal anyway.
        """
        self.approval_gate.discard()
        if self._task is None:
            # A cancel that beats `start()`: run the first step so the handler can finalize.
            self.start()
            await asyncio.sleep(0)
        task = self._task
        assert task is not None
        if task.done():
            return
        if not self._finalizing:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=CANCEL_TIMEOUT_S)
        if not done:
            _log.warning(
                "task_cancel_timed_out",
                context_id=self.context_id,
                task_id=self.task_id,
                timeout_s=CANCEL_TIMEOUT_S,
            )

    def abandon(self) -> None:
        """Cancel the runner without waiting; retention uses it for a Context that was deleted."""
        self.approval_gate.discard()
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    def live_artifacts(self) -> list[ArtifactRecord]:
        """Artifacts with the text accumulated so far (2.10)."""
        return self.emitter.live_artifacts()

    # --- The runner ---------------------------------------------------------------------------

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._closing.set()
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                _log.error(
                    "task_runner_crashed",
                    context_id=self.context_id,
                    task_id=self.task_id,
                    error_type=type(exc).__name__,
                    reason=self._deps.redactor.redact(str(exc)),
                )
        if self._on_finished is not None:
            self._on_finished(self)

    async def _run(self) -> None:
        with bound_contextvars(context_id=self.context_id, task_id=self.task_id):
            try:
                await self._run_steps()
            except asyncio.CancelledError:
                await self._finalize(TaskState.CANCELED, CANCELED_MESSAGE)
                raise
            except LlmUnreachable:
                await self._finalize(TaskState.FAILED, UNREACHABLE_MESSAGE)  # 10.7
            except LlmRejected as exc:
                await self._finalize(TaskState.FAILED, REJECTED_PREFIX + exc.reason)  # 10.9
            except LlmInterrupted:
                await self._finalize(TaskState.FAILED, INTERRUPTED_MESSAGE)  # 3.8, 10.10
            except TurnTooLarge as exc:
                detail = f"{exc.estimate} estimated tokens, budget {exc.budget}"
                await self._finalize(
                    TaskState.FAILED, f"{TURN_TOO_LARGE_PREFIX} ({detail})."
                )  # 11.9
            except StorageWriteFailed:
                await self._finalize(TaskState.FAILED, STORAGE_FAILED_MESSAGE)  # 18.30
            except Exception as exc:
                _log.error(
                    "task_unhandled_error",
                    error_type=type(exc).__name__,
                    reason=self._deps.redactor.redact(str(exc)),
                    exc_info=True,
                )
                await self._finalize(TaskState.FAILED, self._safe_message(exc))  # 2.6

    async def _run_steps(self) -> None:
        await self.emitter.transition(TaskState.WORKING, DISCOVERY_MESSAGE)  # 2.4, 4.1
        await self._refresh_tools()
        await self._append([UserMessage(self._user_text)])  # 11.3
        await self._reasoning_loop()

    async def _refresh_tools(self) -> None:
        """Refresh every available server's tool list for a new Context (6.12, 6.13)."""
        if not self._deps.orchestration.per_session_discovery:
            return
        is_new = self._is_new_context
        if is_new is None:
            _, history = await self._deps.sessions.get_history(self.context_id)
            is_new = not history
        if not is_new:
            return
        try:
            await self._deps.mcp.refresh_available()
        except Exception as exc:  # a failed refresh keeps the old tools and never blocks (6.13)
            _log.warning(
                "mcp_refresh_failed",
                context_id=self.context_id,
                task_id=self.task_id,
                reason=self._deps.redactor.redact(f"{type(exc).__name__}: {exc}"),
            )

    async def _reasoning_loop(self) -> None:
        round_trips = 0
        max_iterations = self._deps.orchestration.max_tool_iterations
        while True:
            tools_enabled = round_trips < max_iterations
            request = await self._build_request(tools_enabled)
            response = await self._stream_response(request, tools_enabled=tools_enabled)
            if not tools_enabled or not response.tool_calls:  # 2.5, 8.8, 8.12
                await self._append_assistant(response, tool_calls=())
                await self._complete()
                return
            await self._append_assistant(response, tool_calls=response.tool_calls)  # 8.10
            results = await self._dispatcher.run_turn(response.tool_calls)
            await self._append_results(results)  # 8.4, 8.5
            round_trips += 1

    # --- One request --------------------------------------------------------------------------

    async def _build_request(self, tools_enabled: bool) -> ChatRequest:
        """Build the next chat-completion request (8.7, 11.8, 12.5, 12.6, 12.7).

        The System_Prompt is rebuilt from the live registry snapshot, written to the Context's
        single slot when it changed, and placed first. The stored history is loaded and trimmed
        to the Context_Window_Budget for this request only; storage keeps the full history. With
        `tools_enabled` false, or with no tools available, the request carries no tool
        definitions (10.4).
        """
        snapshot = self._deps.registry.snapshot()
        self._tools.pinned = snapshot
        availability = self._deps.mcp.availability()
        visible = [
            tool for tools in snapshot.tools_by_server(availability).values() for tool in tools
        ]
        system = SystemMessage(build_system_prompt(self._deps.base_instructions, visible))
        stored_system, history = await self._deps.sessions.get_history(self.context_id)
        if stored_system != system:
            await self.emitter.commit(
                StorageTxn(
                    context_id=self.context_id,
                    now_ms=self._deps.clock.now_ms(),
                    system_prompt=system,
                )
            )
        self._known_call_ids = _tool_call_ids(history)

        tools = snapshot.openai_tools(availability) if tools_enabled else []
        messages = await self._trim(system, history, tools)
        return ChatRequest(messages=messages, tools=tools or None)

    async def _trim(
        self, system: SystemMessage, history: Sequence[ChatMessage], tools: list[dict[str, Any]]
    ) -> list[ChatMessage]:
        """Drop the oldest whole turns until the request fits the budget (11.8, 11.9).

        The estimate can mean a full tokenization per call, so the whole search runs on a worker
        thread.
        """
        deps = self._deps

        def estimate(messages: Sequence[ChatMessage]) -> int:
            return deps.estimate_tokens(deps.model, messages, tools or None)

        return await asyncio.to_thread(
            trim_to_budget,
            system,
            split_turns(history),
            deps.orchestration.context_window_budget,
            estimate,
        )

    async def _stream_response(
        self, request: ChatRequest, *, tools_enabled: bool
    ) -> AssembledResponse:
        """Stream one assistant response and emit its text as Token_Events (3.2, 3.3, 3.4).

        Every response gets a fresh artifact ID. Tool_Call fragments emit nothing. The artifact is
        closed when the stream ends if it received text, or if the response is a final one, so a
        completed Task always has a result artifact even when the answer is empty (2.5). On
        `LlmInterrupted` the text already emitted stays in the artifact (3.8) and nothing of the
        response enters the history (10.7).
        """
        self._responses += 1
        artifact_id = f"response-{self._responses}"
        stream = await self._deps.llm.stream(request.wire_messages(), request.tools)
        emitted = False
        try:
            async for chunk in stream:
                await self.emitter.token(artifact_id, chunk, first=not emitted)
                emitted = True
        except BaseException:
            with contextlib.suppress(Exception):
                await stream.aclose()
            raise
        response = stream.result(self._known_call_ids)
        if emitted or not tools_enabled or not response.tool_calls:
            await self.emitter.close_artifact(artifact_id)
        return response

    # --- History writes -----------------------------------------------------------------------

    async def _append(self, messages: Sequence[ChatMessage]) -> None:
        """Append `messages` in one commit, each tagged with this Task's ID (18.10)."""
        await self.emitter.commit(
            StorageTxn(
                context_id=self.context_id,
                now_ms=self._deps.clock.now_ms(),
                appends=tuple((self.task_id, message) for message in messages),
            )
        )

    async def _append_assistant(
        self, response: AssembledResponse, *, tool_calls: tuple[ToolCall, ...]
    ) -> None:
        """Append the assistant message. A tools-disabled response keeps only its text (8.12)."""
        # No text with Tool_Calls is stored as None; an empty final answer stays "" (18.3).
        content = (response.text or None) if tool_calls else response.text
        await self._append([AssistantMessage(content=content, tool_calls=tool_calls)])
        self._known_call_ids.update(call.id for call in tool_calls)
        self._outstanding = tool_calls

    async def _append_results(self, results: Sequence[ToolResult]) -> None:
        """Append the turn's Tool_Results, in Tool_Call order, as one commit (8.5, 9.10)."""
        await self._append([result.as_message() for result in results])
        self._outstanding = ()

    async def _complete(self) -> None:
        await self.emitter.transition(TaskState.COMPLETED)

    # --- Finalization -------------------------------------------------------------------------

    async def missing_results(self, state: TaskState) -> list[ToolMessage]:
        """A synthetic Tool_Result for every Tool_Call of the history that has none (11.5, 14.5).

        The stored history is authoritative. If it cannot be read, the calls of the last committed
        assistant message stand in.
        """
        text = NOT_COMPLETED_FAILED if state is TaskState.FAILED else NOT_COMPLETED_CANCELED
        try:
            _, history = await self._deps.sessions.get_history(self.context_id)
        except (StorageError, StaleState) as exc:
            _log.warning(
                "missing_results_history_unreadable",
                context_id=self.context_id,
                task_id=self.task_id,
                reason=self._deps.redactor.redact(str(exc)),
            )
            calls = list(self._outstanding)
        else:
            calls = unanswered_tool_calls(history)
        return [ToolMessage(tool_call_id=call.id, content=text) for call in calls]

    async def _finalize(self, state: TaskState, message: str) -> None:
        """Move to a terminal `state`, bounded by 1.5 s so a cancel completes within 2 s (2.7).

        Never raises except for cancellation. A failure here is logged: the Task then stays at its
        last committed state for Restart_Recovery.
        """
        if self.emitter.sealed:
            return
        self._finalizing = True
        try:
            async with asyncio.timeout(FINALIZE_TIMEOUT_S):
                await self._finalize_commit(state, message)
        except TimeoutError:
            _log.warning(
                "task_finalize_timed_out",
                context_id=self.context_id,
                task_id=self.task_id,
                state=state.value,
            )
        except Exception as exc:
            _log.error(
                "task_finalize_failed",
                context_id=self.context_id,
                task_id=self.task_id,
                state=state.value,
                error_type=type(exc).__name__,
                reason=self._deps.redactor.redact(str(exc)),
            )

    async def _finalize_commit(self, state: TaskState, message: str) -> None:
        """Commit the synthetic results, unclosed artifact text, and the terminal state at once.

        A failed commit gets exactly one more attempt as `failed` with "The conversation could not
        be saved" (18.30). When that also fails, the emitter emits the final `failed` event without
        persisting it and seals (18.31).

        A cancel that raced a failed commit ends as `failed` too: the failure is a storage failure
        under 18.30, and the cancel only interrupted the coroutine that awaited the commit.
        """
        if state is TaskState.CANCELED and self.emitter.commit_lost_to_cancel:
            state, message = TaskState.FAILED, STORAGE_FAILED_MESSAGE
        if self.emitter.state is TaskState.SUBMITTED:
            # Cancelled while the first `working` commit was still in flight.
            with contextlib.suppress(StorageWriteFailed):
                await self.emitter.transition(TaskState.WORKING, emit=False)
        results = await self.missing_results(state)
        txn = StorageTxn(
            context_id=self.context_id,
            now_ms=self._deps.clock.now_ms(),
            appends=tuple((self.task_id, result) for result in results),
        )
        try:
            await self.emitter.transition(state, message, with_txn=txn)
        except StorageWriteFailed:
            await self.emitter.transition(TaskState.FAILED, STORAGE_FAILED_MESSAGE, with_txn=txn)

    def _safe_message(self, exc: BaseException) -> str:
        """A human-readable failure text with secrets redacted and no traceback (2.6, 16.11)."""
        text = f"An unexpected error stopped the task. {type(exc).__name__}: {exc}"
        return replace_lone_surrogates(self._deps.redactor.redact(text))


# --- Admission reservations ---------------------------------------------------------------------


async def _wait_for(event: asyncio.Event, timeout_s: float, sleep: Sleep) -> bool:
    """Wait until `event` is set or `timeout_s` passes on `sleep`. Return whether it is set."""
    if event.is_set():
        return True
    waiter = asyncio.ensure_future(event.wait())
    timer = asyncio.ensure_future(sleep(timeout_s))
    try:
        await asyncio.wait({waiter, timer}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiter.cancel()
        timer.cancel()
    return event.is_set()


class AdmissionReservation:
    """Holds a Context while a new message is admitted and its Task is not yet known (1.14).

    The reservation closes the window between "no Task exists" and "the creation commit". It
    carries the `task_id` once `TaskRunnerRegistry.create` claims it. It is released when the
    creation commit succeeds or fails, or when the request ends. Releasing twice is harmless, and
    the object is a context manager that releases on exit.
    """

    def __init__(self, context_id: str, release: Callable[[AdmissionReservation], None]) -> None:
        self.context_id = context_id
        self._release = release
        self._task_id: str | None = None
        self._claimed = asyncio.Event()
        self._released = False

    @property
    def task_id(self) -> str | None:
        """The reserved `task_id`, or `None` until a runner claims the reservation."""
        return self._task_id

    @property
    def released(self) -> bool:
        return self._released

    def claim(self, task_id: str) -> None:
        self._task_id = task_id
        self._claimed.set()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._release(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


# --- Registry -----------------------------------------------------------------------------------


class TaskRunnerRegistry:
    """Owns the runners of the live Tasks and the admission reservations of their Contexts.

    It implements `ForgetContexts` for retention and `LiveArtifacts` for `ProjectedTaskStore`.
    A runner leaves the registry when its `asyncio.Task` ends. A Task whose runner is gone is
    terminal, or was left for Restart_Recovery (18.31).
    """

    def __init__(self, deps: RunnerDeps, *, reservation_wait_s: float = RESERVATION_WAIT_S) -> None:
        self._deps = deps
        self._reservation_wait_s = reservation_wait_s
        self._runners: dict[str, TaskRunner] = {}
        self._reservations: dict[str, AdmissionReservation] = {}

    # --- Reservations -------------------------------------------------------------------------

    def reserve(self, context_id: str) -> AdmissionReservation:
        """Reserve `context_id` for a message being admitted.

        The caller holds the per-Context admission lock and has checked `reservation` first, so a
        second reservation is a bug.
        """
        if context_id in self._reservations:
            raise RuntimeError(f"Context {context_id!r} already has an admission reservation")
        reservation = AdmissionReservation(context_id, self._drop_reservation)
        self._reservations[context_id] = reservation
        return reservation

    def reservation(self, context_id: str) -> AdmissionReservation | None:
        """The live reservation of `context_id`, if any."""
        return self._reservations.get(context_id)

    async def reserved_task_id(self, context_id: str) -> str | None:
        """The `task_id` of `context_id`'s reservation, waiting up to 1 s for a runner to claim it.

        Returns `None` when there is no reservation or nobody claimed it in time.
        """
        reservation = self._reservations.get(context_id)
        if reservation is None:
            return None
        await _wait_for(reservation._claimed, self._reservation_wait_s, self._deps.sleep)
        return reservation.task_id

    def _drop_reservation(self, reservation: AdmissionReservation) -> None:
        if self._reservations.get(reservation.context_id) is reservation:
            del self._reservations[reservation.context_id]

    # --- Runners ------------------------------------------------------------------------------

    def create(self, task_id: str, context_id: str, message: Message) -> TaskRunner:
        """Create the runner of a new Task and claim the Context's reservation, if any.

        The runner is not started; the executor attaches its queue, waits for the creation commit,
        and calls `start`.
        """
        if task_id in self._runners:
            raise RuntimeError(f"Task {task_id!r} already has a runner")
        runner = TaskRunner(
            task_id=task_id,
            context_id=context_id,
            user_text=extract_user_text(message),
            deps=self._deps,
            on_finished=self._runner_finished,
        )
        self._runners[task_id] = runner
        reservation = self._reservations.get(context_id)
        if reservation is not None and reservation.task_id is None:
            reservation.claim(task_id)
        return runner

    def get(self, task_id: str) -> TaskRunner | None:
        return self._runners.get(task_id)

    def discard(self, task_id: str) -> None:
        """Drop a runner that was never started, because its creation commit failed (18.30)."""
        self._runners.pop(task_id, None)

    def _runner_finished(self, runner: TaskRunner) -> None:
        if self._runners.get(runner.task_id) is runner:
            del self._runners[runner.task_id]

    def forget(self, context_ids: Sequence[str]) -> None:
        """Drop all in-memory state for Contexts that retention deleted (11.14).

        Retention never deletes a Context with a live Task, so a runner found here is a bug. It is
        abandoned rather than left running against rows that no longer exist.
        """
        gone = set(context_ids)
        for task_id, runner in list(self._runners.items()):
            if runner.context_id in gone:
                del self._runners[task_id]
                runner.abandon()
        for context_id in gone:
            reservation = self._reservations.pop(context_id, None)
            if reservation is not None:
                reservation.release()

    def live_artifacts(self, task_id: str) -> list[ArtifactRecord]:
        """Artifacts of the Task's runner with the text accumulated so far (2.10).

        Empty when the Task has no runner.
        """
        runner = self._runners.get(task_id)
        return [] if runner is None else runner.live_artifacts()

    async def cancel_all(self) -> None:
        """Cancel every live Task as in 2.7, for Graceful_Shutdown (17.14).

        Runners cancel concurrently, so the whole call takes at most about 2 seconds.
        """
        runners = list(self._runners.values())
        if runners:
            await asyncio.gather(*(runner.cancel() for runner in runners), return_exceptions=True)
