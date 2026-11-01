"""Task_State transition table (2.9, 3.5, 18.15) and the per-Task `TaskEventEmitter`.

The first half of this module holds the internal `TaskState` and the single place where it is
mapped to the pinned `a2a-sdk` enum names (`TASK_STATE_*`). Everything else in the Backend uses the
internal enum, so an SDK rename touches only `_SDK_NAMES`. The functions there are pure: no I/O,
no clock.

The second half is `TaskEventEmitter`. Every Status_Event and Token_Event of a Task goes through
one emitter. It enforces the transition table, commits each state change as one `StorageTxn`
*before* the event is enqueued (11.2, 18.8), and seals itself after a terminal state (3.9, 4.9,
14.6). `session.store` imports `TaskState` from this module, so the emitter imports the
`StorageTxn` types inside `_make_txn` instead of at module level.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part
from a2a.types import TaskState as SdkTaskState

from local_agent.logging_setup import Redactor, get_logger
from local_agent.models import ArtifactRecord, Clock, ToolCall, replace_lone_surrogates
from local_agent.storage import StaleState, StorageError

if TYPE_CHECKING:
    from local_agent.session.store import SessionStore, StorageTxn

_log = get_logger("a2a_server.events")


class TaskState(StrEnum):
    """The six Task_States this Backend uses. The value is the lowercase A2A spelling."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# The only place that names SDK enum members. `SdkTaskState.Value(name)` raises ValueError at
# import time (via `_SDK_VALUES` below) if the pinned SDK renames one of them.
_SDK_NAMES: dict[TaskState, str] = {
    TaskState.SUBMITTED: "TASK_STATE_SUBMITTED",
    TaskState.WORKING: "TASK_STATE_WORKING",
    TaskState.INPUT_REQUIRED: "TASK_STATE_INPUT_REQUIRED",
    TaskState.COMPLETED: "TASK_STATE_COMPLETED",
    TaskState.FAILED: "TASK_STATE_FAILED",
    TaskState.CANCELED: "TASK_STATE_CANCELED",
}

_SDK_VALUES: dict[TaskState, SdkTaskState] = {
    state: SdkTaskState.Value(name) for state, name in _SDK_NAMES.items()
}
_FROM_SDK: dict[int, TaskState] = {int(value): state for state, value in _SDK_VALUES.items()}


def to_sdk(state: TaskState) -> SdkTaskState:
    """Return the `a2a-sdk` enum value for an internal state."""
    return _SDK_VALUES[state]


def sdk_name(state: TaskState) -> str:
    """Return the SDK member name (e.g. `TASK_STATE_WORKING`) used on the wire (1.10)."""
    return _SDK_NAMES[state]


def from_sdk(value: SdkTaskState) -> TaskState:
    """Map an `a2a-sdk` enum value back to the internal state.

    Raises `ValueError` for SDK states this Backend never uses (e.g. rejected, auth-required).
    """
    try:
        return _FROM_SDK[int(value)]
    except KeyError:
        raise ValueError(f"unsupported a2a-sdk task state: {value!r}") from None


ALLOWED: frozenset[tuple[TaskState, TaskState]] = frozenset(
    {
        (TaskState.SUBMITTED, TaskState.WORKING),
        (TaskState.WORKING, TaskState.INPUT_REQUIRED),
        (TaskState.INPUT_REQUIRED, TaskState.WORKING),
        (TaskState.WORKING, TaskState.COMPLETED),
        (TaskState.WORKING, TaskState.FAILED),
        (TaskState.INPUT_REQUIRED, TaskState.FAILED),
        (TaskState.WORKING, TaskState.CANCELED),
        (TaskState.INPUT_REQUIRED, TaskState.CANCELED),
    }
)

# Reachable only through Restart_Recovery (18.15); `is_allowed(..., recovery=True)`.
RECOVERY_ONLY: frozenset[tuple[TaskState, TaskState]] = frozenset(
    {(TaskState.SUBMITTED, TaskState.FAILED)}
)

TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED}
)

# States after which `execute()` ends its event stream (3.5, 3.7).
STREAM_CLOSING_STATES: frozenset[TaskState] = TERMINAL_STATES | {TaskState.INPUT_REQUIRED}

# States of a Task that still occupies its Context (11.7).
LIVE_STATES: frozenset[TaskState] = frozenset(
    {TaskState.SUBMITTED, TaskState.WORKING, TaskState.INPUT_REQUIRED}
)


def is_allowed(current: TaskState, new: TaskState, *, recovery: bool = False) -> bool:
    """Return True when `current -> new` is a permitted Task_State transition (2.9).

    Only `recovery=True` admits the `RECOVERY_ONLY` pairs; `storage/recovery.py` is the only caller
    that passes it.
    """
    pair = (current, new)
    if pair in ALLOWED:
        return True
    return recovery and pair in RECOVERY_ONLY


# --- Event sinks ------------------------------------------------------------------------------


class EventSink(Protocol):
    """Where the emitter sends events that are already committed.

    The emitter owns *what* is sent and *when*. A sink only delivers. `final` marks the Status_Event
    that ends the SSE stream (terminal states and `input-required`, 3.5). The pinned SDK has no
    `final` field and closes the stream on those states itself, so `QueueSink` ignores the flag.
    """

    async def status(
        self,
        state: TaskState,
        text: str | None,
        metadata: Mapping[str, Any] | None,
        *,
        final: bool,
    ) -> None: ...

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None: ...


class DetachedSink:
    """Drops every event.

    Used while no `execute()`/`cancel()` call is attached, for example when the approval timeout
    fires with the Chat_App away. The state those events describe is already committed, and
    `ProjectedTaskStore` serves get-task and later subscriptions from the committed rows.
    """

    async def status(
        self,
        state: TaskState,
        text: str | None,
        metadata: Mapping[str, Any] | None,
        *,
        final: bool,
    ) -> None:
        return None

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        return None


class QueueSink:
    """Delivers events to an attached SDK `EventQueue` through a `TaskUpdater`."""

    def __init__(self, event_queue: EventQueue, task_id: str, context_id: str) -> None:
        self._updater = TaskUpdater(event_queue, task_id, context_id)

    async def status(
        self,
        state: TaskState,
        text: str | None,
        metadata: Mapping[str, Any] | None,
        *,
        final: bool,
    ) -> None:
        message = None
        if text is not None or metadata:
            parts = [] if text is None else [Part(text=text)]
            message = self._updater.new_agent_message(
                parts, None if metadata is None else dict(metadata)
            )
        await self._updater.update_status(to_sdk(state), message)

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        await self._updater.add_artifact(
            [Part(text=text)], artifact_id=artifact_id, append=append, last_chunk=last_chunk
        )


# --- Tool calls as the emitter sees them -------------------------------------------------------


class ResolvedCall(Protocol):
    """A Tool_Call that resolved to a registered tool. `tool_dispatch.Prepared` satisfies this."""

    @property
    def call(self) -> ToolCall: ...
    @property
    def server(self) -> str: ...
    @property
    def tool(self) -> str: ...
    @property
    def args(self) -> Mapping[str, Any]: ...


class UnresolvedCall(Protocol):
    """A Tool_Call resolved without being sent (4.6). `tool_dispatch.Rejected` satisfies this.

    `server` and `tool` are `None` when the LLM-facing name did not resolve.
    """

    @property
    def call(self) -> ToolCall: ...
    @property
    def server(self) -> str | None: ...
    @property
    def tool(self) -> str | None: ...


KIND_TOOL_STARTED = "tool_call_started"
KIND_TOOL_COMPLETED = "tool_call_completed"
KIND_APPROVAL_REQUIRED = "tool_approval_required"

REPROMPT_TEXT = "Please reply with an explicit 'approve' or 'deny' to continue."


class StorageWriteFailed(Exception):
    """A commit failed and was rolled back (18.30). `reason` is already redacted (16.11)."""

    reason: str

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class ArtifactBuffer:
    """The in-memory text of one response artifact.

    Chunks are forwarded as they arrive, and the database sees the concatenated text once, when the
    artifact closes or the Task reaches a terminal state (18.12).
    """

    artifact_id: str
    ordinal: int
    closed: bool = False
    _chunks: list[str] = field(default_factory=list, init=False, repr=False)

    def append(self, text: str) -> None:
        self._chunks.append(text)

    @property
    def started(self) -> bool:
        """True once at least one chunk was appended."""
        return bool(self._chunks)

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    def record(self) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=self.artifact_id, ordinal=self.ordinal, text=self.text, closed=self.closed
        )


# --- The emitter ------------------------------------------------------------------------------


class TaskEventEmitter:
    """The single path for every Status_Event and Token_Event of one Task.

    Persist before emit: each state change is committed as one `StorageTxn` and only then enqueued
    (11.2, 18.8). The one documented exception is the final `failed` event after two failed commits
    (18.31). Every public coroutine takes one lock, so commits and events keep the order in which
    the calls were made, including for Tool_Calls that run concurrently.

    The emitter starts in `submitted`: the executor's first snapshot creates the Task row.
    """

    def __init__(
        self,
        sessions: SessionStore,
        sink: EventSink,
        redactor: Redactor,
        clock: Clock,
        db_path: str,
        *,
        task_id: str,
        context_id: str,
    ) -> None:
        self._sessions = sessions
        self._sink = sink
        self._redactor = redactor
        self._clock = clock
        self._db_path = db_path
        self._task_id = task_id
        self._context_id = context_id
        self._lock = asyncio.Lock()
        self._state = TaskState.SUBMITTED
        # States the committed row may be in. Wider than `{state}` only after a failed
        # `submitted -> working` commit (18.30).
        self._expected_from: frozenset[TaskState] = frozenset({TaskState.SUBMITTED})
        self._last_text: str | None = None
        self._last_metadata: Mapping[str, Any] | None = None
        self._terminal_emitted = False
        self._write_failed = False
        self._commit_lost_to_cancel = False
        self._approval_metadata: Mapping[str, Any] | None = None
        self._buffers: dict[str, ArtifactBuffer] = {}
        self._started_ids: set[str] = set()
        self._completed_ids: set[str] = set()

    # --- State ------------------------------------------------------------------------------

    @property
    def state(self) -> TaskState:
        """The logical Task_State: the last committed one, or `working` after 18.30."""
        return self._state

    @property
    def sealed(self) -> bool:
        """True after a terminal state. Every later call is a no-op (3.9, 4.9, 14.6)."""
        return self._state in TERMINAL_STATES

    @property
    def commit_lost_to_cancel(self) -> bool:
        """True when a commit failed after its awaiting coroutine was cancelled (18.30).

        No `StorageWriteFailed` reached the caller then, because `CancelledError` took its place.
        """
        return self._commit_lost_to_cancel

    def attach(self, sink: EventSink) -> None:
        """Send later events to `sink` (an `execute()` or `cancel()` call attached)."""
        self._sink = sink

    def detach(self) -> None:
        """Drop later events until the next `attach`."""
        self._sink = DetachedSink()

    def live_artifacts(self) -> list[ArtifactRecord]:
        """Every artifact with the text accumulated so far, in emission order (2.10).

        `closed` is True only once the close commit succeeded.
        """
        return [buffer.record() for buffer in self._buffers.values()]

    # --- Status events ----------------------------------------------------------------------

    async def transition(
        self,
        new: TaskState,
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        with_txn: StorageTxn | None = None,
        emit: bool = True,
    ) -> bool:
        """Move the Task to `new`: commit, then emit. Returns False when the move is not allowed.

        A disallowed request logs `transition_rejected` and changes nothing (2.9). `with_txn` is
        merged into the same commit. `emit=False` commits without enqueuing; follow with
        `emit_current_status()`. Raises `StorageWriteFailed` when the commit fails.
        """
        async with self._lock:
            current = self._state
            if not is_allowed(current, new):
                _log.warning(
                    "transition_rejected",
                    **self._ids(),
                    current_state=current.value,
                    requested_state=new.value,
                )
                return False
            text = self._clean_text(message)
            meta = self._clean_metadata(metadata)
            writes: list[tuple[ArtifactBuffer, bool]] = []
            if new in TERMINAL_STATES:
                # Text of every artifact that never got a last-chunk event (18.12).
                supplied = (
                    set()
                    if with_txn is None
                    else {a.artifact_id for a in with_txn.artifacts if a.task_id == self._task_id}
                )
                writes = [
                    (b, False)
                    for b in self._buffers.values()
                    if not b.closed and b.artifact_id not in supplied
                ]
            txn = self._make_txn(new, text, meta, writes, with_txn)
            earlier_failure = self._write_failed

            def applied() -> None:
                self._state = new
                self._expected_from = frozenset({new})
                self._last_text = text
                self._last_metadata = meta
                self._terminal_emitted = False

            try:
                await self._commit(txn, applied)
            except StorageWriteFailed:
                if current is TaskState.SUBMITTED and new is TaskState.WORKING:
                    self._state = TaskState.WORKING
                    self._expected_from = frozenset({TaskState.SUBMITTED, TaskState.WORKING})
                elif new is TaskState.FAILED and earlier_failure:
                    # The one exception to persist-before-emit (18.31): the row stays at its last
                    # committed state for Restart_Recovery.
                    applied()
                    _log.warning("final_status_unpersisted", **self._ids(), db_path=self._db_path)
                    if emit:
                        await self._emit_status(new, text, meta)
                    return True
                raise
            _log.info(
                "task_state_changed",
                **self._ids(),
                previous_state=current.value,
                new_state=new.value,
            )
            if emit:
                await self._emit_status(new, text, meta)
            return True

    async def status(self, message: str, metadata: Mapping[str, Any] | None = None) -> None:
        """Commit a new status message in the current state, then emit it (18.4, 18.8)."""
        async with self._lock:
            await self._status_locked(message, metadata)

    async def emit_current_status(self) -> None:
        """Enqueue the committed state's Status_Event. Commits nothing."""
        async with self._lock:
            if self.sealed and self._terminal_emitted:
                return
            await self._emit_status(self._state, self._last_text, self._last_metadata)

    async def commit(self, txn: StorageTxn) -> None:
        """Commit a history write made on behalf of this Task: messages or the System_Prompt.

        The orchestrator uses this for every `sessions.append`-style write, so a failed history
        commit gets the same treatment as a failed state change: rollback, a `storage_write_failed`
        log record, `StorageWriteFailed`, and the 18.31 rule for a later failing `failed` commit.
        `txn` must not carry a state change or artifact writes; those go through `transition`.
        """
        async with self._lock:
            await self._commit(txn, lambda: None)

    # --- Token events -----------------------------------------------------------------------

    async def token(self, artifact_id: str, text: str, first: bool) -> None:
        """Buffer a chunk and enqueue its Token_Event. Commits nothing (3.2, 3.3).

        Empty chunks emit nothing. `first` sets the append flag to false.
        """
        if not text:
            return
        async with self._lock:
            if self.sealed:
                return
            buffer = self._buffer(artifact_id)
            if buffer.closed:
                raise ValueError(f"artifact {artifact_id!r} is already closed")
            buffer.append(text)
            await self._emit_artifact(artifact_id, text, append=not first, last_chunk=False)

    async def close_artifact(self, artifact_id: str) -> None:
        """Commit the artifact's full text with `closed=1`, then enqueue the last-chunk marker.

        The marker is one empty-text update with `append=true` and `last_chunk=true`, so it adds
        nothing to the reassembled text (3.4). An artifact that never received a chunk is created
        by the marker, so its `append` flag is false.
        """
        async with self._lock:
            if self.sealed:
                return
            buffer = self._buffer(artifact_id)
            if buffer.closed:
                return
            txn = self._make_txn(None, None, None, [(buffer, True)], None)

            def applied() -> None:
                buffer.closed = True

            await self._commit(txn, applied)
            await self._emit_artifact(artifact_id, "", append=buffer.started, last_chunk=True)

    # --- Tool events ------------------------------------------------------------------------

    async def tool_started(self, call: ResolvedCall, message: str | None = None) -> None:
        """Emit `tool_call_started` before the call is sent to its MCP_Server (4.2, 4.5).

        `message` defaults to `Calling <tool>...`; the dispatcher passes `start_status_message`.
        """
        async with self._lock:
            if self.sealed:
                return
            call_id = call.call.id
            if call_id in self._started_ids:
                raise AssertionError(f"tool_started called twice for {call_id!r}")
            self._started_ids.add(call_id)
            metadata = self._tool_metadata(KIND_TOOL_STARTED, call)
            await self._status_locked(message or f"Calling {call.tool}...", metadata)

    async def tool_completed(self, call: ResolvedCall | UnresolvedCall, ok: bool) -> None:
        """Emit `tool_call_completed` exactly once per call ID (4.3, 4.5, 4.6).

        Raises `AssertionError` on a second call for the same ID. Does nothing after a terminal
        state (4.9).
        """
        async with self._lock:
            if self.sealed:
                return
            call_id = call.call.id
            if call_id in self._completed_ids:
                raise AssertionError(f"tool_completed called twice for {call_id!r}")
            self._completed_ids.add(call_id)
            outcome = "succeeded" if ok else "failed"
            metadata = self._tool_metadata(KIND_TOOL_COMPLETED, call)
            metadata["outcome"] = outcome
            name = call.tool if call.tool is not None else call.call.name
            await self._status_locked(f"Tool {name} {outcome}.", metadata)

    async def approval_required(self, call: ResolvedCall) -> bool:
        """Move to `input-required` with the approval prompt (5.6). Returns False if not allowed."""
        metadata = self._tool_metadata(KIND_APPROVAL_REQUIRED, call)
        metadata["llm_tool_name"] = call.call.name
        metadata["arguments"] = dict(call.args)
        arguments = json.dumps(call.args, ensure_ascii=False, default=str)
        text = (
            f"Approval required to run tool '{call.tool}' (LLM tool name '{call.call.name}') "
            f"on MCP server '{call.server}' with arguments {arguments}. "
            "Reply 'approve' or 'deny'."
        )
        moved = await self.transition(TaskState.INPUT_REQUIRED, text, metadata)
        if moved:
            self._approval_metadata = self._clean_metadata(metadata)
        return moved

    async def reprompt_approval(self) -> None:
        """Ask again for an explicit decision: commit the message, then emit it as final (5.11).

        Does nothing unless the Task is `input-required`.
        """
        async with self._lock:
            if self._state is not TaskState.INPUT_REQUIRED:
                return
            await self._status_locked(REPROMPT_TEXT, self._approval_metadata)

    # --- Internals --------------------------------------------------------------------------

    def _ids(self) -> dict[str, str]:
        return {"context_id": self._context_id, "task_id": self._task_id}

    def _clean_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        return self._redactor.redact(replace_lone_surrogates(text))

    def _clean_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        """Redact every string in `metadata`. A missing mapping becomes `{}`, so the stored
        metadata always belongs to the stored status message."""
        cleaned = self._clean_json(metadata or {})
        assert isinstance(cleaned, dict)
        return cleaned

    def _clean_json(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clean_text(value)
        if isinstance(value, Mapping):
            return {key: self._clean_json(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self._clean_json(item) for item in value]
        return value

    @staticmethod
    def _tool_metadata(kind: str, call: ResolvedCall | UnresolvedCall) -> dict[str, Any]:
        """The machine-readable metadata of a tool event (4.4).

        For a call whose name did not resolve, `tool_name` is the LLM-facing name and `server` is
        null.
        """
        return {
            "kind": kind,
            "tool_name": call.tool if call.tool is not None else call.call.name,
            "server": call.server,
            "tool_call_id": call.call.id,
        }

    def _buffer(self, artifact_id: str) -> ArtifactBuffer:
        buffer = self._buffers.get(artifact_id)
        if buffer is None:
            buffer = ArtifactBuffer(artifact_id=artifact_id, ordinal=len(self._buffers))
            self._buffers[artifact_id] = buffer
        return buffer

    def _make_txn(
        self,
        new: TaskState | None,
        text: str | None,
        metadata: Mapping[str, Any] | None,
        writes: list[tuple[ArtifactBuffer, bool]],
        with_txn: StorageTxn | None,
    ) -> StorageTxn:
        """Build the commit: an optional state change plus artifact texts, after `with_txn`."""
        # Imported here because `session.store` imports `TaskState` from this module.
        from local_agent.session.store import ArtifactWrite, StateChange, StorageTxn

        own = StorageTxn(
            context_id=self._context_id,
            now_ms=self._clock.now_ms(),
            state=(
                None
                if new is None
                else StateChange(
                    task_id=self._task_id,
                    expected_from=self._expected_from,
                    new=new,
                    status_message=text,
                    metadata=metadata,
                )
            ),
            artifacts=tuple(
                ArtifactWrite(
                    task_id=self._task_id,
                    artifact_id=buffer.artifact_id,
                    ordinal=buffer.ordinal,
                    text=buffer.text,
                    closed=closed,
                )
                for buffer, closed in writes
            ),
        )
        return own if with_txn is None else with_txn.merge(own)

    async def _commit(self, txn: StorageTxn, applied: Callable[[], None]) -> None:
        """Commit `txn`, then run `applied` to update the in-memory view.

        `StorageError` and `StaleState` mean the store rolled back: nothing dependent is enqueued,
        and `StorageWriteFailed` carries the redacted reason (18.30). If this coroutine is
        cancelled mid-commit, the commit still finishes in the store, so we wait for its outcome
        and keep our view in step with it before re-raising. A commit that fails in that window is
        not dropped: it is logged and counted like any other (18.30, 18.31) and
        `commit_lost_to_cancel` is set, so the runner can end the Task as `failed`. The
        `CancelledError` is still re-raised, so the cancel proceeds.
        """
        commit = asyncio.ensure_future(self._sessions.commit(txn))
        try:
            await asyncio.shield(commit)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(commit)
            except (StorageError, StaleState) as exc:
                self._record_write_failure(exc)
                self._commit_lost_to_cancel = True
            except Exception:
                pass  # the commit failed, so there is nothing to apply
            else:
                applied()
            raise
        except (StorageError, StaleState) as exc:
            raise self._record_write_failure(exc) from exc
        applied()

    def _record_write_failure(self, exc: BaseException) -> StorageWriteFailed:
        """Log a rolled-back commit (18.30, 16.11) and return the error to raise for it."""
        self._write_failed = True
        reason = self._redactor.redact(str(exc))
        _log.error("storage_write_failed", **self._ids(), db_path=self._db_path, reason=reason)
        return StorageWriteFailed(reason)

    async def _status_locked(self, message: str, metadata: Mapping[str, Any] | None) -> None:
        """Same-state status: commit the message, then emit it. Caller holds the lock."""
        if self.sealed:
            return
        state = self._state
        text = self._clean_text(message)
        meta = self._clean_metadata(metadata)
        txn = self._make_txn(state, text, meta, [], None)

        def applied() -> None:
            self._expected_from = frozenset({state})
            self._last_text = text
            self._last_metadata = meta
            self._terminal_emitted = False

        await self._commit(txn, applied)
        await self._emit_status(state, text, meta)

    async def _emit_status(
        self, state: TaskState, text: str | None, metadata: Mapping[str, Any] | None
    ) -> None:
        final = state in STREAM_CLOSING_STATES
        if state in TERMINAL_STATES:
            self._terminal_emitted = True
        try:
            await self._sink.status(state, text, metadata or None, final=final)
        except Exception as exc:
            self._log_sink_failure("status", exc)

    async def _emit_artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        try:
            await self._sink.artifact(artifact_id, text, append=append, last_chunk=last_chunk)
        except Exception as exc:
            self._log_sink_failure("artifact", exc)

    def _log_sink_failure(self, kind: str, exc: Exception) -> None:
        # The state is already committed, so a sink that cannot deliver (for example a closed
        # queue after the Chat_App left) must not stop the Task (3.6).
        _log.warning(
            "event_sink_failed",
            **self._ids(),
            sink_call=kind,
            reason=self._redactor.redact(f"{type(exc).__name__}: {exc}"),
        )
