"""The `a2a-sdk` request handler and agent executor of the Backend (design.md, "executor.py").

`HostRequestHandler` is a `DefaultRequestHandler` subclass. It adds the checks the SDK does not
make: the strict `A2A-Version` gate (1.12), Context admission (1.14, 1.15, 2.14, 11.7), and a cancel
path that never lets the SDK write `CANCELED` straight into the store (2.11, 2.12, 18.17).

`HostAgentExecutor` adapts the SDK's `execute`/`cancel` calls to a `TaskRunner`. The runner owns the
reasoning loop in its own `asyncio.Task`, so `execute` only creates or finds the runner, attaches
the request's `EventQueue`, and waits until a stream-closing state has been sent (3.5, 3.6).

Admission errors (`ContextBusyError`, `TaskNotAwaitingInputError`) are stored in
`call_context.state["host_contract_error"]` before they are raised. `HostJsonRpcDispatcher`
(`dispatch.py`) reads that key to render the exact JSON-RPC body.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, MutableMapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Final

from a2a.server.events import Event
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskStatus,
)

from local_agent.a2a_server.a2a_compat import (
    AgentExecutor,
    DefaultRequestHandler,
    EventQueue,
    InternalError,
    RequestContext,
    ServerCallContext,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskStore,
    VersionNotSupportedError,
    validate_proto_required_fields,
)
from local_agent.a2a_server.dispatch import HOST_CONTRACT_ERROR_KEY
from local_agent.a2a_server.errors import (
    ContextBusyError,
    HostContractError,
    StorageWriteError,
    TaskNotAwaitingInputError,
    accepts_version,
)
from local_agent.a2a_server.events import (
    TERMINAL_STATES,
    StorageWriteFailed,
    TaskState,
    from_sdk,
    to_sdk,
)
from local_agent.a2a_server.sdk_log import install_active_task_log_filter
from local_agent.logging_setup import get_logger
from local_agent.models import Clock, SystemClock, TaskRecord
from local_agent.orchestrator.approval_gate import ApprovalKeywordSet, parse_approval_decision
from local_agent.orchestrator.loop import AdmissionReservation, TaskRunner, TaskRunnerRegistry
from local_agent.session.store import SessionStore
from local_agent.storage import StorageError
from local_agent.storage.task_store import ProjectedTaskStore

__all__ = ["CREATION_TIMEOUT_S", "HostAgentExecutor", "HostRequestHandler"]

_log = get_logger("a2a_server.executor")

CREATION_TIMEOUT_S: Final = 5.0
"""How long `execute` waits for the creation commit of a new Task (18.30)."""

VERSION_HEADER: Final = "a2a-version"
"""The `A2A-Version` header, lower-cased for the case-insensitive lookup."""

_MAX_ADMISSION_ATTEMPTS: Final = 3
"""Rounds of the admission check when a reservation vanishes while a request waits on it."""

_MAX_REPORTED_VERSION_CHARS: Final = 64


def _record_contract_error[E: HostContractError](state: MutableMapping[str, Any], error: E) -> E:
    """Store `error` where `HostJsonRpcDispatcher` finds it, and return it to be raised."""
    state[HOST_CONTRACT_ERROR_KEY] = error
    return error


def _version_header(headers: object) -> str | None:
    """The `A2A-Version` value of the request headers, or `None` when the header is absent.

    The SDK stores `dict(request.headers)`, whose keys are lower-case, but a test or another ASGI
    layer may pass any case. A present header with an empty value is still present.
    """
    if not isinstance(headers, Mapping):
        return None
    for name, value in headers.items():
        if isinstance(name, str) and name.casefold() == VERSION_HEADER:
            return str(value)
    return None


@dataclass(slots=True)
class _ContextLock:
    """One Context's admission lock and the number of requests holding or waiting for it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


# --- Request handler ----------------------------------------------------------------------------


class HostRequestHandler(DefaultRequestHandler):  # DefaultRequestHandlerV2 in the 1.2.x line
    """Adds the version gate and Context admission in front of the SDK handler.

    It reads the Session_Store and the runner registry from the `HostAgentExecutor` it is built
    with, so `HostRequestHandler(HostAgentExecutor(...), task_store=..., agent_card=...)` is the
    whole wiring.
    """

    def __init__(
        self,
        agent_executor: HostAgentExecutor,
        task_store: TaskStore,
        agent_card: AgentCard,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_executor, task_store, agent_card, **kwargs)
        self._sessions: SessionStore = agent_executor.sessions
        self._runners: TaskRunnerRegistry = agent_executor.runners
        self._context_locks: dict[str, _ContextLock] = {}
        install_active_task_log_filter()

    # --- Version gate -------------------------------------------------------------------------

    def _check_version(self, context: ServerCallContext) -> None:
        """Reject a present `A2A-Version` header that is not 1.0 (1.12).

        An absent header is left to the SDK's own negotiation (1.13).
        """
        header = _version_header(context.state.get("headers"))
        if header is None or accepts_version(header):
            return
        shown = header[:_MAX_REPORTED_VERSION_CHARS]
        raise VersionNotSupportedError(
            message=f"A2A version {shown!r} is not supported. This server speaks A2A 1.0."
        )

    # --- Admission ----------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _context_lock(self, context_id: str) -> AsyncIterator[None]:
        """The admission lock of one Context. Entries are dropped when nobody uses them."""
        entry = self._context_locks.get(context_id)
        if entry is None:
            entry = self._context_locks[context_id] = _ContextLock()
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0 and self._context_locks.get(context_id) is entry:
                del self._context_locks[context_id]

    async def _admit(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> AdmissionReservation | None:
        """Admit or reject a message before the SDK sees it.

        Returns the Context's admission reservation for a new Task whose `context_id` the client
        chose, and `None` otherwise. The caller holds the reservation for the length of the
        request. A rejection is stored in `context.state["host_contract_error"]` and raised.
        """
        try:
            return await self._admit_message(params.message)
        except HostContractError as error:
            _record_contract_error(context.state, error)
            raise

    async def _admit_message(self, message: Message) -> AdmissionReservation | None:
        if message.task_id:
            await self._check_continuation(message.task_id)
            return None
        context_id = message.context_id
        if not context_id:
            # The SDK will mint a fresh Context ID, so no other request can hold it yet.
            return None
        for _ in range(_MAX_ADMISSION_ATTEMPTS):
            async with self._context_lock(context_id):
                held = self._runners.reservation(context_id)
                if held is None:
                    active = await self._sessions.active_task(context_id)
                    if active is not None:
                        raise ContextBusyError(context_id, active)  # 1.14, 11.7
                    # Unknown, retention-deleted, or persisted by an earlier process: admitted
                    # (2.2, 2.3, 2.13, 11.14, 18.6).
                    return self._runners.reserve(context_id)
            # Another request is between "admitted" and "creation committed". Wait for its
            # `task_id` outside the lock, so other requests for this Context are not queued
            # behind the wait.
            await self._runners.reserved_task_id(context_id)
            if held.released:
                continue  # the earlier request finished or failed; committed state decides now
            raise ContextBusyError(context_id, held.task_id or "")
        raise ContextBusyError(context_id, await self._sessions.active_task(context_id) or "")

    async def _check_continuation(self, task_id: str) -> None:
        """Reject a continuation for a Task that is not waiting for input (1.15, 2.14).

        An unknown or terminal Task is left to the SDK (2.8, 2.11, 18.17).
        """
        record = await self._sessions.get_task(task_id)
        if record is not None and from_sdk(record.state) in (
            TaskState.SUBMITTED,
            TaskState.WORKING,
        ):
            raise TaskNotAwaitingInputError(task_id, record.state)

    @staticmethod
    def _held(reservation: AdmissionReservation | None) -> AbstractContextManager[Any]:
        return reservation if reservation is not None else contextlib.nullcontext()

    # --- Messages -----------------------------------------------------------------------------

    async def on_message_send(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> Message | Task:
        self._check_version(context)
        reservation = await self._admit(params, context)
        with self._held(reservation):
            result: Message | Task = await super().on_message_send(params, context)
            return result

    async def on_message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event]:
        self._check_version(context)
        reservation = await self._admit(params, context)
        with self._held(reservation):
            async with contextlib.aclosing(
                super().on_message_send_stream(params, context)
            ) as events:
                async for event in events:
                    yield event

    # --- Cancel -------------------------------------------------------------------------------

    async def on_cancel_task(self, params: CancelTaskRequest, context: ServerCallContext) -> Task:
        """Check the Session_Store before the SDK can write `CANCELED` into the store itself.

        The SDK cancels a Task that has no local `ActiveTask` by saving `CANCELED` directly, which
        would bypass the runner (design.md, "Cancel").
        """
        self._check_version(context)
        # The SDK validates required fields in a decorator on its own `on_cancel_task`, which runs
        # after this lookup. Without this call a missing `id` reads as an unknown Task (1.9).
        validate_proto_required_fields(params)
        record = await self._sessions.get_task(params.id)
        if record is None:
            raise TaskNotFoundError  # 2.11, 11.14
        state = from_sdk(record.state)
        if state in TERMINAL_STATES:
            raise TaskNotCancelableError  # 2.12, 18.17
        if self._runners.get(params.id) is None:
            # Restart_Recovery leaves no non-terminal Task, so this cannot happen in one process.
            _log.error(
                "cancel_without_runner",
                context_id=record.context_id,
                task_id=record.task_id,
                state=state.value,
            )
            raise TaskNotCancelableError
        result: Task = await super().on_cancel_task(params, context)
        return result

    # --- Reads --------------------------------------------------------------------------------

    async def on_get_task(self, params: GetTaskRequest, context: ServerCallContext) -> Task | None:
        self._check_version(context)
        result: Task | None = await super().on_get_task(params, context)
        return result

    async def on_list_tasks(
        self, params: ListTasksRequest, context: ServerCallContext
    ) -> ListTasksResponse:
        self._check_version(context)
        result: ListTasksResponse = await super().on_list_tasks(params, context)
        return result

    async def on_subscribe_to_task(
        self, params: SubscribeToTaskRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event]:
        self._check_version(context)
        async with contextlib.aclosing(super().on_subscribe_to_task(params, context)) as events:
            async for event in events:
                yield event


# --- Agent executor -----------------------------------------------------------------------------


class HostAgentExecutor(AgentExecutor):
    """Adapts SDK `execute`/`cancel` calls to the `TaskRunner` of the Task."""

    def __init__(
        self,
        *,
        sessions: SessionStore,
        runners: TaskRunnerRegistry,
        task_store: ProjectedTaskStore,
        approval_keywords: ApprovalKeywordSet,
        clock: Clock | None = None,
        creation_timeout_s: float = CREATION_TIMEOUT_S,
    ) -> None:
        self.sessions = sessions
        self.runners = runners
        self._task_store = task_store
        self._keywords = approval_keywords
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._creation_timeout_s = creation_timeout_s

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id, message = context.task_id, context.context_id, context.message
        if task_id is None or context_id is None or message is None:
            raise InternalError(message="The request carries no message or Task identifiers")
        # The SDK's v2 request builder passes `task=None`, so the executor looks the Task up.
        record = await self.sessions.get_task(task_id)
        if record is None:
            await self._start_task(task_id, context_id, message, event_queue)
        else:
            await self._continue_task(context, record, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Attach the cancel request's queue and stop the runner (2.7, 14.4)."""
        runner = None if context.task_id is None else self.runners.get(context.task_id)
        if runner is None:
            _log.error("cancel_without_runner", task_id=context.task_id)
            raise TaskNotCancelableError
        runner.attach(event_queue)
        await runner.cancel()

    # --- New Task -----------------------------------------------------------------------------

    async def _start_task(
        self, task_id: str, context_id: str, message: Message, event_queue: EventQueue
    ) -> None:
        """Create the runner, commit the Task through the first snapshot, then start the runner."""
        runner = self.runners.create(task_id, context_id, message)  # claims the reservation
        created = self._task_store.expect_creation(task_id)
        try:
            runner.attach(event_queue)
            # The SDK saves this snapshot, which is the creation commit, before it forwards it
            # (3.10, 3.11, 18.30).
            await event_queue.enqueue_event(self._submitted_snapshot(task_id, context_id))
            await self._await_creation(task_id, created)
        except BaseException:
            # The runner was never started, so nothing else will remove it (18.30).
            if not created.done():
                created.cancel()
            self.runners.discard(task_id)
            raise
        finally:
            self._release_reservation(context_id, task_id)
        runner.start()
        await runner.wait_stream_closing()

    async def _await_creation(self, task_id: str, created: asyncio.Future[None]) -> None:
        try:
            await asyncio.wait_for(created, self._creation_timeout_s)
        except TimeoutError:
            _log.error(
                "task_creation_timed_out", task_id=task_id, timeout_s=self._creation_timeout_s
            )
            raise StorageWriteError from None
        except StorageError as exc:
            raise StorageWriteError from exc

    def _submitted_snapshot(self, task_id: str, context_id: str) -> Task:
        status = TaskStatus(state=to_sdk(TaskState.SUBMITTED))
        status.timestamp.FromMilliseconds(self._clock.now_ms())
        return Task(id=task_id, context_id=context_id, status=status)

    def _release_reservation(self, context_id: str, task_id: str) -> None:
        """Release the Context's admission reservation once creation succeeded or failed."""
        reservation = self.runners.reservation(context_id)
        if reservation is not None and reservation.task_id == task_id:
            reservation.release()

    # --- Continuation -------------------------------------------------------------------------

    async def _continue_task(
        self, context: RequestContext, record: TaskRecord, event_queue: EventQueue
    ) -> None:
        """Resume an `input-required` Task from the user's approval decision (5.8, 5.11)."""
        task_id = record.task_id
        message = context.message
        assert message is not None  # checked by `execute`
        state = context.call_context.state
        if from_sdk(record.state) is not TaskState.INPUT_REQUIRED:
            raise _record_contract_error(state, TaskNotAwaitingInputError(task_id, record.state))
        runner = self.runners.get(task_id)
        if runner is None:
            _log.error("continuation_without_runner", task_id=task_id, context_id=record.context_id)
            raise InternalError(message="The Task is no longer running")
        decision = parse_approval_decision(message, self._keywords)
        runner.attach(event_queue)
        try:
            if decision is None:
                await self._reprompt(runner, event_queue)
            else:
                moved = await runner.emitter.transition(TaskState.WORKING, emit=False)
                if not moved:
                    # The approval timeout or a concurrent decision moved the Task first.
                    raise _record_contract_error(
                        state, TaskNotAwaitingInputError(task_id, to_sdk(runner.emitter.state))
                    )
                try:
                    # `working` is committed, so this snapshot never reads `input-required`
                    # (3.12). The Status_Event follows it.
                    await event_queue.enqueue_event(await self._task_store.snapshot(task_id))
                    await runner.emitter.emit_current_status()
                finally:
                    # The Task is `working` now, so the gate must be released even when sending
                    # the events failed.
                    runner.approval_gate.resolve(decision)
        except StorageWriteFailed as exc:
            runner.detach()
            raise StorageWriteError from exc
        except Exception:
            runner.detach()
            raise
        await runner.wait_stream_closing()

    async def _reprompt(self, runner: TaskRunner, event_queue: EventQueue) -> None:
        """Answer an unrecognized reply: the `input-required` snapshot, then the final re-prompt.

        The reply is not appended to the Message_History and the pending Tool_Call is untouched
        (5.11, 3.13).
        """
        await event_queue.enqueue_event(await self._task_store.snapshot(runner.task_id))
        await runner.emitter.reprompt_approval()
