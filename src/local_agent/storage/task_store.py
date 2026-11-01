"""The A2A `TaskStore` as a projection of the Session_Store (2.10, 2.11, 18.1, 18.2, 18.7-18.9).

The `a2a-sdk` request handler wants a `TaskStore`. Rather than keep a second copy of every Task,
this class answers the SDK from the rows the Session_Store already holds. The Backend's event
emitter is the only writer of Task_State, status message, and artifacts, and it writes them
through `SessionStore.commit`. So one commit updates both views (18.8), and they cannot disagree
after a kill (18.9): there is only one copy.

What the SDK may still write through `save()`:

* **The first snapshot of a Task.** An unknown `task_id` saved in `submitted` creates the Context
  (if absent) and the Task through `SessionStore.create_task`. The SDK saves this snapshot before
  it forwards the event, so creation is persist-before-emit (18.30). A failed write raises, so no
  event goes out and no row exists.
* **The SDK's own `history` and `metadata`.** For a known Task they go into the private
  `a2a_snapshot` column, only when they changed. State, status, and artifacts in the saved `Task`
  are never used.

The class depends only on the `SessionStore` ABC and the small `LiveArtifacts` protocol, so
`storage.backend: memory` gets the same class over `InMemorySessionStore` (18.2).
"""

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol

from a2a.types import (
    Artifact,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    Task,
    TaskStatus,
)
from google.protobuf import json_format

from local_agent.a2a_server.a2a_compat import (
    InvalidParamsError,
    ServerCallContext,
    TaskNotFoundError,
    TaskStore,
)
from local_agent.a2a_server.a2a_compat import TaskState as SdkTaskState
from local_agent.a2a_server.errors import StorageWriteError
from local_agent.a2a_server.events import TERMINAL_STATES, TaskState, from_sdk
from local_agent.logging_setup import get_logger
from local_agent.models import ArtifactRecord, Clock, SystemClock, TaskRecord
from local_agent.session.store import SessionStore, StorageError, TaskPage, TaskQuery

__all__ = ["LiveArtifacts", "ProjectedTaskStore", "snapshot_json"]

_log = get_logger("storage.task_store")

# A2A's default `ListTasks` page size, and the page size used to count matches for `total_size`.
_DEFAULT_PAGE_SIZE = 50
_COUNT_PAGE_SIZE = 500
# How many failed creation IDs `ProjectedTaskStore` remembers. IDs are unique per request and the
# SDK's repeat save follows the failure at once, so a small window is enough.
_FAILED_CREATIONS_LIMIT = 256


class LiveArtifacts(Protocol):
    """Where the Task_Store finds artifact text that is buffered but not yet committed (2.10).

    `TaskRunnerRegistry` implements it. It returns every artifact of the Task's live runner with
    the text accumulated so far, in emission order, or an empty sequence when the Task has no
    runner.
    """

    def live_artifacts(self, task_id: str) -> Sequence[ArtifactRecord]: ...


def snapshot_json(task: Task) -> str:
    """Serialize the part of `task` the Task_Store keeps privately: SDK `history` and `metadata`.

    Status and artifacts are dropped on purpose; they come from the Session_Store rows. Keys are
    sorted so equal content gives equal text, which lets `save` skip an unchanged snapshot.
    """
    kept = Task()
    kept.history.extend(task.history)
    if task.HasField("metadata"):
        kept.metadata.CopyFrom(task.metadata)
    return str(json_format.MessageToJson(kept, indent=None, sort_keys=True))


def _state_name(state: SdkTaskState) -> str:
    try:
        return str(SdkTaskState.Name(state))
    except ValueError:
        return str(state)


def _is_terminal(state: SdkTaskState) -> bool:
    try:
        return from_sdk(state) in TERMINAL_STATES
    except ValueError:
        return False


class ProjectedTaskStore(TaskStore):
    """`TaskStore` over a `SessionStore`. Tenant and owner scoping are ignored: single user."""

    def __init__(
        self,
        sessions: SessionStore,
        live: LiveArtifacts | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._sessions = sessions
        self._live = live
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._pending: dict[str, asyncio.Future[None]] = {}
        # IDs whose creation commit failed, oldest first. The SDK's producer cleanup saves the
        # SUBMITTED Task again after the failure; that save must not create a ghost row (18.30).
        self._failed_creations: OrderedDict[str, None] = OrderedDict()

    # --- Creation hand-off ------------------------------------------------------------------

    def expect_creation(self, task_id: str) -> asyncio.Future[None]:
        """Return a future that `save()` resolves once the Task's creation commit is done.

        It holds `None` on success and the `StorageError` when the commit failed. The executor
        calls this before it enqueues the first snapshot, then waits for the future before it
        starts the runner (18.30). Must be called from the running event loop.
        """
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending[task_id] = future

        def forget(done: asyncio.Future[None]) -> None:
            # Drop the entry when the waiter gave up (cancelled or timed out) as well.
            if self._pending.get(task_id) is done:
                del self._pending[task_id]

        future.add_done_callback(forget)
        return future

    # --- TaskStore --------------------------------------------------------------------------

    async def save(self, task: Task, context: ServerCallContext) -> None:
        stored = await self._sessions.get_a2a_snapshot(task.id)
        record: TaskRecord | None = None
        if stored is None:
            # Either an unknown Task or a known one without a snapshot. One read tells them apart.
            record = await self._sessions.get_task(task.id)
            if record is None:
                await self._save_unknown(task)
                return

        incoming = snapshot_json(task)
        if _is_terminal(task.status.state):
            record = record if record is not None else await self._sessions.get_task(task.id)
            if record is None:
                return  # removed by retention while this save was in flight
            self._check_divergence(task, record)
        if incoming != stored:
            await self._store_snapshot(task, incoming)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        record = await self._sessions.get_task(task_id)
        return None if record is None else await self._build(record)

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        if params.status:
            try:
                state: TaskState | None = from_sdk(params.status)
            except ValueError:
                # A state this Backend never uses: no Task can match it.
                return ListTasksResponse(page_size=params.page_size or _DEFAULT_PAGE_SIZE)
        else:
            state = None
        after_ms: int | None = None
        if params.HasField("status_timestamp_after"):
            # The A2A filter is inclusive and the store's is exclusive: ceil to milliseconds,
            # then step one back.
            after_ns = params.status_timestamp_after.ToNanoseconds()
            after_ms = -(-after_ns // 1_000_000) - 1
        page_size = max(1, params.page_size or _DEFAULT_PAGE_SIZE)
        query = TaskQuery(
            context_id=params.context_id or None,
            state=state,
            updated_after_ms=after_ms,
            page_size=page_size,
            page_token=params.page_token or None,
        )
        try:
            page = await self._sessions.list_tasks(query)
        except ValueError:
            raise InvalidParamsError(message="Invalid page token") from None

        total = (
            len(page.tasks)
            if page.next_page_token is None and query.page_token is None
            else await self._count(query)
        )
        return ListTasksResponse(
            tasks=[await self._build(record) for record in page.tasks],
            next_page_token=page.next_page_token,
            page_size=page_size,
            total_size=total,
        )

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Delete one Task and its artifacts. The Context and its history stay."""
        await self._sessions.delete_task(task_id)

    async def snapshot(self, task_id: str) -> Task:
        """The Task as committed now, for the executor's first event (3.10, 3.12, 3.13).

        Raises `TaskNotFoundError` when the Task is absent.
        """
        record = await self._sessions.get_task(task_id)
        if record is None:
            raise TaskNotFoundError
        return await self._build(record)

    # --- save() helpers ---------------------------------------------------------------------

    async def _save_unknown(self, task: Task) -> None:
        if task.status.state != SdkTaskState.TASK_STATE_SUBMITTED:
            # The SDK's bookkeeping for a request whose creation failed (18.30): store nothing.
            _log.debug(
                "task_store_unknown_task_ignored",
                task_id=task.id,
                state=_state_name(task.status.state),
            )
            return
        if task.id in self._failed_creations:
            # The SDK's cleanup after a failed creation commit: do not create the Task now.
            _log.debug("task_store_failed_creation_ignored", task_id=task.id)
            return
        future = self._pending.get(task.id)
        try:
            await self._sessions.create_task(
                task.id, task.context_id, self._clock.now_ms(), snapshot_json(task)
            )
        except StorageError as exc:
            self._failed_creations[task.id] = None
            while len(self._failed_creations) > _FAILED_CREATIONS_LIMIT:
                self._failed_creations.popitem(last=False)
            if future is not None and not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved: the waiter may already be gone
            raise StorageWriteError from exc
        if future is not None and not future.done():
            future.set_result(None)

    def _check_divergence(self, task: Task, record: TaskRecord) -> None:
        if task.status.state == record.state:
            return
        if _is_terminal(task.status.state):
            _log.warning(
                "task_store_divergent_save",
                task_id=task.id,
                context_id=record.context_id,
                saved_state=_state_name(task.status.state),
                committed_state=_state_name(record.state),
            )

    async def _store_snapshot(self, task: Task, snapshot: str) -> None:
        try:
            await self._sessions.set_a2a_snapshot(task.id, snapshot)
        except StorageError as exc:
            # Only the SDK's history/metadata copy is lost. The Backend's own commits are
            # unaffected, so the Task keeps running (3.6); the next save retries.
            _log.error(
                "task_store_snapshot_write_failed",
                task_id=task.id,
                context_id=task.context_id,
                reason=str(exc),
            )

    # --- list() / get() helpers -------------------------------------------------------------

    async def _count(self, query: TaskQuery) -> int:
        """Count every Task matching `query`'s filters, ignoring its page position."""
        total = 0
        token: str | None = None
        while True:
            page: TaskPage = await self._sessions.list_tasks(
                TaskQuery(
                    context_id=query.context_id,
                    state=query.state,
                    updated_after_ms=query.updated_after_ms,
                    page_size=_COUNT_PAGE_SIZE,
                    page_token=token,
                )
            )
            total += len(page.tasks)
            if page.next_page_token is None:
                return total
            token = page.next_page_token

    async def _build(self, record: TaskRecord) -> Task:
        """Build the A2A `Task` proto from a committed row plus the SDK's private snapshot."""
        task = _parse_snapshot(await self._sessions.get_a2a_snapshot(record.task_id))
        task.id = record.task_id
        task.context_id = record.context_id

        status = TaskStatus(state=record.state)
        status.timestamp.FromMilliseconds(record.updated_at_ms)
        message = _status_message(record)
        if message is not None:
            status.message.CopyFrom(message)
        task.status.CopyFrom(status)

        for artifact in self._artifacts(record):
            task.artifacts.append(
                Artifact(artifact_id=artifact.artifact_id, parts=[Part(text=artifact.text)])
            )
        return task

    def _artifacts(self, record: TaskRecord) -> Sequence[ArtifactRecord]:
        """Committed artifacts, replaced by live buffers for a Task that is still running."""
        merged = {artifact.artifact_id: artifact for artifact in record.artifacts}
        if self._live is not None and not _is_terminal(record.state):
            # A live buffer holds at least the committed text, so it wins (2.10).
            for artifact in self._live.live_artifacts(record.task_id):
                merged[artifact.artifact_id] = artifact
        return sorted(merged.values(), key=lambda artifact: artifact.ordinal)


def _status_message(record: TaskRecord) -> Message | None:
    """The agent message of the status: the status text and the event metadata, if any."""
    if record.status_message is None and not record.metadata:
        return None
    message = Message(
        message_id=f"{record.task_id}:status:{record.updated_at_ms}",
        role=Role.ROLE_AGENT,
        context_id=record.context_id,
        task_id=record.task_id,
    )
    if record.status_message is not None:
        message.parts.append(Part(text=record.status_message))
    if record.metadata:
        message.metadata.update(dict(record.metadata))
    return message


def _parse_snapshot(raw: str | None) -> Task:
    """Read the private snapshot back. A missing or unreadable one gives an empty Task."""
    task = Task()
    if raw:
        try:
            json_format.Parse(raw, task, ignore_unknown_fields=True)
        except json_format.ParseError as exc:
            _log.warning("task_store_snapshot_unreadable", reason=str(exc))
            return Task()
    return task
