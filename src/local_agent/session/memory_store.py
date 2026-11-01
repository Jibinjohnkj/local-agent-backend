"""In-memory Session_Store (`storage.backend: memory`, and the store most tests use).

The story of a write: the Orchestrator builds one pure `StorageTxn`. `commit` first walks the
whole value and checks every precondition without touching any state. Only when everything passes
does it apply the writes. There is no `await` between the two phases, and the event loop only
switches tasks at an `await`, so no other coroutine can see a half-applied transaction. That is
the same all-or-nothing contract the SQLite store gets from `BEGIN IMMEDIATE ... COMMIT` (11.10).

Rules this store settles where the interface (`session/store.py`) is silent:

* **Task_State types.** `models.TaskRecord.state` is the `a2a-sdk` enum, while `StateChange`,
  `TaskQuery`, and the Orchestrator use the internal `events.TaskState`. This store keeps the
  internal enum in its own records and converts with `events.to_sdk` only when it builds a
  `TaskRecord` for a reader. Nothing else in the store touches SDK enum values.
* **Unknown Context or Task.** `commit` never creates a Context. Only `create_task` does. A
  transaction that names an unknown Context, or a Task that is unknown or belongs to another
  Context, fails with `StorageError`. This mirrors the foreign keys in SQLite and keeps Contexts
  isolated (11.6).
* **Appends.** The producing `task_id` of an appended message is validated and then dropped,
  because no reader needs it yet. A `SystemMessage` in `appends` is rejected: the System_Prompt
  has its own slot (12.7).
* **StateChange.** `status_message` always replaces the stored one (it belongs to the new status).
  `metadata=None` leaves the stored metadata alone, any mapping replaces it. Every `StateChange`,
  including a same-state one, sets the Task's `updated_at_ms`. Only a change to a different state
  counts as Last_Activity.
* **Artifacts.** A write replaces the artifact with the same `artifact_id`. Two different
  `artifact_id`s in one Task may not share an `ordinal`. Artifact writes do not touch
  `updated_at_ms` or Last_Activity.
* **Paging.** `updated_after_ms` is exclusive. The cursor is keyset based: it encodes the
  `(updated_at_ms, task_id)` of the last row returned, so a page does not shift when Tasks are
  updated between calls. A malformed or foreign cursor raises `ValueError`, and so does
  `page_size < 1`, because those are caller mistakes, not storage failures.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from local_agent.a2a_server.events import LIVE_STATES, TaskState, to_sdk
from local_agent.models import (
    ArtifactRecord,
    ChatMessage,
    Clock,
    ContextRecord,
    SystemMessage,
    TaskRecord,
)
from local_agent.session.store import (
    SessionStore,
    StaleState,
    StorageError,
    StorageTxn,
    TaskPage,
    TaskQuery,
    Unset,
    decode_page_token,
    encode_page_token,
)
from local_agent.storage.retention import ContextActivity, select_expired

__all__ = ["InMemorySessionStore"]

_STORE_NAME = "memory"


@dataclass
class _TaskEntry:
    """A Task as this store holds it. Mutable, private, and never handed to callers."""

    task_id: str
    context_id: str
    state: TaskState
    status_message: str | None
    metadata: dict[str, Any]
    created_at_ms: int
    updated_at_ms: int
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)  # keyed by artifact_id
    a2a_snapshot: str | None = None


class InMemorySessionStore(SessionStore):
    """Session_Store that keeps everything in process memory and loses it on exit."""

    def __init__(self, clock: Clock | None = None) -> None:
        super().__init__(clock)
        self._contexts: dict[str, ContextRecord] = {}
        self._tasks: dict[str, _TaskEntry] = {}
        # context_id -> task_ids in creation order, so `active_task` need not scan every Task.
        self._context_tasks: dict[str, list[str]] = {}

    # --- Writes ----------------------------------------------------------------------------

    async def commit(self, txn: StorageTxn) -> None:
        # No `await` anywhere below: validate everything, then apply everything, atomically.
        context = self._contexts.get(txn.context_id)
        if context is None:
            raise StorageError(_STORE_NAME, f"unknown Context {txn.context_id!r}")
        self._validate(txn)
        self._apply(txn, context)

    async def create_task(
        self, task_id: str, context_id: str, now_ms: int, a2a_snapshot: str | None
    ) -> bool:
        if task_id in self._tasks:
            raise StorageError(_STORE_NAME, f"Task {task_id!r} already exists")
        context = self._contexts.get(context_id)
        context_is_new = context is None
        if context is None:
            context = ContextRecord(
                context_id=context_id,
                system_prompt=None,
                history=[],
                created_at_ms=now_ms,
                last_activity_ms=now_ms,
            )
            self._contexts[context_id] = context
            self._context_tasks[context_id] = []
        context.last_activity_ms = now_ms
        self._tasks[task_id] = _TaskEntry(
            task_id=task_id,
            context_id=context_id,
            state=TaskState.SUBMITTED,
            status_message=None,
            metadata={},
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
            a2a_snapshot=a2a_snapshot,
        )
        self._context_tasks[context_id].append(task_id)
        return context_is_new

    async def set_a2a_snapshot(self, task_id: str, snapshot: str) -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            raise StorageError(_STORE_NAME, f"unknown Task {task_id!r}")
        entry.a2a_snapshot = snapshot

    async def get_a2a_snapshot(self, task_id: str) -> str | None:
        entry = self._tasks.get(task_id)
        return None if entry is None else entry.a2a_snapshot

    async def delete_task(self, task_id: str) -> None:
        entry = self._tasks.pop(task_id, None)
        if entry is not None:
            self._context_tasks[entry.context_id].remove(task_id)

    async def delete_expired(self, cutoff_ms: int) -> list[str]:
        # `last_activity_ms < cutoff_ms` is `select_expired` with now = cutoff and a zero timeout.
        # No `await` here, so the select and the delete are one atomic step (11.11).
        activity = [
            ContextActivity(
                context_id=context.context_id,
                last_activity_ms=context.last_activity_ms,
                has_live_task=any(
                    self._tasks[task_id].state in LIVE_STATES
                    for task_id in self._context_tasks.get(context.context_id, ())
                ),
            )
            for context in self._contexts.values()
        ]
        expired = select_expired(activity, cutoff_ms, 0)
        for context_id in expired:
            del self._contexts[context_id]
            for task_id in self._context_tasks.pop(context_id, ()):
                del self._tasks[task_id]
        return expired

    # --- Reads -----------------------------------------------------------------------------

    async def get_history(self, context_id: str) -> tuple[SystemMessage | None, list[ChatMessage]]:
        context = self._contexts.get(context_id)
        if context is None:
            return None, []
        # A fresh list, so the caller cannot grow the stored history by accident.
        return context.system_prompt, list(context.history)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        entry = self._tasks.get(task_id)
        return None if entry is None else self._to_record(entry)

    async def list_tasks(self, query: TaskQuery) -> TaskPage:
        if query.page_size < 1:
            raise ValueError(f"page_size must be at least 1, got {query.page_size}")
        after = decode_page_token(query.page_token) if query.page_token is not None else None

        matching = [entry for entry in self._tasks.values() if self._matches(entry, query)]
        matching.sort(key=lambda entry: (entry.updated_at_ms, entry.task_id), reverse=True)
        if after is not None:
            matching = [entry for entry in matching if (entry.updated_at_ms, entry.task_id) < after]

        page = matching[: query.page_size]
        has_more = len(matching) > query.page_size
        next_token = (
            encode_page_token(page[-1].updated_at_ms, page[-1].task_id) if has_more else None
        )
        return TaskPage(
            tasks=tuple(self._to_record(entry) for entry in page), next_page_token=next_token
        )

    async def active_task(self, context_id: str) -> str | None:
        for task_id in self._context_tasks.get(context_id, ()):
            if self._tasks[task_id].state in LIVE_STATES:
                return task_id
        return None

    # --- Commit internals ------------------------------------------------------------------

    def _validate(self, txn: StorageTxn) -> None:
        """Check every precondition of `txn` without changing anything."""
        for task_id, message in txn.appends:
            if isinstance(message, SystemMessage):
                raise StorageError(
                    _STORE_NAME, "a SystemMessage cannot be appended; use the system_prompt slot"
                )
            if task_id is not None:
                self._require_task(task_id, txn.context_id)

        if txn.state is not None:
            entry = self._require_task(txn.state.task_id, txn.context_id)
            if entry.state not in txn.state.expected_from:
                raise StaleState(
                    f"Task {entry.task_id!r} is {entry.state.value!r}, expected one of "
                    f"{sorted(state.value for state in txn.state.expected_from)}"
                )

        # artifact_id -> ordinal per Task, as the Task would look after the writes so far.
        ordinals: dict[str, dict[str, int]] = {}
        for write in txn.artifacts:
            entry = self._require_task(write.task_id, txn.context_id)
            task_ordinals = ordinals.setdefault(
                write.task_id, {a.artifact_id: a.ordinal for a in entry.artifacts.values()}
            )
            task_ordinals[write.artifact_id] = write.ordinal
            if len(set(task_ordinals.values())) != len(task_ordinals):
                raise StorageError(
                    _STORE_NAME,
                    f"Task {write.task_id!r}: artifact ordinal {write.ordinal} is already in use",
                )

    def _apply(self, txn: StorageTxn, context: ContextRecord) -> None:
        """Apply `txn`. Every precondition was checked by `_validate`, so this cannot fail."""
        activity = bool(txn.appends)
        context.history.extend(message for _, message in txn.appends)

        if not isinstance(txn.system_prompt, Unset):
            context.system_prompt = txn.system_prompt

        if txn.state is not None:
            entry = self._tasks[txn.state.task_id]
            if txn.state.new != entry.state:
                activity = True
            entry.state = txn.state.new
            entry.status_message = txn.state.status_message
            if txn.state.metadata is not None:
                entry.metadata = copy.deepcopy(dict(txn.state.metadata))
            entry.updated_at_ms = txn.now_ms

        for write in txn.artifacts:
            self._tasks[write.task_id].artifacts[write.artifact_id] = ArtifactRecord(
                artifact_id=write.artifact_id,
                ordinal=write.ordinal,
                text=write.text,
                closed=write.closed,
            )

        if activity:
            context.last_activity_ms = txn.now_ms

    def _require_task(self, task_id: str, context_id: str) -> _TaskEntry:
        entry = self._tasks.get(task_id)
        if entry is None:
            raise StorageError(_STORE_NAME, f"unknown Task {task_id!r}")
        if entry.context_id != context_id:
            raise StorageError(
                _STORE_NAME,
                f"Task {task_id!r} belongs to Context {entry.context_id!r}, not {context_id!r}",
            )
        return entry

    # --- Read helpers ----------------------------------------------------------------------

    @staticmethod
    def _matches(entry: _TaskEntry, query: TaskQuery) -> bool:
        if query.context_id is not None and entry.context_id != query.context_id:
            return False
        if query.state is not None and entry.state != query.state:
            return False
        return query.updated_after_ms is None or entry.updated_at_ms > query.updated_after_ms

    @staticmethod
    def _to_record(entry: _TaskEntry) -> TaskRecord:
        """Build the immutable reader view, converting the internal state to the SDK enum."""
        metadata: Mapping[str, Any] = copy.deepcopy(entry.metadata)
        return TaskRecord(
            task_id=entry.task_id,
            context_id=entry.context_id,
            state=to_sdk(entry.state),
            status_message=entry.status_message,
            metadata=metadata,
            created_at_ms=entry.created_at_ms,
            updated_at_ms=entry.updated_at_ms,
            artifacts=tuple(sorted(entry.artifacts.values(), key=lambda a: a.ordinal)),
        )
