"""The Session_Store interface and the `StorageTxn` write model (11.10).

Every write the Orchestrator, the event emitter, or Restart_Recovery makes is described as one
pure `StorageTxn` value and handed to `SessionStore.commit`. The SQLite store runs it as one
`BEGIN IMMEDIATE ... COMMIT`. The memory store validates the whole value first and then applies
it with no `await` in between. Code above this interface cannot tell the two apart.

This module does no I/O. It holds the data shapes and the abstract class only.
"""

import base64
import binascii
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from local_agent.a2a_server.events import TaskState
from local_agent.models import ChatMessage, Clock, SystemClock, SystemMessage, TaskRecord
from local_agent.storage import StaleState, StorageError

__all__ = [
    "UNSET",
    "ArtifactWrite",
    "SessionStore",
    "StaleState",
    "StateChange",
    "StorageError",
    "StorageTxn",
    "TaskPage",
    "TaskQuery",
    "Unset",
    "decode_page_token",
    "encode_page_token",
]


class Unset(Enum):
    """Type of the `UNSET` sentinel.

    `StorageTxn.system_prompt` needs three outcomes: replace the slot with a prompt, clear it
    (`None`), or leave it alone. A one-member enum gives the "leave it alone" case its own type,
    so mypy narrows `x is UNSET` correctly and `None` keeps its meaning of "clear".
    """

    UNSET = "unset"


UNSET = Unset.UNSET


@dataclass(frozen=True)
class StateChange:
    """One Task_State transition, applied as a compare-and-set.

    The store only moves the Task when its committed state is in `expected_from`. Otherwise it
    rolls the whole transaction back and raises `StaleState`, so two writers can never interleave
    a transition. `expected_from` holds several states when a caller accepts more than one
    starting point. `status_message` must already be redacted (16.11).
    """

    task_id: str
    expected_from: frozenset[TaskState]
    new: TaskState
    status_message: str | None
    metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class ArtifactWrite:
    """The full text of one response artifact, written or overwritten by `artifact_id`.

    `ordinal` is the emission order within the Task (18.7). `closed` records that the last-chunk
    event went out (18.12).
    """

    task_id: str
    artifact_id: str
    ordinal: int
    text: str
    closed: bool


@dataclass(frozen=True)
class StorageTxn:
    """A pure description of ONE atomic commit against a single Context.

    Fields:
        context_id: the Context every write in this transaction belongs to (11.6).
        now_ms: wall-clock epoch milliseconds used for `last_activity_ms` and `updated_at_ms`.
        appends: `(producing task_id, message)` pairs in the order they enter the history.
        system_prompt: replacement for the single System_Prompt slot (12.7). `UNSET` leaves the
            slot alone, `None` clears it.
        state: an optional Task_State transition.
        artifacts: artifact texts to write, in the order given.
    """

    context_id: str
    now_ms: int
    appends: tuple[tuple[str | None, ChatMessage], ...] = ()
    system_prompt: SystemMessage | Unset | None = UNSET
    state: StateChange | None = None
    artifacts: tuple[ArtifactWrite, ...] = ()

    def merge(self, other: StorageTxn) -> StorageTxn:
        """Combine two descriptions of the same Context into one transaction.

        `self` goes first: its appends and artifacts come before `other`'s. `now_ms` is the later
        of the two. A `system_prompt` set on `other` wins over one set on `self`. At most one side
        may carry a `state`, because one transaction moves one Task once.

        Raises `ValueError` when the Contexts differ or both sides carry a `state`.
        """
        if other.context_id != self.context_id:
            raise ValueError(
                f"cannot merge transactions for different Contexts: "
                f"{self.context_id!r} and {other.context_id!r}"
            )
        if self.state is not None and other.state is not None:
            raise ValueError("cannot merge two transactions that both carry a state change")
        return StorageTxn(
            context_id=self.context_id,
            now_ms=max(self.now_ms, other.now_ms),
            appends=self.appends + other.appends,
            system_prompt=(
                self.system_prompt if other.system_prompt is UNSET else other.system_prompt
            ),
            state=self.state if self.state is not None else other.state,
            artifacts=self.artifacts + other.artifacts,
        )


@dataclass(frozen=True)
class TaskQuery:
    """Filters for `SessionStore.list_tasks`, mirroring the A2A `ListTasksRequest` (2.10).

    Results are ordered by `updated_at_ms DESC, task_id DESC`. `page_token` is the opaque cursor
    a previous `TaskPage` returned.
    """

    context_id: str | None = None
    state: TaskState | None = None
    updated_after_ms: int | None = None
    page_size: int = 50
    page_token: str | None = None


@dataclass(frozen=True)
class TaskPage:
    """One page of `list_tasks` results.

    `next_page_token` is `None` on the last page.
    """

    tasks: tuple[TaskRecord, ...]
    next_page_token: str | None = None


def encode_page_token(updated_at_ms: int, task_id: str) -> str:
    """Build the opaque keyset cursor that points just after `(updated_at_ms, task_id)`.

    Both stores use it, so a cursor means the same thing whichever backend issued it.
    """
    raw = json.dumps([updated_at_ms, task_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_page_token(token: str) -> tuple[int, str]:
    """Return the `(updated_at_ms, task_id)` a cursor points after.

    Raises `ValueError` for a token that `encode_page_token` could not have produced.
    """
    try:
        decoded = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
    except binascii.Error, UnicodeError, ValueError:
        raise ValueError("invalid page token") from None
    if (
        not isinstance(decoded, list)
        or len(decoded) != 2
        or not isinstance(decoded[0], int)
        or isinstance(decoded[0], bool)
        or not isinstance(decoded[1], str)
    ):
        raise ValueError("invalid page token")
    return decoded[0], decoded[1]


class SessionStore(ABC):
    """Contexts, Message_History, Tasks, and artifacts behind one transactional interface.

    Subclasses implement the abstract methods. The convenience wrappers at the bottom are built on
    `commit`, so a subclass gets them for free. The optional `clock` supplies `now_ms` to those
    wrappers when the caller does not pass one.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()

    @abstractmethod
    async def commit(self, txn: StorageTxn) -> None:
        """Apply `txn` all or nothing.

        Raises `StaleState` when `txn.state.expected_from` does not contain the committed state,
        and `StorageError` for any other failure, including an unknown Context or Task. A failed
        commit leaves the store unchanged.

        Last_Activity: the store sets `last_activity_ms = txn.now_ms` when the transaction creates
        the Context, creates a Task, changes a Task_State, or appends messages. A same-state status
        message does not count as activity.
        """

    @abstractmethod
    async def create_task(
        self, task_id: str, context_id: str, now_ms: int, a2a_snapshot: str | None
    ) -> bool:
        """In one transaction, ensure the Context exists and insert the Task as `submitted`.

        Returns `True` when the Context was new. A `task_id` that already exists fails the whole
        transaction with `StorageError` and leaves the Context unchanged.
        """

    @abstractmethod
    async def get_history(self, context_id: str) -> tuple[SystemMessage | None, list[ChatMessage]]:
        """Return the System_Prompt slot and the history for a Context.

        An unknown Context gives `(None, [])`.
        """

    @abstractmethod
    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Return the Task with its artifacts in emission order, or `None` when absent."""

    @abstractmethod
    async def list_tasks(self, query: TaskQuery) -> TaskPage:
        """Return one page of Tasks matching `query`, newest update first."""

    @abstractmethod
    async def active_task(self, context_id: str) -> str | None:
        """Return the `task_id` of a Task in `submitted`, `working`, or `input-required`, if any."""

    @abstractmethod
    async def set_a2a_snapshot(self, task_id: str, snapshot: str) -> None:
        """Store the Task_Store's private A2A snapshot (SDK history and metadata) for a Task."""

    @abstractmethod
    async def get_a2a_snapshot(self, task_id: str) -> str | None:
        """Return the stored A2A snapshot, or `None` when the Task is absent or has none.

        The Task_Store reads it to build `history` and to skip a `save` that changes nothing.
        """

    @abstractmethod
    async def delete_task(self, task_id: str) -> None:
        """Delete one Task and its artifacts in one transaction. An unknown `task_id` is a no-op.

        The Context and its Message_History stay. Retention (`delete_expired`) is the normal
        deletion path; the Task_Store's `delete` is the only caller.
        """

    @abstractmethod
    async def delete_expired(self, cutoff_ms: int) -> list[str]:
        """Delete idle Contexts in one transaction and return their `context_id`s (11.11).

        A Context is expired when `last_activity_ms < cutoff_ms` and it has no live Task (11.12).
        Deleting a Context removes its history, Tasks, and artifacts with it.
        """

    # --- Convenience wrappers built on `commit` ------------------------------------------------

    async def append(
        self,
        context_id: str,
        messages: Sequence[ChatMessage],
        *,
        task_id: str | None = None,
        now_ms: int | None = None,
    ) -> None:
        """Append `messages` to the history in one commit, tagged with the producing `task_id`."""
        await self.commit(
            StorageTxn(
                context_id=context_id,
                now_ms=self._resolve_now_ms(now_ms),
                appends=tuple((task_id, message) for message in messages),
            )
        )

    async def set_system_prompt(
        self, context_id: str, prompt: SystemMessage | None, *, now_ms: int | None = None
    ) -> None:
        """Replace the single System_Prompt slot in one commit. `None` clears it (12.7)."""
        await self.commit(
            StorageTxn(
                context_id=context_id,
                now_ms=self._resolve_now_ms(now_ms),
                system_prompt=prompt,
            )
        )

    async def set_state(
        self, context_id: str, change: StateChange, *, now_ms: int | None = None
    ) -> None:
        """Apply one `StateChange` in one commit. Raises `StaleState` if the CAS loses."""
        await self.commit(
            StorageTxn(context_id=context_id, now_ms=self._resolve_now_ms(now_ms), state=change)
        )

    def _resolve_now_ms(self, now_ms: int | None) -> int:
        return now_ms if now_ms is not None else self._clock.now_ms()
