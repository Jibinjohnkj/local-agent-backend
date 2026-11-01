"""Restart_Recovery: finalize Tasks a restart left running (18.15-18.18).

The Backend keeps no runner, approval, or suspended turn on disk, so a Task found in `submitted`,
`working`, or `input-required` at startup can never continue (5.7). Recovery moves each one to
`failed` and answers its unanswered Tool_Calls with a synthetic Tool_Result, so the Message_History
stays valid for the next turn of the Context (11.5).

`plan_recovery` is the pure decision. `run` reads the rows, plans, and then executes each action
as ONE transaction: the synthetic `ToolMessage`s, the `failed` state with `RECOVERY_STATUS`, and
Last_Activity = now (18.16). The transaction is the same `StorageTxn` write the Orchestrator uses,
so the compare-and-set, `seq` numbering, and Last_Activity rules are shared with
`SQLiteSessionStore`. The Task_Store reads the same row, so it shows the same state (18.16).

Recovery sends nothing to an MCP_Server, the LLM, or a client, and starts no runner (18.17). A
second run finds no non-terminal Task and writes nothing (18.18). A failing transaction rolls back
that Task alone, stops recovery, and raises `StorageError`. Tasks recovered earlier stay recovered
and the rest are recovered on the next start (18.24).

`run` blocks on the writer thread through `Database.run_sync`, so it runs at startup, before the
event loop exists. It is the only caller that passes `recovery=True` to `is_allowed` (2.9).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from local_agent.a2a_server.events import LIVE_STATES, TaskState, is_allowed
from local_agent.logging_setup import Redactor, get_logger
from local_agent.models import ToolCall, ToolMessage
from local_agent.session.sqlite_store import _commit
from local_agent.session.store import StaleState, StateChange, StorageTxn
from local_agent.storage import StorageError
from local_agent.storage.database import Database, run_in_transaction

if TYPE_CHECKING:
    from local_agent.models import Clock

__all__ = [
    "RECOVERY_STATUS",
    "RECOVERY_TOOL_RESULT",
    "RecoveryAction",
    "RecoveryReport",
    "StoredMessage",
    "TaskRow",
    "plan_recovery",
    "run",
]

RECOVERY_STATUS: Final = "The backend restarted while this task was running"
RECOVERY_TOOL_RESULT: Final = (
    "Tool call not completed: the backend restarted while this task was running"
)

_log = get_logger("storage.recovery")

# The stored spelling of every state recovery acts on. The names are the lower-case A2A spellings
# the schema's CHECK constraint lists, which are the `TaskState` values.
_LIVE_SQL: Final = ", ".join(sorted(f"'{state.value}'" for state in LIVE_STATES))


@dataclass(frozen=True)
class TaskRow:
    """The columns of a `tasks` row that recovery needs."""

    task_id: str
    context_id: str
    state: TaskState
    created_at_ms: int


@dataclass(frozen=True)
class StoredMessage:
    """The columns of a `messages` row that recovery needs. `seq` orders a Context's history."""

    seq: int
    task_id: str | None
    role: str
    tool_calls: tuple[ToolCall, ...]
    tool_call_id: str | None


@dataclass(frozen=True)
class RecoveryAction:
    """What recovery does to one Task: one transaction (18.16).

    `synthetic_results` holds the `tool_call_id`s that get a synthetic Tool_Result, in history
    order, then Tool_Call order.
    """

    task_id: str
    context_id: str
    from_state: TaskState
    synthetic_results: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryReport:
    """What `run` changed. Empty when the database held no non-terminal Task (18.18)."""

    actions: tuple[RecoveryAction, ...] = ()

    @property
    def tasks_recovered(self) -> int:
        return len(self.actions)

    @property
    def tool_results_added(self) -> int:
        return sum(len(action.synthetic_results) for action in self.actions)


def _unanswered(task_id: str, history: Sequence[StoredMessage]) -> tuple[str, ...]:
    """The IDs of `task_id`'s Tool_Calls that no later tool message answers.

    An assistant message belongs to the Task when its `task_id` matches. A Tool_Call is answered by
    a `tool` message with the same `tool_call_id` and a greater `seq`. A repeated ID is returned
    once, because one synthetic result answers every call that carries it.
    """
    ordered = sorted(history, key=lambda message: message.seq)
    answered_at: dict[str, int] = {}
    for message in ordered:
        if message.role == "tool" and message.tool_call_id is not None:
            answered_at[message.tool_call_id] = message.seq  # the last answer wins
    pending: list[str] = []
    for message in ordered:
        if message.role != "assistant" or message.task_id != task_id:
            continue
        for call in message.tool_calls:
            if answered_at.get(call.id, -1) > message.seq or call.id in pending:
                continue
            pending.append(call.id)
    return tuple(pending)


def plan_recovery(
    tasks: Sequence[TaskRow], histories: Mapping[str, Sequence[StoredMessage]]
) -> list[RecoveryAction]:
    """Decide what recovery does. Pure.

    Selects exactly the Tasks in `submitted`, `working`, or `input-required`, ordered by
    `created_at_ms` and then `task_id`. Terminal Tasks produce no action (18.15). `histories` maps
    a `context_id` to its messages. A Context without an entry has no Tool_Calls to answer.
    """
    live = sorted(
        (task for task in tasks if task.state in LIVE_STATES),
        key=lambda task: (task.created_at_ms, task.task_id),
    )
    return [
        RecoveryAction(
            task_id=task.task_id,
            context_id=task.context_id,
            from_state=task.state,
            synthetic_results=_unanswered(task.task_id, histories.get(task.context_id, ())),
        )
        for task in live
    ]


# --- Reads and writes (run on the writer thread) ----------------------------------------------


def _decode_tool_calls(raw: str | None) -> tuple[ToolCall, ...]:
    if raw is None:
        return ()
    return tuple(
        ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"])
        for call in json.loads(raw)
    )


def _state_from_db(path: str, name: str) -> TaskState:
    try:
        return TaskState(name)
    except ValueError:
        raise StorageError(path, f"unknown stored Task_State {name!r}") from None


def _load(
    conn: sqlite3.Connection, path: str
) -> tuple[list[TaskRow], dict[str, list[StoredMessage]]]:
    """Read the live Tasks and the histories of their Contexts, without writing."""
    tasks = [
        TaskRow(
            task_id=str(row[0]),
            context_id=str(row[1]),
            state=_state_from_db(path, str(row[2])),
            created_at_ms=int(row[3]),
        )
        for row in conn.execute(
            "SELECT task_id, context_id, state, created_at_ms FROM tasks"
            f" WHERE state IN ({_LIVE_SQL})"
        )
    ]
    histories: dict[str, list[StoredMessage]] = {}
    for context_id in sorted({task.context_id for task in tasks}):
        histories[context_id] = [
            StoredMessage(
                seq=int(row[0]),
                task_id=None if row[1] is None else str(row[1]),
                role=str(row[2]),
                tool_calls=_decode_tool_calls(row[3]),
                tool_call_id=None if row[4] is None else str(row[4]),
            )
            for row in conn.execute(
                "SELECT seq, task_id, role, tool_calls, tool_call_id FROM messages"
                " WHERE context_id = ? ORDER BY seq",
                (context_id,),
            )
        ]
    return tasks, histories


def _transaction(action: RecoveryAction, now_ms: int) -> StorageTxn:
    return StorageTxn(
        context_id=action.context_id,
        now_ms=now_ms,
        appends=tuple(
            (action.task_id, ToolMessage(tool_call_id=call_id, content=RECOVERY_TOOL_RESULT))
            for call_id in action.synthetic_results
        ),
        state=StateChange(
            task_id=action.task_id,
            expected_from=frozenset({action.from_state}),
            new=TaskState.FAILED,
            status_message=RECOVERY_STATUS,
            metadata=None,
        ),
    )


def _run_action(db: Database, action: RecoveryAction, now_ms: int, path: str) -> None:
    """Commit one action as one `BEGIN IMMEDIATE ... COMMIT` on the writer thread."""
    txn = _transaction(action, now_ms)
    redactor = Redactor(())  # RECOVERY_STATUS holds no secret, so there is nothing to mask

    def step(conn: sqlite3.Connection) -> None:
        if not is_allowed(action.from_state, TaskState.FAILED, recovery=True):
            raise StorageError(path, f"Task_State {action.from_state.value!r} cannot be recovered")
        _commit(conn, path, redactor, txn)

    db.run_sync(lambda conn: run_in_transaction(conn, step, path))


def run(db: Database, clock: Clock) -> RecoveryReport:
    """Finalize every non-terminal Task as `failed`, one transaction per Task (18.15, 18.16).

    Returns the actions that were committed. Raises `StorageError` when a transaction fails; that
    Task's changes are rolled back, recovery stops there, and the Tasks already recovered stay
    recovered (18.24). Blocks the caller, so call it before the event loop exists.
    """
    path = str(db.path)
    tasks, histories = db.run_sync(lambda conn: _load(conn, path))
    done: list[RecoveryAction] = []
    for action in plan_recovery(tasks, histories):
        try:
            _run_action(db, action, clock.now_ms(), path)
        except StaleState as exc:
            raise StorageError(path, f"Task {action.task_id!r} changed during recovery") from exc
        done.append(action)
        _log.info(
            "task_recovered",
            task_id=action.task_id,
            context_id=action.context_id,
            from_state=action.from_state.value,
            tool_results=len(action.synthetic_results),
        )
    report = RecoveryReport(actions=tuple(done))
    _log.info(
        "restart_recovery",
        tasks_recovered=report.tasks_recovered,
        tool_results_added=report.tool_results_added,
    )
    return report
