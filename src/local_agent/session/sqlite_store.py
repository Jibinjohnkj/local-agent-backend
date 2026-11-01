"""SQLite Session_Store (`storage.backend: sqlite`, the default).

Every method is one `Database.transaction(fn)` or `Database.read(fn)` call. `fn` is a plain
synchronous function over the connection, so all SQL lives in this module and runs on the single
writer thread. A `StorageTxn` becomes one `BEGIN IMMEDIATE ... COMMIT`: the `IMMEDIATE` write lock
makes `seq = MAX(seq) + 1` and the compare-and-set on the Task_State race-free (11.10, 18.8).

The behaviour above `SessionStore` matches `InMemorySessionStore` exactly. The rules the memory
store documents hold here too:

* **Validation order.** The Context must exist. Then every append is checked (no `SystemMessage`,
  and a producing `task_id` must be a Task of this Context). Then the state change (the Task must
  be in this Context, and `StaleState` when its committed state is not in `expected_from`). Only
  then do the writes run. Any failure rolls the whole transaction back.
* **Last_Activity.** `last_activity_ms = now_ms` when the transaction appends messages or moves a
  Task to a different state, and in `create_task`. A same-state status message, a System_Prompt
  replacement, and artifact writes are not activity.
* **StateChange.** `status_message` always replaces the stored one. `metadata=None` keeps the
  stored metadata. Every `StateChange` sets `updated_at_ms`.
* **Artifacts.** A write replaces the artifact with the same `artifact_id`. The
  `UNIQUE (task_id, ordinal)` constraint rejects two artifacts that share an ordinal, which
  surfaces as `StorageError`.
* **Paging.** `updated_after_ms` is exclusive. The cursor is the keyset `(updated_at_ms, task_id)`
  of the last row returned, built by `encode_page_token`. `page_size < 1` and a malformed cursor
  raise `ValueError`.

Encoding (18.3, 18.5): text is bound as a `TEXT` parameter exactly as the Python `str`, never
formatted into SQL, stripped, or normalized. `None` and `""` are distinct (`NULL` vs `''`).
`tool_calls` is a compact JSON array with `arguments` kept as the JSON *string* the LLM sent.
A lone surrogate cannot be encoded as UTF-8, so it never reaches this module: ingress replaces it
with U+FFFD. Should one arrive anyway, the write fails with `StorageError`, and a read by an
identifier that contains one finds nothing.

Redaction (16.10, 16.11): callers redact `status_message` before it enters a `StorageTxn`. This
store redacts it once more on the way in, as a second line of defence. No other secret-bearing
text is accepted, and the database has no config table.
"""

import json
import sqlite3
from collections.abc import Mapping
from typing import Any, Final

from local_agent.a2a_server.events import LIVE_STATES, TaskState, to_sdk
from local_agent.logging_setup import Redactor
from local_agent.models import (
    ArtifactRecord,
    AssistantMessage,
    ChatMessage,
    Clock,
    SystemMessage,
    TaskRecord,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from local_agent.session.store import (
    SessionStore,
    StaleState,
    StateChange,
    StorageError,
    StorageTxn,
    TaskPage,
    TaskQuery,
    Unset,
    decode_page_token,
    encode_page_token,
)
from local_agent.storage.database import Database

__all__ = ["SQLiteSessionStore"]

# The one mapping between `TaskState` and the lower-case names stored in `tasks.state` (the
# Requirement spellings, which the schema's CHECK constraint lists).
_STATE_TO_DB: Final[Mapping[TaskState, str]] = {
    TaskState.SUBMITTED: "submitted",
    TaskState.WORKING: "working",
    TaskState.INPUT_REQUIRED: "input-required",
    TaskState.COMPLETED: "completed",
    TaskState.FAILED: "failed",
    TaskState.CANCELED: "canceled",
}
_STATE_FROM_DB: Final[Mapping[str, TaskState]] = {name: s for s, name in _STATE_TO_DB.items()}

# Built from constants, never from user text. A literal list (not bound parameters) is what lets
# SQLite see that the query implies the predicate of the partial index `tasks_live`; the planner
# still picks `tasks_by_context` for the per-Context lookup, which is the cheaper of the two.
_LIVE_SQL: Final = ", ".join(sorted(f"'{_STATE_TO_DB[state]}'" for state in LIVE_STATES))

_TASK_COLUMNS: Final = (
    "task_id, context_id, state, status_message, metadata, created_at_ms, updated_at_ms"
)

_STORE_NAME: Final = "sqlite"

# Context IDs per DELETE, to stay under SQLite's bound-parameter limit (999 on old builds).
_DELETE_BATCH: Final = 500

_INT64_MAX: Final = 2**63 - 1
_INT64_MIN: Final = -(2**63)


def _state_to_db(state: TaskState) -> str:
    return _STATE_TO_DB[state]


def _state_from_db(name: str) -> TaskState:
    try:
        return _STATE_FROM_DB[name]
    except KeyError:
        raise StorageError(_STORE_NAME, f"unknown stored Task_State {name!r}") from None


def _can_encode(*texts: str | None) -> bool:
    """False when a text holds a lone surrogate, which SQLite cannot store or match."""
    try:
        for text in texts:
            if text is not None:
                text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _clamp(value: int) -> int:
    """Pin an integer filter into SQLite's INTEGER range, which keeps its ordering meaning."""
    return max(_INT64_MIN, min(_INT64_MAX, value))


# --- Encoding ---------------------------------------------------------------------------------


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _encode_tool_calls(calls: tuple[ToolCall, ...]) -> str | None:
    if not calls:
        return None
    return _json([{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls])


def _decode_tool_calls(raw: str | None) -> tuple[ToolCall, ...]:
    if raw is None:
        return ()
    return tuple(
        ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"]) for c in json.loads(raw)
    )


def _message_columns(
    path: str, message: ChatMessage
) -> tuple[str, str | None, str | None, str | None]:
    """Return `(role, content, tool_calls, tool_call_id)` for a history message."""
    match message:
        case UserMessage():
            return "user", message.content, None, None
        case AssistantMessage():
            return "assistant", message.content, _encode_tool_calls(message.tool_calls), None
        case ToolMessage():
            return "tool", message.content, None, message.tool_call_id
        case SystemMessage():
            raise StorageError(
                path, "a SystemMessage cannot be appended; use the system_prompt slot"
            )


def _decode_message(
    role: str, content: str | None, tool_calls: str | None, tool_call_id: str | None
) -> ChatMessage:
    match role:
        case "user":
            return UserMessage(content=content or "")
        case "assistant":
            return AssistantMessage(content=content, tool_calls=_decode_tool_calls(tool_calls))
        case "tool" if tool_call_id is not None:
            return ToolMessage(tool_call_id=tool_call_id, content=content or "")
    raise StorageError(_STORE_NAME, f"unreadable stored message with role {role!r}")


# --- Writes (run inside `BEGIN IMMEDIATE ... COMMIT`) -----------------------------------------


def _require_task(conn: sqlite3.Connection, path: str, task_id: str, context_id: str) -> str:
    """Return the committed state name of `task_id`, which must belong to `context_id`."""
    row = conn.execute(
        "SELECT context_id, state FROM tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        raise StorageError(path, f"unknown Task {task_id!r}")
    if row[0] != context_id:
        raise StorageError(
            path, f"Task {task_id!r} belongs to Context {row[0]!r}, not {context_id!r}"
        )
    return str(row[1])


def _commit(conn: sqlite3.Connection, path: str, redactor: Redactor, txn: StorageTxn) -> None:
    context_id = txn.context_id
    if (
        conn.execute("SELECT 1 FROM contexts WHERE context_id = ?", (context_id,)).fetchone()
        is None
    ):
        raise StorageError(path, f"unknown Context {context_id!r}")

    # Validate first, in the same order as the memory store, so both raise the same error.
    rows = [(task_id, _message_columns(path, message)) for task_id, message in txn.appends]
    for task_id, _ in rows:
        if task_id is not None:
            _require_task(conn, path, task_id, context_id)

    change = txn.state
    previous: str | None = None
    if change is not None:
        previous = _require_task(conn, path, change.task_id, context_id)
        if _state_from_db(previous) not in change.expected_from:
            raise StaleState(
                f"Task {change.task_id!r} is {previous!r}, expected one of "
                f"{sorted(state.value for state in change.expected_from)}"
            )
    for write in txn.artifacts:
        _require_task(conn, path, write.task_id, context_id)

    activity = bool(rows)
    if rows:
        next_seq = int(
            conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE context_id = ?",
                (context_id,),
            ).fetchone()[0]
        )
        for offset, (task_id, (role, content, tool_calls, tool_call_id)) in enumerate(rows):
            conn.execute(
                "INSERT INTO messages (context_id, seq, task_id, role, content, tool_calls,"
                " tool_call_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (context_id, next_seq + offset, task_id, role, content, tool_calls, tool_call_id),
            )

    if not isinstance(txn.system_prompt, Unset):
        prompt = None if txn.system_prompt is None else txn.system_prompt.content
        conn.execute(
            "UPDATE contexts SET system_prompt = ? WHERE context_id = ?", (prompt, context_id)
        )

    if change is not None and previous is not None:
        _apply_state_change(conn, redactor, change, txn.now_ms)
        activity = activity or _state_to_db(change.new) != previous

    for write in txn.artifacts:
        conn.execute(
            "INSERT INTO artifacts (task_id, artifact_id, ordinal, text, closed)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT (task_id, artifact_id) DO UPDATE SET"
            " ordinal = excluded.ordinal, text = excluded.text, closed = excluded.closed",
            (write.task_id, write.artifact_id, write.ordinal, write.text, int(write.closed)),
        )

    if activity:
        conn.execute(
            "UPDATE contexts SET last_activity_ms = ? WHERE context_id = ?",
            (txn.now_ms, context_id),
        )


def _apply_state_change(
    conn: sqlite3.Connection, redactor: Redactor, change: StateChange, now_ms: int
) -> None:
    """Run the compare-and-set. A row count other than one means another writer moved the Task."""
    expected = sorted(_state_to_db(state) for state in change.expected_from)
    message = None if change.status_message is None else redactor.redact(change.status_message)
    metadata = None if change.metadata is None else _json(dict(change.metadata))
    marks = ", ".join("?" * len(expected))
    cursor = conn.execute(
        "UPDATE tasks SET state = ?, status_message = ?, metadata = COALESCE(?, metadata),"
        f" updated_at_ms = ? WHERE task_id = ? AND state IN ({marks})",
        (_state_to_db(change.new), message, metadata, now_ms, change.task_id, *expected),
    )
    if cursor.rowcount != 1:
        raise StaleState(f"Task {change.task_id!r} is not in an expected state")


def _create_task(
    conn: sqlite3.Connection,
    path: str,
    task_id: str,
    context_id: str,
    now_ms: int,
    a2a_snapshot: str | None,
) -> bool:
    if conn.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)).fetchone() is not None:
        raise StorageError(path, f"Task {task_id!r} already exists")
    is_new = (
        conn.execute("SELECT 1 FROM contexts WHERE context_id = ?", (context_id,)).fetchone()
        is None
    )
    if is_new:
        conn.execute(
            "INSERT INTO contexts (context_id, system_prompt, created_at_ms, last_activity_ms)"
            " VALUES (?, NULL, ?, ?)",
            (context_id, now_ms, now_ms),
        )
    else:
        conn.execute(
            "UPDATE contexts SET last_activity_ms = ? WHERE context_id = ?", (now_ms, context_id)
        )
    conn.execute(
        "INSERT INTO tasks (task_id, context_id, state, status_message, metadata, created_at_ms,"
        " updated_at_ms, a2a_snapshot) VALUES (?, ?, ?, NULL, '{}', ?, ?, ?)",
        (task_id, context_id, _state_to_db(TaskState.SUBMITTED), now_ms, now_ms, a2a_snapshot),
    )
    return is_new


def _delete_expired(conn: sqlite3.Connection, cutoff_ms: int) -> list[str]:
    """Delete Contexts idle since before `cutoff_ms` that have no live Task (11.11, 11.12).

    The same predicate as `retention.select_expired`. `ON DELETE CASCADE` removes the messages,
    Tasks, and artifacts of each deleted Context in this transaction.
    """
    expired = [
        str(row[0])
        for row in conn.execute(
            "SELECT context_id FROM contexts WHERE last_activity_ms < ?"
            " AND NOT EXISTS (SELECT 1 FROM tasks"
            f" WHERE tasks.context_id = contexts.context_id AND tasks.state IN ({_LIVE_SQL}))"
            " ORDER BY rowid",
            (_clamp(cutoff_ms),),
        )
    ]
    for start in range(0, len(expired), _DELETE_BATCH):
        batch = expired[start : start + _DELETE_BATCH]
        marks = ", ".join("?" * len(batch))
        conn.execute(f"DELETE FROM contexts WHERE context_id IN ({marks})", batch)
    return expired


def _set_a2a_snapshot(conn: sqlite3.Connection, path: str, task_id: str, snapshot: str) -> None:
    cursor = conn.execute(
        "UPDATE tasks SET a2a_snapshot = ? WHERE task_id = ?", (snapshot, task_id)
    )
    if cursor.rowcount != 1:
        raise StorageError(path, f"unknown Task {task_id!r}")


# --- Reads ------------------------------------------------------------------------------------


def _read_history(
    conn: sqlite3.Connection, context_id: str
) -> tuple[SystemMessage | None, list[ChatMessage]]:
    row = conn.execute(
        "SELECT system_prompt FROM contexts WHERE context_id = ?", (context_id,)
    ).fetchone()
    if row is None:
        return None, []
    prompt = None if row[0] is None else SystemMessage(content=row[0])
    messages = conn.execute(
        "SELECT role, content, tool_calls, tool_call_id FROM messages"
        " WHERE context_id = ? ORDER BY seq",
        (context_id,),
    ).fetchall()
    return prompt, [_decode_message(*message) for message in messages]


def _to_record(conn: sqlite3.Connection, row: tuple[Any, ...]) -> TaskRecord:
    task_id, context_id, state, status_message, metadata, created_at_ms, updated_at_ms = row
    artifacts = conn.execute(
        "SELECT artifact_id, ordinal, text, closed FROM artifacts"
        " WHERE task_id = ? ORDER BY ordinal",
        (task_id,),
    ).fetchall()
    return TaskRecord(
        task_id=task_id,
        context_id=context_id,
        state=to_sdk(_state_from_db(state)),
        status_message=status_message,
        metadata=json.loads(metadata),
        created_at_ms=created_at_ms,
        updated_at_ms=updated_at_ms,
        artifacts=tuple(
            ArtifactRecord(artifact_id=a[0], ordinal=a[1], text=a[2], closed=bool(a[3]))
            for a in artifacts
        ),
    )


def _read_task(conn: sqlite3.Connection, task_id: str) -> TaskRecord | None:
    row = conn.execute(
        f"SELECT {_TASK_COLUMNS} FROM tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    return None if row is None else _to_record(conn, row)


def _read_page(
    conn: sqlite3.Connection, query: TaskQuery, after: tuple[int, str] | None
) -> TaskPage:
    conditions: list[str] = []
    params: list[Any] = []
    if query.context_id is not None:
        conditions.append("context_id = ?")
        params.append(query.context_id)
    if query.state is not None:
        conditions.append("state = ?")
        params.append(_state_to_db(query.state))
    if query.updated_after_ms is not None:
        conditions.append("updated_at_ms > ?")
        params.append(_clamp(query.updated_after_ms))
    if after is not None:
        conditions.append("(updated_at_ms, task_id) < (?, ?)")
        params.extend((_clamp(after[0]), after[1]))
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    # One extra row tells whether another page follows.
    limit = min(query.page_size, _INT64_MAX - 1) + 1
    rows = conn.execute(
        f"SELECT {_TASK_COLUMNS} FROM tasks{where}"
        " ORDER BY updated_at_ms DESC, task_id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    has_more = len(rows) > query.page_size
    page = rows[: query.page_size]
    next_token = encode_page_token(page[-1][6], page[-1][0]) if has_more else None
    return TaskPage(tasks=tuple(_to_record(conn, row) for row in page), next_page_token=next_token)


class SQLiteSessionStore(SessionStore):
    """Session_Store backed by the Storage_Database. Every call is one transaction or one read."""

    def __init__(self, db: Database, redactor: Redactor, clock: Clock) -> None:
        super().__init__(clock)
        self._db = db
        self._redactor = redactor
        self._path = str(db.path)

    # --- Writes ----------------------------------------------------------------------------

    async def commit(self, txn: StorageTxn) -> None:
        def run(conn: sqlite3.Connection) -> None:
            _commit(conn, self._path, self._redactor, txn)

        await self._db.transaction(run)

    async def create_task(
        self, task_id: str, context_id: str, now_ms: int, a2a_snapshot: str | None
    ) -> bool:
        def run(conn: sqlite3.Connection) -> bool:
            return _create_task(conn, self._path, task_id, context_id, now_ms, a2a_snapshot)

        return await self._db.transaction(run)

    async def set_a2a_snapshot(self, task_id: str, snapshot: str) -> None:
        def run(conn: sqlite3.Connection) -> None:
            _set_a2a_snapshot(conn, self._path, task_id, snapshot)

        await self._db.transaction(run)

    async def delete_task(self, task_id: str) -> None:
        if not _can_encode(task_id):
            return

        def run(conn: sqlite3.Connection) -> None:
            # `ON DELETE CASCADE` removes the Task's artifacts in the same transaction.
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

        await self._db.transaction(run)

    async def delete_expired(self, cutoff_ms: int) -> list[str]:
        def run(conn: sqlite3.Connection) -> list[str]:
            return _delete_expired(conn, cutoff_ms)

        return await self._db.transaction(run)

    # --- Reads -----------------------------------------------------------------------------

    async def get_history(self, context_id: str) -> tuple[SystemMessage | None, list[ChatMessage]]:
        if not _can_encode(context_id):
            return None, []
        return await self._db.read(lambda conn: _read_history(conn, context_id))

    async def get_task(self, task_id: str) -> TaskRecord | None:
        if not _can_encode(task_id):
            return None
        return await self._db.read(lambda conn: _read_task(conn, task_id))

    async def list_tasks(self, query: TaskQuery) -> TaskPage:
        if query.page_size < 1:
            raise ValueError(f"page_size must be at least 1, got {query.page_size}")
        after = decode_page_token(query.page_token) if query.page_token is not None else None
        if after is not None and not _can_encode(after[1]):
            raise ValueError("invalid page token")
        if not _can_encode(query.context_id):
            return TaskPage(tasks=())
        return await self._db.read(lambda conn: _read_page(conn, query, after))

    async def active_task(self, context_id: str) -> str | None:
        if not _can_encode(context_id):
            return None

        def run(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                f"SELECT task_id FROM tasks WHERE context_id = ? AND state IN ({_LIVE_SQL})"
                " ORDER BY rowid LIMIT 1",
                (context_id,),
            ).fetchone()
            return None if row is None else str(row[0])

        return await self._db.read(run)

    async def get_a2a_snapshot(self, task_id: str) -> str | None:
        if not _can_encode(task_id):
            return None

        def run(conn: sqlite3.Connection) -> str | None:
            row = conn.execute(
                "SELECT a2a_snapshot FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return None if row is None else row[0]

        return await self._db.read(run)
