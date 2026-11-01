"""Unit tests for the SQLite store and Task_Store (Requirements 2.1, 18.3, 18.5, 18.9, 18.30).

The store tests run against a real Storage_Database in `tmp_path` with the real schema, and read
raw rows only where the stored representation itself is the thing under test (`NULL` vs `''`,
the UTF-8 bytes of a replaced surrogate). The Task_Store tests call `ProjectedTaskStore` the way
the SDK does, with `Task` protos and a `ServerCallContext`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import structlog
from a2a.types import (
    ListTasksRequest,
    Message,
    Part,
    Role,
    Task,
    TaskStatus,
)

from conftest import FakeClock
from local_agent.a2a_server.a2a_compat import (
    InvalidParamsError,
    ServerCallContext,
    TaskNotFoundError,
)
from local_agent.a2a_server.a2a_compat import TaskState as SdkTaskState
from local_agent.a2a_server.errors import StorageWriteError
from local_agent.a2a_server.events import TaskState, to_sdk
from local_agent.config.models import LoggingConfig
from local_agent.llm.stream_assembler import StreamAssembler
from local_agent.logging_setup import Redactor, configure_logging
from local_agent.models import (
    ArtifactRecord,
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import (
    ArtifactWrite,
    StaleState,
    StateChange,
    StorageError,
    StorageTxn,
    TaskQuery,
)
from local_agent.storage import migrations
from local_agent.storage.database import Database
from local_agent.storage.task_store import ProjectedTaskStore, snapshot_json
from storage_faults import FaultFactory

T0 = 1_700_000_000_000
ALL_STATES = frozenset(TaskState)
SERVER_CONTEXT = ServerCallContext()
REPLACEMENT = "\ufffd"


# --- Fixtures and helpers ---------------------------------------------------------------------


def _open(path: Path, factory: FaultFactory | None = None) -> Database:
    database = Database.open(path, create=True, connection_factory=factory)
    try:
        migrations.apply(database, str(path))
    except BaseException:
        database.close()
        raise
    return database


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = _open(tmp_path / "agent.db")
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def store(db: Database, clock: FakeClock) -> SQLiteSessionStore:
    return SQLiteSessionStore(db, Redactor([]), clock)


@pytest.fixture
def tasks(store: SQLiteSessionStore, clock: FakeClock) -> ProjectedTaskStore:
    return ProjectedTaskStore(store, clock=clock)


@pytest.fixture(autouse=True)
def _isolated_logging() -> Iterator[None]:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _capture_logs(level: str = "debug") -> BytesIO:
    stream = BytesIO()
    configure_logging(LoggingConfig(level=level), Redactor([]), stream=stream)
    return stream


def _records(stream: BytesIO) -> list[dict[str, Any]]:
    lines = stream.getvalue().decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def _events(stream: BytesIO, name: str) -> list[dict[str, Any]]:
    return [record for record in _records(stream) if record.get("event") == name]


def _rows(db: Database, sql: str, params: Sequence[object] = ()) -> list[tuple[Any, ...]]:
    def run(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in conn.execute(sql, params).fetchall()]

    return db.run_sync(run)


def _count(db: Database, table: str) -> int:
    return int(_rows(db, f"SELECT count(*) FROM {table}")[0][0])


def _change(
    task_id: str,
    new: TaskState,
    *,
    status_message: str | None = None,
    expected: frozenset[TaskState] = ALL_STATES,
) -> StateChange:
    return StateChange(
        task_id=task_id,
        expected_from=expected,
        new=new,
        status_message=status_message,
        metadata=None,
    )


async def _make_task(
    store: SQLiteSessionStore,
    task_id: str,
    context_id: str,
    *,
    state: TaskState = TaskState.SUBMITTED,
    updated_at_ms: int = T0,
    snapshot: str | None = None,
) -> None:
    await store.create_task(task_id, context_id, T0, snapshot)
    if state is not TaskState.SUBMITTED or updated_at_ms != T0:
        await store.set_state(context_id, _change(task_id, state), now_ms=updated_at_ms)


def _proto(
    task_id: str, context_id: str, state: SdkTaskState = SdkTaskState.TASK_STATE_SUBMITTED
) -> Task:
    return Task(id=task_id, context_id=context_id, status=TaskStatus(state=state))


class _Live:
    """A `LiveArtifacts` fake that returns fixed buffers per Task."""

    def __init__(self, by_task: dict[str, list[ArtifactRecord]]) -> None:
        self._by_task = by_task

    def live_artifacts(self, task_id: str) -> Sequence[ArtifactRecord]:
        return self._by_task.get(task_id, [])


# --- Message encoding (18.3, 18.5) ------------------------------------------------------------


async def test_none_and_empty_assistant_content_stay_distinct(
    store: SQLiteSessionStore, db: Database
) -> None:
    await store.create_task("t1", "c1", T0, None)
    call = ToolCall(id="call_1", name="echo", arguments="{not json")
    await store.append(
        "c1",
        [
            AssistantMessage(content=None, tool_calls=(call,)),
            AssistantMessage(content="", tool_calls=(call,)),
            AssistantMessage(content=""),
            UserMessage(""),
        ],
        task_id="t1",
        now_ms=T0 + 1,
    )

    _, history = await store.get_history("c1")

    assert history == [
        AssistantMessage(content=None, tool_calls=(call,)),
        AssistantMessage(content="", tool_calls=(call,)),
        AssistantMessage(content=""),
        UserMessage(""),
    ]
    stored = _rows(db, "SELECT typeof(content), content FROM messages ORDER BY seq")
    assert stored == [("null", None), ("text", ""), ("text", ""), ("text", "")]


async def test_none_vs_empty_content_survives_close_and_reopen(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "reopen.db"
    first = _open(path)
    try:
        writer = SQLiteSessionStore(first, Redactor([]), clock)
        await writer.create_task("t1", "c1", T0, None)
        call = ToolCall(id="call_1", name="echo", arguments="")
        await writer.append(
            "c1",
            [AssistantMessage(content=None, tool_calls=(call,)), AssistantMessage(content="")],
            task_id="t1",
            now_ms=T0 + 1,
        )
    finally:
        first.close()

    second = _open(path)
    try:
        _, history = await SQLiteSessionStore(second, Redactor([]), clock).get_history("c1")
    finally:
        second.close()

    assert history == [
        AssistantMessage(content=None, tool_calls=(ToolCall("call_1", "echo", ""),)),
        AssistantMessage(content=""),
    ]


async def test_text_is_stored_exactly_as_given(store: SQLiteSessionStore, db: Database) -> None:
    texts = [
        "a\r\nb\rc\nd",
        "\ttabbed\t",
        "nul\x00inside",
        "  padded  ",
        "emoji \U0001f600 and \U0001f9d1\u200d\U0001f4bb",
        "caf\u00e9 vs cafe\u0301",  # composed and decomposed must not be normalized
        "'; DROP TABLE messages; --",
    ]
    await store.create_task("t1", "c1", T0, None)
    await store.append("c1", [UserMessage(text) for text in texts], task_id="t1", now_ms=T0 + 1)

    _, history = await store.get_history("c1")

    assert history == [UserMessage(text) for text in texts]
    assert _count(db, "messages") == len(texts)


async def test_tool_call_arguments_stay_a_json_string(store: SQLiteSessionStore) -> None:
    calls = (
        ToolCall(id="call_1", name="echo", arguments='{"text": "caf\u00e9 \U0001f600"}'),
        ToolCall(id="call_2", name="echo", arguments="not json {"),
        ToolCall(id="call_3", name="noargs", arguments=""),
    )
    await store.create_task("t1", "c1", T0, None)
    await store.append(
        "c1",
        [
            AssistantMessage(content="calling", tool_calls=calls),
            ToolMessage(tool_call_id="call_1", content="done"),
        ],
        task_id="t1",
        now_ms=T0 + 1,
    )

    _, history = await store.get_history("c1")

    assert history == [
        AssistantMessage(content="calling", tool_calls=calls),
        ToolMessage(tool_call_id="call_1", content="done"),
    ]


async def test_lone_surrogate_from_stream_assembler_is_stored_as_replacement_character(
    store: SQLiteSessionStore, db: Database
) -> None:
    assembler = StreamAssembler()
    chunk = assembler.feed(
        {
            "content": "a\ud800b",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "echo", "arguments": '{"x":"\udc00"}'},
                }
            ],
        }
    )
    response = assembler.finish(set())
    assert chunk == f"a{REPLACEMENT}b"

    await store.create_task("t1", "c1", T0, None)
    await store.append(
        "c1",
        [AssistantMessage(content=response.text, tool_calls=response.tool_calls)],
        task_id="t1",
        now_ms=T0 + 1,
    )

    _, history = await store.get_history("c1")

    assert history == [
        AssistantMessage(
            content=f"a{REPLACEMENT}b",
            tool_calls=(ToolCall("call_1", "echo", f'{{"x":"{REPLACEMENT}"}}'),),
        )
    ]
    content, calls = _rows(db, "SELECT CAST(content AS BLOB), tool_calls FROM messages")[0]
    assert content == f"a{REPLACEMENT}b".encode()
    assert REPLACEMENT in json.loads(calls)[0]["arguments"]


async def test_a_surrogate_that_reaches_the_store_fails_the_whole_commit(
    store: SQLiteSessionStore, db: Database
) -> None:
    await store.create_task("t1", "c1", T0, None)

    with pytest.raises(StorageError):
        await store.append(
            "c1", [UserMessage("fine"), UserMessage("bad \ud800")], task_id="t1", now_ms=T0 + 1
        )

    assert await store.get_history("c1") == (None, [])
    assert _count(db, "messages") == 0


async def test_system_prompt_slot_is_single_and_not_part_of_history(
    store: SQLiteSessionStore,
) -> None:
    await store.create_task("t1", "c1", T0, None)
    await store.set_system_prompt("c1", SystemMessage("first"), now_ms=T0 + 1)
    await store.set_system_prompt("c1", SystemMessage("second"), now_ms=T0 + 2)
    await store.append("c1", [UserMessage("hi")], task_id="t1", now_ms=T0 + 3)

    prompt, history = await store.get_history("c1")

    assert prompt == SystemMessage("second")
    assert history == [UserMessage("hi")]

    await store.set_system_prompt("c1", None, now_ms=T0 + 4)
    assert (await store.get_history("c1"))[0] is None


async def test_message_seq_continues_per_context(store: SQLiteSessionStore, db: Database) -> None:
    await store.create_task("t1", "c1", T0, None)
    await store.create_task("t2", "c2", T0, None)
    await store.append("c1", [UserMessage("a"), UserMessage("b")], task_id="t1", now_ms=T0 + 1)
    await store.append("c2", [UserMessage("x")], task_id="t2", now_ms=T0 + 1)
    await store.append("c1", [UserMessage("c")], task_id="t1", now_ms=T0 + 2)

    rows = _rows(db, "SELECT context_id, seq, task_id FROM messages ORDER BY context_id, seq")

    assert rows == [("c1", 0, "t1"), ("c1", 1, "t1"), ("c1", 2, "t1"), ("c2", 0, "t2")]


# --- Task creation and state (2.1, 18.30) -----------------------------------------------------


async def test_task_id_collision_with_a_persisted_row_fails_creation(
    store: SQLiteSessionStore, db: Database
) -> None:
    await store.create_task("t1", "c1", T0, "original")
    await store.append("c1", [UserMessage("kept")], task_id="t1", now_ms=T0 + 1)

    with pytest.raises(StorageError, match="already exists"):
        await store.create_task("t1", "c2", T0 + 5, "other")

    record = await store.get_task("t1")
    assert record is not None
    assert record.context_id == "c1"
    assert record.created_at_ms == T0
    assert await store.get_a2a_snapshot("t1") == "original"
    assert _rows(db, "SELECT context_id FROM contexts") == [("c1",)]
    assert (await store.get_history("c1"))[1] == [UserMessage("kept")]


async def test_task_id_collision_in_the_same_context_leaves_last_activity_alone(
    store: SQLiteSessionStore, db: Database
) -> None:
    await store.create_task("t1", "c1", T0, None)
    before = _rows(db, "SELECT last_activity_ms FROM contexts")

    with pytest.raises(StorageError):
        await store.create_task("t1", "c1", T0 + 99, None)

    assert _rows(db, "SELECT last_activity_ms FROM contexts") == before
    assert _count(db, "tasks") == 1


async def test_failed_creation_through_the_task_store_leaves_no_rows(tmp_path: Path) -> None:
    # A clean run counts the statements of `save`; the second run fails its COMMIT.
    counting = FaultFactory()
    database = _open(tmp_path / "count.db", counting)
    try:
        clock = FakeClock()
        projected = ProjectedTaskStore(
            SQLiteSessionStore(database, Redactor([]), clock), clock=clock
        )
        base = counting.connection.calls
        await projected.save(_proto("t1", "c1"), SERVER_CONTEXT)
        last_statement = counting.connection.calls
    finally:
        database.close()
    assert last_statement > base

    failing = FaultFactory(fail_at=last_statement)
    database = _open(tmp_path / "fail.db", failing)
    try:
        clock = FakeClock()
        projected = ProjectedTaskStore(
            SQLiteSessionStore(database, Redactor([]), clock), clock=clock
        )
        future = projected.expect_creation("t1")

        with pytest.raises(StorageWriteError):
            await projected.save(_proto("t1", "c1"), SERVER_CONTEXT)

        assert failing.connection.fired == (last_statement,)
        assert future.done()
        assert isinstance(future.exception(), StorageError)
        assert _count(database, "tasks") == 0
        assert _count(database, "contexts") == 0
        assert await projected.get("t1", SERVER_CONTEXT) is None
    finally:
        database.close()


async def test_state_change_is_a_compare_and_set(store: SQLiteSessionStore) -> None:
    await _make_task(store, "t1", "c1", state=TaskState.WORKING, updated_at_ms=T0 + 1)

    with pytest.raises(StaleState):
        await store.set_state(
            "c1",
            _change("t1", TaskState.COMPLETED, expected=frozenset({TaskState.SUBMITTED})),
            now_ms=T0 + 2,
        )

    record = await store.get_task("t1")
    assert record is not None
    assert record.state == to_sdk(TaskState.WORKING)
    assert record.updated_at_ms == T0 + 1


async def test_failed_commit_applies_nothing(store: SQLiteSessionStore, db: Database) -> None:
    await _make_task(store, "t1", "c1", state=TaskState.WORKING, updated_at_ms=T0 + 1)
    txn = StorageTxn(
        context_id="c1",
        now_ms=T0 + 5,
        appends=(("t1", UserMessage("lost")),),
        system_prompt=SystemMessage("lost"),
        state=_change("t1", TaskState.COMPLETED, expected=frozenset({TaskState.SUBMITTED})),
        artifacts=(ArtifactWrite("t1", "response-0", 0, "lost", True),),
    )

    with pytest.raises(StaleState):
        await store.commit(txn)

    assert await store.get_history("c1") == (None, [])
    assert _count(db, "artifacts") == 0


async def test_status_message_is_redacted_before_it_is_stored(
    db: Database, clock: FakeClock
) -> None:
    secret_store = SQLiteSessionStore(db, Redactor(["hunter2-secret-token"]), clock)
    await secret_store.create_task("t1", "c1", T0, None)

    await secret_store.set_state(
        "c1",
        _change("t1", TaskState.FAILED, status_message="auth failed: hunter2-secret-token"),
        now_ms=T0 + 1,
    )

    record = await secret_store.get_task("t1")
    assert record is not None
    assert record.status_message is not None
    assert "hunter2-secret-token" not in record.status_message
    assert record.status_message.startswith("auth failed: ")
    assert "hunter2-secret-token" not in str(_rows(db, "SELECT * FROM tasks"))


async def test_artifact_write_replaces_the_artifact_with_the_same_id(
    store: SQLiteSessionStore,
) -> None:
    await store.create_task("t1", "c1", T0, None)
    for text, closed in (("par", False), ("partial", False), ("partial text", True)):
        await store.commit(
            StorageTxn(
                context_id="c1",
                now_ms=T0 + 1,
                artifacts=(
                    ArtifactWrite("t1", "response-0", 0, text, closed),
                    ArtifactWrite("t1", "response-1", 1, "second", True),
                ),
            )
        )

    record = await store.get_task("t1")

    assert record is not None
    assert record.artifacts == (
        ArtifactRecord("response-0", 0, "partial text", True),
        ArtifactRecord("response-1", 1, "second", True),
    )


# --- ProjectedTaskStore.save (2.1, 18.9, 18.30) -----------------------------------------------


async def test_save_of_an_unknown_submitted_task_creates_it_and_resolves_the_future(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore, db: Database
) -> None:
    future = tasks.expect_creation("t1")
    task = _proto("t1", "c1")

    await tasks.save(task, SERVER_CONTEXT)

    assert future.done()
    assert future.result() is None
    record = await store.get_task("t1")
    assert record is not None
    assert record.context_id == "c1"
    assert record.state == to_sdk(TaskState.SUBMITTED)
    assert _count(db, "contexts") == 1
    assert await store.get_a2a_snapshot("t1") == snapshot_json(task)


async def test_save_of_an_unknown_non_submitted_task_stores_nothing(
    tasks: ProjectedTaskStore, db: Database
) -> None:
    logs = _capture_logs()

    for state in (
        SdkTaskState.TASK_STATE_WORKING,
        SdkTaskState.TASK_STATE_FAILED,
        SdkTaskState.TASK_STATE_COMPLETED,
        SdkTaskState.TASK_STATE_INPUT_REQUIRED,
    ):
        await tasks.save(_proto("ghost", "c1", state), SERVER_CONTEXT)

    assert _count(db, "tasks") == 0
    assert _count(db, "contexts") == 0
    assert _count(db, "messages") == 0
    assert await tasks.get("ghost", SERVER_CONTEXT) is None
    ignored = _events(logs, "task_store_unknown_task_ignored")
    assert len(ignored) == 4
    assert {record["level"] for record in ignored} == {"debug"}
    assert _events(logs, "task_store_divergent_save") == []


async def test_divergent_terminal_save_logs_a_warning_and_changes_nothing(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore, db: Database
) -> None:
    submitted = _proto("t1", "c1")
    await store.create_task("t1", "c1", T0, snapshot_json(submitted))
    await store.set_state(
        "c1", _change("t1", TaskState.WORKING, status_message="busy"), now_ms=T0 + 1
    )
    await store.commit(
        StorageTxn(
            context_id="c1",
            now_ms=T0 + 2,
            artifacts=(ArtifactWrite("t1", "response-0", 0, "so far", False),),
        )
    )
    before_tasks = _rows(db, "SELECT * FROM tasks")
    before_artifacts = _rows(db, "SELECT * FROM artifacts")
    before_context = _rows(db, "SELECT * FROM contexts")
    logs = _capture_logs(level="warning")

    divergent = _proto("t1", "c1", SdkTaskState.TASK_STATE_COMPLETED)
    divergent.artifacts.add(artifact_id="sdk-made-up", parts=[Part(text="never committed")])
    await tasks.save(divergent, SERVER_CONTEXT)

    warnings = _events(logs, "task_store_divergent_save")
    assert len(warnings) == 1
    assert warnings[0]["level"] == "warning"
    assert warnings[0]["task_id"] == "t1"
    assert warnings[0]["committed_state"] == "TASK_STATE_WORKING"
    assert _rows(db, "SELECT * FROM tasks") == before_tasks
    assert _rows(db, "SELECT * FROM artifacts") == before_artifacts
    assert _rows(db, "SELECT * FROM contexts") == before_context
    record = await store.get_task("t1")
    assert record is not None
    assert record.state == to_sdk(TaskState.WORKING)


@pytest.mark.parametrize(
    "state",
    [SdkTaskState.TASK_STATE_WORKING, SdkTaskState.TASK_STATE_INPUT_REQUIRED],
)
async def test_non_terminal_lag_is_not_a_divergent_save(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore, state: SdkTaskState
) -> None:
    await store.create_task("t1", "c1", T0, snapshot_json(_proto("t1", "c1")))
    logs = _capture_logs(level="warning")

    await tasks.save(_proto("t1", "c1", state), SERVER_CONTEXT)

    assert _events(logs, "task_store_divergent_save") == []
    record = await store.get_task("t1")
    assert record is not None
    assert record.state == to_sdk(TaskState.SUBMITTED)


async def test_terminal_save_that_matches_the_committed_state_does_not_warn(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await _make_task(
        store,
        "t1",
        "c1",
        state=TaskState.COMPLETED,
        updated_at_ms=T0 + 1,
        snapshot=snapshot_json(_proto("t1", "c1")),
    )
    logs = _capture_logs(level="warning")

    await tasks.save(_proto("t1", "c1", SdkTaskState.TASK_STATE_COMPLETED), SERVER_CONTEXT)

    assert _records(logs) == []


async def test_save_of_a_known_task_keeps_only_sdk_history_and_metadata(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore, db: Database
) -> None:
    await tasks.save(_proto("t1", "c1"), SERVER_CONTEXT)
    saved = _proto("t1", "c1", SdkTaskState.TASK_STATE_WORKING)
    saved.history.append(Message(message_id="m1", role=Role.ROLE_USER, parts=[Part(text="hi")]))
    saved.artifacts.add(artifact_id="sdk-artifact", parts=[Part(text="ignored")])

    await tasks.save(saved, SERVER_CONTEXT)

    record = await store.get_task("t1")
    assert record is not None
    assert record.state == to_sdk(TaskState.SUBMITTED)
    assert record.artifacts == ()
    assert _count(db, "artifacts") == 0
    served = await tasks.get("t1", SERVER_CONTEXT)
    assert served is not None
    assert [message.message_id for message in served.history] == ["m1"]
    assert served.status.state == SdkTaskState.TASK_STATE_SUBMITTED
    assert len(served.artifacts) == 0


async def test_save_skips_the_write_when_the_snapshot_is_unchanged(
    tasks: ProjectedTaskStore, db: Database
) -> None:
    await tasks.save(_proto("t1", "c1"), SERVER_CONTEXT)
    changes_before = int(db.run_sync(lambda conn: conn.total_changes))

    await tasks.save(_proto("t1", "c1", SdkTaskState.TASK_STATE_WORKING), SERVER_CONTEXT)

    assert int(db.run_sync(lambda conn: conn.total_changes)) == changes_before


# --- ProjectedTaskStore.get -------------------------------------------------------------------


async def test_get_builds_the_task_from_the_committed_rows(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await store.create_task("t1", "c1", T0, None)
    await store.commit(
        StorageTxn(
            context_id="c1",
            now_ms=T0 + 7,
            state=StateChange(
                task_id="t1",
                expected_from=ALL_STATES,
                new=TaskState.COMPLETED,
                status_message="all done",
                metadata={"k": "v"},
            ),
            artifacts=(
                ArtifactWrite("t1", "response-1", 1, "second", True),
                ArtifactWrite("t1", "response-0", 0, "first", True),
            ),
        )
    )

    task = await tasks.get("t1", SERVER_CONTEXT)

    assert task is not None
    assert (task.id, task.context_id) == ("t1", "c1")
    assert task.status.state == SdkTaskState.TASK_STATE_COMPLETED
    assert task.status.timestamp.ToMilliseconds() == T0 + 7
    assert [part.text for part in task.status.message.parts] == ["all done"]
    assert task.status.message.metadata["k"] == "v"
    assert [(a.artifact_id, a.parts[0].text) for a in task.artifacts] == [
        ("response-0", "first"),
        ("response-1", "second"),
    ]


async def test_get_of_an_absent_task_is_none(tasks: ProjectedTaskStore) -> None:
    assert await tasks.get("nope", SERVER_CONTEXT) is None


async def test_get_overlays_live_artifacts_for_a_running_task_only(
    store: SQLiteSessionStore, clock: FakeClock
) -> None:
    live = _Live(
        {
            "running": [
                ArtifactRecord("response-0", 0, "committed and more", False),
                ArtifactRecord("response-1", 1, "only live", False),
            ],
            "done": [ArtifactRecord("response-0", 0, "stale live buffer", False)],
        }
    )
    projected = ProjectedTaskStore(store, live, clock)
    for task_id, state in (("running", TaskState.WORKING), ("done", TaskState.COMPLETED)):
        await _make_task(store, task_id, f"ctx-{task_id}", state=state, updated_at_ms=T0 + 1)
        await store.commit(
            StorageTxn(
                context_id=f"ctx-{task_id}",
                now_ms=T0 + 2,
                artifacts=(ArtifactWrite(task_id, "response-0", 0, "committed", False),),
            )
        )

    running = await projected.get("running", SERVER_CONTEXT)
    done = await projected.get("done", SERVER_CONTEXT)

    assert running is not None
    assert [a.parts[0].text for a in running.artifacts] == ["committed and more", "only live"]
    assert done is not None
    assert [a.parts[0].text for a in done.artifacts] == ["committed"]


async def test_snapshot_returns_the_committed_task_or_raises(tasks: ProjectedTaskStore) -> None:
    await tasks.save(_proto("t1", "c1"), SERVER_CONTEXT)

    task = await tasks.snapshot("t1")

    assert task.id == "t1"
    assert task.status.state == SdkTaskState.TASK_STATE_SUBMITTED
    with pytest.raises(TaskNotFoundError):
        await tasks.snapshot("nope")


# --- list: paging and ordering ----------------------------------------------------------------


async def _seed_list(store: SQLiteSessionStore) -> None:
    """Five Tasks. By `(updated_at_ms DESC, task_id DESC)` they list as e, c, b, d, a."""
    plan: list[tuple[str, str, TaskState, int]] = [
        ("a", "c1", TaskState.COMPLETED, T0 + 100),
        ("b", "c1", TaskState.WORKING, T0 + 300),
        ("c", "c2", TaskState.WORKING, T0 + 300),
        ("d", "c2", TaskState.WORKING, T0 + 200),
        ("e", "c1", TaskState.WORKING, T0 + 400),
    ]
    for task_id, context_id, state, updated in plan:
        await _make_task(store, task_id, context_id, state=state, updated_at_ms=updated)


async def test_store_lists_newest_update_first_with_task_id_as_tie_break(
    store: SQLiteSessionStore,
) -> None:
    await _seed_list(store)

    page = await store.list_tasks(TaskQuery())

    assert [task.task_id for task in page.tasks] == ["e", "c", "b", "d", "a"]
    assert page.next_page_token is None


async def test_store_pages_follow_the_cursor_without_gaps_or_repeats(
    store: SQLiteSessionStore,
) -> None:
    await _seed_list(store)
    seen: list[list[str]] = []
    token: str | None = None

    while True:
        page = await store.list_tasks(TaskQuery(page_size=2, page_token=token))
        seen.append([task.task_id for task in page.tasks])
        token = page.next_page_token
        if token is None:
            break

    assert seen == [["e", "c"], ["b", "d"], ["a"]]


async def test_store_page_that_ends_exactly_at_the_last_task_has_no_next_token(
    store: SQLiteSessionStore,
) -> None:
    await _seed_list(store)

    page = await store.list_tasks(TaskQuery(page_size=5))

    assert len(page.tasks) == 5
    assert page.next_page_token is None


@pytest.mark.parametrize("token", ["", "not base64!", "bnVsbA=="])
async def test_store_rejects_a_malformed_page_token(store: SQLiteSessionStore, token: str) -> None:
    with pytest.raises(ValueError, match="page token"):
        await store.list_tasks(TaskQuery(page_token=token))


async def test_store_rejects_a_page_size_below_one(store: SQLiteSessionStore) -> None:
    with pytest.raises(ValueError, match="page_size"):
        await store.list_tasks(TaskQuery(page_size=0))


async def test_task_store_list_pages_in_order_and_reports_the_total(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await _seed_list(store)
    pages: list[list[str]] = []
    totals: list[int] = []
    token = ""

    while True:
        response = await tasks.list(ListTasksRequest(page_size=2, page_token=token), SERVER_CONTEXT)
        pages.append([task.id for task in response.tasks])
        totals.append(response.total_size)
        assert response.page_size == 2
        token = response.next_page_token
        if not token:
            break

    assert pages == [["e", "c"], ["b", "d"], ["a"]]
    assert totals == [5, 5, 5]


async def test_task_store_list_without_paging_returns_everything(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await _seed_list(store)

    response = await tasks.list(ListTasksRequest(), SERVER_CONTEXT)

    assert [task.id for task in response.tasks] == ["e", "c", "b", "d", "a"]
    assert response.total_size == 5
    assert response.next_page_token == ""


async def test_task_store_list_filters_by_context_and_status(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await _seed_list(store)

    by_context = await tasks.list(ListTasksRequest(context_id="c2"), SERVER_CONTEXT)
    by_status = await tasks.list(
        ListTasksRequest(status=SdkTaskState.TASK_STATE_COMPLETED), SERVER_CONTEXT
    )
    both = await tasks.list(
        ListTasksRequest(context_id="c1", status=SdkTaskState.TASK_STATE_WORKING), SERVER_CONTEXT
    )
    unused_state = await tasks.list(
        ListTasksRequest(status=SdkTaskState.TASK_STATE_REJECTED), SERVER_CONTEXT
    )

    assert [task.id for task in by_context.tasks] == ["c", "d"]
    assert [task.id for task in by_status.tasks] == ["a"]
    assert [task.id for task in both.tasks] == ["e", "b"]
    assert list(unused_state.tasks) == []


async def test_task_store_status_timestamp_filter_includes_the_boundary(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await _seed_list(store)
    request = ListTasksRequest()
    request.status_timestamp_after.FromMilliseconds(T0 + 300)

    response = await tasks.list(request, SERVER_CONTEXT)

    assert [task.id for task in response.tasks] == ["e", "c", "b"]
    assert response.total_size == 3


async def test_task_store_list_rejects_a_bad_page_token(tasks: ProjectedTaskStore) -> None:
    with pytest.raises(InvalidParamsError):
        await tasks.list(ListTasksRequest(page_token="%%%"), SERVER_CONTEXT)


# --- delete -----------------------------------------------------------------------------------


async def test_delete_removes_the_task_and_artifacts_but_keeps_the_context_and_history(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore, db: Database
) -> None:
    await _make_task(store, "t1", "c1", state=TaskState.COMPLETED, updated_at_ms=T0 + 1)
    await store.create_task("t2", "c1", T0 + 2, None)
    await store.set_system_prompt("c1", SystemMessage("prompt"), now_ms=T0 + 2)
    await store.commit(
        StorageTxn(
            context_id="c1",
            now_ms=T0 + 3,
            appends=(("t1", UserMessage("q")), ("t1", AssistantMessage("a"))),
            artifacts=(
                ArtifactWrite("t1", "response-0", 0, "a", True),
                ArtifactWrite("t2", "response-0", 0, "other", False),
            ),
        )
    )

    await tasks.delete("t1", SERVER_CONTEXT)

    assert await tasks.get("t1", SERVER_CONTEXT) is None
    assert _rows(db, "SELECT task_id FROM artifacts") == [("t2",)]
    assert _rows(db, "SELECT context_id FROM contexts") == [("c1",)]
    prompt, history = await store.get_history("c1")
    assert prompt == SystemMessage("prompt")
    assert history == [UserMessage("q"), AssistantMessage("a")]
    # The message rows keep their producing task id as written; nothing cascades to them.
    assert _rows(db, "SELECT count(*) FROM messages") == [(2,)]
    survivor = await tasks.get("t2", SERVER_CONTEXT)
    assert survivor is not None
    assert [a.parts[0].text for a in survivor.artifacts] == ["other"]


async def test_delete_of_an_unknown_task_is_a_no_op(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await store.create_task("t1", "c1", T0, None)

    await tasks.delete("nope", SERVER_CONTEXT)
    await tasks.delete("lone \ud800 surrogate", SERVER_CONTEXT)

    assert await tasks.get("t1", SERVER_CONTEXT) is not None


async def test_a_context_survives_deleting_its_only_task_and_accepts_a_new_one(
    tasks: ProjectedTaskStore, store: SQLiteSessionStore
) -> None:
    await store.create_task("t1", "c1", T0, None)
    await store.append("c1", [UserMessage("earlier")], task_id="t1", now_ms=T0 + 1)
    await tasks.delete("t1", SERVER_CONTEXT)

    created_new_context = await store.create_task("t2", "c1", T0 + 2, None)

    assert created_new_context is False
    assert (await store.get_history("c1"))[1] == [UserMessage("earlier")]
