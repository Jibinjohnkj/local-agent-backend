"""Property 49: persistence round trip (Requirements 18.3-18.7, 2.13).

Every example writes a generated set of Contexts and Tasks through `SQLiteSessionStore` into a
fresh database file, closes the `Database`, opens a new one on the same file, and reads everything
back. The generated text covers CR, LF, CRLF, tabs, NUL, non-BMP characters, empty and
whitespace-only strings, `None` versus `""` assistant content, Tool_Call arguments that are not
JSON, System_Prompt present or absent, metadata objects, and artifacts in any number and order.

What must come back equal: the Message_History (length, order, roles, contents, Tool_Calls,
`tool_call_id`s), the System_Prompt slot, each Task's state, status message, metadata, and artifacts
in emission order, the private A2A snapshot, and the integer timestamps (`created_at_ms`,
`updated_at_ms`, `last_activity_ms`) exactly. `ProjectedTaskStore.get` on the reopened store must
return a Task with the same state, status message, and artifacts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from a2a.types import Message, Part, Role, Task
from google.protobuf import json_format
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import FakeClock
from local_agent.a2a_server.a2a_compat import ServerCallContext
from local_agent.a2a_server.events import TaskState, to_sdk
from local_agent.logging_setup import Redactor
from local_agent.models import (
    ArtifactRecord,
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from local_agent.orchestrator.context_window import split_turns, trim_to_budget
from local_agent.orchestrator.prompt_builder import build_system_prompt
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import ArtifactWrite, StateChange, StorageTxn
from local_agent.storage import migrations
from local_agent.storage.database import Database
from local_agent.storage.task_store import ProjectedTaskStore, snapshot_json

# Largest millisecond value an A2A `Timestamp` can hold (the end of year 9999). Timestamps above
# 2**53 would also exercise exactness, but `ProjectedTaskStore.get` must be able to render them.
MAX_MS = 253_402_300_799_999

SERVER_CONTEXT = ServerCallContext()


# --- Strategies -------------------------------------------------------------------------------

_SPECIAL_PIECES = ["\r", "\n", "\r\n", "\t", "\x00", "\U0001f600", "\U0010ffff", "a", " ", "\u2028"]


def stored_texts() -> st.SearchStrategy[str]:
    """Text as the store sees it: any Unicode except lone surrogates, which ingress replaces.

    Mixes free Unicode, the awkward fixed strings the requirement names (empty, whitespace-only,
    CR, LF, CRLF, NUL), and runs built only from those awkward pieces.
    """
    return st.one_of(
        st.text(max_size=30),
        st.sampled_from(["", " ", "\t", "\r", "\n", "\r\n", "\x00", " \t\r\n ", "a\x00b", "\n\r"]),
        st.lists(st.sampled_from(_SPECIAL_PIECES), max_size=10).map("".join),
    )


def _json_values() -> st.SearchStrategy[Any]:
    scalar = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-(2**31), 2**31),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        stored_texts(),
    )
    return st.recursive(
        scalar,
        lambda inner: st.one_of(
            st.lists(inner, max_size=3),
            st.dictionaries(stored_texts(), inner, max_size=3),
        ),
        max_leaves=6,
    )


def _metadata() -> st.SearchStrategy[dict[str, Any]]:
    return st.dictionaries(stored_texts(), _json_values(), max_size=3)


_tool_calls = st.builds(ToolCall, id=stored_texts(), name=stored_texts(), arguments=stored_texts())

_messages: st.SearchStrategy[ChatMessage] = st.one_of(
    st.builds(UserMessage, content=stored_texts()),
    st.builds(
        AssistantMessage,
        content=st.one_of(st.none(), stored_texts()),
        tool_calls=st.lists(_tool_calls, max_size=3).map(tuple),
    ),
    st.builds(ToolMessage, tool_call_id=stored_texts(), content=stored_texts()),
)


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    ordinal: int
    text: str
    closed: bool


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    final_state: TaskState
    status_message: str | None
    metadata: dict[str, Any]
    messages: tuple[ChatMessage, ...]
    artifacts: tuple[ArtifactSpec, ...]  # in emission order
    write_order: tuple[int, ...]  # the order the artifacts are written in (a permutation)
    snapshot_text: str | None  # text of one SDK history message in the A2A snapshot, if any
    created_at_ms: int
    appended_at_ms: int
    finished_at_ms: int


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    prompts: tuple[str | None, ...]  # System_Prompt slot values written in order; none = never set
    tasks: tuple[TaskSpec, ...]
    new_user_text: str  # the user message of a new Task started after the reopen


@st.composite
def _task_specs(draw: st.DrawFn, task_id: str) -> TaskSpec:
    count = draw(st.integers(0, 4))
    ids = draw(st.lists(stored_texts(), min_size=count, max_size=count, unique=True))
    gaps = draw(st.lists(st.integers(1, 5), min_size=count, max_size=count))
    artifacts: list[ArtifactSpec] = []
    ordinal = draw(st.integers(0, 3)) - 1
    for artifact_id, gap in zip(ids, gaps, strict=True):
        ordinal += gap
        artifacts.append(
            ArtifactSpec(artifact_id, ordinal, draw(stored_texts()), draw(st.booleans()))
        )
    return TaskSpec(
        task_id=task_id,
        final_state=draw(st.sampled_from(list(TaskState))),
        status_message=draw(st.one_of(st.none(), stored_texts())),
        metadata=draw(_metadata()),
        messages=tuple(draw(st.lists(_messages, max_size=4))),
        artifacts=tuple(artifacts),
        write_order=tuple(draw(st.permutations(range(count)))),
        snapshot_text=draw(st.one_of(st.none(), stored_texts())),
        created_at_ms=draw(st.integers(0, MAX_MS)),
        appended_at_ms=draw(st.integers(0, MAX_MS)),
        finished_at_ms=draw(st.integers(0, MAX_MS)),
    )


@st.composite
def context_task_sets(draw: st.DrawFn) -> tuple[ContextSpec, ...]:
    """One to three Contexts, each with one to three Tasks, with globally unique identifiers."""
    task_counts = draw(st.lists(st.integers(1, 3), min_size=1, max_size=3))
    context_ids = draw(
        st.lists(stored_texts(), min_size=len(task_counts), max_size=len(task_counts), unique=True)
    )
    task_ids = iter(
        draw(
            st.lists(
                stored_texts(),
                min_size=sum(task_counts),
                max_size=sum(task_counts),
                unique=True,
            )
        )
    )
    contexts: list[ContextSpec] = []
    for context_id, task_count in zip(context_ids, task_counts, strict=True):
        tasks = tuple([draw(_task_specs(next(task_ids))) for _ in range(task_count)])
        prompts = tuple(draw(st.lists(st.one_of(st.none(), stored_texts()), max_size=2)))
        contexts.append(ContextSpec(context_id, prompts, tasks, draw(stored_texts())))
    return tuple(contexts)


# --- Model of what the store must hold --------------------------------------------------------


@dataclass
class ExpectedContext:
    system_prompt: SystemMessage | None
    history: list[ChatMessage]
    created_at_ms: int
    last_activity_ms: int


@dataclass
class Expected:
    contexts: dict[str, ExpectedContext] = field(default_factory=dict)


def _snapshot(spec: TaskSpec) -> str | None:
    if spec.snapshot_text is None:
        return None
    history = Message(message_id="m", role=Role.ROLE_USER, parts=[Part(text=spec.snapshot_text)])
    return snapshot_json(Task(history=[history]))


def _expected_artifacts(spec: TaskSpec) -> tuple[ArtifactRecord, ...]:
    return tuple(ArtifactRecord(a.artifact_id, a.ordinal, a.text, a.closed) for a in spec.artifacts)


# --- Driving the store ------------------------------------------------------------------------


def _open(path: Path, *, create: bool) -> Database:
    database = Database.open(path, create=create)
    try:
        migrations.apply(database, str(path))
    except BaseException:
        database.close()
        raise
    return database


def _raw_contexts(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    return conn.execute(
        "SELECT context_id, system_prompt, created_at_ms, last_activity_ms"
        " FROM contexts ORDER BY context_id"
    ).fetchall()


async def _write(store: SQLiteSessionStore, contexts: Sequence[ContextSpec]) -> Expected:
    """Write every Context and Task the way the Orchestrator does, and model the result."""
    expected = Expected()
    for context in contexts:
        model: ExpectedContext | None = None
        for task in context.tasks:
            await store.create_task(
                task.task_id, context.context_id, task.created_at_ms, _snapshot(task)
            )
            if model is None:
                model = ExpectedContext(None, [], task.created_at_ms, task.created_at_ms)
                expected.contexts[context.context_id] = model
                for prompt in context.prompts:
                    await store.set_system_prompt(
                        context.context_id,
                        None if prompt is None else SystemMessage(content=prompt),
                        now_ms=task.created_at_ms,
                    )
                    model.system_prompt = None if prompt is None else SystemMessage(prompt)
            model.last_activity_ms = task.created_at_ms

            if task.messages:
                await store.append(
                    context.context_id,
                    task.messages,
                    task_id=task.task_id,
                    now_ms=task.appended_at_ms,
                )
                model.history.extend(task.messages)
                model.last_activity_ms = task.appended_at_ms

            # Write each artifact first as a half-finished buffer, then in full, in a shuffled
            # order. The upsert must leave the full text at the original emission ordinal.
            partial = tuple(
                ArtifactWrite(
                    task.task_id, a.artifact_id, a.ordinal, a.text[: len(a.text) // 2], False
                )
                for a in (task.artifacts[i] for i in task.write_order)
            )
            if partial:
                await store.commit(
                    StorageTxn(
                        context_id=context.context_id,
                        now_ms=task.appended_at_ms,
                        artifacts=partial,
                    )
                )
            await store.commit(
                StorageTxn(
                    context_id=context.context_id,
                    now_ms=task.finished_at_ms,
                    state=StateChange(
                        task_id=task.task_id,
                        expected_from=frozenset({TaskState.SUBMITTED}),
                        new=task.final_state,
                        status_message=task.status_message,
                        metadata=task.metadata,
                    ),
                    artifacts=tuple(
                        ArtifactWrite(task.task_id, a.artifact_id, a.ordinal, a.text, a.closed)
                        for a in (task.artifacts[i] for i in task.write_order)
                    ),
                )
            )
            if task.final_state is not TaskState.SUBMITTED:
                model.last_activity_ms = task.finished_at_ms
    return expected


async def _check_session_store(
    store: SQLiteSessionStore, contexts: Sequence[ContextSpec], expected: Expected
) -> None:
    for context in contexts:
        want = expected.contexts[context.context_id]
        prompt, history = await store.get_history(context.context_id)
        assert prompt == want.system_prompt
        assert history == want.history
        # Equality alone would let `None` and `""` blur if both compared equal; they do not in
        # Python, but the explicit check documents what 18.3 asks for.
        for loaded, original in zip(history, want.history, strict=True):
            if isinstance(original, AssistantMessage):
                assert isinstance(loaded, AssistantMessage)
                assert loaded.content == original.content
                assert (loaded.content is None) == (original.content is None)

        for task in context.tasks:
            record = await store.get_task(task.task_id)
            assert record is not None
            assert record.task_id == task.task_id
            assert record.context_id == context.context_id
            assert record.state == to_sdk(task.final_state)
            assert record.status_message == task.status_message
            assert (record.status_message is None) == (task.status_message is None)
            assert dict(record.metadata) == task.metadata
            assert record.artifacts == _expected_artifacts(task)
            assert record.created_at_ms == task.created_at_ms
            assert record.updated_at_ms == task.finished_at_ms
            assert await store.get_a2a_snapshot(task.task_id) == _snapshot(task)


async def _check_task_store(tasks: ProjectedTaskStore, contexts: Sequence[ContextSpec]) -> None:
    for context in contexts:
        for spec in context.tasks:
            task = await tasks.get(spec.task_id, SERVER_CONTEXT)
            assert task is not None
            assert task.id == spec.task_id
            assert task.context_id == context.context_id
            assert task.status.state == to_sdk(spec.final_state)
            assert task.status.timestamp.ToMilliseconds() == spec.finished_at_ms

            message = task.status.message
            if spec.status_message is None:
                assert not message.parts
            else:
                assert [part.text for part in message.parts] == [spec.status_message]
            assert json_format.MessageToDict(message.metadata) == spec.metadata

            assert [(a.artifact_id, [p.text for p in a.parts]) for a in task.artifacts] == [
                (a.artifact_id, [a.text]) for a in spec.artifacts
            ]
            if spec.snapshot_text is not None:
                assert [p.text for m in task.history for p in m.parts] == [spec.snapshot_text]


async def _verify(
    database: Database,
    contexts: Sequence[ContextSpec],
    expected: Expected,
    raw_before: list[tuple[Any, ...]],
) -> None:
    clock = FakeClock()
    store = SQLiteSessionStore(database, Redactor([]), clock)
    await _check_session_store(store, contexts, expected)
    await _check_task_store(ProjectedTaskStore(store, clock=clock), contexts)

    raw_after = await database.read(_raw_contexts)
    assert raw_after == raw_before
    for row in raw_after:
        want = expected.contexts[row[0]]
        assert row[2] == want.created_at_ms
        assert row[3] == want.last_activity_ms

    # Last, because it writes: a new Task in each reopened Context.
    for context in contexts:
        await _check_first_request(store, context, expected.contexts[context.context_id])


def _estimate(messages: Sequence[ChatMessage]) -> int:
    return sum(len(json.dumps(message.to_openai(), ensure_ascii=False)) for message in messages)


async def _check_first_request(
    store: SQLiteSessionStore, context: ContextSpec, want: ExpectedContext
) -> None:
    """The first request for a new Task is the persisted history plus the new user message.

    This follows the steps `TaskRunner._build_request` performs (the runner is not part of the
    codebase yet): append the user message as the runner's first commit, rebuild the System_Prompt
    into the single slot, load the persisted history, and trim whole turns to the budget.
    """
    task_id = f"new-task-{uuid.uuid4().hex}"
    await store.create_task(task_id, context.context_id, MAX_MS, None)
    new_message = UserMessage(content=context.new_user_text)
    await store.append(context.context_id, [new_message], task_id=task_id, now_ms=MAX_MS)

    prompt = SystemMessage(content=build_system_prompt(None, []))
    await store.set_system_prompt(context.context_id, prompt, now_ms=MAX_MS)
    slot, history = await store.get_history(context.context_id)
    assert slot == prompt
    budget = _estimate([prompt, *want.history, new_message])
    request = trim_to_budget(slot, split_turns(history), budget, _estimate)

    assert request == [prompt, *want.history, new_message]
    assert [m for m in request if isinstance(m, SystemMessage)] == [prompt]


async def _before_close(
    database: Database, contexts: Sequence[ContextSpec]
) -> tuple[Expected, list[tuple[Any, ...]]]:
    store = SQLiteSessionStore(database, Redactor([]), FakeClock())
    expected = await _write(store, contexts)
    # The same checks before closing localize a failure to the write path or the reopen.
    await _check_session_store(store, contexts, expected)
    return expected, await database.read(_raw_contexts)


# Feature: local-agent-backend, Property 49: Persistence round trip
# **Validates: Requirements 18.3, 18.4, 18.5, 18.6, 18.7, 2.13**
@settings(max_examples=100)
@given(contexts=context_task_sets())
def test_persistence_round_trip(contexts: tuple[ContextSpec, ...]) -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "agent.db"
        database = _open(path, create=True)
        try:
            expected, raw_before = asyncio.run(_before_close(database, contexts))
        finally:
            database.close()

        reopened = _open(path, create=False)
        try:
            asyncio.run(_verify(reopened, contexts, expected, raw_before))
        finally:
            reopened.close()
