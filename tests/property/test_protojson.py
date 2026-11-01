"""Property 60: ProtoJSON enum form (task 20.16).

The SDK dispatchers serialize every response and stream event with protobuf `MessageToDict`, after
`proto_utils.to_stream_response` for events. This test builds Tasks, Status_Events, Token_Events,
and get-task and cancel results with the Backend's own builders, then serializes them the way the
dispatchers do and checks the enum fields of the JSON:

* the Backend's builders are `TaskEventEmitter` + `QueueSink` (Status_Events and Token_Events, which
  reach a real SDK `TaskUpdater`), `ProjectedTaskStore.get` (the get-task and cancel result) and
  `ProjectedTaskStore.snapshot` (the first event of a stream);
* every `status.state` is the `TASK_STATE_*` name of the Task's state, every `role` is `ROLE_USER`
  or `ROLE_AGENT`, and no integer or lower-case spelling stands where an enum belongs.

Two tests run over both Session_Stores. One generates a Task lifecycle (state path, status texts,
history roles, artifact chunks). The other replays the events of a generated scenario script.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from a2a.server.events import Event, EventQueue
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.utils.proto_utils import to_stream_response
from google.protobuf.json_format import MessageToDict
from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    CONTEXT,
    TASK,
    ScenarioScript,
    StoreEnv,
    StoreKind,
    open_store,
    run_scenario,
    scenario_scripts,
    store_kind,
)
from strategies import stored_texts

from fakes import ArtifactEvent, StatusEvent
from local_agent.a2a_server.a2a_compat import ServerCallContext
from local_agent.a2a_server.events import (
    QueueSink,
    TaskEventEmitter,
    TaskState,
    sdk_name,
)
from local_agent.models import ArtifactRecord
from local_agent.storage.task_store import ProjectedTaskStore, snapshot_json

__all__ = ["store_kind"]  # the fixture is used by name below

ROLE_NAMES = {Role.ROLE_USER: "ROLE_USER", Role.ROLE_AGENT: "ROLE_AGENT"}
STATE_NAMES = frozenset(sdk_name(state) for state in TaskState)
LOWER_CASE_STATES = frozenset(state.value for state in TaskState)
ARTIFACT = "response-1"


# --- The serializers the SDK dispatchers use ----------------------------------------------------


def serialize_task(task: Task) -> dict[str, Any]:
    """The get-task and cancel result: `MessageToDict(task)` (JSON-RPC and REST)."""
    return dict(MessageToDict(task, preserving_proto_field_name=False))


def serialize_send_result(task: Task) -> dict[str, Any]:
    """The `SendMessage` result of a Task."""
    return dict(MessageToDict(SendMessageResponse(task=task)))


def serialize_event(event: Event) -> dict[str, Any]:
    """One SSE data object: `StreamResponse` through `MessageToDict`."""
    return dict(MessageToDict(to_stream_response(event), preserving_proto_field_name=False))


# --- Reading the enum fields of a JSON document --------------------------------------------------


def enum_values(doc: Any) -> tuple[list[Any], list[Any]]:
    """Every value under a `state` key and every value under a `role` key, in document order.

    Free text sits under `text` and tool metadata under `metadata`, which the walk skips, so a
    status text of "working" or a tool argument named `state` cannot be mistaken for an enum.
    """
    states: list[Any] = []
    roles: list[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "metadata":
                    continue
                if key == "state":
                    states.append(value)
                elif key == "role":
                    roles.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(doc)
    return states, roles


def assert_enum_form(doc: Any) -> tuple[list[str], list[str]]:
    """Check that every state and role in `doc` is a ProtoJSON enum name. Return both lists."""
    states, roles = enum_values(doc)
    for state in states:
        assert isinstance(state, str), f"state is not a name: {state!r}"  # no integer enum value
        assert state in STATE_NAMES, f"unknown state spelling: {state!r}"
        assert state not in LOWER_CASE_STATES, f"lower-case state: {state!r}"
    for role in roles:
        assert isinstance(role, str), f"role is not a name: {role!r}"
        assert role in {"ROLE_USER", "ROLE_AGENT"}, f"unknown role spelling: {role!r}"
    return list(states), list(roles)


# --- What the Backend sends the SDK -------------------------------------------------------------


class CollectingQueue(EventQueue):
    """An `EventQueue` that keeps what the executor side enqueues, in order."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def enqueue_event(self, event: Event) -> None:
        self.events.append(event)


class EmitterArtifacts:
    """`LiveArtifacts` for one emitter: the buffered text of a Task that is still running."""

    def __init__(self, emitter: TaskEventEmitter) -> None:
        self._emitter = emitter

    def live_artifacts(self, task_id: str) -> Sequence[ArtifactRecord]:
        return self._emitter.live_artifacts()


# --- Strategies ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    state: TaskState
    text: str | None


@dataclass(frozen=True)
class Lifecycle:
    """A Task's history roles, its state path after `submitted`, and the chunks of one artifact."""

    history: tuple[tuple[Role, str], ...]
    steps: tuple[Step, ...]
    chunks: tuple[str, ...]  # non-empty; emitted once the Task first reaches `working`


@st.composite
def lifecycles(draw: st.DrawFn, *, end: TaskState | None = None) -> Lifecycle:
    """A path through the transition table starting `submitted -> working`.

    `end` fixes the last state (for example `canceled`, the cancel result); otherwise the path may
    also stop in `working` or `input-required`, so all six states occur.
    """
    path = [TaskState.WORKING]
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        path.append(
            TaskState.INPUT_REQUIRED if path[-1] is TaskState.WORKING else TaskState.WORKING
        )
    final = draw(
        st.sampled_from([end])
        if end is not None
        else st.sampled_from([None, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED])
    )
    if final is not None:
        if final is TaskState.COMPLETED and path[-1] is TaskState.INPUT_REQUIRED:
            path.append(TaskState.WORKING)
        path.append(final)
    texts = st.one_of(st.none(), stored_texts(max_size=12))
    steps = tuple(Step(state, draw(texts)) for state in path)
    roles = st.sampled_from([Role.ROLE_USER, Role.ROLE_AGENT])
    history = tuple(draw(st.lists(st.tuples(roles, stored_texts(max_size=12)), max_size=4)))
    chunks = tuple(draw(st.lists(stored_texts(max_size=12).filter(bool), max_size=3)))
    return Lifecycle(history, steps, chunks)


# --- Test 1: a generated lifecycle ----------------------------------------------------------------


async def drive_lifecycle(env: StoreEnv, plan: Lifecycle) -> None:
    seed = Task(id=TASK, context_id=CONTEXT)
    for number, (role, text) in enumerate(plan.history):
        seed.history.append(Message(message_id=f"m{number}", role=role, parts=[Part(text=text)]))
    await env.store.create_task(TASK, CONTEXT, env.clock.now_ms(), snapshot_json(seed))

    queue = CollectingQueue()
    emitter = TaskEventEmitter(
        env.store,
        QueueSink(queue, TASK, CONTEXT),
        env.redactor,
        env.clock,
        env.db_path,
        task_id=TASK,
        context_id=CONTEXT,
    )
    store = ProjectedTaskStore(env.store, EmitterArtifacts(emitter), env.clock)
    call_context = ServerCallContext()
    history_roles = [ROLE_NAMES[role] for role, _ in plan.history]
    seen = 0

    async def check_task(
        state: TaskState, status_text: str | None, artifact_text: str | None = None
    ) -> None:
        """The get-task / cancel result and the first-event snapshot of the Task right now."""
        task = await store.get(TASK, call_context)
        assert task is not None
        doc = serialize_task(task)
        states, roles = assert_enum_form(doc)
        assert states == [sdk_name(state)]
        expected_roles = history_roles + (["ROLE_AGENT"] if status_text is not None else [])
        assert Counter(roles) == Counter(expected_roles)
        if artifact_text is not None:
            assert [p["text"] for p in doc["artifacts"][0]["parts"]] == [artifact_text]
        else:
            assert "artifacts" not in doc

        assert serialize_task(await store.snapshot(TASK)) == doc  # the stream's first event
        send_doc = serialize_send_result(task)
        assert send_doc["task"] == doc
        assert_enum_form(send_doc)

    def take_events() -> list[dict[str, Any]]:
        nonlocal seen
        fresh = queue.events[seen:]
        seen = len(queue.events)
        return [serialize_event(event) for event in fresh]

    await check_task(TaskState.SUBMITTED, None)
    status_text: str | None = None
    artifact_text: str | None = None
    for index, step in enumerate(plan.steps):
        assert await emitter.transition(step.state, step.text)
        status_text = step.text
        (doc,) = take_events()  # one Status_Event per transition
        assert list(doc) == ["statusUpdate"]
        states, roles = assert_enum_form(doc)
        assert states == [sdk_name(step.state)]
        assert roles == (["ROLE_AGENT"] if step.text is not None else [])
        await check_task(step.state, status_text, artifact_text)

        if index == 0 and plan.chunks:
            for number, chunk in enumerate(plan.chunks):
                await emitter.token(ARTIFACT, chunk, number == 0)
            await emitter.close_artifact(ARTIFACT)
            token_docs = take_events()
            assert len(token_docs) == len(plan.chunks) + 1  # the chunks and the last-chunk marker
            for token_doc, chunk in zip(token_docs, plan.chunks, strict=False):
                assert list(token_doc) == ["artifactUpdate"]
                assert assert_enum_form(token_doc) == ([], [])
                assert token_doc["artifactUpdate"]["artifact"]["parts"][0]["text"] == chunk
            assert assert_enum_form(token_docs[-1]) == ([], [])
            artifact_text = "".join(plan.chunks)
            await check_task(step.state, status_text, artifact_text)

    assert all(isinstance(e, TaskStatusUpdateEvent | TaskArtifactUpdateEvent) for e in queue.events)


@settings(max_examples=100)
@given(plan=lifecycles())
def test_lifecycle_uses_protojson_enum_names(store_kind: StoreKind, plan: Lifecycle) -> None:
    # Feature: local-agent-backend, Property 60: ProtoJSON enum form
    # Validates: Requirements 1.10
    with open_store(store_kind) as env:
        asyncio.run(drive_lifecycle(env, plan))


@settings(max_examples=100)
@given(plan=lifecycles(end=TaskState.CANCELED))
def test_cancel_result_uses_protojson_enum_names(store_kind: StoreKind, plan: Lifecycle) -> None:
    # Feature: local-agent-backend, Property 60: ProtoJSON enum form
    # Validates: Requirements 1.10
    with open_store(store_kind) as env:
        asyncio.run(drive_lifecycle(env, plan))


# --- Test 2: the events of a generated scenario -----------------------------------------------


async def replay(
    env: StoreEnv, events: Sequence[StatusEvent | ArtifactEvent], task_id: str, context_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Send the events a scenario's emitter delivered through a `QueueSink` into a real SDK
    `TaskUpdater`, then serialize them, the stored Task, and the stream's first-event snapshot."""
    queue = CollectingQueue()
    sink = QueueSink(queue, task_id, context_id)
    for event in events:
        if isinstance(event, StatusEvent):
            await sink.status(event.state, event.text, event.metadata, final=event.final)
        else:
            await sink.artifact(
                event.artifact_id, event.text, append=event.append, last_chunk=event.last_chunk
            )
    store = ProjectedTaskStore(env.reader, None, env.clock)
    task = await store.get(task_id, ServerCallContext())
    assert task is not None
    docs = [serialize_event(event) for event in queue.events]
    return docs, serialize_task(task), serialize_task(await store.snapshot(task_id))


@settings(max_examples=100)
@given(script=scenario_scripts())
def test_scenario_events_use_protojson_enum_names(
    store_kind: StoreKind, script: ScenarioScript
) -> None:
    # Feature: local-agent-backend, Property 60: ProtoJSON enum form
    # Validates: Requirements 1.10
    with open_store(store_kind) as env:
        run = run_scenario(script, env)
        docs, task_doc, snapshot_doc = asyncio.run(
            replay(env, run.sink.events, run.task_id, run.context_id)
        )

        assert len(docs) == len(run.sink.events)
        emitted: dict[str, str] = {}
        for event, doc in zip(run.sink.events, docs, strict=True):
            states, roles = assert_enum_form(doc)
            if isinstance(event, StatusEvent):
                assert list(doc) == ["statusUpdate"]
                assert states == [sdk_name(event.state)]
                has_message = event.text is not None or bool(event.metadata)
                assert roles == (["ROLE_AGENT"] if has_message else [])
            else:
                assert list(doc) == ["artifactUpdate"]
                assert (states, roles) == ([], [])
                artifact = doc["artifactUpdate"]["artifact"]
                text = "".join(part.get("text", "") for part in artifact["parts"])
                emitted[artifact["artifactId"]] = emitted.get(artifact["artifactId"], "") + text
        assert emitted == run.sink.emitted

        states, roles = assert_enum_form(task_doc)
        assert states == [sdk_name(run.state)]
        assert set(roles) <= {"ROLE_AGENT"}  # the harness seeds no SDK history
        assert snapshot_doc == task_doc
