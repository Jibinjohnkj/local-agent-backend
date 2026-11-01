"""Property 59: the first event of every SendStreamingMessage stream is a Task_Snapshot (20.15).

The tests drive the real ASGI app (`build_app`) through `httpx.ASGITransport`, over a real
Session_Store of both kinds, the real `TaskRunner`, `ProjectedTaskStore`, and executor, and the
scripted LLM and MCP fakes of `tests/fakes.py`. Each example draws a scenario script, a binding
(JSON-RPC or REST), and a plan of replies. It then plays the whole conversation: the first message
creates a Task, and while the Task is `input-required` it sends the next planned reply (an
approval decision the parser recognizes, or one it does not). The SSE bytes of every stream are
parsed and checked:

* a new Task starts with a `Task` in `TASK_STATE_SUBMITTED` (3.10, 3.11);
* a continuation with a recognizable decision starts with a `Task` in `TASK_STATE_WORKING`,
  followed by a `working` Status_Event (3.10, 3.12);
* a continuation without one is exactly a `Task` in `TASK_STATE_INPUT_REQUIRED` and one
  `input-required` Status_Event carrying the re-prompt, after which the stream ends (3.13, 5.11);
* the snapshot carries the Task's `task_id`, `context_id`, and accumulated artifacts, and every
  later event of the stream belongs to that Task (3.1).

Requirements: 3.1, 3.10, 3.11, 3.12, 3.13, 5.11.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import httpx
from a2a.types import Message, Part, Role, StreamResponse
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct, Value
from hypothesis import event as record_event
from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    CallKind,
    CallScript,
    ScenarioScript,
    StoreEnv,
    StoreKind,
    TurnScript,
    build_fakes,
    char_estimate,
    open_store,
    scenario_scripts,
    store_kind,
    tagged_arguments,
)
from starlette.applications import Starlette
from strategies import text_chars

from fakes import ClockSleep, ManualTimers
from local_agent.a2a_server.app import build_app, build_deps
from local_agent.a2a_server.events import (
    REPROMPT_TEXT,
    STREAM_CLOSING_STATES,
    TaskState,
    from_sdk,
    sdk_name,
)
from local_agent.config.models import (
    ApprovalKeywords,
    BackendConfig,
    LLMConfig,
    OrchestrationConfig,
    StorageConfig,
)
from local_agent.orchestrator.loop import RunnerDeps, TaskRunnerRegistry
from local_agent.storage.task_store import ProjectedTaskStore

__all__ = ["store_kind"]  # the fixture is used by name below

# Feature: local-agent-backend, Property 59: First stream event is a Task_Snapshot

KEYWORDS: Final = ApprovalKeywords()
REQUEST_TIMEOUT_S: Final = 20.0
MAX_ROUNDS: Final = 40
"""A script has at most a few protected calls, so a conversation this long is a runaway."""

JSONRPC: Final = "jsonrpc"
REST: Final = "rest"
APPROVAL_WORDS: Final = frozenset(word.casefold() for word in (*KEYWORDS.approve, *KEYWORDS.deny))


# --- Replies ------------------------------------------------------------------------------------


class PartKind(StrEnum):
    TEXT = "text"
    DECISION = "decision"  # a data part {"decision": <value>}


@dataclass(frozen=True)
class PartSpec:
    kind: PartKind
    value: str

    def build(self) -> Part:
        if self.kind is PartKind.TEXT:
            return Part(text=self.value)
        fields = Struct(fields={"decision": Value(string_value=self.value)})
        return Part(data=Value(struct_value=fields))


@dataclass(frozen=True)
class Reply:
    """A continuation message. `recognized` is what the parser must make of it (5.12)."""

    recognized: bool
    parts: tuple[PartSpec, ...]


def _casefold_is_keyword(text: str) -> bool:
    return text.strip().casefold() in APPROVAL_WORDS


_padding = st.sampled_from(["", " ", "\t", "\n", "  \n"])
_keyword_texts = st.builds(
    lambda word, case, left, right: left + case(word) + right,
    st.sampled_from([*KEYWORDS.approve, *KEYWORDS.deny]),
    st.sampled_from([str.lower, str.upper, str.title]),
    _padding,
    _padding,
)
_other_texts = st.one_of(
    st.text(alphabet=text_chars, max_size=12), st.sampled_from(["approved", "yes please", "nope"])
).filter(lambda text: not _casefold_is_keyword(text))
_other_decisions = st.one_of(
    st.sampled_from(["maybe", "Approve", "DENY", "approve ", ""]),
    st.text(alphabet=text_chars, max_size=6).filter(lambda value: value not in ("approve", "deny")),
)

_recognized_parts = st.one_of(
    st.builds(lambda text: (PartSpec(PartKind.TEXT, text),), _keyword_texts),
    st.builds(
        lambda value: (PartSpec(PartKind.DECISION, value),), st.sampled_from(["approve", "deny"])
    ),
    # The first matching part wins, so unrecognized parts in front of it do not matter.
    st.builds(
        lambda before, keyword: (
            PartSpec(PartKind.TEXT, before),
            PartSpec(PartKind.DECISION, "maybe"),
            PartSpec(PartKind.TEXT, keyword),
        ),
        _other_texts,
        _keyword_texts,
    ),
)
_unrecognized_parts = st.lists(
    st.one_of(
        st.builds(lambda text: PartSpec(PartKind.TEXT, text), _other_texts),
        st.builds(lambda value: PartSpec(PartKind.DECISION, value), _other_decisions),
    ),
    min_size=1,
    max_size=3,
).map(tuple)

replies = st.one_of(
    st.builds(Reply, st.just(True), _recognized_parts),
    st.builds(Reply, st.just(False), _unrecognized_parts),
)


# --- Cases --------------------------------------------------------------------------------------


def protected_script(chunks: tuple[str, ...], more: int) -> ScenarioScript:
    """A Task that streams `chunks`, then asks for approval of `more + 1` protected calls."""
    turns = tuple(
        TurnScript(
            chunks=chunks if number == 0 else (f"again {number}",),
            calls=(
                CallScript(
                    f"w{number}",
                    CallKind.PROTECTED,
                    "files__write_file",
                    tagged_arguments({"path": "out.txt", "content": "x"}, f"w{number}"),
                    call_id=f"call-w{number}",
                ),
            ),
        )
        for number in range(more + 1)
    )
    return ScenarioScript("write", turns)


@dataclass(frozen=True)
class Case:
    script: ScenarioScript
    binding: str
    user_text: str
    context_id: str | None
    replies: tuple[Reply, ...]


_scripts = st.one_of(
    # Any script: the Task may complete, fail, or stop at an approval.
    scenario_scripts(cancels=False),
    # A script that always stops at an approval, so the continuation paths are always reached.
    st.builds(
        protected_script,
        st.lists(st.text(alphabet=text_chars, min_size=1, max_size=6), max_size=3).map(tuple),
        st.integers(min_value=0, max_value=2),
    ),
)

_user_texts = st.one_of(
    st.text(alphabet=text_chars, min_size=1, max_size=10),
    st.sampled_from(["a\r\nb", "\r", "\n", "\t", "\x00", "\U0001f389"]),
)

cases = st.builds(
    Case,
    script=_scripts,
    binding=st.sampled_from([JSONRPC, REST]),
    user_text=_user_texts,
    context_id=st.none() | st.text(alphabet=text_chars, min_size=1, max_size=8),
    replies=st.lists(replies, max_size=4).map(tuple),
)


# --- The app under test -------------------------------------------------------------------------


def build_test_app(env: StoreEnv, script: ScenarioScript) -> tuple[Starlette, TaskRunnerRegistry]:
    """The production app over `env`'s store with the scripted LLM and MCP fakes."""
    clock = env.clock
    orchestration = OrchestrationConfig(
        max_tool_iterations=script.max_tool_iterations,
        approval_timeout_s=script.approval_timeout_s,
        context_window_budget=script.context_window_budget,
        per_session_discovery=True,
    )
    cfg = BackendConfig(
        llm=LLMConfig(model="fake/model"),
        orchestration=orchestration,
        storage=StorageConfig(backend="memory"),
    )
    llm, mcp = build_fakes(script, clock, ClockSleep(clock))
    runners = TaskRunnerRegistry(
        RunnerDeps(
            sessions=env.store,
            llm=llm,
            registry=mcp.registry,
            mcp=mcp,
            orchestration=orchestration,
            base_instructions=script.base_instructions,
            model="fake/model",
            redactor=env.redactor,
            clock=clock,
            db_path=env.db_path,
            sleep=ManualTimers(clock),
            estimate_tokens=char_estimate,
        )
    )
    deps = dataclasses.replace(
        build_deps(cfg, None, clock=clock),
        session_store=env.store,
        task_store=ProjectedTaskStore(env.store, runners, clock),
        registry=mcp.registry,
        runners=runners,
    )
    return build_app(cfg, deps), runners


# --- SSE and requests ---------------------------------------------------------------------------


def parse_sse(body: bytes) -> list[dict[str, Any]]:
    """The JSON `data` payload of every SSE event in `body`, in order.

    Lines end with CRLF, LF, or CR. An event ends at a blank line. Several `data` lines of one
    event are joined with a newline. Comment lines and the `event`, `id`, and `retry` fields carry
    no payload. A last event without a closing blank line is still delivered.
    """
    text = body.decode("utf-8")
    payloads: list[dict[str, Any]] = []
    data: list[str] = []

    def flush() -> None:
        if data:
            payloads.append(json.loads("\n".join(data)))
            data.clear()

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line:
            flush()
        elif line.startswith("data:"):
            value = line[len("data:") :]
            data.append(value[1:] if value.startswith(" ") else value)
    flush()
    return payloads


_message_numbers = itertools.count(1)


def message_json(
    parts: Sequence[Part], *, context_id: str = "", task_id: str = ""
) -> dict[str, Any]:
    message = Message(
        message_id=f"m{next(_message_numbers)}",
        role=Role.ROLE_USER,
        parts=list(parts),
        context_id=context_id,
        task_id=task_id,
    )
    encoded: dict[str, Any] = MessageToDict(message)
    return encoded


@dataclass
class Stream:
    """One parsed SendStreamingMessage response."""

    events: list[StreamResponse]
    first_raw: dict[str, Any]

    @property
    def last_state(self) -> TaskState:
        for event in reversed(self.events):
            kind = event.WhichOneof("payload")
            if kind == "task":
                return from_sdk(event.task.status.state)
            if kind == "status_update":
                return from_sdk(event.status_update.status.state)
        raise AssertionError("the stream carried no Task or Status_Event")


async def send_stream(client: httpx.AsyncClient, binding: str, message: dict[str, Any]) -> Stream:
    """Send one streaming message and parse the SSE response."""
    headers = {"A2A-Version": "1.0"}
    request_id = next(_message_numbers)
    async with asyncio.timeout(REQUEST_TIMEOUT_S):
        if binding == JSONRPC:
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "SendStreamingMessage",
                "params": {"message": message},
            }
            response = await client.post("/", json=body, headers=headers)
        else:
            response = await client.post(
                "/message:stream", json={"message": message}, headers=headers
            )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.text
    payloads = parse_sse(response.content)
    assert payloads, "the stream carried no events"
    results: list[dict[str, Any]] = []
    for payload in payloads:
        if binding == JSONRPC:
            assert "error" not in payload, payload
            assert payload["jsonrpc"] == "2.0"
            assert payload["id"] == request_id  # the request ID is echoed on every event
            results.append(payload["result"])
        else:
            results.append(payload)
    events = [json_format.ParseDict(result, StreamResponse()) for result in results]
    return Stream(events, results[0])


# --- The property -------------------------------------------------------------------------------


def artifact_texts(task_events: Sequence[StreamResponse]) -> dict[str, str]:
    """The text of every artifact the Token_Events of `task_events` built, by artifact ID."""
    texts: dict[str, str] = {}
    for event in task_events:
        if event.WhichOneof("payload") != "artifact_update":
            continue
        update = event.artifact_update
        text = "".join(part.text for part in update.artifact.parts)
        key = update.artifact.artifact_id
        texts[key] = texts.get(key, "") + text if update.append else text
    return texts


def check_stream(
    stream: Stream,
    *,
    expected: TaskState,
    task_id: str | None,
    context_id: str | None,
    streamed_before: Sequence[StreamResponse],
) -> tuple[str, str]:
    """Assert the first-event rules for one stream. Return the Task's `task_id` and `context_id`."""
    events = stream.events
    first = events[0]
    assert first.WhichOneof("payload") == "task", f"first event is {first.WhichOneof('payload')}"
    snapshot = first.task

    # The state is in ProtoJSON form on the wire (1.10).
    assert stream.first_raw["task"]["status"]["state"] == sdk_name(expected)
    assert from_sdk(snapshot.status.state) is expected
    record_event(f"first event: Task {expected.value}")
    assert snapshot.id
    assert snapshot.context_id
    if task_id is not None:
        assert snapshot.id == task_id
    if context_id is not None:
        assert snapshot.context_id == context_id

    # Accumulated artifacts: exactly what the Task's earlier streams emitted.
    snapshot_texts = {
        artifact.artifact_id: "".join(part.text for part in artifact.parts)
        for artifact in snapshot.artifacts
    }
    assert snapshot_texts == artifact_texts(streamed_before)
    if expected is not TaskState.SUBMITTED:
        record_event(f"continuation snapshot artifacts: {'some' if snapshot_texts else 'none'}")

    # Every later event is a Status_Event or Token_Event of this Task, never a bare Message.
    for event in events[1:]:
        kind = event.WhichOneof("payload")
        assert kind in ("status_update", "artifact_update"), kind
        update = event.status_update if kind == "status_update" else event.artifact_update
        assert update.task_id == snapshot.id
        assert update.context_id == snapshot.context_id

    if expected is TaskState.SUBMITTED:
        assert not snapshot.artifacts  # nothing has been emitted for a new Task yet
    elif expected is TaskState.WORKING:
        second = events[1] if len(events) > 1 else None
        assert second is not None
        assert second.WhichOneof("payload") == "status_update"
        assert from_sdk(second.status_update.status.state) is TaskState.WORKING
    else:
        # An unrecognized reply: the snapshot and one final re-prompt, and nothing else (3.13).
        assert expected is TaskState.INPUT_REQUIRED
        assert len(events) == 2, [event.WhichOneof("payload") for event in events]
        second = events[1]
        assert second.WhichOneof("payload") == "status_update"
        reprompt = second.status_update
        assert from_sdk(reprompt.status.state) is TaskState.INPUT_REQUIRED
        assert reprompt.status.message.parts[0].text == REPROMPT_TEXT
    # The stream ends on a stream-closing state (3.5), which for a re-prompt is `input-required`.
    assert stream.last_state in STREAM_CLOSING_STATES
    return snapshot.id, snapshot.context_id


async def converse(env: StoreEnv, case: Case) -> int:
    """Play `case` against the app and check the first event of every stream. Return the count."""
    app, runners = build_test_app(env, case.script)
    transport = httpx.ASGITransport(app=app)
    checked = 0
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
            first_message = message_json(
                [Part(text=case.user_text)], context_id=case.context_id or ""
            )
            stream = await send_stream(client, case.binding, first_message)
            task_id, context_id = check_stream(
                stream,
                expected=TaskState.SUBMITTED,
                task_id=None,
                context_id=case.context_id,
                streamed_before=[],
            )
            checked += 1
            streamed = list(stream.events[1:])

            plan = list(case.replies)
            while stream.last_state is TaskState.INPUT_REQUIRED:
                assert checked < MAX_ROUNDS, "the conversation did not end"
                reply = plan.pop(0) if plan else Reply(True, (PartSpec(PartKind.TEXT, "approve"),))
                message = message_json(
                    [spec.build() for spec in reply.parts], context_id=context_id, task_id=task_id
                )
                stream = await send_stream(client, case.binding, message)
                expected = TaskState.WORKING if reply.recognized else TaskState.INPUT_REQUIRED
                check_stream(
                    stream,
                    expected=expected,
                    task_id=task_id,
                    context_id=context_id,
                    streamed_before=streamed,
                )
                checked += 1
                streamed += stream.events[1:]
    finally:
        await runners.cancel_all()
    return checked


@settings(max_examples=100, deadline=None)
@given(case=cases)
def test_first_event_of_every_stream_is_a_task_snapshot(store_kind: StoreKind, case: Case) -> None:
    """For any scenario script, every stream begins with the Task_Snapshot its request calls for."""
    with open_store(store_kind) as env:
        checked = asyncio.run(converse(env, case))
    assert checked >= 1
