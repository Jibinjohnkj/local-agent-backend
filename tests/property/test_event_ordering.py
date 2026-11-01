"""Property test for event ordering (task 17.12).

Feature: local-agent-backend, Property 3: Event ordering

For any scenario script, the recorded event stream satisfies the ordering and shape rules of
Requirements 2.4, 4.1, 4.3, 4.4, 4.5, and 4.6, on both Session_Stores.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from unittest import mock

from hypothesis import HealthCheck, given, settings
from scenarios import (
    CallKind,
    CallScript,
    ScenarioRun,
    ScenarioScript,
    StoreKind,
    open_store,
    run_scenario,
    scenario_scripts,
    store_kind,
)

from fakes import (
    ArtifactEvent,
    FakeLlmGateway,
    RecordingEventSink,
    SinkEvent,
    StatusEvent,
    ToolOutcome,
)
from local_agent.a2a_server.events import (
    KIND_TOOL_COMPLETED,
    KIND_TOOL_STARTED,
    TaskState,
)
from local_agent.models import AssistantMessage, ChatMessage, ToolCall, ToolMessage

__all__ = ["store_kind"]  # the fixture is used by name below

DISCOVERY_TEXT = "Discovering local tools..."
REQUIRED_KEYS = ("kind", "tool_name", "server", "tool_call_id")
MARK_REQUEST = "request"
MARK_TOKEN = "token"
MARK_DISCOVERY = "discovery"
MARK_OTHER = "other"


@contextlib.contextmanager
def spy_on_order() -> Iterator[list[str]]:
    """Record, in one list, every sink event (by kind) and every chat-completion request.

    The sink and the fake LLM share no clock, so this is how the test sees whether the
    "Discovering local tools..." event came before the first request.
    """
    timeline: list[str] = []
    record = RecordingEventSink._record
    stream = FakeLlmGateway.stream

    def spy_record(self: RecordingEventSink, event: SinkEvent) -> None:
        if isinstance(event, ArtifactEvent):
            timeline.append(MARK_TOKEN)
        elif event.state is TaskState.WORKING and event.text == DISCOVERY_TEXT:
            timeline.append(MARK_DISCOVERY)
        else:
            timeline.append(MARK_OTHER)
        record(self, event)

    async def spy_stream(
        self: FakeLlmGateway, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Any:
        timeline.append(MARK_REQUEST)
        return await stream(self, messages, tools)

    with (
        mock.patch.object(RecordingEventSink, "_record", spy_record),
        mock.patch.object(FakeLlmGateway, "stream", spy_stream),
    ):
        yield timeline


def issued_calls(history: Sequence[ChatMessage]) -> list[ToolCall]:
    return [
        call
        for message in history
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    ]


def result_positions(history: Sequence[ChatMessage]) -> dict[str, int]:
    return {
        message.tool_call_id: position
        for position, message in enumerate(history)
        if isinstance(message, ToolMessage)
    }


def check_start(run: ScenarioRun, timeline: list[str]) -> None:
    """The single `working` "Discovering local tools..." event comes first of all (2.4)."""
    statuses = run.sink.statuses
    assert statuses, "no Status_Event was emitted"
    first = statuses[0]
    assert first.state is TaskState.WORKING
    assert first.text == DISCOVERY_TEXT
    assert sum(1 for status in statuses if status.text == DISCOVERY_TEXT) == 1
    # Nothing, not even a Token_Event, precedes it, and no request was made before it.
    assert run.sink.events[0] is first
    assert timeline[0] == MARK_DISCOVERY, timeline
    assert timeline.count(MARK_DISCOVERY) == 1
    if MARK_REQUEST in timeline:
        assert timeline.index(MARK_DISCOVERY) < timeline.index(MARK_REQUEST)
    if MARK_TOKEN in timeline:
        assert timeline.index(MARK_DISCOVERY) < timeline.index(MARK_TOKEN)


def check_metadata(event: StatusEvent, kind: str) -> Mapping[str, Any]:
    metadata = event.metadata
    assert metadata is not None
    for key in REQUIRED_KEYS:
        assert key in metadata, (key, metadata)
    assert metadata["kind"] == kind
    assert isinstance(metadata["tool_name"], str)
    assert metadata["server"] is None or isinstance(metadata["server"], str)
    assert isinstance(metadata["tool_call_id"], str)
    if kind == KIND_TOOL_COMPLETED:
        assert metadata.get("outcome") in ("succeeded", "failed"), metadata
    return metadata


def expected_outcome(call: CallScript, *, sent: bool) -> str:
    """`failed` exactly when the call was rejected, error-flagged, timed out, or raised (4.6)."""
    if sent and call.behavior.outcome is ToolOutcome.SUCCESS:
        return "succeeded"
    return "failed"


def check_tool_events(run: ScenarioRun) -> None:
    """Rules 2 to 4 of the property, call by call (4.1, 4.3, 4.4, 4.5, 4.6)."""
    canceled = run.state is TaskState.CANCELED
    started = run.sink.tool_events(KIND_TOOL_STARTED)
    completed = run.sink.tool_events(KIND_TOOL_COMPLETED)
    history = run.history
    issued = issued_calls(history)
    scripted = run.script.calls
    assert len(issued) <= len(scripted)
    positions = result_positions(history)
    sent_tags = {call.args.get("tag") for call in run.mcp.calls}

    # Every tool event carries the four keys, and belongs to a Tool_Call of this Task.
    issued_ids = {call.id for call in issued}
    for event, kind in [(e, KIND_TOOL_STARTED) for e in started] + [
        (e, KIND_TOOL_COMPLETED) for e in completed
    ]:
        metadata = check_metadata(event, kind)
        assert metadata["tool_call_id"] in issued_ids

    started_by_id: dict[str, list[StatusEvent]] = {}
    completed_by_id: dict[str, list[StatusEvent]] = {}
    for event in started:
        assert event.metadata is not None
        started_by_id.setdefault(event.metadata["tool_call_id"], []).append(event)
    for event in completed:
        assert event.metadata is not None
        completed_by_id.setdefault(event.metadata["tool_call_id"], []).append(event)

    for issue, script_call in zip(issued, scripted, strict=False):
        assert issue.name == script_call.name  # the history lines up with the script
        sent = script_call.tag in sent_tags
        starts = started_by_id.get(issue.id, [])
        ends = completed_by_id.get(issue.id, [])

        if sent:
            assert len(starts) == 1, (script_call, starts)
        else:
            assert starts == [], (script_call, starts)  # resolved without sending: no start
        if canceled:
            assert len(ends) <= 1, (script_call, ends)
        else:
            assert len(ends) == 1, (script_call, ends)
        if not ends:
            continue
        end = ends[0]
        assert end.metadata is not None

        if sent:
            assert starts[0].index < end.index  # started comes first
            expect_server, _, expect_tool = script_call.name.partition("__")
            for event in (starts[0], end):
                assert event.metadata is not None
                assert event.metadata["server"] == expect_server
                assert event.metadata["tool_name"] == expect_tool
        elif script_call.kind is CallKind.UNKNOWN_TOOL:
            assert end.metadata["server"] is None
            assert end.metadata["tool_name"] == script_call.name
        assert end.metadata["outcome"] == expected_outcome(script_call, sent=sent), script_call

        # The completed event arrives before the Tool_Result is appended to the history (4.3):
        # the store held no more messages than precede the result.
        position = positions.get(issue.id)
        if position is not None:
            assert end.history_len is not None
            assert end.history_len <= position, (end.history_len, position, script_call)


# Feature: local-agent-backend, Property 3: Event ordering
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(script=scenario_scripts(world_changes=False))
def test_event_ordering(store_kind: StoreKind, script: ScenarioScript) -> None:
    """**Validates: Requirements 2.4, 4.1, 4.3, 4.4, 4.5, 4.6**"""
    with open_store(store_kind) as env, spy_on_order() as timeline:
        run = run_scenario(script, env, probe_history=True)

        assert run.terminal
        assert not run.crashed
        check_start(run, timeline)
        check_tool_events(run)
