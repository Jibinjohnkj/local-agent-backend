"""Property test for the Max_Tool_Iterations limit (task 17.15).

The LLM fake answers every tools-enabled request with Tool_Calls. After N tool round trips the
runner must make one more request without tool definitions, ignore any Tool_Call in its answer,
store only its text, and complete the Task.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    READ_FILE,
    WRITE_FILE,
    ApprovalChoice,
    CallKind,
    CallScript,
    ScenarioScript,
    StoreKind,
    TurnScript,
    open_store,
    run_scenario,
    store_kind,
    tagged_arguments,
)
from strategies import text_chars

from local_agent.a2a_server.events import TaskState, to_sdk
from local_agent.models import AssistantMessage, ToolMessage

__all__ = ["store_kind"]  # the fixture is used by name below

texts = st.text(alphabet=text_chars, max_size=16)


@dataclass(frozen=True)
class Case:
    limit: int  # Max_Tool_Iterations N
    tool_turns: tuple[tuple[int, str], ...]  # per tools-enabled turn: (valid calls, text)
    final_calls: int  # Tool_Calls the tools-disabled response carries
    final_protected: bool  # whether the first of those is a protected call
    final_text: str
    drop_calls: bool  # the model obeys the missing tools and omits its Tool_Calls


@st.composite
def cases(draw: st.DrawFn) -> Case:
    limit = draw(st.integers(min_value=1, max_value=10))
    turns = draw(
        st.lists(
            st.tuples(st.integers(min_value=1, max_value=3), texts),
            min_size=limit,
            max_size=limit,
        )
    )
    return Case(
        limit=limit,
        tool_turns=tuple(turns),
        final_calls=draw(st.integers(min_value=0, max_value=3)),
        final_protected=draw(st.booleans()),
        final_text=draw(texts),
        drop_calls=draw(st.booleans()),
    )


def valid_call(tag: str) -> CallScript:
    return CallScript(
        tag=tag,
        kind=CallKind.VALID,
        name=READ_FILE,
        arguments=tagged_arguments({"path": "a.txt"}, tag),
        call_id=f"call-{tag}",
    )


def protected_call(tag: str) -> CallScript:
    return CallScript(
        tag=tag,
        kind=CallKind.PROTECTED,
        name=WRITE_FILE,
        arguments=tagged_arguments({"path": "a.txt", "content": "x"}, tag),
        decision=ApprovalChoice.APPROVE,
        call_id=f"call-{tag}",
    )


def build_script(case: Case) -> tuple[ScenarioScript, list[str], list[str]]:
    """The script, the tags of the calls that may run, and the tags of the final, ignored calls."""
    turns: list[TurnScript] = []
    allowed: list[str] = []
    for index, (count, text) in enumerate(case.tool_turns):
        tags = [f"t{index}c{position}" for position in range(count)]
        allowed += tags
        turns.append(TurnScript(chunks=(text,) if text else (), calls=tuple(map(valid_call, tags))))
    ignored = [f"final-c{position}" for position in range(case.final_calls)]
    final_calls = tuple(
        protected_call(tag) if position == 0 and case.final_protected else valid_call(tag)
        for position, tag in enumerate(ignored)
    )
    turns.append(
        TurnScript(chunks=(case.final_text,) if case.final_text else (), calls=final_calls)
    )
    script = ScenarioScript(
        "go",
        tuple(turns),
        max_tool_iterations=case.limit,
        drop_calls_without_tools=case.drop_calls,
    )
    return script, allowed, ignored


# Feature: local-agent-backend, Property 14: Iteration limit
# **Validates: Requirements 8.8, 8.12**
@settings(max_examples=100, deadline=None)
@given(case=cases())
def test_iteration_limit(store_kind: StoreKind, case: Case) -> None:
    script, allowed, ignored = build_script(case)
    limit = case.limit

    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        # Exactly N tools-enabled requests, then exactly one request without tool definitions.
        requests = run.llm.requests
        assert len(requests) == limit + 1
        assert all(request.tools for request in requests[:limit])
        assert requests[limit].tools is None

        # Only the Tool_Calls of the first N responses ran; none from the final response, and
        # the final response triggered no approval request.
        assert [call.args["tag"] for call in run.mcp.calls] == allowed
        assert not any(call.args["tag"] in ignored for call in run.mcp.calls)
        assert not any(event.kind == "approval_required" for event in run.sink.statuses)

        # The final response's text is the final assistant message, stored without Tool_Calls.
        assert run.state is TaskState.COMPLETED
        assert run.task is not None
        assert run.task.state == to_sdk(TaskState.COMPLETED)
        assistants = [m for m in run.history if isinstance(m, AssistantMessage)]
        assert len(assistants) == limit + 1
        assert all(m.tool_calls for m in assistants[:limit])
        assert not assistants[-1].tool_calls
        assert (assistants[-1].content or "") == case.final_text
        assert run.history[-1] is assistants[-1]  # no Tool_Result follows it
        assert sum(isinstance(m, ToolMessage) for m in run.history) == len(allowed)
        stored_ids = {call.id for m in assistants for call in m.tool_calls}
        assert not any(tag in call_id for call_id in stored_ids for tag in ignored)

        # The final text was emitted as the last response artifact, which is closed.
        artifact_id = f"response-{limit + 1}"
        assert run.sink.emitted.get(artifact_id, "") == case.final_text
        stored = {a.artifact_id: a for a in run.task.artifacts}
        assert stored[artifact_id].text == case.final_text
        assert stored[artifact_id].closed
