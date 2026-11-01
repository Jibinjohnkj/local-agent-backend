"""Property 12: History integrity and result ordering (task 17.13).

The scenario harness runs the real TaskRunner over both Session_Stores, with random cancellation
(an injected cancel, a hanging tool, a stalled stream) and random LLM failures. After the Task is
terminal, the stored Message_History must satisfy (a) to (f) of the property. (a) to (d) are
checked on the stored history alone. (e) is checked against a history derived from the script:
the exact messages for a `completed` or `failed` Task, and one of the legal cut points for a
`canceled` Task, where the calls of the last assistant message carry the "not completed" result.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from scenarios import (
    ApprovalChoice,
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

from fakes import LlmFailure, ToolOutcome
from local_agent.a2a_server.events import TERMINAL_STATES, TaskState
from local_agent.models import AssistantMessage, ChatMessage, ToolMessage, UserMessage
from local_agent.orchestrator.loop import NOT_COMPLETED_CANCELED

__all__ = ["store_kind"]  # the fixture is used by name below


# --- (a) to (d): what any stored history must satisfy ---------------------------------------------


def assert_history_integrity(history: list[ChatMessage]) -> None:
    issued: set[str] = set()
    answers: dict[str, int] = {}
    position = 0
    while position < len(history):
        message = history[position]
        if isinstance(message, ToolMessage):
            # (a) and (d): a tool message that no assistant message of its turn announced.
            raise AssertionError(f"tool message {message.tool_call_id!r} has no preceding turn")
        position += 1
        if not isinstance(message, AssistantMessage) or not message.tool_calls:
            continue
        ids = [call.id for call in message.tool_calls]
        issued.update(ids)
        # (c): the turn's tool messages follow the assistant message, in Tool_Call order.
        turn = []
        while position < len(history) and isinstance(history[position], ToolMessage):
            tool = history[position]
            assert isinstance(tool, ToolMessage)
            turn.append(tool.tool_call_id)
            answers[tool.tool_call_id] = answers.get(tool.tool_call_id, 0) + 1
            position += 1
        assert turn == ids, f"results {turn} do not follow the call order {ids}"
    # (a) every tool message answers an issued call; (b) every call has exactly one answer.
    assert set(answers) <= issued
    assert all(answers.get(call_id) == 1 for call_id in issued), answers


# --- (e): the history the script derives ----------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One LLM response the reasoning loop would process."""

    text: str
    calls: tuple[CallScript, ...]  # the Tool_Calls the loop dispatches; empty for a final answer


def derive_turns(script: ScenarioScript, run: ScenarioRun) -> tuple[list[Turn], bool]:
    """The turns the loop processes, and whether it ends in a failed LLM request.

    A turn whose request the LLM fake fails ends the sequence. The fake offers no tools to the
    request that follows the last tools-enabled one, and a fake with `drop_calls_without_tools`
    then streams text only.
    """
    turns: list[Turn] = []
    index = 0
    while True:
        scripted = script.turns[index] if index < len(script.turns) else None
        if scripted is not None and scripted.failure is not None:
            return turns, True
        text = "".join(scripted.chunks) if scripted is not None else "ok"
        calls = scripted.calls if scripted is not None else ()
        tools_enabled = index < script.max_tool_iterations
        requests = run.llm.requests
        offered = requests[index].tools is not None if index < len(requests) else tools_enabled
        if script.drop_calls_without_tools and not offered:
            calls = ()
        if not (tools_enabled and calls):
            turns.append(Turn(text, ()))
            return turns, False
        turns.append(Turn(text, calls))
        index += 1


def check_result(call: CallScript, content: str) -> None:
    """The Tool_Result text a call must have when the registry did not change under it."""
    if call.kind is CallKind.UNKNOWN_TOOL:
        assert content.startswith(f"Unknown tool '{call.name}'"), content
    elif call.kind is CallKind.BAD_JSON:
        assert content.startswith(("Invalid JSON in arguments", "Invalid arguments")), content
    elif call.kind is CallKind.SCHEMA_INVALID:
        assert "failed validation" in content, content
    elif call.kind is CallKind.UNAVAILABLE_SERVER:
        assert content == "MCP server 'flaky' is unavailable.", content
    elif call.decision is not ApprovalChoice.APPROVE and call.kind is CallKind.PROTECTED:
        assert "not executed" in content, content
    else:
        outcome, text = call.behavior.outcome, call.behavior.text
        if outcome in (ToolOutcome.SUCCESS, ToolOutcome.ERROR):
            assert content == text
        elif outcome is ToolOutcome.TIMEOUT:
            assert content.startswith(f"Tool '{call.name}' timed out"), content
        elif outcome is ToolOutcome.EXCEPTION:
            assert content == f"RuntimeError: {text}"
        elif outcome is ToolOutcome.UNAVAILABLE:
            assert content.endswith("is unavailable."), content


def assert_matches(
    history: list[ChatMessage],
    script: ScenarioScript,
    turns: list[Turn],
    *,
    with_user: bool,
    cut: int | None,
    synthetic_last: bool,
    exact_results: bool,
) -> None:
    """`history` is the user message (if `with_user`) and the first `cut` turns (all if `None`).

    With `synthetic_last`, the last of those turns carries the "not completed" results instead of
    the results its calls would have had. A `cut` of `n` and `synthetic_last` mean: `n` turns, the
    last one without its real results.
    """
    expected_turns = turns if cut is None else turns[:cut]
    position = 0
    if with_user:
        assert history[0] == UserMessage(script.user_text)
        position = 1
    for number, turn in enumerate(expected_turns):
        assistant = history[position]
        position += 1
        assert isinstance(assistant, AssistantMessage)
        content = turn.text if not turn.calls else (turn.text or None)
        assert assistant.content == content
        assert len(assistant.tool_calls) == len(turn.calls)
        for stored, scripted in zip(assistant.tool_calls, turn.calls, strict=True):
            assert (stored.name, stored.arguments) == (scripted.name, scripted.arguments)
            if scripted.call_id is None:
                assert stored.id.startswith("call_")
            else:
                assert stored.id == scripted.call_id
        for stored, scripted in zip(assistant.tool_calls, turn.calls, strict=True):
            result = history[position]
            position += 1
            assert isinstance(result, ToolMessage)
            assert result.tool_call_id == stored.id
            if synthetic_last and number == len(expected_turns) - 1:
                assert result.content == NOT_COMPLETED_CANCELED
            elif exact_results:
                check_result(scripted, result.content)
    assert position == len(history), f"{len(history) - position} unexpected trailing messages"


def history_matches_a_cut(
    run: ScenarioRun, turns: list[Turn], *, exact_results: bool
) -> str | None:
    """Try every legal cut of a canceled Task. Return None on a match, else the last failure."""
    history = run.history
    script = run.script
    failures: list[str] = []
    if not history:
        return None  # canceled before the user message was committed
    candidates: list[tuple[int, bool]] = [(0, False)]
    for count in range(1, len(turns) + 1):
        candidates.append((count, False))  # the turn is complete: assistant and real results
        if turns[count - 1].calls:
            candidates.append((count, True))  # the assistant message stored, results synthetic
    for count, synthetic in candidates:
        try:
            assert_matches(
                history,
                script,
                turns,
                with_user=True,
                cut=count,
                synthetic_last=synthetic,
                exact_results=exact_results,
            )
        except (AssertionError, IndexError) as exc:
            failures.append(f"cut {count} synthetic={synthetic}: {exc}")
        else:
            return None
    return "; ".join(failures)


# --- The property ---------------------------------------------------------------------------------


def has_cancel_source(script: ScenarioScript) -> bool:
    return (
        script.cancel_after_events is not None
        or any(call.behavior.outcome is ToolOutcome.HANG for call in script.calls)
        or any(turn.failure is LlmFailure.STALL for turn in script.turns)
    )


# Feature: local-agent-backend, Property 12: History integrity and result ordering
@settings(max_examples=150, deadline=None)
@given(script=scenario_scripts())
def test_history_integrity_and_result_ordering(
    store_kind: StoreKind, script: ScenarioScript
) -> None:
    """**Validates: Requirements 8.4, 8.5, 8.10, 9.8, 9.10, 11.4, 11.5, 14.5, 5.13**"""
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

    assert run.state in TERMINAL_STATES
    assert run.task is not None
    history = run.history

    # (a) to (d)
    assert_history_integrity(history)

    # (f) a Tool_Call failure alone never fails the Task; only an LLM failure does.
    llm_failure_reached = any(
        turn.failure is not None for turn in script.turns[: len(run.llm.requests)]
    )
    if run.state is TaskState.FAILED:
        assert llm_failure_reached
    if run.state is TaskState.CANCELED:
        assert has_cancel_source(script)
    if not llm_failure_reached and not has_cancel_source(script):
        assert run.state is TaskState.COMPLETED

    # (e) the stored history equals the script-derived sequence.
    turns, ends_in_failure = derive_turns(script, run)
    changes = any(turn.world_changes for turn in script.turns)
    exact_results = not changes
    if run.state is TaskState.CANCELED:
        mismatch = history_matches_a_cut(run, turns, exact_results=exact_results)
        assert mismatch is None, mismatch
    else:
        assert (run.state is TaskState.FAILED) == ends_in_failure
        assert_matches(
            history,
            script,
            turns,
            with_user=True,
            cut=None,
            synthetic_last=False,
            exact_results=exact_results,
        )
