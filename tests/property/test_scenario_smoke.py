"""Smoke tests for the scenario fakes and harness (task 17.9).

They prove the pieces work end to end on both Session_Stores: a streamed answer, a tool turn, an
approval, a cancel, an LLM failure, a kill, and the transcript normalizer. The properties that use
the harness (P1-P4, P12-P14, P23, P50-P52, P59, P61) live in their own modules.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path

from hypothesis import given, settings
from scenarios import (
    FETCH,
    LIST_DIR,
    READ_FILE,
    WRITE_FILE,
    ApprovalChoice,
    CallKind,
    CallScript,
    ContextPlan,
    ScenarioRun,
    ScenarioScript,
    StoreKind,
    TurnScript,
    build_transcript,
    kill_run,
    normalize_transcript,
    open_store,
    run_plans,
    run_scenario,
    scenario_scripts,
    store_kind,
    tagged_arguments,
    transcript_json,
)

from fakes import DEFAULT_BEHAVIOR, LlmFailure, ToolBehavior, ToolOutcome
from local_agent.a2a_server.events import (
    KIND_TOOL_COMPLETED,
    KIND_TOOL_STARTED,
    TaskState,
)
from local_agent.models import AssistantMessage, ChatMessage, ToolCall, ToolMessage, UserMessage

__all__ = ["store_kind"]  # the fixture is used by name below

GENERATED_ID = r"call_[0-9a-f]{32}"


def read_call(tag: str, behavior: ToolBehavior = DEFAULT_BEHAVIOR) -> CallScript:
    return CallScript(
        tag=tag,
        kind=CallKind.VALID,
        name=READ_FILE,
        arguments=tagged_arguments({"path": "a.txt"}, tag),
        behavior=behavior,
        call_id=f"call-{tag}",
    )


def write_call(tag: str, decision: ApprovalChoice) -> CallScript:
    return CallScript(
        tag=tag,
        kind=CallKind.PROTECTED,
        name=WRITE_FILE,
        arguments=tagged_arguments({"path": "out.txt", "content": "x"}, tag),
        decision=decision,
        call_id=f"call-{tag}",
    )


def assert_history_integrity(history: list[ChatMessage]) -> None:
    """(a) every tool message answers an earlier call; (b) every call has exactly one answer."""
    issued: list[str] = []
    answered: list[str] = []
    for message in history:
        if isinstance(message, AssistantMessage):
            issued += [call.id for call in message.tool_calls]
        elif isinstance(message, ToolMessage):
            assert message.tool_call_id in issued
            answered.append(message.tool_call_id)
    assert sorted(answered) == sorted(issued)


def test_streamed_answer(store_kind: StoreKind) -> None:
    script = ScenarioScript("hi", (TurnScript(chunks=("Hel", "lo")),))
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        assert run.state is TaskState.COMPLETED
        assert run.sink.states == [TaskState.WORKING, TaskState.COMPLETED]
        assert run.sink.statuses[0].text == "Discovering local tools..."
        tokens = [(e.text, e.append, e.last_chunk) for e in run.sink.artifact_events]
        assert tokens == [("Hel", False, False), ("lo", True, False), ("", True, True)]
        assert run.sink.emitted == {"response-1": "Hello"}
        assert run.history == [UserMessage("hi"), AssistantMessage("Hello")]
        assert run.task is not None
        assert [(a.text, a.closed) for a in run.task.artifacts] == [("Hello", True)]
        assert run.system_prompt is not None
        assert [len(r.system_messages) for r in run.llm.requests] == [1]
        assert run.mcp.refreshes == 1


def test_tool_turn(store_kind: StoreKind) -> None:
    call = read_call("t0c0", behavior=ToolBehavior(text="contents"))
    script = ScenarioScript(
        "read it",
        (TurnScript(chunks=("Let me look. ",), calls=(call,)), TurnScript(chunks=("Done",))),
    )
    with open_store(store_kind) as env:
        run = run_scenario(script, env, probe_history=True)

        assert run.state is TaskState.COMPLETED
        assert run.history == [
            UserMessage("read it"),
            AssistantMessage("Let me look. ", (ToolCall("call-t0c0", READ_FILE, call.arguments),)),
            ToolMessage("call-t0c0", "contents"),
            AssistantMessage("Done"),
        ]
        assert [(c.server, c.tool, c.args["path"]) for c in run.mcp.calls] == [
            ("files", "read_file", "a.txt")
        ]
        started = run.sink.tool_events(KIND_TOOL_STARTED)
        completed = run.sink.tool_events(KIND_TOOL_COMPLETED)
        assert [e.metadata["tool_call_id"] for e in started + completed if e.metadata] == [
            "call-t0c0",
            "call-t0c0",
        ]
        assert started[0].text == "Calling read_file (a.txt)..."
        # tool_call_completed arrives before the Tool_Result is stored: user + assistant only.
        assert completed[0].history_len == 2
        assert [len(r.messages) for r in run.llm.requests] == [2, 4]  # system+user, then +2
        assert run.llm.requests[0].tools is not None
        assert_history_integrity(run.history)


def test_concurrent_calls_keep_call_order(store_kind: StoreKind) -> None:
    slow = read_call("t0c0", behavior=ToolBehavior(text="slow", latency=40))
    fast = CallScript(
        tag="t0c1",
        kind=CallKind.VALID,
        name=LIST_DIR,
        arguments=tagged_arguments({}, "t0c1"),
        behavior=ToolBehavior(text="fast"),
        call_id="call-t0c1",
        pieces=2,
    )
    script = ScenarioScript("go", (TurnScript(calls=(slow, fast), interleave=True),))
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        results = [m.content for m in run.history if isinstance(m, ToolMessage)]
        assert results == ["slow", "fast"]  # call order, whichever finished first
        assert run.mcp.max_in_flight == 2
        assert run.mcp.finish_order == [1, 0]  # the second call finished first
        assert_history_integrity(run.history)


def test_unsendable_calls_get_failed_results(store_kind: StoreKind) -> None:
    calls = (
        CallScript("t0c0", CallKind.UNKNOWN_TOOL, "nope", "{}", call_id="a"),
        CallScript("t0c1", CallKind.BAD_JSON, READ_FILE, "{", call_id="b"),
        CallScript("t0c2", CallKind.SCHEMA_INVALID, FETCH, "{}", call_id="c"),
        CallScript("t0c3", CallKind.UNAVAILABLE_SERVER, "flaky__ping", "{}", call_id=None),
    )
    with open_store(store_kind) as env:
        run = run_scenario(ScenarioScript("go", (TurnScript(calls=calls),)), env)

        assert run.state is TaskState.COMPLETED
        assert run.mcp.calls == []
        assert not run.sink.tool_events(KIND_TOOL_STARTED)
        outcomes = [e.text for e in run.sink.tool_events(KIND_TOOL_COMPLETED)]
        assert len(outcomes) == 4
        assert all(text is not None and text.endswith("failed.") for text in outcomes)
        assert_history_integrity(run.history)


def test_protected_call_waits_for_the_scripted_decision(store_kind: StoreKind) -> None:
    for choice, expected in (
        (ApprovalChoice.APPROVE, 1),
        (ApprovalChoice.DENY, 0),
        (ApprovalChoice.TIMEOUT, 0),
    ):
        script = ScenarioScript(
            "write",
            (TurnScript(calls=(write_call("t0c0", choice),)),),
            approval_timeout_s=5 if choice is ApprovalChoice.TIMEOUT else None,
        )
        with open_store(store_kind) as env:
            run = run_scenario(script, env)

            assert run.state is TaskState.COMPLETED
            assert TaskState.INPUT_REQUIRED in run.sink.states
            assert len(run.mcp.calls) == expected
            result = run.history[2]
            assert isinstance(result, ToolMessage)
            assert (result.content == "ok") if expected else ("denied" in result.content)
            assert_history_integrity(run.history)
            if choice is ApprovalChoice.TIMEOUT:
                assert run.timers.requested == [5]


def test_cancel_during_a_hanging_tool(store_kind: StoreKind) -> None:
    hang = read_call("t0c0", behavior=ToolBehavior(ToolOutcome.HANG))
    with open_store(store_kind) as env:
        run = run_scenario(ScenarioScript("go", (TurnScript(calls=(hang,)),)), env)

        assert run.state is TaskState.CANCELED
        assert run.sink.last_state is TaskState.CANCELED
        assert run.mcp.calls[0].cancelled
        # The unanswered call got a synthetic result in the terminal commit (11.5).
        assert isinstance(run.history[-1], ToolMessage)
        assert run.history[-1].tool_call_id == "call-t0c0"
        assert_history_integrity(run.history)


def test_cancel_after_the_nth_event(store_kind: StoreKind) -> None:
    script = ScenarioScript("go", (TurnScript(chunks=("a", "b", "c", "d")),), cancel_after_events=2)
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        assert run.state is TaskState.CANCELED
        assert run.task is not None
        emitted = run.sink.emitted.get("response-1", "")
        # The persisted artifact text is a prefix of what was emitted (P52).
        assert all(emitted.startswith(a.text) for a in run.task.artifacts)


def test_llm_failures_end_the_task(store_kind: StoreKind) -> None:
    cases = {
        LlmFailure.UNREACHABLE: (3, "unreachable"),
        LlmFailure.REJECTED: (1, "rejected"),
        LlmFailure.INTERRUPTED: (1, "interrupted"),
    }
    for failure, (attempts, word) in cases.items():
        turn = TurnScript(chunks=("one", "two"), failure=failure, fail_after=1, reason="bad model")
        with open_store(store_kind) as env:
            run = run_scenario(ScenarioScript("go", (turn,)), env)

            assert run.state is TaskState.FAILED, word
            assert run.llm.attempts == attempts, word
            assert not any(isinstance(m, AssistantMessage) for m in run.history), word
            if failure is LlmFailure.INTERRUPTED:
                # The one chunk emitted before the break stays in the artifact (P2).
                assert run.sink.emitted == {"response-1": "one"}


def test_stalled_stream_is_cancelled(store_kind: StoreKind) -> None:
    turn = TurnScript(chunks=("one", "two"), failure=LlmFailure.STALL, fail_after=1)
    with open_store(store_kind) as env:
        run = run_scenario(ScenarioScript("go", (turn,)), env)

        assert run.state is TaskState.CANCELED
        assert run.sink.emitted == {"response-1": "one"}


def test_iteration_limit_ends_with_a_tools_disabled_request() -> None:
    turns = tuple(
        TurnScript(calls=(read_call(f"t{i}c0"),)) for i in range(3)
    )  # the model keeps asking for tools
    script = ScenarioScript("loop", turns, max_tool_iterations=2)
    with open_store("memory") as env:
        run = run_scenario(script, env)

        assert run.state is TaskState.COMPLETED
        assert [r.tools is not None for r in run.llm.requests] == [True, True, False]
        assert len(run.mcp.calls) == 2
        assert_history_integrity(run.history)


def test_kill_run_leaves_a_reopenable_database() -> None:
    script = ScenarioScript(
        "read", (TurnScript(chunks=("x",), calls=(read_call("t0c0"),)), TurnScript(chunks=("y",)))
    )
    with tempfile.TemporaryDirectory() as tmp:
        clean = kill_run(script, Path(tmp) / "clean", None)
        assert not clean.crashed
        assert clean.runs[0].state is TaskState.COMPLETED
        assert clean.statements > 5

        killed = kill_run(script, Path(tmp) / "killed", clean.statements // 2)

        assert killed.crashed
        assert not killed.runs[0].terminal  # nothing finalized the Task after the kill
        with sqlite3.connect(killed.db_path) as conn:
            states = [row[0] for row in conn.execute("SELECT state FROM tasks")]
        assert states != ["completed"]  # nothing finalized the Task after the kill


def test_contexts_run_concurrently_and_stay_apart(store_kind: StoreKind) -> None:
    def script(word: str) -> ScenarioScript:
        return ScenarioScript(
            word,
            (TurnScript(chunks=(word,), calls=(read_call("t0c0"),)), TurnScript(chunks=("!",))),
        )

    plans = [
        ContextPlan("ctx-a", (script("alpha"), script("again"))),
        ContextPlan("ctx-b", (script("beta"),)),
    ]
    with open_store(store_kind) as env:
        runs = run_plans(plans, env)

        assert [r.task_id for r in runs] == ["ctx-a/task-1", "ctx-a/task-2", "ctx-b/task-1"]
        assert all(r.state is TaskState.COMPLETED for r in runs)
        users = {
            r.context_id: [m.content for m in r.history if isinstance(m, UserMessage)] for r in runs
        }
        assert users == {"ctx-a": ["alpha", "again"], "ctx-b": ["beta"]}
        # The second Task of a Context sees the first Task's four messages before its own four.
        assert len(runs[1].history) == 8
        assert (env.statements > 0) == (store_kind == "sqlite")


def test_transcripts_normalize_generated_ids() -> None:
    generated = CallScript("t0c0", CallKind.VALID, LIST_DIR, tagged_arguments({}, "t0c0"))
    script = ScenarioScript("go", (TurnScript(calls=(generated,)),))
    transcripts: list[str] = []
    for _ in range(2):
        with open_store("memory") as env:
            run = run_scenario(script, env)
            raw = build_transcript(run)
            assert re.search(GENERATED_ID, str(raw))  # the assembler invented an ID
            transcripts.append(transcript_json(raw))
    assert transcripts[0] == transcripts[1]
    assert "<call-id-1>" in transcripts[0]
    assert not re.search(GENERATED_ID, transcripts[0])

    sample = {"updated_at_ms": 1, "duration_ms": 5, "text": "at 2024-01-02T03:04:05Z"}
    assert normalize_transcript(sample) == {
        "updated_at_ms": "<timestamp>",
        "duration_ms": "<duration>",
        "text": "at <timestamp>",
    }


@settings(max_examples=40)
@given(script=scenario_scripts())
def test_generated_scripts_run_to_a_terminal_state(
    store_kind: StoreKind, script: ScenarioScript
) -> None:
    with open_store(store_kind) as env:
        run: ScenarioRun = run_scenario(script, env)

        assert run.terminal
        assert not run.crashed
        assert run.sink.last_state is run.state
        assert_history_integrity(run.history)


@settings(max_examples=25)
@given(script=scenario_scripts(secrets=["hunter2-secret"]))
def test_generated_scripts_are_repeatable(store_kind: StoreKind, script: ScenarioScript) -> None:
    """Two runs of one script produce the same normalized transcript."""
    texts = []
    for _ in range(2):
        with open_store(store_kind, secrets=["hunter2-secret"]) as env:
            texts.append(transcript_json(build_transcript(run_scenario(script, env))))
    assert len(set(texts)) == 1
