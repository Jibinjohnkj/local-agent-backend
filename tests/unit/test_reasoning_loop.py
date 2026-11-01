"""Edge cases of the reasoning loop (task 17.20).

These tests run the real `TaskRunner`, `ToolDispatcher`, `ApprovalGate`, and `TaskEventEmitter`
through the scenario harness, on both Session_Stores where the behavior is store independent.
They cover the cases the properties do not pin down one by one: an empty final answer, a failure
message without a traceback, the order of the user message and the first request, approval resume
without a new LLM request, identical arguments after approval, a cancel between approval and
send, a failed commit, and the `task_id` of every stored message.

Requirements: 2.5, 2.6, 4.7, 4.8, 5.7, 5.9, 11.3, 18.10, 18.30.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import pytest
from scenarios import (
    CONTEXT,
    READ_FILE,
    WRITE_FILE,
    ApprovalChoice,
    CallKind,
    CallScript,
    ContextPlan,
    ScenarioScript,
    StoreKind,
    TurnScript,
    open_store,
    run_plans,
    run_scenario,
    store_kind,
    tagged_arguments,
)
from structlog.testing import capture_logs

from fakes import ArtifactEvent, FakeLlmGateway, StatusEvent, ToolBehavior, ToolOutcome
from local_agent.a2a_server.events import (
    KIND_APPROVAL_REQUIRED,
    KIND_TOOL_COMPLETED,
    KIND_TOOL_STARTED,
    TaskState,
    to_sdk,
)
from local_agent.llm.gateway import LlmStream
from local_agent.models import (
    AssistantMessage,
    ChatMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from local_agent.orchestrator.loop import (
    CANCELED_MESSAGE,
    DISCOVERY_MESSAGE,
    NOT_COMPLETED_CANCELED,
    NOT_COMPLETED_FAILED,
    STORAGE_FAILED_MESSAGE,
)
from storage_faults import disk_io_error

__all__ = ["store_kind"]  # the fixture is used by name below


# --- Script helpers -----------------------------------------------------------------------------


def read_call(tag: str, behavior: ToolBehavior | None = None) -> CallScript:
    return CallScript(
        tag=tag,
        kind=CallKind.VALID,
        name=READ_FILE,
        arguments=tagged_arguments({"path": "a.txt"}, tag),
        behavior=behavior or ToolBehavior(),
        call_id=f"call-{tag}",
    )


def write_call(
    tag: str, decision: ApprovalChoice, *, content: str = "x", path: str = "out.txt"
) -> CallScript:
    return CallScript(
        tag=tag,
        kind=CallKind.PROTECTED,
        name=WRITE_FILE,
        arguments=tagged_arguments({"path": path, "content": content}, tag),
        decision=decision,
        call_id=f"call-{tag}",
    )


def tool_turn_script() -> ScenarioScript:
    """A tool round trip and a final answer: user, assistant with a call, result, assistant."""
    return ScenarioScript(
        "read it",
        (
            TurnScript(chunks=("Let me look. ",), calls=(read_call("t0c0"),)),
            TurnScript(chunks=("Done",)),
        ),
    )


# --- Empty final response (2.5) -----------------------------------------------------------------


def test_empty_final_response_still_produces_a_closed_artifact(store_kind: StoreKind) -> None:
    with open_store(store_kind) as env:
        run = run_scenario(ScenarioScript("hello", (TurnScript(),)), env)

        assert run.state is TaskState.COMPLETED
        # No Token_Event carried text; the last-chunk marker creates the artifact (append false).
        assert [
            (e.artifact_id, e.text, e.append, e.last_chunk) for e in run.sink.artifact_events
        ] == [("response-1", "", False, True)]
        assert run.task is not None
        assert [(a.artifact_id, a.text, a.closed) for a in run.task.artifacts] == [
            ("response-1", "", True)
        ]
        # The empty answer is stored as "" and not as "no text" (18.3).
        assert run.history == [UserMessage("hello"), AssistantMessage("")]
        # The artifact is closed before the Task completes (2.5).
        marker = run.sink.events.index(run.sink.artifact_events[-1])
        completed = run.sink.events.index(run.sink.statuses[-1])
        assert marker < completed
        assert run.sink.statuses[-1].state is TaskState.COMPLETED
        assert run.sink.statuses[-1].final


def test_empty_answer_after_a_tool_round_trip_closes_only_the_last_artifact(
    store_kind: StoreKind,
) -> None:
    """A tool turn without text opens no artifact; the empty final answer still gets one."""
    script = ScenarioScript("go", (TurnScript(calls=(read_call("t0c0"),)), TurnScript()))
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        assert run.state is TaskState.COMPLETED
        assert run.task is not None
        assert [(a.artifact_id, a.text, a.closed) for a in run.task.artifacts] == [
            ("response-2", "", True)
        ]
        assert run.history[-1] == AssistantMessage("")
        assert run.history[1] == AssistantMessage(
            None, (ToolCall("call-t0c0", READ_FILE, read_call("t0c0").arguments),)
        )


# --- Failure message without a traceback (2.6) --------------------------------------------------

SECRET = "sk-live-4f9a7c21d0b8"


def _deep_failure(secret: str) -> None:
    raise RuntimeError(f"gateway exploded while using key {secret}")


def test_unhandled_error_fails_the_task_without_traceback_or_secret(
    store_kind: StoreKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exploding_stream(
        self: FakeLlmGateway, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LlmStream:
        _deep_failure(SECRET)  # a frame below the awaiting one, so a traceback would show it
        raise AssertionError("unreachable")

    monkeypatch.setattr(FakeLlmGateway, "stream", exploding_stream)
    with open_store(store_kind, secrets=[SECRET]) as env, capture_logs():
        run = run_scenario(ScenarioScript("hello", (TurnScript(chunks=("x",)),)), env)

        assert run.state is TaskState.FAILED
        last = run.sink.statuses[-1]
        assert (last.state, last.final) == (TaskState.FAILED, True)
        text = last.text
        assert text is not None
        assert text.startswith("An unexpected error stopped the task. RuntimeError: ")
        assert SECRET not in text
        assert "Traceback" not in text
        assert 'File "' not in text
        assert "_deep_failure" not in text
        assert "\n" not in text
        assert run.task is not None
        assert run.task.status_message == text
        assert run.history == [UserMessage("hello")]


# --- User message committed before the first request (11.3, 18.10) ------------------------------


def test_user_message_is_committed_before_each_first_request(
    store_kind: StoreKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots: list[list[ChatMessage]] = []
    original = FakeLlmGateway.stream

    async def spying_stream(
        self: FakeLlmGateway, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LlmStream:
        # `reader` is a separate, never-faulted connection for SQLite: it sees committed rows only.
        _, committed = await env.reader.get_history(CONTEXT)
        snapshots.append(committed)
        return await original(self, messages, tools)

    monkeypatch.setattr(FakeLlmGateway, "stream", spying_stream)
    first = ScenarioScript("first question", (TurnScript(chunks=("one",)),))
    second = ScenarioScript("second question", (TurnScript(chunks=("two",)),))
    with open_store(store_kind) as env:
        runs = run_plans([ContextPlan(CONTEXT, (first, second))], env, probe_history=True)

        assert [run.state for run in runs] == [TaskState.COMPLETED, TaskState.COMPLETED]
        # Task 1: the user message alone. Task 2: the earlier turn, then the new user message.
        assert snapshots == [
            [UserMessage("first question")],
            [
                UserMessage("first question"),
                AssistantMessage("one"),
                UserMessage("second question"),
            ],
        ]
        for run in runs:
            first_status = run.sink.statuses[0]
            assert (first_status.state, first_status.text) == (TaskState.WORKING, DISCOVERY_MESSAGE)
            # The first token arrives after the request, so the user message is already stored.
            first_token = run.sink.artifact_events[0]
            assert first_token.history_len == len(
                snapshots[0] if run.task_id.endswith("task-1") else snapshots[1]
            )


# --- No new LLM call on approval resume (5.7) ---------------------------------------------------


def test_approval_resume_issues_no_llm_request(store_kind: StoreKind) -> None:
    script = ScenarioScript(
        "write twice",
        (
            TurnScript(
                chunks=("Writing. ",),
                calls=(
                    write_call("t0c0", ApprovalChoice.APPROVE),
                    write_call("t0c1", ApprovalChoice.DENY),
                ),
            ),
            TurnScript(chunks=("Done",)),
        ),
    )
    with open_store(store_kind) as env:
        run = run_scenario(script, env, probe_history=True)

        assert run.state is TaskState.COMPLETED
        # One request for the turn with both calls and one for the answer: neither approval, nor
        # the not-yet-executed second call, triggered a request of its own.
        assert len(run.llm.requests) == 2
        second_request = run.llm.requests[1].messages
        assert [m.get("tool_call_id") for m in second_request if m["role"] == "tool"] == [
            "call-t0c0",
            "call-t0c1",
        ]

        events = run.sink.events
        asks = [
            i
            for i, e in enumerate(events)
            if isinstance(e, StatusEvent) and e.state is TaskState.INPUT_REQUIRED
        ]
        assert len(asks) == 2
        for index in asks:
            ask = events[index]
            assert isinstance(ask, StatusEvent)
            assert ask.kind == KIND_APPROVAL_REQUIRED
            # The history is retained while waiting: the user message and the assistant message.
            assert ask.history_len == 2
            # The very next event is the `working` transition of the resume, not a token.
            resume = events[index + 1]
            assert isinstance(resume, StatusEvent)
            assert (resume.state, resume.kind) == (TaskState.WORKING, None)
        # Tokens of the second response come only after the last resume.
        second_response = [
            i
            for i, e in enumerate(events)
            if isinstance(e, ArtifactEvent) and e.artifact_id == "response-2"
        ]
        assert second_response
        assert min(second_response) > asks[-1] + 1
        # Only the approved call was sent, and the denied one is answered without a send.
        assert [call.args["tag"] for call in run.mcp.calls] == ["t0c0"]


# --- Approve uses identical arguments (5.9) -----------------------------------------------------


def test_approved_call_is_sent_with_the_original_arguments(store_kind: StoreKind) -> None:
    content = "  first line\r\n\u00e9\u4e2d\u6587 \U0001f600  \n"
    call = write_call("t0c0", ApprovalChoice.APPROVE, content=content, path="dir/\u00fcber.txt")
    script = ScenarioScript("write", (TurnScript(calls=(call,)), TurnScript(chunks=("Saved",))))
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        assert run.state is TaskState.COMPLETED
        original = json.loads(call.arguments)
        assert len(run.mcp.calls) == 1
        sent = run.mcp.calls[0]
        assert (sent.server, sent.tool) == ("files", "write_file")
        assert sent.args == original
        assert sent.args["content"] == content

        # The user was shown the same arguments that were then sent.
        asked = run.sink.tool_events(KIND_APPROVAL_REQUIRED)
        assert len(asked) == 1
        assert asked[0].metadata is not None
        assert asked[0].metadata["arguments"] == original

        # Order: approval, `working`, `tool_call_started`, then the send (4.8).
        kinds = [(e.state, e.kind) for e in run.sink.statuses]
        ask = kinds.index((TaskState.INPUT_REQUIRED, KIND_APPROVAL_REQUIRED))
        assert kinds[ask + 1] == (TaskState.WORKING, None)
        assert kinds[ask + 2] == (TaskState.WORKING, KIND_TOOL_STARTED)
        # The call is stored exactly as the LLM emitted it.
        assert run.history[1] == AssistantMessage(
            None, (ToolCall("call-t0c0", WRITE_FILE, call.arguments),)
        )


# --- Cancel between approval and send (5.7, 2.7, 14.4) ------------------------------------------


def test_cancel_after_approval_and_before_send_sends_nothing(store_kind: StoreKind) -> None:
    call = write_call("t0c0", ApprovalChoice.APPROVE)
    # Events: 1 discovery `working`, 2 `input-required`, 3 `working` after the approval. The cancel
    # request arrives with event 3, while the approved call has not been sent yet.
    script = ScenarioScript("write", (TurnScript(calls=(call,)),), cancel_after_events=3)
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        assert run.sink.states[:3] == [
            TaskState.WORKING,
            TaskState.INPUT_REQUIRED,
            TaskState.WORKING,
        ]
        assert run.state is TaskState.CANCELED
        assert run.mcp.calls == []
        assert not run.sink.tool_events(KIND_TOOL_STARTED)
        assert not run.sink.tool_events(KIND_TOOL_COMPLETED)  # 4.9
        last = run.sink.statuses[-1]
        assert (last.state, last.text, last.final) == (TaskState.CANCELED, CANCELED_MESSAGE, True)
        assert run.task is not None
        assert run.task.state == to_sdk(TaskState.CANCELED)
        # The approved but unsent call is answered, so the history stays consistent (11.5).
        assert run.history == [
            UserMessage("write"),
            AssistantMessage(None, (ToolCall("call-t0c0", WRITE_FILE, call.arguments),)),
            ToolMessage("call-t0c0", NOT_COMPLETED_CANCELED),
        ]


# --- A failed commit ends the Task `failed` (18.30) ---------------------------------------------


@dataclass(frozen=True)
class Txn:
    """One Storage_Database transaction: the 1-based statement index of its COMMIT and its SQL."""

    commit_index: int
    sql: tuple[str, ...]

    def writes(self, prefix: str) -> bool:
        return any(statement.startswith(prefix) for statement in self.sql)


def transactions(statements: Sequence[str]) -> list[Txn]:
    """Split the counted statements into committed transactions (`BEGIN` ... `COMMIT`)."""
    found: list[Txn] = []
    current: list[str] | None = None
    for number, raw in enumerate(statements, start=1):
        sql = " ".join(raw.split())
        verb = sql.split(" ", 1)[0].upper()
        if verb == "BEGIN":
            current = []
        elif current is None:
            continue
        elif verb in {"COMMIT", "END"}:
            found.append(Txn(number, tuple(current)))
            current = None
        elif verb == "ROLLBACK":
            current = None
        else:
            current.append(sql)
    return found


def nth_message_commit(n: int) -> Callable[[list[Txn]], Txn]:
    def pick(txns: list[Txn]) -> Txn:
        return [t for t in txns if t.writes("INSERT INTO messages")][n]

    return pick


def first_state_commit(txns: list[Txn]) -> Txn:
    return next(t for t in txns if t.writes("UPDATE tasks SET state"))


def last_commit(txns: list[Txn]) -> Txn:
    return txns[-1]


FAILING_COMMITS: dict[str, tuple[Callable[[list[Txn]], Txn], list[TaskState] | None]] = {
    # name: (which commit fails, the exact Status_Event states, or None to skip that check)
    "first working transition": (first_state_commit, [TaskState.FAILED]),
    "user message": (nth_message_commit(0), [TaskState.WORKING, TaskState.FAILED]),
    "assistant message with calls": (nth_message_commit(1), [TaskState.WORKING, TaskState.FAILED]),
    "tool results": (nth_message_commit(2), None),
    "final assistant message": (nth_message_commit(3), None),
    "completed transition": (last_commit, None),
}


@pytest.mark.parametrize("name", list(FAILING_COMMITS))
def test_failed_commit_ends_the_task_failed_and_could_not_be_saved(name: str) -> None:
    pick, expected_states = FAILING_COMMITS[name]
    script = tool_turn_script()

    with open_store("sqlite") as clean:
        clean_run = run_scenario(script, clean)
        assert clean_run.state is TaskState.COMPLETED
        assert clean.factory is not None
        target = pick(transactions(clean.factory.connection.statements))

    with open_store("sqlite", fail_at=target.commit_index, error=disk_io_error) as env:
        with capture_logs() as logs:
            run = run_scenario(script, env)
        assert env.factory is not None
        assert env.factory.connection.fired == (target.commit_index,)

        # The Task is `failed` with the fixed message, committed and emitted (the commit after the
        # failed one succeeds), and nothing that depended on the failed commit was emitted.
        assert run.admitted
        assert run.state is TaskState.FAILED
        assert run.task is not None
        assert run.task.state == to_sdk(TaskState.FAILED)
        assert run.task.status_message == STORAGE_FAILED_MESSAGE
        last = run.sink.statuses[-1]
        assert (last.state, last.text, last.final) == (
            TaskState.FAILED,
            STORAGE_FAILED_MESSAGE,
            True,
        )
        assert run.sink.states.count(TaskState.FAILED) == 1
        assert TaskState.COMPLETED not in run.sink.states
        if expected_states is not None:
            assert run.sink.states == expected_states

        # One error record names the database and the Task (16.11).
        failures = [r for r in logs if r["event"] == "storage_write_failed"]
        assert len(failures) == 1
        assert failures[0]["db_path"] == env.db_path
        assert failures[0]["task_id"] == run.task_id
        assert failures[0]["log_level"] == "error"

        # Every Tool_Call has exactly one answer, and the failed commit left no half-written turn.
        issued = [
            c.id for m in run.history if isinstance(m, AssistantMessage) for c in m.tool_calls
        ]
        answered = [m.tool_call_id for m in run.history if isinstance(m, ToolMessage)]
        assert sorted(answered) == sorted(issued)
        if name == "tool results":
            assert run.history[-1] == ToolMessage("call-t0c0", NOT_COMPLETED_FAILED)
        if name == "user message":
            assert run.history == []


# --- Every message row carries the producing task_id (11.3, 18.10) ------------------------------


def test_every_message_row_carries_the_producing_task_id() -> None:
    hang = read_call("t0c0", ToolBehavior(ToolOutcome.HANG))
    plans = [
        ContextPlan(
            "ctx-a",
            (
                tool_turn_script(),
                ScenarioScript("hang", (TurnScript(calls=(hang,)),)),  # canceled: synthetic result
                ScenarioScript("again", (TurnScript(chunks=("fine",)),)),
            ),
        ),
        ContextPlan("ctx-b", (tool_turn_script(),)),
    ]
    with open_store("sqlite") as env:
        runs = run_plans(plans, env)
        assert [run.state for run in runs] == [
            TaskState.COMPLETED,
            TaskState.CANCELED,
            TaskState.COMPLETED,
            TaskState.COMPLETED,
        ]
        assert env.path is not None
        with closing(sqlite3.connect(env.path)) as conn:
            rows = conn.execute(
                "SELECT context_id, task_id, role FROM messages ORDER BY context_id, seq"
            ).fetchall()

    tool_turn = ["user", "assistant", "tool", "assistant"]
    expected: list[tuple[str, str, str]] = (
        [("ctx-a", "ctx-a/task-1", role) for role in tool_turn]
        # user, assistant with the call, and the synthetic result of the canceled Task
        + [("ctx-a", "ctx-a/task-2", role) for role in ("user", "assistant", "tool")]
        + [("ctx-a", "ctx-a/task-3", role) for role in ("user", "assistant")]
        + [("ctx-b", "ctx-b/task-1", role) for role in tool_turn]
    )
    assert rows == expected
