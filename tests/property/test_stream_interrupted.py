"""Property test for interrupted LLM streams (task 17.11).

The harness runs the real `TaskRunner` over the real `LlmGateway` (only `litellm.acompletion` is
scripted) and both Session_Stores. A turn streams a generated chunk sequence, mixed with empty
deltas, and the stream raises right after the k-th non-empty chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    READ_FILE,
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

from fakes import LlmFailure, StatusEvent
from local_agent.a2a_server.events import TaskState, to_sdk
from local_agent.models import AssistantMessage, ToolMessage
from local_agent.orchestrator.loop import INTERRUPTED_MESSAGE

__all__ = ["store_kind"]  # the fixture is used by name below

non_empty_chunks = st.text(alphabet=text_chars, min_size=1, max_size=12)


@dataclass(frozen=True)
class Interruption:
    """A response whose stream breaks after `k` non-empty chunks."""

    chunks: tuple[str, ...]  # every chunk the LLM would send, empty ones included
    k: int
    prior_turns: int  # tool turns that complete before the interrupted one

    @property
    def non_empty(self) -> list[str]:
        return [chunk for chunk in self.chunks if chunk]

    @property
    def fail_after(self) -> int:
        """The number of deltas streamed before the break: up to and including the k-th text."""
        seen = 0
        for position, chunk in enumerate(self.chunks, start=1):
            seen += bool(chunk)
            if seen == self.k:
                return position
        raise AssertionError("k exceeds the number of non-empty chunks")


@st.composite
def interruptions(draw: st.DrawFn) -> Interruption:
    texts = draw(st.lists(non_empty_chunks, min_size=1, max_size=8))
    k = draw(st.integers(min_value=1, max_value=len(texts)))
    # Empty deltas anywhere, including after the break position (never delivered) and before the
    # first chunk.
    gaps = draw(
        st.lists(
            st.integers(min_value=0, max_value=2), min_size=len(texts) + 1, max_size=len(texts) + 1
        )
    )
    chunks: list[str] = []
    for gap, text in zip(gaps, [*texts, ""], strict=True):
        chunks += [""] * gap
        if text:
            chunks.append(text)
    return Interruption(tuple(chunks), k, draw(st.integers(min_value=0, max_value=2)))


def prior_turn(index: int) -> TurnScript:
    tag = f"t{index}c0"
    call = CallScript(
        tag=tag,
        kind=CallKind.VALID,
        name=READ_FILE,
        arguments=tagged_arguments({"path": "a.txt"}, tag),
        call_id=f"call-{tag}",
    )
    return TurnScript(chunks=(f"step {index}. ",), calls=(call,))


# Feature: local-agent-backend, Property 2: Interrupted streams keep the emitted prefix
# **Validates: Requirements 3.8, 10.10, 10.7**
@settings(max_examples=100, deadline=None)
@given(case=interruptions())
def test_interrupted_streams_keep_the_emitted_prefix(
    store_kind: StoreKind, case: Interruption
) -> None:
    interrupted = TurnScript(
        chunks=case.chunks, failure=LlmFailure.INTERRUPTED, fail_after=case.fail_after
    )
    script = ScenarioScript("go", (*(prior_turn(i) for i in range(case.prior_turns)), interrupted))
    prefix = case.non_empty[: case.k]
    artifact_id = f"response-{case.prior_turns + 1}"

    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        # The Task ends failed, and the final Status_Event says the response was interrupted.
        assert run.state is TaskState.FAILED
        last = run.sink.events[-1]
        assert isinstance(last, StatusEvent)
        assert last.state is TaskState.FAILED
        assert last.final
        assert last.text == INTERRUPTED_MESSAGE
        assert "interrupted" in (last.text or "")
        failed_events = [e for e in run.sink.statuses if e.state is TaskState.FAILED]
        assert failed_events == [last]  # exactly one final Status_Event

        # Exactly the first k non-empty chunks were emitted as Token_Events, in order, and no
        # Token_Event follows the final Status_Event.
        tokens = [e for e in run.sink.artifact_events if e.artifact_id == artifact_id]
        assert [e.text for e in tokens] == prefix
        assert [e.append for e in tokens] == [False] + [True] * (len(prefix) - 1)
        assert not any(e.last_chunk for e in tokens)
        assert run.sink.events.index(last) == len(run.sink.events) - 1
        assert run.sink.emitted[artifact_id] == "".join(prefix)

        # The gateway made exactly one attempt for the interrupted request (10.10).
        request = run.llm.requests[-1]
        assert len(run.llm.requests) == case.prior_turns + 1
        assert request.attempts == 1
        assert run.llm.attempts == case.prior_turns + 1

        # The history holds no assistant message from that request, and the stored artifact
        # keeps the emitted prefix.
        assistants = [m for m in run.history if isinstance(m, AssistantMessage)]
        assert [m.content for m in assistants] == [f"step {i}. " for i in range(case.prior_turns)]
        assert sum(isinstance(m, ToolMessage) for m in run.history) == case.prior_turns
        assert run.task is not None
        assert run.task.state == to_sdk(TaskState.FAILED)
        assert run.task.status_message == INTERRUPTED_MESSAGE
        stored = {a.artifact_id: a.text for a in run.task.artifacts}
        assert stored[artifact_id] == "".join(prefix)
