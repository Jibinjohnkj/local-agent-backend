"""Property 1: stream reassembly and token framing (task 17.10).

For any scripted LLM stream (non-empty text, empty text, and Tool_Call fragments in any
interleaving), the Token_Events of a response are exactly its non-empty text deltas in order, under
one artifact ID that is unique within the Task. The first event has `append=false`, later ones
`append=true`, only the final event has `last_chunk=true`, and the emitted texts add up to the
assistant text stored in the Message_History. Runs on both Session_Stores.
"""

from __future__ import annotations

from hypothesis import given, settings
from scenarios import (
    ScenarioScript,
    StoreKind,
    TurnScript,
    open_store,
    run_scenario,
    scenario_scripts,
    store_kind,
)

from fakes import ArtifactEvent
from local_agent.a2a_server.events import TaskState
from local_agent.models import AssistantMessage

__all__ = ["store_kind"]  # the fixture is used by name below

DEFAULT_ANSWER = ("ok",)
"""What the fake LLM streams once its script has run out."""


def expected_chunks(script: ScenarioScript, index: int) -> list[str]:
    """The non-empty text deltas the LLM streams in response number `index` (0-based)."""
    turn: TurnScript | None = script.turns[index] if index < len(script.turns) else None
    chunks = turn.chunks if turn is not None else DEFAULT_ANSWER
    return [chunk for chunk in chunks if chunk]


# Feature: local-agent-backend, Property 1: Stream reassembly and token framing
@settings(max_examples=200, deadline=None)
@given(script=scenario_scripts(failures=False, cancels=False, world_changes=False))
def test_stream_reassembly_and_token_framing(store_kind: StoreKind, script: ScenarioScript) -> None:
    """**Validates: Requirements 3.2, 3.3, 3.4**"""
    with open_store(store_kind) as env:
        run = run_scenario(script, env)

        assert run.state is TaskState.COMPLETED
        assistants = [m for m in run.history if isinstance(m, AssistantMessage)]
        responses = len(run.llm.requests)
        assert len(assistants) == responses  # one stored assistant message per LLM response

        by_artifact: dict[str, list[ArtifactEvent]] = {}
        for event in run.sink.artifact_events:
            by_artifact.setdefault(event.artifact_id, []).append(event)
        valid_ids = {f"response-{n}" for n in range(1, responses + 1)}
        assert set(by_artifact) <= valid_ids  # no foreign or reused IDs

        for index in range(responses):
            artifact_id = f"response-{index + 1}"
            stored = assistants[index].content or ""
            chunks = expected_chunks(script, index)
            events = by_artifact.get(artifact_id, [])

            if not events:
                # Only a response with Tool_Calls and no text at all may leave no artifact (2.5).
                assert stored == ""
                assert not chunks
                assert assistants[index].tool_calls
                continue

            # Every event but the last carries one non-empty delta; the last is the closing marker.
            *tokens, marker = events
            assert [e.text for e in tokens] == chunks
            assert marker.text == ""
            assert marker.last_chunk
            assert not any(e.last_chunk for e in tokens)

            assert events[0].append is False
            assert all(e.append for e in events[1:])

            assert "".join(e.text for e in events) == stored

        # IDs appear in response order; a closed artifact in the Task holds the emitted text.
        order = list(dict.fromkeys(e.artifact_id for e in run.sink.artifact_events))
        assert order == sorted(order, key=lambda name: int(name.removeprefix("response-")))
        assert run.task is not None
        emitted = run.sink.emitted
        for artifact in run.task.artifacts:
            assert artifact.closed
            assert artifact.text == emitted[artifact.artifact_id]
