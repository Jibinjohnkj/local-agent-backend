"""Property test for context window trimming (Requirements 11.8, 11.9).

`split_turns` groups a history into turns, and `trim_to_budget` keeps the system prompt plus the
longest suffix of whole turns that fits a budget. The test draws a random history, a random
monotone token estimator, and a budget that lands on both sides of the "last turn fits" line. It
then checks the three promises: `TurnTooLarge` when the last turn cannot fit, otherwise a maximal
contiguous suffix of whole turns that includes the last one, and an untouched stored history.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from local_agent.models import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from local_agent.orchestrator.context_window import TurnTooLarge, split_turns, trim_to_budget

_TEXT = st.text(max_size=30)
_TOOL_CALLS = st.builds(ToolCall, id=st.text(max_size=8), name=st.text(max_size=8), arguments=_TEXT)
_ASSISTANT = st.builds(
    AssistantMessage,
    content=st.one_of(st.none(), _TEXT),
    tool_calls=st.lists(_TOOL_CALLS, max_size=3).map(tuple),
)
_TOOL = st.builds(ToolMessage, tool_call_id=st.text(max_size=8), content=_TEXT)
_USER = st.builds(UserMessage, content=_TEXT)
# A turn is a user message followed by assistant and tool messages. Histories are flattened turns.
_TURN = st.tuples(_USER, st.lists(st.one_of(_ASSISTANT, _TOOL), max_size=5)).map(
    lambda pair: [pair[0], *pair[1]]
)
_HISTORIES = st.lists(_TURN, max_size=8).map(lambda turns: [m for turn in turns for m in turn])
_SYSTEM = st.builds(SystemMessage, content=_TEXT)


@dataclass(frozen=True)
class _Estimator:
    """Monotone by construction: a fixed overhead plus a non-negative cost for every message."""

    overhead: int
    scale: int

    def __call__(self, messages: Sequence[ChatMessage]) -> int:
        return self.overhead + sum(self._cost(message) for message in messages)

    def _cost(self, message: ChatMessage) -> int:
        cost = 1
        if isinstance(message, AssistantMessage):
            cost += len(message.content or "") * self.scale
            cost += sum(len(c.name) + len(c.arguments) + 1 for c in message.tool_calls)
        else:
            cost += len(message.content) * self.scale
        return cost


@dataclass(frozen=True)
class Case:
    system: SystemMessage
    history: Sequence[ChatMessage]
    estimate: _Estimator
    budget: int


@st.composite
def cases(draw: st.DrawFn) -> Case:
    system = draw(_SYSTEM)
    history = draw(_HISTORIES)
    estimate = _Estimator(overhead=draw(st.integers(0, 5)), scale=draw(st.integers(1, 3)))
    # The budget runs up to just past the full history, so it covers "nothing dropped" and "some
    # turns dropped". Half the time the floor is the smallest sendable size (system prompt plus last
    # turn), which keeps the interesting trimming cases from being drowned out by `TurnTooLarge`.
    turns = split_turns(history)
    start_at_sendable = draw(st.booleans())
    floor = estimate([system, *turns[-1]]) if turns and start_at_sendable else 0
    budget = draw(st.integers(floor, estimate([system, *history]) + 5))
    return Case(system, history, estimate, budget)


def _flatten(turns: Sequence[Sequence[ChatMessage]]) -> list[ChatMessage]:
    return [message for turn in turns for message in turn]


# Feature: local-agent-backend, Property 20: Context window trimming
# **Validates: Requirements 11.8, 11.9**
@given(case=cases())
def test_context_window_trimming(case: Case) -> None:
    system, history, estimate, budget = case.system, case.history, case.estimate, case.budget
    stored = list(history)

    turns = split_turns(history)
    # Splitting loses no message and starts a new turn at exactly the user messages.
    assert _flatten(turns) == stored
    assert [turn[0] for turn in turns if isinstance(turn[0], UserMessage)] == [
        m for m in stored if isinstance(m, UserMessage)
    ]
    assert all(not isinstance(m, UserMessage) for turn in turns for m in turn[1:]), (
        "a user message must open a new turn"
    )
    turns_before = [list(turn) for turn in turns]

    tightest = estimate([system, *turns[-1]]) if turns else estimate([system])

    if tightest > budget:
        event("outcome=too_large")
        with pytest.raises(TurnTooLarge) as raised:
            trim_to_budget(system, turns, budget, estimate)
        assert raised.value.budget == budget
        assert raised.value.estimate == tightest
    else:
        result = trim_to_budget(system, turns, budget, estimate)

        # System prompt first, then a contiguous suffix of whole turns.
        assert result[0] == system
        tail = result[1:]
        if turns:
            kept = next(k for k in range(len(turns)) if len(_flatten(turns[k:])) == len(tail))
            assert tail == _flatten(turns[kept:])
            # The last turn is always kept (kept <= len(turns) - 1 by construction).
            assert tail[-len(turns[-1]) :] == turns[-1]
            event(f"dropped={'none' if kept == 0 else 'some'}")
            # The result fits, and putting the next older dropped turn back would not.
            assert estimate(result) <= budget
            if kept > 0:
                assert estimate([system, *turns[kept - 1], *tail]) > budget
        else:
            assert tail == []
            assert estimate(result) <= budget

    # The stored history and the turns handed in are unchanged.
    assert history == stored
    assert [list(turn) for turn in turns] == turns_before
