"""Context window trimming: split history into turns, then drop the oldest turns to fit a budget.

`split_turns` and `trim_to_budget` are pure. They never touch the stored Message_History, so the
Session_Store keeps the full conversation while each chat-completion request carries only what fits.
`estimate_tokens` is the I/O-facing estimator. It runs `litellm.token_counter` on a worker thread so
tokenization never blocks the event loop (14.1).
"""

import asyncio
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from local_agent.models import ChatMessage, UserMessage

_log = logging.getLogger(__name__)

TokenEstimator = Callable[[Sequence[ChatMessage]], int]


class TurnTooLarge(Exception):
    """The System_Prompt plus the most recent turn alone exceed the Context_Window_Budget (11.9)."""

    def __init__(self, estimate: int, budget: int) -> None:
        super().__init__(
            f"System_Prompt plus the most recent turn need an estimated {estimate} tokens, "
            f"which exceeds the context window budget of {budget}"
        )
        self.estimate = estimate
        self.budget = budget


def split_turns(history: Sequence[ChatMessage]) -> list[list[ChatMessage]]:
    """Group `history` into turns. Each `UserMessage` opens a new turn.

    A turn is a user message plus every assistant message, Tool_Call, and Tool_Result that follows
    it before the next user message (11.8). Messages that appear before the first user message
    (which should not happen in practice) form a turn of their own, so no message is ever lost:
    concatenating the returned turns always gives back `history`.
    """
    turns: list[list[ChatMessage]] = []
    for message in history:
        if isinstance(message, UserMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def trim_to_budget(
    system: ChatMessage,
    turns: Sequence[Sequence[ChatMessage]],
    budget: int,
    estimate: TokenEstimator,
) -> list[ChatMessage]:
    """Return `system` followed by the longest suffix of whole `turns` whose estimate fits `budget`.

    We drop turns from the oldest side only. The system prompt and the last turn always stay. If
    those two alone are over budget we raise `TurnTooLarge` and the caller sends no request (11.9).
    The result is a new list; neither `turns` nor the messages in it are modified.

    `estimate` must be monotone: adding messages never lowers it. That lets us binary-search the
    number of dropped turns, so a long history costs O(log n) estimator calls instead of O(n).
    Each call can mean a full tokenization, so this matters.
    """
    if not turns:
        base = estimate([system])
        if base > budget:
            raise TurnTooLarge(base, budget)
        return [system]

    # The tightest case we are allowed to send: system prompt plus the last turn.
    tightest = estimate([system, *turns[-1]])
    if tightest > budget:
        raise TurnTooLarge(tightest, budget)

    # `lo` is the smallest drop count we have not ruled out; `hi` always fits (the last turn does).
    # Invariant: dropping `hi` oldest turns fits the budget.
    lo, hi = 0, len(turns) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if estimate(_assemble(system, turns[mid:])) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return _assemble(system, turns[hi:])


def _assemble(system: ChatMessage, turns: Sequence[Sequence[ChatMessage]]) -> list[ChatMessage]:
    result: list[ChatMessage] = [system]
    for turn in turns:
        result.extend(turn)
    return result


def estimate_tokens_sync(
    model: str,
    messages: Sequence[ChatMessage],
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Blocking token estimate for `messages` plus the `tools` definitions.

    Uses `litellm.token_counter`. If it cannot tokenize (unknown model, missing tiktoken cache that
    cannot be downloaded offline, or litellm failing to import), we fall back to `ceil(chars / 4)`.
    A broken tokenizer then only makes the estimate rougher. It never fails the Task.
    Call this from a worker thread; `estimate_tokens` does that for you.
    """
    wire_messages = [message.to_openai() for message in messages]
    wire_tools = [dict(tool) for tool in tools] if tools else None
    try:
        import litellm  # imported here: slow import, and a failure must fall back

        count = litellm.token_counter(model=model, messages=wire_messages, tools=wire_tools)
    except Exception as exc:  # any tokenizer failure must degrade, not fail
        _log.debug("token_counter failed (%s); using the chars/4 estimate", type(exc).__name__)
        return _fallback_estimate(wire_messages, wire_tools)
    return int(count)


async def estimate_tokens(
    model: str,
    messages: Sequence[ChatMessage],
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Estimate the tokens of one chat-completion request without blocking the event loop."""
    return await asyncio.to_thread(estimate_tokens_sync, model, messages, tools)


def _fallback_estimate(
    wire_messages: Sequence[Mapping[str, Any]],
    wire_tools: Sequence[Mapping[str, Any]] | None,
) -> int:
    """`ceil(chars / 4)` over the JSON wire form, so Tool_Calls and tool definitions count too."""
    chars = len(json.dumps(wire_messages, ensure_ascii=False))
    if wire_tools:
        chars += len(json.dumps(wire_tools, ensure_ascii=False))
    return math.ceil(chars / 4)


__all__ = [
    "TokenEstimator",
    "TurnTooLarge",
    "estimate_tokens",
    "estimate_tokens_sync",
    "split_turns",
    "trim_to_budget",
]
