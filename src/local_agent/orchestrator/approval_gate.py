"""Approval_Gate: the user's approve/deny decision for a Protected_Tool Tool_Call.

The module has two parts. The pure decision parser reads a continuation message and answers one
question: did the user approve, deny, or say something we can't recognize (Requirement 5.12)?
`ApprovalGate` is the per-Task gate that suspends one Protected_Tool Tool_Call at a time, moves
the Task to `input-required`, and waits for the decision or for the approval timeout.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from a2a.types import Message

from local_agent.a2a_server.events import TaskEventEmitter, TaskState

if TYPE_CHECKING:
    # `tool_dispatch` imports `Decision` from this module, so it is imported for typing only.
    from local_agent.orchestrator.tool_dispatch import Prepared

_DECISION_KEY = "decision"

Sleep = Callable[[float], Awaitable[None]]


class Decision(Enum):
    """How a pending Protected_Tool Tool_Call was resolved."""

    APPROVE = "approve"
    DENY = "deny"
    TIMEOUT = "timeout"


class ApprovalKeywordSet(Protocol):
    """The slice of the configured Approval_Keywords the parser needs.

    The config model `ApprovalKeywords` satisfies this protocol, so the parser does not import
    the config package.
    """

    @property
    def approve(self) -> Iterable[str]: ...

    @property
    def deny(self) -> Iterable[str]: ...


def _structured_decision(message: Message) -> Decision | None:
    """Return the first data part whose `decision` is exactly "approve" or "deny"."""
    for part in message.parts:
        if part.WhichOneof("content") != "data" or part.data.WhichOneof("kind") != "struct_value":
            continue
        value = part.data.struct_value.fields.get(_DECISION_KEY)
        if value is None or value.WhichOneof("kind") != "string_value":
            continue
        if value.string_value == Decision.APPROVE.value:
            return Decision.APPROVE
        if value.string_value == Decision.DENY.value:
            return Decision.DENY
    return None


def _keyword_decision(message: Message, keywords: ApprovalKeywordSet) -> Decision | None:
    """Return the decision of the first text part that equals a keyword after trim + casefold."""
    approve = {word.casefold() for word in keywords.approve}
    deny = {word.casefold() for word in keywords.deny}
    for part in message.parts:
        if part.WhichOneof("content") != "text":
            continue
        text = part.text.strip().casefold()
        if text in approve:
            return Decision.APPROVE
        if text in deny:
            return Decision.DENY
    return None


def parse_approval_decision(message: Message, keywords: ApprovalKeywordSet) -> Decision | None:
    """Find the approval decision in `message`, or return None if there isn't one (5.12).

    A structured data part with `decision` exactly "approve" or "deny" wins over any text part,
    and the first such data part wins. Otherwise we compare each text part, trimmed and
    casefolded, with the casefolded keyword lists, and the first match wins.
    """
    structured = _structured_decision(message)
    if structured is not None:
        return structured
    return _keyword_decision(message, keywords)


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """The Protected_Tool Tool_Call that waits for a decision. Held in memory only (5.7).

    `future` completes with the user's `Decision`. The Message_History and the not-yet-executed
    Tool_Calls of the turn stay in the Session_Store and in the suspended `run_turn` frame.
    """

    call_id: str
    llm_name: str
    args: dict[str, Any]
    future: asyncio.Future[Decision]


class ApprovalGate:
    """Suspends Protected_Tool Tool_Calls for approval, one at a time per Task (5.13).

    `request` is awaited by `ToolDispatcher.run_turn`. `resolve` is called by the executor after
    it has committed and emitted the `working` state of a continuation message (5.8). `discard`
    is called when the Task is canceled (2.7).

    `sleep` is the delay used for the approval timeout. Tests pass a fake that advances a fake
    clock instead of waiting.
    """

    def __init__(
        self,
        emitter: TaskEventEmitter,
        *,
        approval_timeout_s: float | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._emitter = emitter
        self._timeout_s = approval_timeout_s
        self._sleep = sleep
        self.pending: PendingApproval | None = None

    async def request(self, prepared: Prepared) -> Decision:
        """Move the Task to `input-required` and wait for the decision of `prepared` (5.4, 5.6).

        Returns `APPROVE` or `DENY` when `resolve` delivers one. When the approval timeout
        expires first, this method moves the Task back to `working` and returns `TIMEOUT` (5.14).
        Without a timeout it waits until `resolve` or `discard` (5.15). The caller sends the call
        to its MCP_Server only after `APPROVE`, so no send can precede the approval (5.4).

        Returns `DENY` without waiting when the Task cannot enter `input-required`, which happens
        only after a terminal state. A failed commit raises `StorageWriteFailed`. `discard`
        makes this method raise `asyncio.CancelledError`. The pending entry is cleared on every
        exit.
        """
        if self.pending is not None:
            raise RuntimeError("an approval is already pending for this Task")
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self.pending = PendingApproval(
            call_id=prepared.call.id,
            llm_name=prepared.call.name,
            args=dict(prepared.args),
            future=future,
        )
        try:
            if not await self._emitter.approval_required(prepared):
                return Decision.DENY
            decision = await self._wait(future)
            if decision is Decision.TIMEOUT:
                await self._emitter.transition(TaskState.WORKING)
            return decision
        finally:
            if self.pending is not None and self.pending.future is future:
                self.pending = None
            if not future.done():
                future.cancel()

    def resolve(self, decision: Decision) -> None:
        """Deliver the user's decision to the pending request. Does nothing without one."""
        pending = self.pending
        if pending is not None and not pending.future.done():
            pending.future.set_result(decision)

    def discard(self) -> None:
        """Drop the pending approval because the Task was canceled (2.7).

        A later `resolve` is a no-op, and the waiting `request` raises `asyncio.CancelledError`.
        """
        pending = self.pending
        self.pending = None
        if pending is not None and not pending.future.done():
            pending.future.cancel()

    async def _wait(self, future: asyncio.Future[Decision]) -> Decision:
        """Wait for `future`, or return `TIMEOUT` when the approval timeout passes first."""
        if self._timeout_s is None:
            return await future
        timer = asyncio.ensure_future(self._sleep(self._timeout_s))
        waiting: set[asyncio.Future[Any]] = {future, timer}
        try:
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        finally:
            timer.cancel()
        # A decision that arrived together with the timer wins over the timeout.
        return future.result() if future.done() else Decision.TIMEOUT
