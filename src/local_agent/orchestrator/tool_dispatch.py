"""Tool_Call preparation: decide whether a Tool_Call can be sent, and write its status text.

Every Tool_Call the LLM emits passes through ``prepare_call`` first. It is a pure function. It
runs the checks of Requirements 9.3 to 9.6 in a fixed order and answers with one of two values:

* ``Rejected`` means the call is resolved without contacting any MCP_Server. It carries the
  error text that becomes the Tool_Result (4.6).
* ``Prepared`` means the call is valid. It carries the parsed arguments and whether the tool is a
  Protected_Tool, so the dispatcher knows if it must ask for approval first.

Because validation happens here, a Protected_Tool call with bad arguments is rejected before
approval is requested (5.5).
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from jsonschema.protocols import Validator
from jsonschema.validators import Draft202012Validator, validator_for

from local_agent.a2a_server.events import TaskEventEmitter
from local_agent.logging_setup import Redactor, get_logger
from local_agent.mcp_client.manager import CallOutcome, McpServerUnavailable
from local_agent.mcp_client.registry import RegistrySnapshot
from local_agent.mcp_client.result_formatter import (
    content_to_text,
    exception_text,
    timeout_text,
    truncate,
)
from local_agent.models import Clock, RegisteredTool, ToolCall, ToolResult
from local_agent.orchestrator.approval_gate import ApprovalGate, Decision

_log = get_logger("orchestrator")

MAX_PATH_ARGUMENT_CHARS = 120
"""Longest argument value shown in a tool-start status message (4.2)."""

_PATH_KEY_MARKERS = ("path", "file", "uri", "url")
"""A top-level argument name containing any of these (case-insensitive) is shown to the user."""


@dataclass(frozen=True)
class Prepared:
    """A valid Tool_Call, ready to send to its MCP_Server (or to the Approval_Gate first)."""

    call: ToolCall
    server: str
    tool: str  # original MCP_Tool name
    args: dict[str, Any]
    protected: bool


@dataclass(frozen=True)
class Rejected:
    """A Tool_Call resolved without being sent to any MCP_Server (4.6)."""

    call: ToolCall
    result_text: str
    server: str | None
    tool: str | None
    outcome: Literal["error", "denied"]

    def as_result(self) -> ToolResult:
        """The Tool_Result that goes into the Message_History for this call."""
        return ToolResult(
            call_id=self.call.id,
            llm_name=self.call.name,
            server=self.server,
            tool=self.tool,
            text=self.result_text,
            outcome=self.outcome,
        )


def _unknown_tool_text(name: str, registry: RegistrySnapshot) -> str:
    names = [tool.llm_name for tool in registry.tools]
    if not names:
        return f"Unknown tool '{name}'. No tools are available."
    return f"Unknown tool '{name}'. Available tools: {', '.join(names)}."


def _validation_errors(schema: Mapping[str, Any], args: dict[str, Any]) -> list[str]:
    """Every violation of ``schema`` by ``args``, one line each (9.5).

    A line reads ``<json path>: <keyword> <en dash> <message>``.

    The validator class comes from the schema's ``$schema`` dialect and defaults to Draft 2020-12.
    Lines are sorted by JSON path, then keyword and message, so the output is deterministic.
    """
    validator_class = validator_for(schema, default=Draft202012Validator)
    validator: Validator = validator_class(schema)
    found = sorted(
        (str(error.json_path), str(error.validator), error.message)
        for error in validator.iter_errors(args)
    )
    return [f"{path}: {keyword} \u2013 {message}" for path, keyword, message in found]


def _reject(call: ToolCall, text: str, tool: RegisteredTool | None) -> Rejected:
    return Rejected(
        call=call,
        result_text=text,
        server=None if tool is None else tool.server,
        tool=None if tool is None else tool.tool,
        outcome="error",
    )


def prepare_call(
    call: ToolCall, registry: RegistrySnapshot, availability: Mapping[str, bool]
) -> Prepared | Rejected:
    """Check ``call`` and return ``Prepared`` or the ``Rejected`` result to hand back to the LLM.

    The checks run in this order, and the first failure wins:

    1. The name is not in the registry (9.3).
    2. The arguments are not valid JSON, or are valid JSON but not an object (9.4).
    3. The arguments violate the tool's input schema; every violation is listed (9.5).
    4. The owning MCP_Server is unavailable (9.6). A server missing from ``availability`` counts
       as unavailable.
    """
    tool = registry.get(call.name)
    if tool is None:
        return _reject(call, _unknown_tool_text(call.name, registry), None)

    try:
        parsed: object = json.loads(call.arguments)
    except json.JSONDecodeError as exc:
        text = (
            f"Invalid JSON in arguments for tool '{call.name}': {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        )
        return _reject(call, text, tool)
    if not isinstance(parsed, dict):
        text = (
            f"Invalid arguments for tool '{call.name}': arguments must be a JSON object, "
            f"got {type(parsed).__name__}."
        )
        return _reject(call, text, tool)
    args: dict[str, Any] = parsed

    errors = _validation_errors(tool.openai_def["function"]["parameters"], args)
    if errors:
        text = f"Arguments for tool '{call.name}' failed validation:\n" + "\n".join(errors)
        return _reject(call, text, tool)

    if not availability.get(tool.server, False):
        return _reject(call, f"MCP server '{tool.server}' is unavailable.", tool)

    return Prepared(
        call=call,
        server=tool.server,
        tool=tool.tool,
        args=args,
        protected=tool.protected,
    )


def denied_result(prepared: Prepared, reason: Decision = Decision.DENY) -> ToolResult:
    """The Tool_Result for a Protected_Tool call that will not run (5.10, 5.14).

    ``reason`` is ``Decision.DENY`` when the user said no and ``Decision.TIMEOUT`` when the
    approval timeout expired. ``Decision.APPROVE`` is a caller bug, because an approved call is
    executed, not denied.
    """
    if reason is Decision.DENY:
        text = f"The user denied the Tool_Call to '{prepared.call.name}'. It was not executed."
    elif reason is Decision.TIMEOUT:
        text = (
            f"The Tool_Call to '{prepared.call.name}' was denied because the approval "
            "timeout expired. It was not executed."
        )
    else:
        raise ValueError("denied_result cannot be built for an approved Tool_Call")
    return ToolResult(
        call_id=prepared.call.id,
        llm_name=prepared.call.name,
        server=prepared.server,
        tool=prepared.tool,
        text=text,
        outcome="denied",
    )


def start_status_message(tool: str, args: Mapping[str, Any]) -> str:
    """The status text shown just before a tool runs (4.2).

    ``tool`` is the original MCP_Tool name. We scan the top-level arguments in the order the
    JSON object gave them and take the first one whose name contains ``path``, ``file``,
    ``uri`` or ``url`` (case-insensitive) and whose value is a string. That value, cut to 120
    characters, goes in parentheses: ``Calling read_file (config.json)...``. With no such
    argument the message is ``Calling read_file...``.
    """
    for key, value in args.items():
        lowered = key.lower()
        if isinstance(value, str) and any(marker in lowered for marker in _PATH_KEY_MARKERS):
            return f"Calling {tool} ({value[:MAX_PATH_ARGUMENT_CHARS]})..."
    return f"Calling {tool}..."


class SnapshotSource(Protocol):
    """Where the dispatcher gets the Tool_Registry view. `ToolRegistry` satisfies this."""

    def snapshot(self) -> RegistrySnapshot: ...


class ToolBackend(Protocol):
    """The part of `McpManager` the dispatcher uses. Tests can supply a fake."""

    def availability(self) -> Mapping[str, bool]: ...

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> CallOutcome: ...


class ToolDispatcher:
    """Runs the Tool_Calls of one LLM turn for one Task (4.2 to 4.8, 5.9, 5.10, 5.13, 8.2 to 8.5).

    There is one dispatcher per Task, because it is bound to the Task's emitter and Approval_Gate.
    `run_turn` never raises for a failed Tool_Call (9.8). It raises `asyncio.CancelledError` when
    the Task is canceled, and `StorageWriteFailed` when the emitter cannot commit a status.
    """

    def __init__(
        self,
        registry: SnapshotSource,
        manager: ToolBackend,
        gate: ApprovalGate,
        emitter: TaskEventEmitter,
        redactor: Redactor,
        clock: Clock,
        *,
        tool_timeout_s: float,
        max_output_chars: int,
    ) -> None:
        self._registry = registry
        self._manager = manager
        self._gate = gate
        self._emitter = emitter
        self._redactor = redactor
        self._clock = clock
        self._timeout_s = tool_timeout_s
        self._max_chars = max_output_chars

    async def run_turn(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Resolve every Tool_Call of the turn and return one Tool_Result per call, in call order.

        1. Calls that cannot be sent (unknown tool, bad arguments, unavailable server) get their
           error result and a failed `tool_call_completed` event (4.6).
        2. Non-protected calls start together and run concurrently (8.3).
        3. Protected calls go through the Approval_Gate one at a time in call order (5.13). An
           approved call is executed (5.9). A denied or timed-out call gets a denied result (5.10).

        The results are ordered like `calls`, whatever order they completed in (8.5, 9.10).
        """
        snapshot = self._registry.snapshot()
        availability = self._manager.availability()
        prepared = [prepare_call(call, snapshot, availability) for call in calls]
        results: dict[int, ToolResult] = {}

        for index, item in enumerate(prepared):
            if isinstance(item, Rejected):
                results[index] = item.as_result()
                await self._emitter.tool_completed(item, ok=False)

        plain = [
            (index, item)
            for index, item in enumerate(prepared)
            if isinstance(item, Prepared) and not item.protected
        ]
        for (index, _), result in zip(plain, await self._run_concurrently(plain), strict=True):
            results[index] = result

        for index, item in enumerate(prepared):
            if not (isinstance(item, Prepared) and item.protected):
                continue
            decision = await self._gate.request(item)
            if decision is Decision.APPROVE:
                results[index] = await self._execute(item)
            else:
                results[index] = denied_result(item, reason=decision)
                await self._emitter.tool_completed(item, ok=False)

        return [results[index] for index in range(len(calls))]

    async def _run_concurrently(self, items: Sequence[tuple[int, Prepared]]) -> list[ToolResult]:
        """Execute `items` at the same time. A failure of one cancels the rest and propagates.

        `_execute` raises only `CancelledError` and `StorageWriteFailed`, so no sibling keeps
        running after the turn has stopped.
        """
        tasks = [asyncio.ensure_future(self._execute(item)) for _, item in items]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _execute(self, prepared: Prepared) -> ToolResult:
        """Send one valid Tool_Call to its MCP_Server and build its Tool_Result.

        The order is `tool_started`, the call, `tool_completed` (4.2, 4.3, 4.5). A tool error,
        a timeout, an unavailable server, and any other exception become a Tool_Result and leave
        the Task `working` (9.1, 9.2, 9.7, 9.8).
        """
        await self._emitter.tool_started(
            prepared, start_status_message(prepared.tool, prepared.args)
        )
        started = self._clock.monotonic()
        result = await self._call(prepared)
        await self._emitter.tool_completed(prepared, ok=result.outcome == "success")
        _log.info(
            "tool_call_finished",
            tool=prepared.call.name,
            server=prepared.server,
            tool_call_id=prepared.call.id,
            outcome=result.outcome,
            duration_ms=round((self._clock.monotonic() - started) * 1000),
        )
        return result

    async def _call(self, prepared: Prepared) -> ToolResult:
        """Call the MCP_Server and map the outcome or the exception to a Tool_Result."""
        text: str
        outcome: Literal["success", "error", "timeout"]
        try:
            reply = await self._manager.call_tool(
                prepared.server, prepared.tool, prepared.args, self._timeout_s
            )
        except McpServerUnavailable:
            text, outcome = f"MCP server '{prepared.server}' is unavailable.", "error"
        except TimeoutError:
            text, outcome = timeout_text(prepared.call.name, self._timeout_s), "timeout"
        except Exception as exc:  # every failure goes back to the LLM (9.7)
            text, outcome = exception_text(exc, self._redactor), "error"
        else:
            text = truncate(content_to_text(reply.content), self._max_chars)
            outcome = "error" if reply.is_error else "success"
        return ToolResult(
            call_id=prepared.call.id,
            llm_name=prepared.call.name,
            server=prepared.server,
            tool=prepared.tool,
            text=text,
            outcome=outcome,
        )


__all__ = [
    "MAX_PATH_ARGUMENT_CHARS",
    "Prepared",
    "Rejected",
    "SnapshotSource",
    "ToolBackend",
    "ToolDispatcher",
    "denied_result",
    "prepare_call",
    "start_status_message",
]
