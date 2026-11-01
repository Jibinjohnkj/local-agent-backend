"""Property test for name round-trip and faithful dispatch (Requirements 7.4, 8.1, 8.2).

The test registers generated MCP_Servers and MCP_Tools in a real `ToolRegistry`. Names are drawn
from a small alphabet that includes characters the sanitizer replaces, so many names collide and
need `_k` suffixes. For every registered tool the test checks two things:

* Resolving the tool's LLM-facing name through the `RegistrySnapshot` returns exactly the original,
  unsanitized server name and tool name (7.4, 8.1).
* A `ToolDispatcher` turn that holds one valid non-protected Tool_Call per tool calls the backend
  once per Tool_Call, with that server, that original tool name, and arguments equal to the ones
  the LLM sent (8.2). The dispatcher runs over a fake manager, a real `ApprovalGate`, and a real
  `TaskEventEmitter` on an `InMemorySessionStore`.
"""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from local_agent.a2a_server.events import TaskEventEmitter, TaskState
from local_agent.logging_setup import Redactor
from local_agent.mcp_client.manager import CallOutcome
from local_agent.mcp_client.registry import ToolRegistry
from local_agent.models import McpToolInfo, ToolCall
from local_agent.orchestrator.approval_gate import ApprovalGate
from local_agent.orchestrator.tool_dispatch import ToolDispatcher
from local_agent.session.memory_store import InMemorySessionStore

TASK = "task-1"
CONTEXT = "ctx-1"
TOOL_TIMEOUT_S = 12.5
MAX_OUTPUT_CHARS = 1000

# A small alphabet with characters the sanitizer rewrites, so distinct names often collide.
_ALPHABET = ["a", "b", "_", "-", "2", ".", " ", "/", "\u00e9", "\u4e2d"]
_COLLIDING = st.lists(st.sampled_from(_ALPHABET), min_size=1, max_size=5).map("".join)
_LONG = st.integers(min_value=55, max_value=90).map(lambda n: ("a." * n)[:n])
_NAMES = st.one_of(_COLLIDING, _LONG, st.text(min_size=1, max_size=20))

_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=10),
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(max_size=6), children, max_size=3),
    ),
    max_leaves=6,
)
_ARGS = st.dictionaries(st.text(max_size=6), _JSON_VALUES, max_size=4)

# One server's tools, as MCP_Tool names. Repeats within a server are allowed.
_SERVERS = st.dictionaries(_NAMES, st.lists(_NAMES, min_size=1, max_size=4), min_size=1, max_size=4)


class Clock:
    """A fixed clock; this test does not depend on time passing."""

    def now_ms(self) -> int:
        return 1_700_000_000_000

    def monotonic(self) -> float:
        return 1_000.0


class NullSink:
    """An event sink that drops everything."""

    async def status(
        self, state: TaskState, text: str | None, metadata: Mapping[str, Any] | None, *, final: bool
    ) -> None:
        return None

    async def artifact(
        self, artifact_id: str, text: str, *, append: bool, last_chunk: bool
    ) -> None:
        return None


@dataclass(frozen=True)
class BackendCall:
    server: str
    tool: str
    args: dict[str, Any]
    timeout: float


class FakeManager:
    """The `ToolBackend` the dispatcher uses. It records each `call_tool` and answers success."""

    def __init__(self, servers: list[str]) -> None:
        self._servers = servers
        self.calls: list[BackendCall] = []

    def availability(self) -> Mapping[str, bool]:
        return dict.fromkeys(self._servers, True)

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> CallOutcome:
        self.calls.append(BackendCall(server, tool, args, timeout))
        return CallOutcome(content=(), is_error=False)


async def dispatch_all(
    registry: ToolRegistry, manager: FakeManager, calls: list[ToolCall]
) -> list[str]:
    """Run one turn with `calls` through a real dispatcher and return the outcomes."""
    clock = Clock()
    store = InMemorySessionStore(clock)
    await store.create_task(TASK, CONTEXT, clock.now_ms(), None)
    emitter = TaskEventEmitter(
        store, NullSink(), Redactor([]), clock, "/tmp/db", task_id=TASK, context_id=CONTEXT
    )
    assert await emitter.transition(TaskState.WORKING)
    dispatcher = ToolDispatcher(
        registry,
        manager,
        ApprovalGate(emitter),
        emitter,
        Redactor([]),
        clock,
        tool_timeout_s=TOOL_TIMEOUT_S,
        max_output_chars=MAX_OUTPUT_CHARS,
    )
    results = await dispatcher.run_turn(calls)
    assert [result.call_id for result in results] == [call.id for call in calls]
    return [result.outcome for result in results]


# Feature: local-agent-backend, Property 6: Name round-trip and faithful dispatch
# **Validates: Requirements 7.4, 8.1, 8.2**
@given(servers=_SERVERS, data=st.data())
def test_names_round_trip_and_dispatch_is_faithful(
    servers: dict[str, list[str]], data: st.DataObject
) -> None:
    registry = ToolRegistry(list(servers))
    for server, tool_names in servers.items():
        registry.replace_server(
            server,
            [
                McpToolInfo(name=t, description=None, input_schema={"type": "object"})
                for t in tool_names
            ],
        )
    snapshot = registry.snapshot()

    # Every original (server, tool) pair is registered, in Backend_Config order (7.4).
    expected = [(server, tool) for server, tool_names in servers.items() for tool in tool_names]
    assert [(t.server, t.tool) for t in snapshot.tools] == expected

    # Resolving each LLM-facing name returns the original, unsanitized names (7.4, 8.1).
    for registered, (server, tool) in zip(snapshot.tools, expected, strict=True):
        assert snapshot.resolve(registered.llm_name) == (server, tool)
        assert not registered.protected

    # One valid non-protected Tool_Call per registered tool, each with its own arguments.
    sent_args = [data.draw(_ARGS) for _ in snapshot.tools]
    calls = [
        ToolCall(id=f"call-{index}", name=registered.llm_name, arguments=json.dumps(args))
        for index, (registered, args) in enumerate(zip(snapshot.tools, sent_args, strict=True))
    ]
    manager = FakeManager(list(servers))

    outcomes = asyncio.run(dispatch_all(registry, manager, calls))

    assert outcomes == ["success"] * len(calls)
    # The backend got exactly one call per Tool_Call. Calls run concurrently, so match them by
    # their position in the turn rather than by arrival order.
    assert len(manager.calls) == len(calls)
    remaining = list(manager.calls)
    for (server, tool), args in zip(expected, sent_args, strict=True):
        wanted = BackendCall(server, tool, args, TOOL_TIMEOUT_S)
        assert wanted in remaining
        remaining.remove(wanted)
    assert remaining == []
