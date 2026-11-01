"""Property test for availability filtering of the tool list (Requirement 7.9).

A `RegistrySnapshot` holds tools from every server that registered. When the LLM asks for tools,
`openai_tools(availability)` and `tools_by_server(availability)` must return only the tools of
servers marked available. The test builds a registry from generated servers and tools, draws an
availability argument in each accepted shape (a `{server: bool}` mapping or a collection of
names), and checks that the exposed definitions are exactly the registered definitions of the
available servers, in the same order.
"""

from collections.abc import Collection, Mapping

from hypothesis import given
from hypothesis import strategies as st

from local_agent.mcp_client.registry import ToolRegistry
from local_agent.models import McpToolInfo

# A small server pool makes it likely that some servers are available and some are not.
# "a.b" and "a_b" sanitize to the same prefix, so tool names collide across servers too.
_SERVERS = ["alpha", "beta", "gamma", "a.b", "a_b"]
_TOOL_NAMES = st.sampled_from(["read", "write", "list", "r.d", "r_d", "x" * 70])
_TOOLS = st.lists(
    st.builds(
        McpToolInfo,
        name=_TOOL_NAMES,
        description=st.one_of(st.none(), st.text(max_size=20)),
        input_schema=st.just({"type": "object", "properties": {"q": {"type": "string"}}}),
    ),
    max_size=4,
)


@st.composite
def _registry_contents(draw: st.DrawFn) -> list[tuple[str, list[McpToolInfo]]]:
    """Servers in registration order, each with its tool list (possibly empty)."""
    servers = draw(st.lists(st.sampled_from(_SERVERS), unique=True, max_size=len(_SERVERS)))
    return [(server, draw(_TOOLS)) for server in servers]


@st.composite
def _availability(draw: st.DrawFn) -> Mapping[str, bool] | Collection[str]:
    """An availability argument covering registered, unregistered and unmentioned servers."""
    known = draw(st.lists(st.sampled_from([*_SERVERS, "ghost"]), unique=True))
    if draw(st.booleans()):
        return {server: draw(st.booleans()) for server in known}
    as_collection: Collection[str] = draw(st.sampled_from([list, set, frozenset, tuple]))(known)
    return as_collection


def _available_servers(availability: Mapping[str, bool] | Collection[str]) -> set[str]:
    if isinstance(availability, Mapping):
        return {server for server, up in availability.items() if up}
    return set(availability)


# Feature: local-agent-backend, Property 8: Only available servers' tools are exposed
# **Validates: Requirements 7.9**
@given(
    contents=_registry_contents(),
    availability=_availability(),
    server_order=st.permutations(_SERVERS),
)
def test_only_available_servers_tools_are_exposed(
    contents: list[tuple[str, list[McpToolInfo]]],
    availability: Mapping[str, bool] | Collection[str],
    server_order: list[str],
) -> None:
    registry = ToolRegistry(server_order)
    for server, tools in contents:
        registry.replace_server(server, tools)
    snapshot = registry.snapshot()
    available = _available_servers(availability)

    expected = [tool for tool in snapshot.tools if tool.server in available]
    exposed = snapshot.openai_tools(availability)

    # Exactly the translated tools of available servers, in snapshot order (7.9).
    assert exposed == [tool.openai_def for tool in expected]
    exposed_names = [definition["function"]["name"] for definition in exposed]
    assert exposed_names == [tool.llm_name for tool in expected]

    # No tool of an unavailable server appears, by name or by resolution.
    for tool in snapshot.tools:
        if tool.server not in available:
            assert tool.llm_name not in exposed_names
    for name in exposed_names:
        resolved = snapshot.resolve(name)
        assert resolved is not None
        assert resolved[0] in available

    # The per-server grouping used for the prompt follows the same rule.
    grouped = snapshot.tools_by_server(availability)
    assert set(grouped) <= available
    assert [tool for tools in grouped.values() for tool in tools] == sorted(
        expected, key=lambda tool: list(grouped).index(tool.server)
    )
    assert sum(len(tools) for tools in grouped.values()) == len(expected)
