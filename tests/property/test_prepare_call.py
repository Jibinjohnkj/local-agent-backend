"""Property test for rejected Tool_Calls (Requirements 9.3, 9.4, 9.5 and 5.5).

`prepare_call` is the gate every Tool_Call passes before it can reach an MCP_Server. A call with an
unknown name, arguments that are not JSON, arguments that are JSON but not an object, or arguments
that break the tool's input schema must come back as `Rejected`. A `Prepared` result is the only
thing the dispatcher sends, so `Rejected` here means the server receives no request. Because the
checks happen before approval, a Protected_Tool call with bad arguments is rejected too (5.5).

The test builds a registry from generated servers and tools (some protected), draws one kind of bad
Tool_Call, and checks the rejection. For an unknown name the text must list every LLM-facing name,
or say no tools are available. For a schema failure the text must name the JSON path and keyword
of every error that `jsonschema` reports, computed here independently of the implementation.
"""

import json
from collections.abc import Mapping
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from jsonschema.validators import Draft202012Validator

from local_agent.mcp_client.registry import RegistrySnapshot, ToolRegistry
from local_agent.models import McpToolInfo, ToolCall
from local_agent.orchestrator.tool_dispatch import Prepared, Rejected, prepare_call

_SERVERS = ["alpha", "beta", "gamma", "a.b", "a_b"]
_TOOL_NAMES = ["read", "write", "list", "r.d", "r_d", "x" * 70]
_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}, "n": {"type": "integer", "minimum": 0}},
    "required": ["q"],
}
_TOOLS = st.lists(
    st.builds(
        McpToolInfo,
        name=st.sampled_from(_TOOL_NAMES),
        description=st.one_of(st.none(), st.text(max_size=20)),
        input_schema=st.just(_SCHEMA),
    ),
    min_size=1,
    max_size=3,
)
_KINDS = ["unknown_name", "invalid_json", "non_object", "schema_violation"]

# Values that are never a JSON string, used to break the `q` property.
_NON_STRINGS = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(), max_size=3),
)
# JSON documents whose top level is not an object (9.4).
_NON_OBJECTS = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.text(max_size=10),
    st.lists(st.integers(), max_size=3),
)


def _is_invalid_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return True
    return False


_INVALID_JSON = st.text(max_size=20).filter(_is_invalid_json)


@st.composite
def _violating_args(draw: st.DrawFn) -> dict[str, Any]:
    """An object that breaks `_SCHEMA` in at least one way: bad `q`, no `q`, or bad `n`."""
    args: dict[str, Any] = {}
    problem = draw(st.sampled_from(["q_type", "q_missing", "n_type", "n_range", "both"]))
    if problem in ("q_type", "both"):
        args["q"] = draw(_NON_STRINGS)
    elif problem != "q_missing":
        args["q"] = draw(st.text(max_size=5))
    if problem in ("n_type", "both"):
        args["n"] = draw(st.one_of(st.text(max_size=5), st.none(), st.booleans()))
    elif problem == "n_range":
        args["n"] = draw(st.integers(max_value=-1))
    if draw(st.booleans()):
        args["extra"] = draw(st.integers())
    return args


@st.composite
def _scenario(
    draw: st.DrawFn,
) -> tuple[str, RegistrySnapshot, ToolCall, Mapping[str, bool]]:
    kind = draw(st.sampled_from(_KINDS))
    servers = draw(
        st.lists(
            st.sampled_from(_SERVERS),
            unique=True,
            min_size=0 if kind == "unknown_name" else 1,
            max_size=len(_SERVERS),
        )
    )
    policy = {
        server: draw(st.lists(st.sampled_from(["*", "r*", "write"]), max_size=2))
        for server in servers
    }
    registry = ToolRegistry(draw(st.permutations(_SERVERS)), policy)
    for server in servers:
        registry.replace_server(server, draw(_TOOLS))
    snapshot = registry.snapshot()
    names = [tool.llm_name for tool in snapshot.tools]

    if kind == "unknown_name":
        name = draw(st.text(max_size=30).filter(lambda text: text not in names))
        arguments = draw(st.one_of(_INVALID_JSON, st.just('{"q": "ok"}')))
    else:
        name = draw(st.sampled_from(names))
        if kind == "invalid_json":
            arguments = draw(_INVALID_JSON)
        elif kind == "non_object":
            arguments = json.dumps(draw(_NON_OBJECTS))
        else:
            arguments = json.dumps(draw(_violating_args()))
    call = ToolCall(id=draw(st.text(min_size=1, max_size=8)), name=name, arguments=arguments)
    # Availability must not matter: every check above it fails first (9.6 comes last).
    availability = {server: draw(st.booleans()) for server in [*servers, "ghost"]}
    return kind, snapshot, call, availability


# Feature: local-agent-backend, Property 15: Rejected Tool_Calls are never sent
# **Validates: Requirements 9.3, 9.4, 9.5, 5.5**
@settings(max_examples=200)
@given(scenario=_scenario())
def test_rejected_tool_calls_are_never_sent(
    scenario: tuple[str, RegistrySnapshot, ToolCall, Mapping[str, bool]],
) -> None:
    kind, snapshot, call, availability = scenario

    outcome = prepare_call(call, snapshot, availability)

    # Only `Prepared` is sent to an MCP_Server; every bad call must be `Rejected`.
    assert not isinstance(outcome, Prepared)
    assert isinstance(outcome, Rejected)
    assert outcome.call == call
    assert outcome.outcome == "error"
    assert outcome.result_text
    result = outcome.as_result()
    assert result.call_id == call.id
    assert result.llm_name == call.name
    assert result.text == outcome.result_text
    assert result.outcome == "error"

    names = [tool.llm_name for tool in snapshot.tools]
    registered = snapshot.get(call.name)
    if kind == "unknown_name":
        assert registered is None
        assert outcome.server is None
        assert outcome.tool is None
        if names:
            assert all(name in outcome.result_text for name in names)
        else:
            assert "no tools are available" in outcome.result_text.lower()
        return

    assert registered is not None
    assert outcome.server == registered.server
    assert outcome.tool == registered.tool

    if kind == "schema_violation":
        parameters = registered.openai_def["function"]["parameters"]
        errors = list(Draft202012Validator(parameters).iter_errors(json.loads(call.arguments)))
        assert errors
        for error in errors:
            assert str(error.json_path) in outcome.result_text
            assert f"{error.json_path}: {error.validator}" in outcome.result_text
