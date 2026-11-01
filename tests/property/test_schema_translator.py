"""Property test for MCP tool schema translation (Requirements 7.2, 7.5, 7.6, 7.8)."""

from copy import deepcopy
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from local_agent.mcp_client.schema_translator import SchemaRejected, translate
from local_agent.models import McpToolInfo

_prop_names = st.text(alphabet="abcxyz_0", min_size=1, max_size=4)
_leaf_schemas = st.one_of(
    st.just({"type": "string"}),
    st.just({"type": "integer", "minimum": 0}),
    st.just({"type": "boolean"}),
    st.just({"type": "array", "items": {"type": "string"}}),
    st.just({"enum": ["a", "b"]}),
)
_properties = st.dictionaries(_prop_names, _leaf_schemas, max_size=4)
_extras = st.fixed_dictionaries(
    {},
    optional={
        "description": st.text(max_size=10),
        "title": st.text(max_size=10),
        "additionalProperties": st.booleans(),
        "$comment": st.text(max_size=10),
    },
)


@st.composite
def _valid_schema(draw: st.DrawFn, *, declare_type: bool) -> dict[str, Any]:
    """A schema that passes the meta-schema check and is either ``type: object`` or untyped."""
    schema: dict[str, Any] = dict(draw(_extras))
    if declare_type:
        schema["type"] = "object"
    if draw(st.booleans()):
        schema["properties"] = draw(_properties)
    if draw(st.booleans()):
        # ``required`` may name properties that are not declared; it is still a valid schema.
        schema["required"] = draw(st.lists(_prop_names, unique=True, max_size=4))
    return schema


# Each of these fails the JSON Schema meta-schema check.
_invalid_schemas = st.sampled_from(
    [
        {"type": "object", "properties": 5},
        {"type": "object", "required": "name"},
        {"type": "object", "required": [1, 2]},
        {"type": 12},
        {"type": "nonsense"},
        {"type": "object", "properties": {"a": {"type": "nonsense"}}},
        {"properties": ["a"]},
        {"type": "object", "additionalProperties": "yes"},
    ]
)

# Each of these is a valid JSON Schema whose ``type`` is not ``object``.
_non_object_schemas = st.builds(
    lambda t: {"type": t},
    st.sampled_from(["string", "array", "integer", "number", "boolean", "null"]),
)

# kind: "object" (declares type object), "untyped" (no type), or an excluded kind.
_schema_cases = st.one_of(
    _valid_schema(declare_type=True).map(lambda s: ("object", s)),
    _valid_schema(declare_type=False).map(lambda s: ("untyped", s)),
    _invalid_schemas.map(lambda s: ("invalid", s)),
    _non_object_schemas.map(lambda s: ("non_object", s)),
)
_descriptions = st.one_of(st.none(), st.just(""), st.text(max_size=20))
_llm_names = st.text(alphabet="abcxyz_-0123", min_size=1, max_size=12)

_server_tools = st.lists(
    st.tuples(_schema_cases, _descriptions, _llm_names), min_size=1, max_size=6
)


# Feature: local-agent-backend, Property 7: Schema translation preserves the input schema
# Validates: Requirements 7.2, 7.5, 7.6, 7.8
@given(_server_tools)
def test_translation_preserves_input_schema(
    cases: list[tuple[tuple[str, dict[str, Any]], str | None, str]],
) -> None:
    server = "srv"
    for index, ((kind, schema), description, llm_name) in enumerate(cases):
        original = deepcopy(schema)
        tool = McpToolInfo(name=f"tool{index}", description=description, input_schema=schema)

        # Each tool is translated on its own, so a rejected tool cannot affect the others (7.8).
        result = translate(server, tool, llm_name)

        # The translator never mutates the input schema.
        assert schema == original

        if kind in ("invalid", "non_object"):
            assert isinstance(result, SchemaRejected)
            assert result.server == server
            assert result.tool == tool.name
            continue

        assert isinstance(result, dict)
        assert set(result) == {"type", "function"}
        assert result["type"] == "function"
        function = result["function"]
        assert function["name"] == llm_name
        assert function["description"] == (description or "")

        parameters = function["parameters"]
        assert parameters["type"] == "object"
        if kind == "object":
            # 7.5: type, properties and required equal the input; absence is preserved.
            assert ("properties" in parameters) == ("properties" in schema)
            assert parameters.get("properties") == schema.get("properties")
        else:
            # 7.6: an untyped schema gets type object and the input properties, or ``{}``.
            assert parameters["properties"] == schema.get("properties", {})
        assert ("required" in parameters) == ("required" in schema)
        assert parameters.get("required") == schema.get("required")
        # Every other keyword is copied unchanged.
        for key, value in schema.items():
            assert parameters[key] == value
        # The output does not alias the input.
        assert parameters is not schema
