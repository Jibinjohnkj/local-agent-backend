"""Convert an MCP tool definition into an OpenAI-compatible function tool definition (7.2).

``translate`` is pure. When a tool's input schema cannot be used, it returns a
``SchemaRejected`` value instead of logging. The caller (the Tool_Registry) logs the warning
that names the MCP_Server and MCP_Tool, then keeps registering the remaining tools (7.8).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from local_agent.models import McpToolInfo


@dataclass(frozen=True)
class SchemaRejected:
    """The tool's input schema was excluded from the registry (7.8)."""

    server: str
    tool: str
    reason: str


def translate(server: str, tool: McpToolInfo, llm_name: str) -> dict[str, Any] | SchemaRejected:
    """Build ``{"type": "function", "function": {...}}`` for ``tool``, or reject its schema.

    The input schema is rejected when it fails the JSON Schema meta-schema check, or when it
    declares any ``type`` other than ``"object"`` (7.8). A schema with no ``type`` gets
    ``type: "object"`` and an empty ``properties`` object when it has none (7.6). Every other
    keyword is copied unchanged, so ``type``, ``properties`` and ``required`` equal the input
    for object schemas (7.5).
    """
    schema = tool.input_schema

    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as exc:
        return SchemaRejected(
            server=server,
            tool=tool.name,
            reason=f"input schema is not valid JSON Schema: {exc.message}",
        )

    if "type" in schema and schema["type"] != "object":
        return SchemaRejected(
            server=server,
            tool=tool.name,
            reason=f"input schema type must be 'object', got {schema['type']!r}",
        )

    parameters = deepcopy(schema)
    if "type" not in parameters:
        parameters["type"] = "object"
        parameters.setdefault("properties", {})

    return {
        "type": "function",
        "function": {
            "name": llm_name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }
