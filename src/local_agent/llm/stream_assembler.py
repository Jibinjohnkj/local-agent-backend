"""Pure, stateful accumulator that turns streamed LLM deltas into one assembled response.

LiteLLM streams OpenAI-style deltas. Text arrives as `content` strings. Tool_Calls arrive as
fragments keyed by `index`: the call ID and name usually show up once, and the JSON arguments are
cut into pieces that must be concatenated in arrival order. This module does no I/O. The gateway
feeds each delta in and calls `finish` once the stream ends.
"""

import json
import re
import uuid
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

from local_agent.models import AssembledResponse, ToolCall, replace_lone_surrogates

# A LiteLLM `Delta` is a pydantic object in production and a plain mapping in tests and in some
# provider adapters, so the assembler reads both shapes through `_get`.
Delta = Any

_MISSING_INDEX = 0


def _get(obj: Any, name: str) -> Any:
    """Read `name` from a mapping or an object, returning None when it is absent."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _non_empty_str(value: Any) -> str | None:
    """Return `value` when it is a non-empty string, otherwise None."""
    if isinstance(value, str) and value:
        return value
    return None


@dataclass
class _CallFragments:
    """Everything seen so far for one Tool_Call index."""

    id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)


_FENCE = re.compile(r"\A```[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)\r?\n?```\Z", re.DOTALL)


def _as_call(obj: Any, known: AbstractSet[str]) -> tuple[str, str] | None:
    """Return `(name, arguments_json)` when `obj` is a call to one of the `known` tools."""
    if not isinstance(obj, dict):
        return None
    # Some models wrap the call as {"function": {...}} the way the OpenAI wire format does.
    function = obj.get("function")
    if isinstance(function, dict) and "name" not in obj:
        obj = function
    name = obj.get("name")
    if not isinstance(name, str) or name not in known:
        return None
    arguments = obj["arguments"] if "arguments" in obj else obj.get("parameters", {})
    if isinstance(arguments, str):
        return name, arguments
    if isinstance(arguments, dict):
        return name, json.dumps(arguments, ensure_ascii=False)
    return None


def parse_prompted_tool_calls(text: str, known: AbstractSet[str]) -> list[tuple[str, str]] | None:
    """Read a prompt-based function-calling reply (10.8).

    A model without native tool support is told to answer with JSON such as
    `{"name": "echo", "arguments": {"text": "hi"}}`, optionally inside a Markdown code fence,
    or a list of such objects. Returns `(name, arguments_json)` pairs when the whole reply is
    one or more calls to a tool in `known`. Returns None for anything else, so ordinary prose
    and JSON that merely looks similar stay text.
    """
    if not known:
        return None
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced is not None:
        body = fenced.group(1).strip()
    if not body or body[0] not in "{[":
        return None
    try:
        document = json.loads(body)
    except ValueError:
        return None
    items = document if isinstance(document, list) else [document]
    calls: list[tuple[str, str]] = []
    for item in items:
        call = _as_call(item, known)
        if call is None:
            return None
        calls.append(call)
    return calls or None


class StreamAssembler:
    """Collects text and Tool_Call fragments from a stream of deltas.

    `prompted_tools` are the names of the tools offered to a model that has no native function
    calling (`llm.native_function_calling: false`). LiteLLM turns such a reply into `tool_calls`
    only for non-streaming requests, so when a stream ends with no Tool_Call fragments at all,
    `finish` reads the text as a prompted call (10.8). Left empty, the text is never parsed.
    """

    def __init__(self, prompted_tools: AbstractSet[str] = frozenset()) -> None:
        self._text: list[str] = []
        self._calls: dict[int, _CallFragments] = {}
        self._prompted_tools = frozenset(prompted_tools)

    def feed(self, delta: Delta) -> str | None:
        """Absorb one delta and return its text, or None when it carries no text.

        Text is cleaned with `replace_lone_surrogates` before it is returned, so the chunk the
        emitter forwards is the same text that later lands in the Message_History. Tool_Call
        fragments are grouped by `index`: the first non-empty `id` wins, the first non-empty
        `function.name` wins, and `function.arguments` pieces are concatenated in arrival order.
        """
        for fragment in _get(delta, "tool_calls") or ():
            self._absorb_fragment(fragment)

        content = _get(delta, "content")
        if not isinstance(content, str) or not content:
            return None
        chunk = replace_lone_surrogates(content)
        self._text.append(chunk)
        return chunk

    def finish(self, existing_ids: AbstractSet[str]) -> AssembledResponse:
        """Build the final response with Tool_Calls in ascending index order.

        A call that never received an ID gets `call_<uuid4 hex>`, regenerated until it collides
        with neither `existing_ids` (IDs already in the Message_History) nor another ID in this
        response. Argument text is passed through unchanged, even when it is not valid JSON,
        except that lone surrogates become U+FFFD.
        """
        text = "".join(self._text)
        if not self._calls:
            prompted = parse_prompted_tool_calls(text, self._prompted_tools)
            if prompted is not None:
                return self._finish_prompted(prompted, existing_ids)

        taken = set(existing_ids)
        taken.update(call.id for call in self._calls.values() if call.id is not None)

        tool_calls: list[ToolCall] = []
        for index in sorted(self._calls):
            fragments = self._calls[index]
            call_id = fragments.id
            if call_id is None:
                call_id = self._new_call_id(taken)
                taken.add(call_id)
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=fragments.name or "",
                    arguments=replace_lone_surrogates("".join(fragments.arguments)),
                )
            )
        return AssembledResponse(text=text, tool_calls=tuple(tool_calls))

    def _finish_prompted(
        self, prompted: list[tuple[str, str]], existing_ids: AbstractSet[str]
    ) -> AssembledResponse:
        """Tool_Calls read from the reply text. The JSON was the whole reply, so the text is
        dropped from the result and the Message_History keeps only the Tool_Calls."""
        taken = set(existing_ids)
        tool_calls: list[ToolCall] = []
        for name, arguments in prompted:
            call_id = self._new_call_id(taken)
            taken.add(call_id)
            tool_calls.append(
                ToolCall(id=call_id, name=name, arguments=replace_lone_surrogates(arguments))
            )
        return AssembledResponse(text="", tool_calls=tuple(tool_calls))

    def _absorb_fragment(self, fragment: Any) -> None:
        raw_index = _get(fragment, "index")
        index = raw_index if isinstance(raw_index, int) else _MISSING_INDEX
        call = self._calls.setdefault(index, _CallFragments())

        if call.id is None:
            call.id = _non_empty_str(_get(fragment, "id"))

        function = _get(fragment, "function")
        if call.name is None:
            call.name = _non_empty_str(_get(function, "name"))

        arguments = _get(function, "arguments")
        if isinstance(arguments, str) and arguments:
            call.arguments.append(arguments)

    @staticmethod
    def _new_call_id(taken: AbstractSet[str]) -> str:
        while True:
            candidate = f"call_{uuid.uuid4().hex}"
            if candidate not in taken:
                return candidate
