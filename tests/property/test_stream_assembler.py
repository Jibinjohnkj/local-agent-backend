"""Property test for Tool_Call fragment assembly (Requirements 10.3, 10.11)."""

from itertools import pairwise
from types import SimpleNamespace
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.llm.stream_assembler import StreamAssembler

# Arguments are arbitrary text on purpose: valid JSON, truncated JSON and plain prose must all
# pass through byte for byte.
_argument_text = st.one_of(
    st.just(""),
    st.just('{"path": "a.txt", "n": 1}'),
    st.just('{"unterminated": '),
    st.text(max_size=30),
)
_names = st.text(min_size=1, max_size=8)
_given_ids = st.one_of(st.none(), st.text(min_size=1, max_size=8))


@st.composite
def _scenario(draw: st.DrawFn) -> dict[str, Any]:
    """Complete Tool_Calls, the stream indices they use, and an interleaved delta order."""
    count = draw(st.integers(min_value=0, max_value=5))
    indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=10),
            min_size=count,
            max_size=count,
            unique=True,
        )
    )
    # Original IDs must be distinct from each other, so duplicates are dropped to None.
    seen: set[str] = set()
    calls: list[dict[str, Any]] = []
    for index in indices:
        call_id = draw(_given_ids)
        if call_id is not None:
            if call_id in seen:
                call_id = None
            else:
                seen.add(call_id)
        calls.append(
            {
                "index": index,
                "id": call_id,
                "name": draw(_names),
                "arguments": draw(_argument_text),
            }
        )

    # Per index: the first fragment carries the ID and name, the rest carry argument pieces.
    queues: dict[int, list[dict[str, Any]]] = {}
    for call in calls:
        text: str = call["arguments"]
        cuts = sorted(draw(st.lists(st.integers(min_value=0, max_value=len(text)), max_size=4)))
        bounds = [0, *cuts, len(text)]
        pieces = [text[a:b] for a, b in pairwise(bounds)]
        first: dict[str, Any] = {
            "index": call["index"],
            "function": {"name": call["name"], "arguments": pieces[0]},
        }
        if call["id"] is not None:
            first["id"] = call["id"]
        rest = [{"index": call["index"], "function": {"arguments": piece}} for piece in pieces[1:]]
        queues[call["index"]] = [first, *rest]

    # Shuffling one label per fragment interleaves indices while each queue stays in order.
    labels = [index for index, queue in queues.items() for _ in queue]
    order = draw(st.permutations(labels))

    as_objects = draw(st.booleans())
    return {
        "calls": calls,
        "queues": queues,
        "order": order,
        "as_objects": as_objects,
        "existing_ids": draw(st.sets(st.text(min_size=1, max_size=8), max_size=5)),
    }


def _as_object(fragment: dict[str, Any]) -> SimpleNamespace:
    """Rebuild a fragment mapping as attribute-style objects, like a LiteLLM delta."""
    function = SimpleNamespace(
        name=fragment["function"].get("name"),
        arguments=fragment["function"].get("arguments"),
    )
    return SimpleNamespace(index=fragment["index"], id=fragment.get("id"), function=function)


# Feature: local-agent-backend, Property 18: Tool-call fragment assembly
@settings(max_examples=100)
@given(_scenario())
def test_tool_call_fragment_assembly(scenario: dict[str, Any]) -> None:
    """**Validates: Requirements 10.3, 10.11**"""
    calls: list[dict[str, Any]] = scenario["calls"]
    queues: dict[int, list[dict[str, Any]]] = {
        index: list(queue) for index, queue in scenario["queues"].items()
    }
    existing_ids: set[str] = scenario["existing_ids"]

    assembler = StreamAssembler()
    for index in scenario["order"]:
        fragment = queues[index].pop(0)
        if scenario["as_objects"]:
            delta: Any = SimpleNamespace(content=None, tool_calls=[_as_object(fragment)])
        else:
            delta = {"content": None, "tool_calls": [fragment]}
        assert assembler.feed(delta) is None

    response = assembler.finish(existing_ids)

    expected = sorted(calls, key=lambda call: call["index"])
    assert len(response.tool_calls) == len(expected)

    for actual, original in zip(response.tool_calls, expected, strict=True):
        assert actual.name == original["name"]
        assert actual.arguments == original["arguments"]
        if original["id"] is not None:
            assert actual.id == original["id"]
        else:
            assert actual.id
            assert actual.id not in existing_ids

    response_ids = [call.id for call in response.tool_calls]
    assert len(set(response_ids)) == len(response_ids)
    generated = [
        actual.id
        for actual, original in zip(response.tool_calls, expected, strict=True)
        if original["id"] is None
    ]
    given = {original["id"] for original in expected if original["id"] is not None}
    assert not set(generated) & given
