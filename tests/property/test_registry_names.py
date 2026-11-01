"""Property test for LLM-facing tool names (Requirements 7.3, 7.7, 7.12).

`assign_names` turns ordered `(server, tool)` pairs into names that an OpenAI-compatible API
accepts. The test generates pairs that stress the three promises: arbitrary Unicode, names longer
than 64 characters, and names that collide after sanitization or after a `_k` suffix is added. It
checks the result against a small model of the rule, then checks that `ToolRegistry` publishes the
same names whatever order its servers register in.
"""

import re

from hypothesis import given
from hypothesis import strategies as st

from local_agent.mcp_client.registry import MAX_NAME_LENGTH, ToolRegistry, assign_names, sanitize
from local_agent.models import McpToolInfo

_LLM_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SUFFIXED = re.compile(r"(.*)_([0-9]+)", re.DOTALL)

# Short names over a tiny alphabet collide constantly: "a.b" and "a_b" sanitize alike, and "x_2"
# can equal the suffixed form of a repeated "x".
_COLLIDING = st.lists(
    st.sampled_from(["a", "b", "_", "-", "2", "3", ".", " ", "/", "\u00e9"]), max_size=5
).map("".join)
# Names whose sanitized base is cut at 64 characters, so a suffix must shorten the base.
_LONG = st.integers(min_value=55, max_value=90).flatmap(
    lambda n: st.sampled_from(["a", "a.", "_2"]).map(lambda unit: (unit * n)[:n])
)
_NAMES = st.one_of(_COLLIDING, _LONG, st.text(max_size=80))
_PAIRS = st.tuples(_NAMES, _NAMES)


@st.composite
def _entries(draw: st.DrawFn) -> list[tuple[str, str]]:
    """Ordered `(server, tool)` pairs, with repeats drawn from a small pool to force collisions."""
    pool = draw(st.lists(_PAIRS, min_size=1, max_size=4))
    return draw(st.lists(st.one_of(st.sampled_from(pool), _PAIRS), max_size=12))


def _base(server: str, tool: str) -> str:
    return sanitize(f"{server}__{tool}")[:MAX_NAME_LENGTH]


def _tool(name: str) -> McpToolInfo:
    return McpToolInfo(name=name, description=None, input_schema={"type": "object"})


# Feature: local-agent-backend, Property 5: LLM-facing name format, uniqueness, and determinism
# **Validates: Requirements 7.3, 7.7, 7.12**
@given(entries=_entries(), data=st.data())
def test_llm_facing_names_are_valid_unique_and_deterministic(
    entries: list[tuple[str, str]], data: st.DataObject
) -> None:
    names = assign_names(entries)

    # One name per pair, in input order, and the same input gives the same output (7.12).
    assert len(names) == len(entries)
    assert assign_names(list(entries)) == names

    # Every name is a valid, non-empty identifier of at most 64 characters (7.3).
    for name in names:
        assert _LLM_NAME.fullmatch(name)

    # Names are pairwise distinct (7.7).
    assert len(set(names)) == len(names)

    for index, ((server, tool), name) in enumerate(zip(entries, names, strict=True)):
        base = _base(server, tool)
        if base not in names[:index]:
            # The first pair to claim a base keeps it unchanged (7.7).
            assert name == base
        else:
            # A later pair gets the base shortened to fit a `_k` suffix with k >= 2.
            match = _SUFFIXED.fullmatch(name)
            assert match is not None
            suffix = f"_{match.group(2)}"
            assert int(match.group(2)) >= 2
            assert name == base[: MAX_NAME_LENGTH - len(suffix)] + suffix

    # The registry applies the same rule in Backend_Config order, however servers register (7.12).
    server_order = list(dict.fromkeys(server for server, _ in entries))
    tools_by_server = {
        server: [tool for s, tool in entries if s == server] for server in server_order
    }
    config_ordered = [(s, t) for s in server_order for t in tools_by_server[s]]
    registration_order = data.draw(st.permutations(server_order))

    registry = ToolRegistry(server_order)
    for server in registration_order:
        registry.replace_server(server, [_tool(t) for t in tools_by_server[server]])

    published = [tool.llm_name for tool in registry.snapshot().tools]
    assert published == assign_names(config_ordered)
    assert [(t.server, t.tool) for t in registry.snapshot().tools] == config_ordered
