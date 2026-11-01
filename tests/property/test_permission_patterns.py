"""Property test for Permission_Policy pattern matching (Requirements 5.1, 5.2).

A pattern is plain text where `*` means "any sequence of characters" and `?` means "any one
character". Everything else, including `[`, `]`, `.`, `(` and the other regex metacharacters,
stands for itself. `compile_pattern` turns such a pattern into a regex, so the test checks that
regex against a tiny recursive matcher that knows nothing about regexes. It then registers tools
in a `ToolRegistry` and checks that a tool is protected exactly when one of the patterns listed
for its own server matches its original name.
"""

from functools import cache

from hypothesis import given
from hypothesis import strategies as st

from local_agent.mcp_client.registry import ToolRegistry, compile_pattern, is_protected
from local_agent.models import McpToolInfo

# A small alphabet with regex metacharacters, brackets and a newline, so that literal handling is
# exercised and generated names and patterns overlap often enough to produce real matches.
_LITERALS = ["a", "b", "A", ".", "[", "]", "^", "$", "\\", "(", "+", "|", "-", "\n", "\u00e9"]
_NAME_CHARS = st.sampled_from([*_LITERALS, "*", "?"])
_PATTERN_CHARS = st.sampled_from([*_LITERALS, "*", "?"])

_NAMES = st.text(_NAME_CHARS, max_size=8)
_RANDOM_PATTERNS = st.text(_PATTERN_CHARS, max_size=8)
_SERVERS = st.sampled_from(["fs", "git", "web", "db"])


def _reference_match(pattern: str, name: str) -> bool:
    """Whole-string wildcard match: `*` is any sequence, `?` is one character, case-sensitive."""

    @cache
    def go(i: int, j: int) -> bool:
        if i == len(pattern):
            return j == len(name)
        head = pattern[i]
        if head == "*":
            return go(i + 1, j) or (j < len(name) and go(i, j + 1))
        if j == len(name):
            return False
        if head == "?" or head == name[j]:
            return go(i + 1, j + 1)
        return False

    return go(0, 0)


@st.composite
def _patterns_for(draw: st.DrawFn, name: str) -> str:
    """A pattern that matches `name` by construction: keep, generalize or widen each character."""
    parts: list[str] = []
    for ch in name:
        action = draw(st.sampled_from(["keep", "keep", "any_one", "any_run"]))
        if action == "keep":
            parts.append(ch)
        elif action == "any_one":
            parts.append("?")
        else:
            parts.append("*")
    return "".join(parts)


@st.composite
def _pattern_and_name(draw: st.DrawFn) -> tuple[str, str]:
    """A pattern with a name, drawn from unrelated, derived (likely matching) or edited cases."""
    name = draw(_NAMES)
    kind = draw(st.sampled_from(["random", "derived", "edited"]))
    if kind == "random":
        return draw(_RANDOM_PATTERNS), name
    derived = draw(_patterns_for(name))
    if kind == "derived":
        return derived, name
    # Edited: append a character to the pattern so many near-misses appear.
    return derived + draw(_PATTERN_CHARS), name


@st.composite
def _scenario(
    draw: st.DrawFn,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """A Permission_Policy and the tools each server exposes.

    Tool names are unique within a server. Patterns come partly from the server's own tools and
    partly from tools of other servers, so a pattern that only fits another server is covered.
    """
    tools = draw(
        st.dictionaries(
            _SERVERS,
            st.lists(st.text(_NAME_CHARS, min_size=1, max_size=8), unique=True, max_size=5),
            max_size=4,
        )
    )
    every_name = [name for names in tools.values() for name in names]
    sources = st.sampled_from(every_name) if every_name else _NAMES
    pattern = st.one_of(
        _RANDOM_PATTERNS,
        sources.flatmap(_patterns_for),
    )
    # The policy may name servers that have no tools, and omit servers that do.
    policy = draw(st.dictionaries(_SERVERS, st.lists(pattern, max_size=3), max_size=4))
    return policy, tools


def _tool(name: str) -> McpToolInfo:
    return McpToolInfo(name=name, description=None, input_schema={"type": "object"})


# Feature: local-agent-backend, Property 9: Permission pattern matching
# **Validates: Requirements 5.1, 5.2**
@given(case=_pattern_and_name(), scenario=_scenario())
def test_permission_pattern_matching(
    case: tuple[str, str],
    scenario: tuple[dict[str, list[str]], dict[str, list[str]]],
) -> None:
    # 5.1: the compiled pattern agrees with the reference matcher on the whole name.
    pattern, name = case
    assert (compile_pattern(pattern).fullmatch(name) is not None) == _reference_match(pattern, name)

    # 5.2: a tool is protected iff a pattern listed for its own server matches its original name.
    policy, tools = scenario
    registry = ToolRegistry(server_order=list(tools), policy=policy)
    for server, names in tools.items():
        registry.replace_server(server, [_tool(n) for n in names])

    registered = {(t.server, t.tool): t.protected for t in registry.snapshot().tools}
    assert len(registered) == sum(len(names) for names in tools.values())
    for server, names in tools.items():
        for tool_name in names:
            expected = any(_reference_match(p, tool_name) for p in policy.get(server, []))
            assert registered[(server, tool_name)] is expected
            assert is_protected(policy, server, tool_name) is expected
