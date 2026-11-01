"""Property test for System_Prompt structure (Requirements 12.1, 12.2, 12.4, 12.8, 12.9).

`build_system_prompt` takes optional base instructions and the tools of the available servers. The
test generates arbitrary bases (including `None` and `""`) and tool lists with repeated servers,
missing or multi-line descriptions, and then parses the prompt back into its parts. It checks the
parsed structure instead of comparing against a second copy of the builder.
"""

from hypothesis import given
from hypothesis import strategies as st

from local_agent.models import RegisteredTool
from local_agent.orchestrator.prompt_builder import (
    DEFAULT_BASE_INSTRUCTIONS,
    NO_DESCRIPTION_NOTE,
    NO_TOOLS_SENTENCE,
    TOOL_USAGE_HEADING,
    TOOL_USAGE_INSTRUCTIONS,
    build_system_prompt,
)

# A base may span several lines and may even contain tool-looking lines. It must not contain the
# tool-usage heading itself, because then "exactly one section" would be ambiguous by construction.
_BASE = st.one_of(
    st.none(),
    st.just(""),
    st.text(max_size=200).filter(lambda text: TOOL_USAGE_HEADING not in text),
)

# Server names come from Backend_Config, so they are single-line. A small pool makes servers repeat
# and interleave across the tool list.
_SERVER_LINE = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs", "Zl", "Zp")), max_size=12
)
_SERVER_POOL = st.lists(_SERVER_LINE, min_size=1, max_size=4, unique=True)

# Descriptions are free text: absent, empty, blank, or spread over several lines.
_DESCRIPTION = st.one_of(st.none(), st.just(""), st.text(max_size=80))


@st.composite
def _tools(draw: st.DrawFn) -> list[RegisteredTool]:
    """Registered tools with distinct `llm_name`s, drawn from a small pool of servers."""
    servers = draw(_SERVER_POOL)
    picks = draw(
        st.lists(st.tuples(st.sampled_from(servers), _DESCRIPTION), min_size=0, max_size=10)
    )
    return [
        RegisteredTool(
            llm_name=f"tool_{index}",
            server=server,
            tool=f"tool_{index}",
            description=description,
            openai_def={},
            protected=False,
        )
        for index, (server, description) in enumerate(picks)
    ]


def _expected_description(description: str | None) -> str:
    """The one-line text a listing must carry for this description (12.9)."""
    flattened = " ".join((description or "").splitlines()).strip()
    return flattened or NO_DESCRIPTION_NOTE


# Feature: local-agent-backend, Property 22: System prompt structure
# **Validates: Requirements 12.1, 12.2, 12.4, 12.8, 12.9**
@given(base=_BASE, tools=_tools())
def test_system_prompt_structure(base: str | None, tools: list[RegisteredTool]) -> None:
    prompt = build_system_prompt(base, tools)

    # The prompt starts with the base instructions, or the default when they are empty (12.1, 12.8).
    expected_base = base or DEFAULT_BASE_INSTRUCTIONS
    assert prompt.startswith(expected_base)

    # Exactly one tool-usage section follows the base (12.1).
    tail_lines = prompt[len(expected_base) :].split("\n")
    assert tail_lines.count(TOOL_USAGE_HEADING) == 1
    assert prompt.count(TOOL_USAGE_HEADING) == 1
    section = tail_lines[tail_lines.index(TOOL_USAGE_HEADING) + 1 :]
    for instruction in TOOL_USAGE_INSTRUCTIONS:
        assert section.count(instruction) == 1
    body = [line for line in section if line and line not in TOOL_USAGE_INSTRUCTIONS]

    if not tools:
        # An empty snapshot says so, and lists no server heading and no tool (12.4).
        assert body == [NO_TOOLS_SENTENCE]
        return

    assert NO_TOOLS_SENTENCE not in body

    # Parse the listing into (server heading, [tool lines]) groups.
    groups: list[tuple[str, list[str]]] = []
    for line in body:
        if line.startswith("### "):
            groups.append((line.removeprefix("### "), []))
        else:
            assert groups, f"tool line before any server heading: {line!r}"
            groups[-1][1].append(line)

    # Each server has one heading, in order of first appearance (12.2).
    expected_servers = list(dict.fromkeys(tool.server for tool in tools))
    assert [server for server, _ in groups] == expected_servers

    # Under its own server's heading, every tool appears once with its description or the note
    # (12.2, 12.9), in registry order.
    for server, lines in groups:
        expected_lines = [
            f"- {tool.llm_name}: {_expected_description(tool.description)}"
            for tool in tools
            if tool.server == server
        ]
        assert lines == expected_lines

    # Every LLM-facing name is listed exactly once across the whole prompt section.
    listed = [line.split(":", 1)[0].removeprefix("- ") for _, lines in groups for line in lines]
    assert sorted(listed) == sorted(tool.llm_name for tool in tools)
