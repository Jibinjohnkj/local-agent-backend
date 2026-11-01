"""Property test for paginated tool discovery (Requirement 7.1).

`McpSessionAdapter.list_all_tools` runs against a fake `ToolSession` that serves a scripted list of
pages. Each page carries some tools and, except for the last page, an opaque cursor that the
adapter must send back to get the next page. The fake records the cursor of every request, so the
test checks three things from the script alone:

- the first request carries no cursor, and each later request carries the cursor of the page
  before it, unchanged (cursors are any text, including the empty string);
- the adapter makes exactly one request per page, so it stops when no cursor comes back;
- the returned tools are the concatenation of all pages in order.

Half of the examples spell the page's cursor field as the 2.x ``next_cursor`` and half as the 1.x
``nextCursor``, since the adapter accepts both.
"""

from dataclasses import dataclass
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.mcp_client.manager import McpSessionAdapter
from local_agent.models import McpToolInfo

_TEXT = st.text(max_size=12)


@dataclass(frozen=True)
class _SdkTool:
    """A tool as the SDK returns it (2.x field spelling)."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class _SnakePage:
    tools: list[_SdkTool]
    next_cursor: str | None


@dataclass(frozen=True)
class _CamelPage:
    tools: list[_SdkTool]
    nextCursor: str | None  # the 1.x field spelling


class _FakeSession:
    """Serves `pages` in order and records the cursor of each `tools/list` request."""

    def __init__(self, pages: list[_SnakePage | _CamelPage]) -> None:
        self._pages = pages
        self.cursors: list[str | None] = []

    async def list_tools(self, *, params: Any = None) -> Any:
        self.cursors.append(None if params is None else params.cursor)
        # A request past the last page means the adapter ignored the missing cursor.
        assert len(self.cursors) <= len(self._pages), "requested a page past the end"
        return self._pages[len(self.cursors) - 1]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: Any = None,
    ) -> Any:
        raise AssertionError("list_all_tools must not call a tool")


_TOOL = st.builds(
    _SdkTool,
    name=_TEXT,
    description=st.none() | _TEXT,
    input_schema=st.dictionaries(_TEXT, st.integers() | _TEXT, max_size=3),
)


@st.composite
def _scenario(draw: st.DrawFn) -> tuple[list[list[_SdkTool]], list[str], bool]:
    """Draw the tools of each page, the cursors between pages, and the field spelling."""
    pages = draw(st.lists(st.lists(_TOOL, max_size=4), min_size=1, max_size=6))
    cursors = draw(st.lists(_TEXT, min_size=len(pages) - 1, max_size=len(pages) - 1))
    return pages, cursors, draw(st.booleans())


# Feature: local-agent-backend, Property 32: Paginated tool discovery
# **Validates: Requirements 7.1**
@settings(max_examples=100)
@given(_scenario())
async def test_paginated_tool_discovery(
    scenario: tuple[list[list[_SdkTool]], list[str], bool],
) -> None:
    tool_pages, cursors, camel_case = scenario
    next_cursors: list[str | None] = [*cursors, None]
    pages: list[_SnakePage | _CamelPage] = [
        _CamelPage(tools=tools, nextCursor=cursor)
        if camel_case
        else _SnakePage(tools=tools, next_cursor=cursor)
        for tools, cursor in zip(tool_pages, next_cursors, strict=True)
    ]
    session = _FakeSession(pages)

    result = await McpSessionAdapter(session).list_all_tools(timeout=5.0)

    expected = [
        McpToolInfo(name=t.name, description=t.description, input_schema=t.input_schema)
        for tools in tool_pages
        for t in tools
    ]
    assert result == expected
    assert session.cursors == [None, *cursors]
