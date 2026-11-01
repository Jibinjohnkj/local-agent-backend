"""Property test for tool output truncation (Requirements 8.9, 9.1).

`truncate` returns the text unchanged when it fits the limit. Otherwise it keeps the first `max`
characters and appends a notice that states the original length and the limit.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.mcp_client.result_formatter import truncate

_MIN_LIMIT = 1_000
_MAX_LIMIT = 1_000_000


@st.composite
def _text_and_limit(draw: st.DrawFn) -> tuple[str, int]:
    """Draw a limit in the allowed range and a text whose length is short, near, or above it."""
    limit = draw(st.integers(min_value=_MIN_LIMIT, max_value=_MAX_LIMIT))
    length = draw(
        st.one_of(
            st.integers(min_value=0, max_value=limit),
            st.integers(min_value=max(0, limit - 3), max_value=limit + 3),
            st.integers(min_value=limit + 1, max_value=limit + 5_000),
        )
    )
    chunk = draw(st.text(min_size=1, max_size=40))
    text = (chunk * (length // len(chunk) + 1))[:length]
    return text, limit


# Feature: local-agent-backend, Property 17: Tool output truncation
# **Validates: Requirements 8.9, 9.1**
@settings(max_examples=200)
@given(case=_text_and_limit())
def test_tool_output_truncation(case: tuple[str, int]) -> None:
    text, limit = case

    result = truncate(text, limit)

    if len(text) <= limit:
        assert result == text
    else:
        assert result.startswith(text[:limit])
        notice = result[limit:]
        assert notice != ""
        assert str(len(text)) in notice
        assert str(limit) in notice
