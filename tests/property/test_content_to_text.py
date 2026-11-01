"""Property test for MCP content conversion (Requirements 8.6, 8.11).

`content_to_text` joins, in order and with a newline, each text item's text and each non-text
item's description. A description names the item type and carries the MIME type and URI when the
item has them. An empty list gives the "no content" text.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.mcp_client.result_formatter import NO_CONTENT_TEXT, content_to_text
from local_agent.models import (
    AudioItem,
    ContentItem,
    EmbeddedResourceItem,
    ImageItem,
    ResourceLinkItem,
    TextItem,
)

# `st.text()` never draws surrogates, so the lone-surrogate replacement does not apply here.
_TEXT = st.text(max_size=40)
_OPTIONAL = st.none() | st.text(min_size=1, max_size=20)

_NON_TEXT_ITEMS = st.one_of(
    st.builds(ImageItem, mime_type=_OPTIONAL),
    st.builds(AudioItem, mime_type=_OPTIONAL),
    st.builds(EmbeddedResourceItem, uri=_OPTIONAL, mime_type=_OPTIONAL),
    st.builds(ResourceLinkItem, uri=_OPTIONAL, mime_type=_OPTIONAL),
)
_ITEMS = st.lists(st.builds(TextItem, text=_TEXT) | _NON_TEXT_ITEMS, max_size=8)


# Feature: local-agent-backend, Property 16: MCP content conversion
# **Validates: Requirements 8.6, 8.11**
@settings(max_examples=200)
@given(items=_ITEMS)
def test_mcp_content_conversion(items: list[ContentItem]) -> None:
    result = content_to_text(items)

    if not items:
        assert result == NO_CONTENT_TEXT
        return

    expected_parts: list[str] = []
    for item in items:
        if isinstance(item, TextItem):
            expected_parts.append(item.text)
            continue
        description = content_to_text([item])
        assert item.type in description
        mime_type = getattr(item, "mime_type", None)
        if mime_type is not None:
            assert mime_type in description
        uri = getattr(item, "uri", None)
        if uri is not None:
            assert uri in description
        expected_parts.append(description)

    assert result == "\n".join(expected_parts)
