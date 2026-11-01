"""Turn MCP tool results and tool failures into the text the LLM sees.

All functions are pure. `content_to_text` joins the text of an MCP response and describes the
non-text items (8.6, 8.11). `truncate` enforces the maximum tool output size (8.9). Error-flagged
results use the same conversion and truncation (9.1). `timeout_text` and `exception_text` build
the Tool_Result text for a timed-out call (9.2) and for a transport or protocol exception (9.7).
"""

from collections.abc import Sequence

from local_agent.logging_setup import Redactor
from local_agent.models import ContentItem, TextItem, replace_lone_surrogates

__all__ = [
    "NO_CONTENT_TEXT",
    "content_to_text",
    "exception_text",
    "timeout_text",
    "truncate",
]

NO_CONTENT_TEXT = "The tool returned no content."
"""Tool_Result text for a successful response without content items (8.11)."""


def _describe(item: ContentItem) -> str:
    """Describe a non-text item: its type, then the MIME type and URI when present (8.6)."""
    description = f"[{item.type} content"
    mime_type = getattr(item, "mime_type", None)
    if mime_type:
        description += f", mime type {mime_type}"
    uri = getattr(item, "uri", None)
    if uri:
        description += f", uri {uri}"
    return description + "]"


def content_to_text(items: Sequence[ContentItem]) -> str:
    """Convert MCP content items to one text, in received order (8.6, 8.11).

    Text items keep their text and every other item becomes a description. The parts are joined
    with a single newline. An empty list gives `NO_CONTENT_TEXT`. Lone surrogates are replaced
    with U+FFFD so the text can be stored and sent as UTF-8 (18.5).
    """
    if not items:
        return NO_CONTENT_TEXT
    parts = [item.text if isinstance(item, TextItem) else _describe(item) for item in items]
    return replace_lone_surrogates("\n".join(parts))


def truncate(text: str, max_chars: int) -> str:
    """Keep the first `max_chars` characters and append a notice when `text` is longer (8.9)."""
    if len(text) <= max_chars:
        return text
    return (
        f"{text[:max_chars]}\n"
        f"[Output truncated: original length {len(text)} characters, limit {max_chars} characters]"
    )


def _format_seconds(seconds: float) -> str:
    """Show whole-number durations without a fraction (`60`, not `60.0`)."""
    if float(seconds).is_integer():
        return str(int(seconds))
    return str(seconds)


def timeout_text(llm_name: str, seconds: float) -> str:
    """Tool_Result text for a call that did not finish within the Tool_Timeout (9.2)."""
    return f"Tool '{llm_name}' timed out after {_format_seconds(seconds)} seconds."


def exception_text(exc: BaseException, redactor: Redactor) -> str:
    """Return `"<TypeName>: <message>"` with configured secrets redacted (9.7, 16.11).

    No traceback is included. Lone surrogates are replaced with U+FFFD (18.5).
    """
    text = f"{type(exc).__name__}: {exc}"
    return replace_lone_surrogates(redactor.redact(text))
