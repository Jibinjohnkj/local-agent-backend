"""Shared runtime data models, the injectable clock, and the lone-surrogate helper.

Everything here is plain data or a pure function. No module in this file does I/O, so the
orchestrator, the session stores, the MCP client, and the A2A server can all import it without
creating import cycles.
"""

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar, Literal, Protocol

from a2a.types import TaskState

# --- LLM output -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One Tool_Call emitted by the LLM."""

    id: str
    name: str
    arguments: str  # raw JSON text, unchanged (10.11)


@dataclass(frozen=True)
class AssembledResponse:
    """A finished assistant response: the full text plus its Tool_Calls in index order."""

    text: str
    tool_calls: tuple[ToolCall, ...]


# --- Chat messages kept in Message_History ---------------------------------------------------


@dataclass(frozen=True)
class SystemMessage:
    content: str

    def to_openai(self) -> dict[str, Any]:
        return {"role": "system", "content": self.content}


@dataclass(frozen=True)
class UserMessage:
    content: str

    def to_openai(self) -> dict[str, Any]:
        return {"role": "user", "content": self.content}


@dataclass(frozen=True)
class AssistantMessage:
    content: str | None  # None (no text) and "" (empty text) stay distinct (18.3)
    tool_calls: tuple[ToolCall, ...] = ()

    def to_openai(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return wire


@dataclass(frozen=True)
class ToolMessage:
    tool_call_id: str
    content: str

    def to_openai(self) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}


ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


# --- Session and storage records --------------------------------------------------------------


@dataclass
class ContextRecord:
    """A Context as the Session_Store holds it. Mutable: `history` grows as messages arrive."""

    context_id: str
    system_prompt: SystemMessage | None  # single slot (12.7)
    history: list[ChatMessage]  # user/assistant/tool only
    created_at_ms: int  # integer epoch milliseconds, wall clock (18.3)
    last_activity_ms: int  # integer epoch milliseconds, wall clock (11.11)


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str  # "response-<n>", unique within the Task (3.3)
    ordinal: int  # emission order within the Task (18.4, 18.7)
    text: str  # full reassembled text (3.4); prefix of emitted text after a crash (18.12)
    closed: bool  # last-chunk event emitted


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    context_id: str
    state: TaskState
    status_message: str | None  # latest, redacted (16.11)
    metadata: Mapping[str, Any]
    created_at_ms: int  # integer epoch milliseconds
    updated_at_ms: int  # integer epoch milliseconds
    artifacts: tuple[ArtifactRecord, ...]  # ordered by ordinal

    def __post_init__(self) -> None:
        ordinals = [artifact.ordinal for artifact in self.artifacts]
        if any(later <= earlier for earlier, later in pairwise(ordinals)):
            raise ValueError(
                f"TaskRecord {self.task_id!r}: artifacts must be ordered by strictly "
                f"increasing ordinal, got {ordinals}"
            )


# --- MCP tools and results --------------------------------------------------------------------


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class RegisteredTool:
    llm_name: str  # sanitized, unique, at most 64 characters (7.3, 7.7)
    server: str
    tool: str  # original MCP_Tool name
    description: str | None
    openai_def: dict[str, Any]  # {"type": "function", "function": {...}} sent to the LLM
    protected: bool


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    llm_name: str
    server: str | None  # None when the LLM-facing name did not resolve
    tool: str | None
    text: str
    outcome: Literal["success", "error", "timeout", "denied"]

    def as_message(self) -> ToolMessage:
        """The Message_History entry for this result (8.4)."""
        return ToolMessage(tool_call_id=self.call_id, content=self.text)


@dataclass(frozen=True)
class ServerStatus:
    """In-memory view of one configured MCP_Server, as reported by the health endpoint (15.8)."""

    name: str
    enabled: bool
    available: bool


# --- MCP content items ------------------------------------------------------------------------
# A small, SDK-independent description of the content an MCP tool call returns. The MCP session
# adapter converts the SDK's content blocks into these, so `content_to_text` stays pure.


@dataclass(frozen=True)
class TextItem:
    type: ClassVar[str] = "text"
    text: str


@dataclass(frozen=True)
class ImageItem:
    type: ClassVar[str] = "image"
    mime_type: str | None = None


@dataclass(frozen=True)
class AudioItem:
    type: ClassVar[str] = "audio"
    mime_type: str | None = None


@dataclass(frozen=True)
class EmbeddedResourceItem:
    type: ClassVar[str] = "resource"
    uri: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True)
class ResourceLinkItem:
    type: ClassVar[str] = "resource_link"
    uri: str | None = None
    mime_type: str | None = None


ContentItem = TextItem | ImageItem | AudioItem | EmbeddedResourceItem | ResourceLinkItem


# --- Time -------------------------------------------------------------------------------------


class Clock(Protocol):
    """Injected time source. Tests substitute a fake that advances without waiting."""

    def now_ms(self) -> int:
        """Wall-clock integer epoch milliseconds, for stored timestamps (Last_Activity)."""
        ...

    def monotonic(self) -> float:
        """Seconds on a clock that never goes backwards, for timeouts and backoff."""
        ...


@dataclass(frozen=True)
class SystemClock:
    """The real clock."""

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def monotonic(self) -> float:
        return time.monotonic()


# --- Text hygiene -----------------------------------------------------------------------------

_SURROGATE = re.compile("[\ud800-\udfff]")
REPLACEMENT_CHARACTER = "\ufffd"


def replace_lone_surrogates(text: str) -> str:
    """Replace every surrogate code point in `text` with U+FFFD.

    A Python `str` holds code points, so a surrogate can never be half of a valid pair: valid
    non-BMP characters are single code points. Any U+D800..U+DFFF is therefore unpaired and
    cannot be encoded as UTF-8. Text that is already valid is returned unchanged.
    """
    if not _SURROGATE.search(text):
        return text
    return _SURROGATE.sub(REPLACEMENT_CHARACTER, text)


__all__ = [
    "REPLACEMENT_CHARACTER",
    "ArtifactRecord",
    "AssembledResponse",
    "AssistantMessage",
    "AudioItem",
    "ChatMessage",
    "Clock",
    "ContentItem",
    "ContextRecord",
    "EmbeddedResourceItem",
    "ImageItem",
    "McpToolInfo",
    "RegisteredTool",
    "ResourceLinkItem",
    "ServerStatus",
    "SystemClock",
    "SystemMessage",
    "TaskRecord",
    "TextItem",
    "ToolCall",
    "ToolMessage",
    "ToolResult",
    "UserMessage",
    "replace_lone_surrogates",
]
