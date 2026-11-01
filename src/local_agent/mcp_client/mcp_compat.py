"""The one place that knows where the `mcp` SDK keeps the names this Backend uses.

`mcp` 2.x moved the protocol types into the standalone `mcp_types` package and left `mcp.types`
as a mirror (every name is the same object). It also changed the timeout arguments of
`ClientSession` and `call_tool` from `timedelta` (1.x) to `float | None` seconds. The rest of the
`mcp_client` package imports SDK names from this module only, so a version change touches this
file and nothing else.

The pinned range is `mcp>=2.3,<2.4`. The import fallbacks below keep 1.x spellings working for
the names that exist under both; the 2.x timeout type is the only one supported.
"""

from typing import Any

from mcp import ClientSession
from mcp.client._transport import ReadStream, WriteStream
from mcp.client.sse import sse_client
from mcp.shared.message import SessionMessage
from pydantic import TypeAdapter

try:
    from mcp_types import REQUEST_TIMEOUT, JSONRPCMessage, PaginatedRequestParams
except ImportError:  # pragma: no cover - only reached on an SDK without `mcp_types`
    from mcp.types import REQUEST_TIMEOUT, JSONRPCMessage, PaginatedRequestParams

try:
    from mcp.shared.exceptions import MCPError
except ImportError:  # pragma: no cover - 1.x spelling
    from mcp.shared.exceptions import McpError as MCPError  # type: ignore[attr-defined,no-redef]

__all__ = [
    "ClientSession",
    "JSONRPCMessage",
    "MCPError",
    "ReadStream",
    "ReadTimeout",
    "SessionMessage",
    "WriteStream",
    "is_read_timeout",
    "jsonrpc_message_adapter",
    "page_params",
    "read_attr",
    "sse_client",
    "to_read_timeout",
]

jsonrpc_message_adapter: TypeAdapter[JSONRPCMessage] = TypeAdapter(JSONRPCMessage)
"""Validates one JSON-RPC message from text or bytes; the SDK's stdio transport uses the same."""

ReadTimeout = float | None
"""`ClientSession(read_timeout_seconds=...)` and `call_tool(read_timeout_seconds=...)` argument."""


def to_read_timeout(seconds: float | None) -> ReadTimeout:
    """Convert seconds (`None` for no limit) to the SDK's timeout argument type."""
    return seconds


_HTTP_REQUEST_TIMEOUT = 408
"""The code `mcp` 1.x used for a request read timeout; 2.x uses `REQUEST_TIMEOUT` (-32001)."""


def is_read_timeout(exc: BaseException) -> bool:
    """True when `exc` is the SDK's "request timed out" error (``MCPError`` with a timeout code)."""
    return isinstance(exc, MCPError) and getattr(exc, "code", None) in (
        REQUEST_TIMEOUT,
        _HTTP_REQUEST_TIMEOUT,
    )


def page_params(cursor: str | None) -> PaginatedRequestParams | None:
    """Pagination parameters for ``ClientSession.list_tools``; ``None`` asks for the first page."""
    if cursor is None:
        return None
    return PaginatedRequestParams(cursor=cursor)


def read_attr(obj: Any, snake_name: str) -> Any:
    """Read a field by its 2.x snake_case name, falling back to the 1.x camelCase spelling.

    ``next_cursor``/``nextCursor``, ``is_error``/``isError``, ``input_schema``/``inputSchema`` and
    ``mime_type``/``mimeType`` are the names that differ between the two lines.
    """
    if hasattr(obj, snake_name):
        return getattr(obj, snake_name)
    head, *rest = snake_name.split("_")
    return getattr(obj, head + "".join(part.capitalize() for part in rest), None)
