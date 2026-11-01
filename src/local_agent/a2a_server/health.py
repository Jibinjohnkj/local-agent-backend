"""The `/health` route (Requirement 15.8, 15.9).

The payload is built only from in-memory MCP_Server status, so the route answers at once and never
contacts the LLM or any MCP_Server. The route sits behind the bearer-token middleware like every
other path except the Agent_Card (16.2); this module does no authentication itself.
"""

from collections.abc import Sequence
from typing import Any, Final, Protocol

from starlette.requests import Request
from starlette.responses import JSONResponse

from local_agent.models import ServerStatus

__all__ = ["HealthStatusSource", "health_endpoint", "health_payload"]

_AVAILABLE: Final = "available"
_UNAVAILABLE: Final = "unavailable"


class HealthStatusSource(Protocol):
    """What the endpoint needs from the `McpManager`: a synchronous, in-memory status snapshot."""

    def statuses(self) -> Sequence[ServerStatus]: ...


def health_payload(servers: Sequence[ServerStatus]) -> dict[str, Any]:
    """Build the health response body from the configured servers (pure).

    Every configured server is listed once, in the given order, as `available` or `unavailable`.
    A disabled server is always listed as `unavailable` but does not affect the Backend status.
    The Backend status is `ok` iff every enabled server is available, otherwise `degraded` (15.9).
    """
    listed = [
        {
            "name": server.name,
            "status": _AVAILABLE if server.enabled and server.available else _UNAVAILABLE,
        }
        for server in servers
    ]
    degraded = any(server.enabled and not server.available for server in servers)
    return {"status": "degraded" if degraded else "ok", "mcp_servers": listed}


async def health_endpoint(request: Request) -> JSONResponse:
    """Answer `GET /health` from the in-memory `McpManager` state (15.8).

    `build_app` stores the manager on `app.state.mcp_manager`. The handler is `async` and the
    snapshot call is synchronous and does no I/O, so the event loop is never blocked.
    """
    source: HealthStatusSource = request.app.state.mcp_manager
    return JSONResponse(health_payload(source.statuses()))
