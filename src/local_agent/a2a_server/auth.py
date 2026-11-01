"""Loopback guard and bearer-token authentication for the A2A_Server (Requirement 16).

Both classes are plain ASGI middleware, not `BaseHTTPMiddleware`, so they never buffer a streaming
(SSE) response and they run before routing. A rejected request therefore never reaches the
`DefaultRequestHandler`, which means no Task is created, changed, or cancelled and no
Message_History is appended to (16.3).

`app.py` stacks them as `LoopbackGuardMiddleware` -> `CORSMiddleware` -> `BearerAuthMiddleware`.
"""

import hmac
import ipaddress
from collections.abc import Collection
from typing import Final

from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from local_agent.logging_setup import get_logger

__all__ = ["BearerAuthMiddleware", "LoopbackGuardMiddleware", "is_loopback"]

_log = get_logger("a2a_server.auth")

_BEARER_PREFIX: Final = b"Bearer "
_AUTHORIZATION: Final = b"authorization"
_WS_POLICY_VIOLATION: Final = 1008


def is_loopback(host: str) -> bool:
    """Return True when `host` names the loopback interface (16.2).

    Loopback means the name `localhost` (any letter case), any address in `127.0.0.0/8`, or the
    IPv6 address `::1`. Everything else is not loopback, including the wildcard binds `0.0.0.0`
    and `::`, other host names, the empty string, and text that is not an IP address. An IPv6
    literal may carry the URL square brackets.
    """
    candidate = host
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if candidate.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.ip_network("127.0.0.0/8")
    return address == ipaddress.IPv6Address("::1")


def _is_loopback_peer(host: str) -> bool:
    """Classify a peer address taken from the ASGI `client` tuple.

    A dual-stack socket reports an IPv4 peer as an IPv4-mapped IPv6 address such as
    `::ffff:127.0.0.1`. We unwrap it first, so a local IPv4 client still passes when the server
    listens on an IPv6 socket.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return is_loopback(str(address.ipv4_mapped))
    return is_loopback(host)


def _peer_host(scope: Scope) -> str | None:
    """Return the client host from `scope["client"]`, or None when the server gives none."""
    client = scope.get("client")
    if not client:
        return None
    host = client[0]
    return host if isinstance(host, str) else None


async def _reject(scope: Scope, receive: Receive, send: Send, response: Response) -> None:
    """Answer a rejected connection: an HTTP response, or a policy-violation close for WebSocket."""
    if scope["type"] == "websocket":
        await WebSocketClose(code=_WS_POLICY_VIOLATION)(scope, receive, send)
    else:
        await response(scope, receive, send)


class LoopbackGuardMiddleware:
    """Refuse connections whose peer is not on the loopback interface (16.1).

    The bind address already keeps remote hosts out. This check is defense in depth: if the server
    is ever reached through a non-loopback interface, the request is answered with HTTP 403. When
    the server gives no client address (for example a Unix socket), we fail closed and refuse.
    With `enabled=False`, the middleware passes everything through.
    """

    def __init__(self, app: ASGIApp, *, enabled: bool = True) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.enabled or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        peer = _peer_host(scope)
        if peer is not None and _is_loopback_peer(peer):
            await self.app(scope, receive, send)
            return
        _log.warning("loopback_guard_rejected", client=peer, path=scope.get("path"))
        await _reject(
            scope, receive, send, Response("Forbidden", status_code=403, media_type="text/plain")
        )


class BearerAuthMiddleware:
    """Require `Authorization: Bearer <token>` on every non-exempt request (16.2, 16.3).

    The check compares raw header bytes against the UTF-8 bytes of the configured token with
    `hmac.compare_digest`, so the match is exact, case-sensitive, full-string, and takes the same
    time whatever prefix is correct. A missing header, a second `Authorization` header, a scheme
    other than `Bearer`, or a wrong token all answer HTTP 401 before routing.

    `OPTIONS` requests pass through to the CORS layer and the routes (which answer 405), and carry
    no side effects. With `enabled=False` (a loopback bind, 16.7) every request passes.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str | None,
        enabled: bool = True,
        exempt_paths: Collection[str] = (),
    ) -> None:
        if enabled and not token:
            raise ValueError("BearerAuthMiddleware needs a non-empty token when enabled")
        self.app = app
        self.enabled = enabled
        self._token = (token or "").encode("utf-8")
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.enabled or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http" and scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        _log.warning("bearer_auth_rejected", client=_peer_host(scope), path=scope.get("path"))
        await _reject(
            scope,
            receive,
            send,
            Response(
                "Unauthorized",
                status_code=401,
                media_type="text/plain",
                headers={"WWW-Authenticate": "Bearer"},
            ),
        )

    def _authorized(self, scope: Scope) -> bool:
        """Return True when the request carries exactly one header with the exact token."""
        values = [
            value for name, value in scope.get("headers", ()) if name.lower() == _AUTHORIZATION
        ]
        if len(values) != 1 or not values[0].startswith(_BEARER_PREFIX):
            return False
        return hmac.compare_digest(values[0][len(_BEARER_PREFIX) :], self._token)
