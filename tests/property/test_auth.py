"""Property test for loopback classification and the bearer check (Requirements 16.1-16.3)."""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass, field
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from local_agent.a2a_server.auth import BearerAuthMiddleware, LoopbackGuardMiddleware, is_loopback

Kind = Literal["name", "ip", "text"]
_HostCase = tuple[str, bool, Kind]

_V6_LOOPBACK_FORMS = ["::1", "0:0:0:0:0:0:0:1", "0000:0000:0000:0000:0000:0000:0000:0001", "::0001"]
_V6_LOOPBACK = ipaddress.IPv6Address("::1")


@st.composite
def _localhost_names(draw: st.DrawFn) -> str:
    flags = draw(st.lists(st.booleans(), min_size=9, max_size=9))
    return "".join(c.upper() if up else c for c, up in zip("localhost", flags, strict=True))


def _is_plain_text(text: str) -> bool:
    """Free text that is neither `localhost` nor an IP address, with or without URL brackets."""
    candidate = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    if text.lower() == "localhost" or candidate.lower() == "localhost":
        return False
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return True
    return False


def _is_other_v4(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return isinstance(address, ipaddress.IPv4Address) and int(address) >> 24 != 127


def _is_other_v6(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # IPv4-mapped IPv6 forms are left out: the guard unwraps them on purpose (see auth.py).
    return (
        isinstance(address, ipaddress.IPv6Address)
        and address != _V6_LOOPBACK
        and address.ipv4_mapped is None
    )


def _loopback_v4(octets: tuple[int, int, int]) -> _HostCase:
    return ("127." + ".".join(map(str, octets)), True, "ip")


_TEXT_HOSTS = ["0.0.0.0", "::", "", "localhost.", "localhost.localdomain", "127.0.0.1.x"]

_loopback_hosts: st.SearchStrategy[_HostCase] = st.one_of(
    _localhost_names().map(lambda h: (h, True, "name")),
    st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)).map(_loopback_v4),
    st.sampled_from(_V6_LOOPBACK_FORMS).map(lambda h: (h, True, "ip")),
)

_non_loopback_hosts: st.SearchStrategy[_HostCase] = st.one_of(
    st.ip_addresses(v=4).filter(_is_other_v4).map(lambda a: (str(a), False, "ip")),
    st.ip_addresses(v=6).filter(_is_other_v6).map(lambda a: (str(a), False, "ip")),
    st.sampled_from(_TEXT_HOSTS).map(lambda h: (h, False, "text")),
    st.text(max_size=20).filter(_is_plain_text).map(lambda h: (h, False, "text")),
)

_hosts = st.one_of(_loopback_hosts, _non_loopback_hosts)

_tokens = st.text(min_size=1, max_size=40)


@st.composite
def _headers(draw: st.DrawFn, token: str) -> str:
    """A presented header value: the exact one half the time, otherwise a near miss or noise."""
    exact = "Bearer " + token
    value: str = draw(
        st.one_of(
            st.just(exact),
            st.just(exact),
            st.just(exact + " "),
            st.just(" " + exact),
            st.just(exact[:-1]),
            st.just(exact + "x"),
            st.just("bearer " + token),
            st.just("BEARER " + token),
            st.just("Bearer  " + token),
            st.just("Basic " + token),
            st.just(token),
            st.just("Bearer"),
            st.just(""),
            st.just("Bearer " + token.swapcase()),
            st.text(max_size=40).map(lambda s: "Bearer " + s),
            st.text(max_size=60),
        )
    )
    return value


@dataclass
class _FakeStores:
    """Stands in for the Session_Store and TaskStore: records every write made through it."""

    writes: list[str] = field(default_factory=list)


async def _drive(app: ASGIApp, scope: Scope) -> int:
    """Send one request through `app` and return the HTTP status it answered with."""
    status = 0

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]

    await app(scope, receive, send)
    return status


def _scope(peer: str, authorization: str | None) -> Scope:
    headers: list[tuple[bytes, bytes]] = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("utf-8")))
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": headers,
        "client": (peer, 50000),
        "server": ("127.0.0.1", 8000),
    }


def _downstream(stores: _FakeStores) -> ASGIApp:
    """The handler behind the middleware. Reaching it writes to both fake stores and answers 200."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        stores.writes.extend(["session_store", "task_store"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


# Feature: local-agent-backend, Property 30: Loopback classification and bearer check
# **Validates: Requirements 16.1, 16.2, 16.3**
@settings(max_examples=200, deadline=None)
@given(case=_hosts, token=_tokens, data=st.data())
def test_loopback_classification_and_bearer_check(
    case: _HostCase, token: str, data: st.DataObject
) -> None:
    host, expected_loopback, kind = case
    presented = data.draw(_headers(token))

    # is_loopback is true exactly for `localhost`, 127.0.0.0/8 and ::1.
    assert is_loopback(host) is expected_loopback

    # The loopback guard admits a loopback IP peer and answers 403 for any other peer, with no
    # writes. A name such as `localhost` is not a socket peer address, so it is not an admit case.
    guard_stores = _FakeStores()
    guard = LoopbackGuardMiddleware(_downstream(guard_stores))
    guard_status = asyncio.run(_drive(guard, _scope(host, None)))
    if expected_loopback and kind == "ip":
        assert guard_status == 200
        assert guard_stores.writes == ["session_store", "task_store"]
    else:
        assert guard_status == 403
        assert guard_stores.writes == []

    # The bearer check admits iff the header equals "Bearer " + token exactly; otherwise 401 and
    # nothing is written.
    auth_stores = _FakeStores()
    auth = BearerAuthMiddleware(_downstream(auth_stores), token=token)
    auth_status = asyncio.run(_drive(auth, _scope("127.0.0.1", presented)))
    if presented == "Bearer " + token:
        assert auth_status == 200
        assert auth_stores.writes == ["session_store", "task_store"]
    else:
        assert auth_status == 401
        assert auth_stores.writes == []

    # A request with no Authorization header at all is also refused with no writes.
    missing_stores = _FakeStores()
    missing = BearerAuthMiddleware(_downstream(missing_stores), token=token)
    assert asyncio.run(_drive(missing, _scope("127.0.0.1", None))) == 401
    assert missing_stores.writes == []
