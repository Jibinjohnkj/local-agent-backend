"""Property test for the bind plan and socket options (Requirements 17.30, 17.31, 17.32, 1.8)."""

from __future__ import annotations

import errno
import socket
from dataclasses import dataclass, field
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.oscompat.net import SO_EXCLUSIVEADDRUSE, BindableSocket, BindError, bind_all

Outcome = Literal["ok", "conflict", "unavailable", "other"]

# Error codes per outcome class, including the Windows socket codes `classify_bind_error` reads.
_ERRNOS: dict[Outcome, list[int]] = {
    "ok": [],
    "conflict": [errno.EADDRINUSE, errno.EACCES, 10048, 10013],
    "unavailable": [errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, 10049, 10047],
    "other": [errno.EIO, errno.EINVAL, errno.ENOBUFS],
}

_V4 = socket.AF_INET
_V6 = socket.AF_INET6
_PLATFORMS = ["win32", "linux", "linux2", "darwin", "freebsd14"]

# (level, option) pairs, compared as a set so the order of `setsockopt` calls does not matter.
_Option = tuple[int, int]


@dataclass
class _FakeSocket:
    family: int
    outcome_error: OSError | None
    options: list[tuple[int, int, int]] = field(default_factory=list)
    bound_to: tuple[str, int] | None = None
    closed: bool = False

    def setsockopt(self, level: int, optname: int, value: int, /) -> None:
        self.options.append((level, optname, value))

    def bind(self, address: tuple[str, int], /) -> None:
        if self.outcome_error is not None:
            raise self.outcome_error
        self.bound_to = address

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeLayer:
    """Scripts the bind outcome of each family and records every socket it creates."""

    has_ipv6: bool
    errors: dict[int, OSError | None]
    created: list[_FakeSocket] = field(default_factory=list)

    def create(self, family: int) -> BindableSocket:
        sock = _FakeSocket(family, self.errors[family])
        self.created.append(sock)
        return sock

    def resolve(self, host: str, port: int) -> tuple[int, str]:
        raise AssertionError(f"name resolution must not run for {host!r}")


@dataclass
class _Warnings:
    events: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def warning(self, event: str, /, **fields: object) -> None:
        self.events.append((event, fields))


@dataclass(frozen=True)
class _Case:
    host: str
    port: int
    platform: str
    has_ipv6: bool
    # Bind outcome and the OS error that produces it, per address family.
    v4: tuple[Outcome, int]
    v6: tuple[Outcome, int]


@st.composite
def _outcome(draw: st.DrawFn) -> tuple[Outcome, int]:
    kinds: list[Outcome] = ["ok", "ok", "conflict", "unavailable", "other"]
    kind = draw(st.sampled_from(kinds))
    codes = _ERRNOS[kind]
    return kind, (draw(st.sampled_from(codes)) if codes else 0)


@st.composite
def _cases(draw: st.DrawFn) -> _Case:
    host = draw(
        st.one_of(
            st.sampled_from(["localhost", "LocalHost", "LOCALHOST"]),
            st.ip_addresses(v=4).map(str),
            st.ip_addresses(v=6).map(str),
        )
    )
    return _Case(
        host=host,
        port=draw(st.integers(min_value=1, max_value=65535)),
        platform=draw(st.sampled_from(_PLATFORMS)),
        has_ipv6=draw(st.booleans()),
        v4=draw(_outcome()),
        v6=draw(_outcome()),
    )


def _error(outcome: tuple[Outcome, int]) -> OSError | None:
    kind, code = outcome
    return None if kind == "ok" else OSError(code, f"simulated {kind}")


def _expected_options(platform: str, family: int) -> set[_Option]:
    options: set[_Option] = set()
    if platform == "win32":
        options.add((socket.SOL_SOCKET, SO_EXCLUSIVEADDRUSE))
    if platform.startswith("linux"):
        options.add((socket.SOL_SOCKET, socket.SO_REUSEADDR))
    if family == _V6:
        options.add((socket.IPPROTO_IPV6, socket.IPV6_V6ONLY))
    return options


def _expected_targets(host: str) -> list[tuple[int, str, bool]]:
    """(family, address, required) in plan order."""
    if host.lower() == "localhost":
        return [(_V4, "127.0.0.1", True), (_V6, "::1", False)]
    return [(_V6 if ":" in host else _V4, host, True)]


# Feature: local-agent-backend, Property 45: Bind plan and socket options
# **Validates: Requirements 17.30, 17.31, 17.32, 1.8**
@settings(max_examples=300)
@given(_cases())
def test_bind_all_plan_options_and_failure_handling(case: _Case) -> None:
    layer = _FakeLayer(
        has_ipv6=case.has_ipv6,
        errors={_V4: _error(case.v4), _V6: _error(case.v6)},
    )
    warnings = _Warnings()
    outcomes: dict[int, Outcome] = {_V4: case.v4[0], _V6: case.v6[0]}

    # Walk the plan the way the spec describes it, to get the expected result.
    expected_bound: list[tuple[int, str]] = []
    expected_skips = 0
    expected_created = 0
    failed = False
    for family, address, required in _expected_targets(case.host):
        if family == _V6 and not case.has_ipv6:
            kind: Outcome = "unavailable"  # no socket is created at all
        else:
            expected_created += 1
            kind = outcomes[family]
        if kind == "ok":
            expected_bound.append((family, address))
        elif not required and kind == "unavailable":
            expected_skips += 1
        else:
            failed = True
            break

    error: BindError | None = None
    result: list[BindableSocket] = []
    try:
        result = bind_all(case.host, case.port, case.platform, layer=layer, logger=warnings)
    except BindError as exc:
        error = exc

    if error is not None:
        assert failed, "bind_all failed although every failure was a skippable optional one"
        assert error.host == case.host
        assert error.port == case.port
        assert case.host in str(error)
        assert str(case.port) in str(error)
        assert error.os_error
        # Every socket created so far, bound or not, is closed.
        assert all(sock.closed for sock in layer.created)
        assert warnings.events == []
    else:
        assert not failed, "bind_all succeeded although a target failed"
        # Sockets are returned in plan order, bound to (address, port).
        assert [(sock.family, sock.bound_to) for sock in layer.created if not sock.closed] == [
            (family, (address, case.port)) for family, address in expected_bound
        ]
        assert result == [sock for sock in layer.created if not sock.closed]
        # A skipped optional socket produces exactly one warning, and is not leaked open.
        assert len(warnings.events) == expected_skips
        assert all(event == "bind_optional_unavailable" for event, _ in warnings.events)
        assert all(sock.closed for sock in layer.created if sock.bound_to is None)
        # `localhost` always binds 127.0.0.1 first.
        if case.host.lower() == "localhost":
            assert expected_bound[0] == (_V4, "127.0.0.1")

    # `localhost` and literals never resolve (the fake layer raises if asked), and the sockets
    # created are exactly the ones the plan reaches.
    assert len(layer.created) == expected_created

    # Socket options: exclusive-use on Windows only, SO_REUSEADDR on Linux only, never
    # SO_REUSEPORT, IPV6_V6ONLY on IPv6. Each option is set to 1.
    reuse_port = getattr(socket, "SO_REUSEPORT", None)
    for sock in layer.created:
        assert all(value == 1 for _, _, value in sock.options)
        applied = {(level, name) for level, name, _ in sock.options}
        assert len(applied) == len(sock.options)
        assert applied == _expected_options(case.platform, sock.family)
        if reuse_port is not None:
            assert (socket.SOL_SOCKET, reuse_port) not in applied
