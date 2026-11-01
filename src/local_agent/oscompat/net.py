"""Bind planning for the A2A_Server's listening sockets (Requirements 1.8, 17.30, 17.31, 17.32).

The Backend binds its listening sockets itself, before the event loop starts, and hands them to
uvicorn. Three pure functions decide what to bind and how:

- `plan_binds(host, port)` turns the configured host into `BindTarget`s. `localhost` becomes
  `127.0.0.1` (required) plus `::1` (optional), without any name resolution. An IP literal becomes
  one required target. Any other host name is resolved once and the first result is used.
- `socket_options(platform, family)` picks the socket options. Windows gets
  `SO_EXCLUSIVEADDRUSE` only. Linux gets `SO_REUSEADDR`, which there never lets two sockets listen
  on one address and port. macOS gets no reuse option, because BSD semantics would let a
  specific-address bind coexist with another process's wildcard listener. `SO_REUSEPORT` is never
  set, and IPv6 sockets set `IPV6_V6ONLY`.
- `classify_bind_error(exc)` sorts an `OSError` from `bind` into `"conflict"`, `"unavailable"`, or
  `"other"`.

`bind_all` runs the plan against a `SocketLayer`. The default layer uses the real `socket` module;
a test passes a fake layer that scripts each bind outcome. An optional target that is unavailable
(IPv6 turned off) is skipped with one warning. Any other failure closes every socket already bound
and raises `BindError` naming the host, the port, and the OS error. `bind_all` only binds. The
caller (or uvicorn) calls `listen`.

`platform` takes `sys.platform` values: `"win32"` selects the Windows rules, a value starting with
`"linux"` selects the Linux rules, and every other value gets the macOS rules.
"""

from __future__ import annotations

import contextlib
import errno
import ipaddress
import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast, overload

__all__ = [
    "SO_EXCLUSIVEADDRUSE",
    "BindError",
    "BindTarget",
    "BindableSocket",
    "SockOpts",
    "SocketLayer",
    "bind_all",
    "classify_bind_error",
    "plan_binds",
    "socket_options",
]

BindErrorKind = Literal["conflict", "unavailable", "other"]

# `socket.SO_EXCLUSIVEADDRUSE` exists on Windows only. Its value there is `~SO_REUSEADDR`, which
# Python exposes as -5. The fallback lets a test plan Windows options on another host.
SO_EXCLUSIVEADDRUSE: Final[int] = getattr(socket, "SO_EXCLUSIVEADDRUSE", -5)

_LOCALHOST: Final = "localhost"
_LOOPBACK_V4: Final = "127.0.0.1"
_LOOPBACK_V6: Final = "::1"

# Windows socket error codes. Python on Windows reports them as `errno` (and `winerror`), but the
# `errno` module's POSIX-named constants differ from them on other hosts, so list both.
_WSAEACCES: Final = 10013
_WSAEADDRINUSE: Final = 10048
_WSAEADDRNOTAVAIL: Final = 10049
_WSAEAFNOSUPPORT: Final = 10047

_CONFLICT_CODES: Final = frozenset({errno.EADDRINUSE, errno.EACCES, _WSAEADDRINUSE, _WSAEACCES})
_UNAVAILABLE_CODES: Final = frozenset(
    {errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, _WSAEADDRNOTAVAIL, _WSAEAFNOSUPPORT}
)

_logger = logging.getLogger("local_agent.oscompat.net")


# --- Plan types ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindTarget:
    """One socket to bind. `required` is false only for the `::1` socket of `localhost`."""

    family: int
    address: str
    required: bool


@dataclass(frozen=True, slots=True)
class SockOpts:
    """The socket options to set on one listening socket (all `1` when true, unset when false)."""

    exclusive_addr_use: bool
    reuse_addr: bool
    reuse_port: bool
    v6only: bool


class BindError(Exception):
    """The A2A_Server cannot bind its configured host and port (1.8).

    `os_error` is the text of the OS error that stopped the bind.
    """

    host: str
    port: int
    os_error: str

    def __init__(self, host: str, port: int, os_error: str) -> None:
        super().__init__(host, port, os_error)
        self.host = host
        self.port = port
        self.os_error = os_error

    def __str__(self) -> str:
        return f"cannot bind {self.host}:{self.port}: {self.os_error}"


# --- Pure functions ------------------------------------------------------------------------------


def _system_resolve(host: str, port: int) -> tuple[int, str]:
    """First `getaddrinfo` result for `host` as `(family, address)`."""
    results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not results:
        raise OSError(errno.EADDRNOTAVAIL, f"no address found for {host}")
    family, _kind, _proto, _canon, sockaddr = results[0]
    return int(family), str(sockaddr[0])


def plan_binds(
    host: str,
    port: int,
    *,
    resolve: Callable[[str, int], tuple[int, str]] | None = None,
) -> list[BindTarget]:
    """The sockets to bind for the configured `host` (17.30).

    `localhost` gives `127.0.0.1` (required) and `::1` (optional) and never resolves a name. An
    IPv4 or IPv6 literal gives one required target. Any other host name goes through `resolve`
    (default: the first `getaddrinfo` result for a stream socket), which may raise `OSError`.
    """
    if host.lower() == _LOCALHOST:
        return [
            BindTarget(socket.AF_INET, _LOOPBACK_V4, required=True),
            BindTarget(socket.AF_INET6, _LOOPBACK_V6, required=False),
        ]
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        family, address = (resolve or _system_resolve)(host, port)
        return [BindTarget(family, address, required=True)]
    family = socket.AF_INET if literal.version == 4 else socket.AF_INET6
    return [BindTarget(family, host, required=True)]


def socket_options(platform: str, family: int) -> SockOpts:
    """Socket options for a listening socket of `family` on `platform` (17.32)."""
    return SockOpts(
        exclusive_addr_use=platform == "win32",
        reuse_addr=platform.startswith("linux"),
        reuse_port=False,
        v6only=family == socket.AF_INET6,
    )


def classify_bind_error(exc: OSError) -> BindErrorKind:
    """Sort a bind failure (1.8, 17.31).

    `"conflict"`: the address is in use, or access is denied (also what an exclusive-use conflict
    and a Windows excluded port range return). `"unavailable"`: the address or the address family
    does not exist on this host. Everything else is `"other"`. Both `errno` and `winerror` are
    read, and a conflict code wins when the two disagree.
    """
    codes = {code for code in (exc.errno, getattr(exc, "winerror", None)) if isinstance(code, int)}
    if codes & _CONFLICT_CODES:
        return "conflict"
    if codes & _UNAVAILABLE_CODES:
        return "unavailable"
    return "other"


# --- Socket layer --------------------------------------------------------------------------------


class BindableSocket(Protocol):
    """The part of `socket.socket` that `bind_all` uses."""

    def setsockopt(self, level: int, optname: int, value: int, /) -> None: ...

    def bind(self, address: tuple[str, int], /) -> None: ...

    def close(self) -> None: ...


class SocketLayer(Protocol):
    """Everything `bind_all` needs from the operating system."""

    @property
    def has_ipv6(self) -> bool:
        """Whether this host's Python build supports IPv6 sockets."""
        ...

    def create(self, family: int) -> BindableSocket:
        """A new, unbound TCP socket of `family`. May raise `OSError`."""
        ...

    def resolve(self, host: str, port: int) -> tuple[int, str]:
        """First `getaddrinfo` result for `host` as `(family, address)`. May raise `OSError`."""
        ...


class _SystemSocketLayer:
    """`SocketLayer` backed by the real `socket` module."""

    @property
    def has_ipv6(self) -> bool:
        return socket.has_ipv6

    def create(self, family: int) -> socket.socket:
        return socket.socket(family, socket.SOCK_STREAM)

    def resolve(self, host: str, port: int) -> tuple[int, str]:
        return _system_resolve(host, port)


class _WarningLogger(Protocol):
    def warning(self, event: str, /, **fields: object) -> None: ...


class _StdlibWarningLogger:
    def warning(self, event: str, /, **fields: object) -> None:
        rendered = " ".join(f"{key}={value!r}" for key, value in fields.items())
        _logger.warning("%s %s", event, rendered, extra={"event": event, "fields": dict(fields)})


# --- I/O -----------------------------------------------------------------------------------------


def _apply_options(sock: BindableSocket, opts: SockOpts) -> None:
    if opts.exclusive_addr_use:
        sock.setsockopt(socket.SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1)
    if opts.reuse_addr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if opts.v6only:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    # `opts.reuse_port` is never true (17.32), so `SO_REUSEPORT` is never set.


def _close_quietly(sock: BindableSocket) -> None:
    with contextlib.suppress(OSError):
        sock.close()


def _bind_one(layer: SocketLayer, target: BindTarget, port: int, platform: str) -> BindableSocket:
    """Create, configure, and bind one socket. Closes it and re-raises when anything fails."""
    if target.family == socket.AF_INET6 and not layer.has_ipv6:
        raise OSError(errno.EAFNOSUPPORT, "IPv6 is not available on this host")
    sock = layer.create(target.family)
    try:
        _apply_options(sock, socket_options(platform, target.family))
        sock.bind((target.address, port))
    except BaseException:
        _close_quietly(sock)
        raise
    return sock


@overload
def bind_all(host: str, port: int, platform: str) -> list[socket.socket]: ...


@overload
def bind_all(
    host: str,
    port: int,
    platform: str,
    *,
    layer: SocketLayer,
    logger: _WarningLogger | None = None,
) -> list[BindableSocket]: ...


def bind_all(
    host: str,
    port: int,
    platform: str,
    *,
    layer: SocketLayer | None = None,
    logger: _WarningLogger | None = None,
) -> list[socket.socket] | list[BindableSocket]:
    """Bind every `plan_binds(host, port)` target; return the sockets in plan order (17.30-17.32).

    An optional target that is unavailable is skipped with one `bind_optional_unavailable` warning
    (17.31). Any other failure, on any target, closes the sockets already bound and raises
    `BindError(host, port, os_error)`. `layer` replaces the real socket module for tests.
    """
    active: SocketLayer = layer if layer is not None else _SystemSocketLayer()
    log: _WarningLogger = logger if logger is not None else _StdlibWarningLogger()
    bound: list[BindableSocket] = []
    try:
        targets = plan_binds(host, port, resolve=active.resolve)
        for target in targets:
            try:
                bound.append(_bind_one(active, target, port, platform))
            except OSError as exc:
                if not target.required and classify_bind_error(exc) == "unavailable":
                    log.warning(
                        "bind_optional_unavailable",
                        host=host,
                        port=port,
                        address=target.address,
                        reason=str(exc),
                    )
                    continue
                raise
    except OSError as exc:
        for sock in bound:
            _close_quietly(sock)
        raise BindError(host, port, str(exc)) from exc
    except BaseException:
        for sock in bound:
            _close_quietly(sock)
        raise
    if layer is None:
        return cast("list[socket.socket]", bound)
    return bound
