"""Socket bind integration tests (task 22.13): real sockets and real `python -m local_agent` runs.

* `test_exclusive_bind_conflict*`: a Backend that cannot bind its host and port exits 1 and names
  `host:port`, whatever already holds the port: another Backend, or a plain listener on the same
  or the wildcard address, with or without `SO_REUSEADDR` (1.8, 17.32).
* `test_bind_all_sockets_follow_the_os_rules`: the options on the real sockets match the rules of
  the host OS: `SO_REUSEADDR` only on Linux, never `SO_REUSEPORT`, `IPV6_V6ONLY` on `::1` (17.32).
* `test_localhost_dual_stack`: `localhost` serves on `127.0.0.1` and on `::1`, or writes one
  `bind_optional_unavailable` warning and serves on `127.0.0.1` alone (17.30, 17.31). The `::1`
  assertions are skipped when Python has no IPv6 (`socket.has_ipv6` is false).
* A conflict on the optional `::1` target is not "unavailable": the Backend still exits 1 (1.8).

Bind failure through the real process, with no MCP_Server started, is also in `test_process.py`.
A restart within seconds of a crash can hit `TIME_WAIT` on macOS. The design accepts that
trade-off (design.md, `socket_options`), so no test expects a restart to rebind at once.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from integration_harness import A2A_HEADERS, LOOPBACK, BackendLauncher, BackendProcess, free_port
from local_agent.oscompat.net import BindError, bind_all

EXIT_FAILURE = 1
EXIT_TIMEOUT_S = 60.0
LOCALHOST = "localhost"
LOOPBACK_V6 = "::1"
SO_REUSEPORT: int | None = getattr(socket, "SO_REUSEPORT", None)


# --- Helpers ------------------------------------------------------------------------------------


@contextmanager
def _foreign_listener(address: str, port: int, *, reuse_addr: bool = False) -> Iterator[None]:
    """A listening socket of another party on `address:port` (plain bind, as a dev server does)."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
        if reuse_addr:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((address, port))
        sock.listen()
        yield


def _can_bind_ipv6_loopback() -> bool:
    """True when this host lets a socket bind `::1` (IPv6 is built in and enabled)."""
    if not socket.has_ipv6:
        return False
    try:
        with closing(socket.socket(socket.AF_INET6, socket.SOCK_STREAM)) as sock:
            sock.bind((LOOPBACK_V6, 0))
    except OSError:
        return False
    return True


def _run_to_exit(process: BackendProcess) -> int:
    """Wait for a Backend that must stop by itself. It must not serve."""
    try:
        return process.wait(EXIT_TIMEOUT_S)
    finally:
        process.close()


def _events(process: BackendProcess, event: str) -> list[dict[str, Any]]:
    return [record for record in process.log_records() if record.get("event") == event]


def _health_status(host: str, port: int) -> int:
    url_host = f"[{host}]" if ":" in host else host
    return httpx.get(
        f"http://{url_host}:{port}/health", headers=A2A_HEADERS, timeout=5.0
    ).status_code


def _assert_bind_failure(process: BackendProcess, host: str, port: int) -> None:
    """The Backend ended with status 1 and named `host:port` in its log and `bind_failed`."""
    assert _run_to_exit(process) == EXIT_FAILURE
    assert f"{host}:{port}" in process.stderr_text()
    (record,) = _events(process, "bind_failed")
    assert record["host"] == host
    assert record["port"] == port


# --- 1.8, 17.32: exclusive bind -----------------------------------------------------------------


def test_exclusive_bind_conflict_with_running_backend(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    first = backend_launcher.launch()

    second = backend_launcher.launch(
        port=first.port, storage_dir=tmp_path / "second-storage", wait=False
    )
    _assert_bind_failure(second, LOOPBACK, first.port)

    assert first.running, "the first Backend must keep serving"
    assert _health_status(LOOPBACK, first.port) == 200


@pytest.mark.parametrize("reuse_addr", [False, True], ids=["plain", "reuseaddr"])
@pytest.mark.parametrize("address", [LOOPBACK, "0.0.0.0"], ids=["same-address", "wildcard"])
def test_exclusive_bind_conflict_with_foreign_listener(
    backend_launcher: BackendLauncher, address: str, reuse_addr: bool
) -> None:
    port = free_port()
    with _foreign_listener(address, port, reuse_addr=reuse_addr):
        process = backend_launcher.launch(port=port, wait=False)
        _assert_bind_failure(process, LOOPBACK, port)


def test_exclusive_bind_conflict_in_process() -> None:
    port = free_port()
    first = bind_all(LOOPBACK, port, sys.platform)
    try:
        with pytest.raises(BindError) as excinfo:
            bind_all(LOOPBACK, port, sys.platform)
        assert excinfo.value.host == LOOPBACK
        assert excinfo.value.port == port
        assert f"{LOOPBACK}:{port}" in str(excinfo.value)
        for sock in first:  # the failed attempt left the first listener's socket usable
            sock.listen()
    finally:
        for sock in first:
            sock.close()


# --- 17.32: options on real sockets -------------------------------------------------------------


def test_bind_all_sockets_follow_the_os_rules() -> None:
    port = free_port()
    sockets = bind_all(LOCALHOST, port, sys.platform)
    try:
        assert sockets[0].getsockname()[:2] == (LOOPBACK, port)
        for sock in sockets:
            reuse_addr = sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
            assert bool(reuse_addr) == sys.platform.startswith("linux")
            if SO_REUSEPORT is not None:
                assert sock.getsockopt(socket.SOL_SOCKET, SO_REUSEPORT) == 0
            if sock.family == socket.AF_INET6:
                assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
    finally:
        for sock in sockets:
            sock.close()


# --- 17.30, 17.31: localhost dual stack ---------------------------------------------------------


def test_localhost_dual_stack(backend_launcher: BackendLauncher) -> None:
    ipv6_enabled = _can_bind_ipv6_loopback()
    process = backend_launcher.launch(overrides={"server": {"host": LOCALHOST}})

    assert _health_status(LOOPBACK, process.port) == 200
    warnings = _events(process, "bind_optional_unavailable")
    if ipv6_enabled:
        assert _health_status(LOOPBACK_V6, process.port) == 200
        assert warnings == []
    else:
        assert len(warnings) == 1, "a missing ::1 gives exactly one warning"
        assert warnings[0]["address"] == LOOPBACK_V6
        if socket.has_ipv6:
            with pytest.raises(httpx.TransportError):
                _health_status(LOOPBACK_V6, process.port)


@pytest.mark.skipif(not socket.has_ipv6, reason="Python has no IPv6 support (socket.has_ipv6)")
def test_localhost_ipv6_conflict_is_fatal(backend_launcher: BackendLauncher) -> None:
    if not _can_bind_ipv6_loopback():
        pytest.skip("::1 cannot be bound on this host")
    port = free_port()
    with _foreign_listener(LOOPBACK_V6, port):
        process = backend_launcher.launch(
            port=port, overrides={"server": {"host": LOCALHOST}}, wait=False
        )
        _assert_bind_failure(process, LOCALHOST, port)
    # Nothing may listen on 127.0.0.1 after the failed start.
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((LOOPBACK, port), timeout=1.0).close()
