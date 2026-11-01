"""POSIX-only process and loop integration tests (task 22.12; Requirements 17.10, 17.11, 17.29).

Each test runs the real Backend as a child (`python -m local_agent`, or the loop probe) and looks at
it from outside with `os.getsid`, `os.getpgid`, signals, and `ps`.

* `test_new_session_isolation` (17.10): a `stdio` MCP_Server and its descendants lead a session of
  their own, not the Backend's. A `SIGINT` sent to the Backend's process group does not reach
  them. The Backend is stopped (`SIGSTOP`) while the signal is sent and for the check, so a
  Backend that forwarded the signal could not hide that. The same `SIGINT` sent to the server's
  own group is the control: it does end the descendants.
* `test_sigterm_ignoring_child` (17.11): a server that ignores `SIGTERM` and stdin EOF gets
  `SIGKILL` 5 s (+-0.5 s) after the graceful `SIGTERM`. The descendants die at the `SIGTERM` and
  the server at the `SIGKILL`, and the gap between the two is measured.
* `test_uvloop_selected` (17.29): the loop the Backend serves on is a `uvloop.Loop`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
import yaml

from integration_harness import (
    FIXTURES_DIR,
    PIDS,
    BackendLauncher,
    BackendProcess,
    config_dict,
    free_port,
    isolated_env,
    wait_until_gone,
)

# Same definition as `posix_only` in tests/conftest.py. `from conftest import posix_only` fails
# here: tests/integration/conftest.py takes the module name `conftest` once pytest has loaded it.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific OS behaviour")

pytestmark = [pytest.mark.integration, posix_only]

SPAWNING_SERVER: Final = FIXTURES_DIR / "mcp_spawning_server.py"
LOOP_PROBE: Final = FIXTURES_DIR / "loop_probe_backend.py"
SERVER_NAME: Final = "spawn"

GRACE_S: Final = 5.0
GRACE_TOLERANCE_S: Final = 0.5
POLL_S: Final = 0.02
STARTUP_WAIT_S: Final = 30.0
# Stdin wait (1 s) + grace (5 s) + reap (1.5 s) after the shutdown began, plus time to start it.
KILL_WAIT_S: Final = 20.0
SIGNAL_SETTLE_S: Final = 1.0
CONTROL_WAIT_S: Final = 5.0
EXIT_WAIT_S: Final = 30.0


@dataclass(frozen=True)
class Tree:
    """The pids of one server Process_Tree: the server, its child, and its grandchild."""

    server: int
    grandchild: int
    great_grandchild: int

    @property
    def descendants(self) -> tuple[int, int]:
        return (self.grandchild, self.great_grandchild)

    @property
    def pids(self) -> tuple[int, int, int]:
        return (self.server, self.grandchild, self.great_grandchild)


def spawning_entry(pid_file: Path, *flags: str) -> dict[str, Any]:
    return {
        "name": SERVER_NAME,
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SPAWNING_SERVER), *flags, "--pid-file", str(pid_file)],
    }


def running_pids(pids: Collection[int]) -> set[int]:
    """The pids that exist and have not exited. A zombie has exited, so `ps` state `Z` is out.

    `os.kill(pid, 0)` cannot tell the two apart: a descendant killed while its parent does not
    reap it stays a zombie, and so does a server whose stopped parent has not reaped it.
    """
    done = subprocess.run(
        ["ps", "-o", "pid=,stat=", "-p", ",".join(str(pid) for pid in pids)],
        capture_output=True,
        text=True,
        check=False,
    )
    alive: set[int] = set()
    for line in done.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and not fields[1].startswith("Z"):
            alive.add(int(fields[0]))
    return alive


def wait_until_available(backend: BackendProcess, timeout_s: float = STARTUP_WAIT_S) -> None:
    """Wait until `/health` lists the server as available (the MCP handshake is complete)."""
    deadline = time.monotonic() + timeout_s
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        code = backend.popen.poll()
        if code is not None:
            pytest.fail(f"the Backend exited with status {code}\n{backend.stderr_text()}")
        try:
            response = httpx.get(f"{backend.url}/health", timeout=1.0)
            body = response.json()
        except httpx.TransportError, ValueError:
            body = {}
        servers = body.get("mcp_servers", [])
        if servers and all(server["status"] == "available" for server in servers):
            return
        time.sleep(0.1)
    pytest.fail(f"the server never became available; last /health: {body}\n{backend.stderr_text()}")


def read_tree(pid_file: Path, timeout_s: float = 10.0) -> Tree:
    """The tree in the pid file once all three pids are known. Records every pid for cleanup."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            data = json.loads(pid_file.read_text(encoding="utf-8"))
        except OSError, ValueError:
            data = {}
        descendants = data.get("descendants", [])
        if "server" in data and len(descendants) == 2:
            tree = Tree(int(data["server"]), int(descendants[0]), int(descendants[1]))
            PIDS.record(*tree.pids)
            return tree
        time.sleep(0.05)
    pytest.fail(f"the server did not publish its process tree in {timeout_s} s")


def start_tree(
    launcher: BackendLauncher, tmp_path: Path, *flags: str
) -> tuple[BackendProcess, Tree]:
    """A running Backend with one spawning server that is available; returns it and its tree."""
    pid_file = tmp_path / "pids.json"
    backend = launcher.launch(mcp_servers=[spawning_entry(pid_file, *flags)])
    wait_until_available(backend)
    tree = read_tree(pid_file)
    assert running_pids(tree.pids) == set(tree.pids)
    return backend, tree


# --- 17.10: a new session ----------------------------------------------------------------------


def test_new_session_isolation(backend_launcher: BackendLauncher, tmp_path: Path) -> None:
    # Feature: local-agent-backend, POSIX process isolation; Validates: Requirements 17.10
    backend, tree = start_tree(backend_launcher, tmp_path)

    # The harness starts the Backend in a session of its own, so its session and group are its pid
    # and a signal to that group cannot reach the test run.
    backend_group = os.getpgid(backend.pid)
    assert os.getsid(backend.pid) == backend.pid == backend_group
    assert os.getsid(backend.pid) != os.getsid(0)

    # The server leads a new session and group, and its descendants stay in them.
    server_session = os.getsid(tree.server)
    assert server_session == tree.server
    assert server_session != os.getsid(backend.pid)
    assert os.getpgid(tree.server) == tree.server != backend_group
    for descendant in tree.descendants:
        assert os.getsid(descendant) == server_session
        assert os.getpgid(descendant) == tree.server

    # Stop the Backend, so it cannot act on the signal while the check runs. The pending `SIGINT`
    # is delivered when it continues.
    os.kill(backend.pid, signal.SIGSTOP)
    try:
        os.killpg(backend_group, signal.SIGINT)
        time.sleep(SIGNAL_SETTLE_S)
        assert running_pids(tree.pids) == set(tree.pids), "SIGINT to the Backend's group reached"

        # Control: the same signal to the server's own group does end the descendants (they run
        # the Python default handler), so the survival above is not because they ignore it.
        os.killpg(tree.server, signal.SIGINT)
        deadline = time.monotonic() + CONTROL_WAIT_S
        while running_pids(tree.descendants) and time.monotonic() < deadline:
            time.sleep(POLL_S)
        assert running_pids(tree.descendants) == set(), "SIGINT to a group did not end its members"
    finally:
        os.kill(backend.pid, signal.SIGCONT)

    backend.wait(EXIT_WAIT_S)
    assert wait_until_gone(tree.pids, 5.0) == []


# --- 17.11: SIGKILL after the grace period -----------------------------------------------------


def test_sigterm_ignoring_child(backend_launcher: BackendLauncher, tmp_path: Path) -> None:
    # Feature: local-agent-backend, POSIX tree termination; Validates: Requirements 17.11
    backend, tree = start_tree(backend_launcher, tmp_path, "--ignore-graceful")

    # The Backend closes stdin (ignored), waits 1 s, sends SIGTERM to the group (ignored by the
    # server, fatal to the descendants), waits the grace period, and then sends SIGKILL.
    started = time.monotonic()
    backend.send_shutdown_signal()
    terminated_at: float | None = None
    killed_at: float | None = None
    while killed_at is None and time.monotonic() - started < KILL_WAIT_S:
        alive = running_pids(tree.pids)
        now = time.monotonic()
        if terminated_at is None and tree.grandchild not in alive:
            terminated_at = now
        if tree.server not in alive:
            killed_at = now
        time.sleep(POLL_S)

    assert killed_at is not None, "the server that ignores SIGTERM was never killed"
    assert terminated_at is not None, "the descendants were not ended by the SIGTERM"
    assert terminated_at < killed_at
    assert killed_at - terminated_at == pytest.approx(GRACE_S, abs=GRACE_TOLERANCE_S)

    assert backend.wait(EXIT_WAIT_S) is not None
    assert wait_until_gone(tree.pids, 5.0) == []
    stopped = [r for r in backend.log_records() if r.get("event") == "mcp_process_tree_stopped"]
    assert len(stopped) == 1
    assert stopped[0]["forced"] is True
    assert stopped[0]["survivors"] == []


# --- 17.29: uvloop ----------------------------------------------------------------------------


def test_uvloop_selected(tmp_path: Path, isolated_home: Path) -> None:
    # Feature: local-agent-backend, event loop selection; Validates: Requirements 17.29
    uvloop = pytest.importorskip("uvloop")

    port = free_port()
    config_path = tmp_path / "config.yaml"
    config = config_dict(storage_dir=tmp_path / "storage", port=port)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    probe_file = tmp_path / "loop.json"
    log_file = tmp_path / "probe-stderr.log"

    with log_file.open("wb") as stderr:
        backend = subprocess.Popen(
            [sys.executable, str(LOOP_PROBE), str(probe_file)],
            cwd=tmp_path,
            env=isolated_env(isolated_home, {"AGENT_CONFIG": str(config_path)}),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            start_new_session=True,
        )
    PIDS.track_process(backend)
    try:
        deadline = time.monotonic() + STARTUP_WAIT_S
        healthy = False
        while not healthy and time.monotonic() < deadline:
            assert backend.poll() is None, log_file.read_text(encoding="utf-8", errors="replace")
            try:
                healthy = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).is_success
            except httpx.TransportError:
                time.sleep(0.1)
        assert healthy, log_file.read_text(encoding="utf-8", errors="replace")

        loop_class = json.loads(probe_file.read_text(encoding="utf-8"))
        assert (loop_class["module"], loop_class["name"]) == (
            uvloop.Loop.__module__,
            uvloop.Loop.__qualname__,
        )
    finally:
        if backend.poll() is None:
            backend.send_signal(signal.SIGTERM)
        try:
            backend.wait(EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(10)
