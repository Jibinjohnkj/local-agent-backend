"""Graceful_Shutdown on a Shutdown_Signal, against a real `python -m local_agent` (task 22.14).

Each test starts the Backend with the spawning MCP fixture (a three-level Process_Tree) and the
OpenAI-compatible stub LLM. A gated `StubTurn` keeps Tasks in `working` while the signal arrives:
the stub sends one chunk and then waits, so only the Backend's own shutdown can end them.

* POSIX sends `SIGTERM` and `SIGINT`; Windows sends `CTRL_BREAK_EVENT` to a Backend started with
  `CREATE_NEW_PROCESS_GROUP` (the harness does both through `send_shutdown_signal`).
* `test_console_close` needs a person to close a console window. It is marked `manual` and runs
  only on Windows.

The tests check: shutdown starts within 1 s of the signal, every active Task ends `canceled` (also
in the Session_Store after a restart), every process of the MCP trees is gone, the exit status is 0,
and a second signal during shutdown still exits 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import pytest
import yaml

from integration_harness import (
    FIXTURES_DIR,
    PIDS,
    BackendLauncher,
    BackendProcess,
    FakeLlmServer,
    StubTurn,
    config_dict,
    free_port,
    isolated_env,
    jsonrpc_call,
    message_params,
    pid_alive,
    wait_for_task_state,
    wait_until_gone,
)
from local_agent.oscompat.signals import CONSOLE_CLOSE_GRACE_S, DEFAULT_GRACE_S

# Requirements: 6.11, 17.12, 17.13, 17.14, 17.15

SPAWNING_SERVER: Final = FIXTURES_DIR / "mcp_spawning_server.py"
SERVER_NAME: Final = "spawn"
WORKING: Final = "TASK_STATE_WORKING"
CANCELED: Final = "TASK_STATE_CANCELED"
SHUTDOWN_START_LIMIT_S: Final = 1.0
"""Graceful_Shutdown starts within 1 s of the signal (17.13)."""
TREE_MARGIN_S: Final = 1.5
"""Time the OS gets, after the Backend exits, to remove the processes of a tree (17.12)."""
EXIT_LIMIT_S: Final = 1.0 + DEFAULT_GRACE_S + 5.0
SSE_HEADERS: Final = {"A2A-Version": "1.0", "Accept": "text/event-stream"}

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific OS behaviour")

type StorageBackend = Literal["memory", "sqlite"]

# POSIX has two Shutdown_Signals; Windows has CTRL_BREAK_EVENT only (the harness maps both to it).
SIGNAL_CASES: Final = (
    [pytest.param(False, id="CTRL_BREAK")]
    if sys.platform == "win32"
    else [pytest.param(False, id="SIGTERM"), pytest.param(True, id="SIGINT")]
)


# --- Helpers ------------------------------------------------------------------------------------


def spawning_entry(mode: str, pid_file: Path) -> dict[str, Any]:
    """The `mcp_servers` entry that runs the spawning fixture in one termination mode."""
    return {
        "name": SERVER_NAME,
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SPAWNING_SERVER), f"--{mode}", "--pid-file", str(pid_file)],
    }


def read_tree(pid_file: Path) -> list[int] | None:
    """The server, grandchild, and great-grandchild pids once all three are known."""
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    descendants = data.get("descendants", [])
    if "server" not in data or len(descendants) != 2:
        return None
    return [int(data["server"]), *(int(pid) for pid in descendants)]


def wait_for_tree(pid_file: Path, timeout_s: float = 20.0) -> list[int]:
    """Wait for the whole tree to be published. Records its pids for the leftover killer."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pids = read_tree(pid_file)
        if pids is not None:
            PIDS.record(*pids)
            return pids
        time.sleep(0.05)
    raise TimeoutError(f"the MCP server did not publish its process tree to {pid_file}")


def wait_for_log_event(backend: BackendProcess, event: str, timeout_s: float) -> float | None:
    """Seconds until the Backend logs `event`, or `None` when it does not within `timeout_s`."""
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if any(record.get("event") == event for record in backend.log_records()):
            return time.monotonic() - started
        time.sleep(0.02)
    return None


def log_events(backend: BackendProcess) -> list[str]:
    return [str(record.get("event")) for record in backend.log_records()]


@dataclass
class Scenario:
    """A running Backend with `working` Tasks held open by the stub LLM."""

    backend: BackendProcess
    llm: FakeLlmServer
    gate: threading.Event
    pids: list[int]
    pid_file: Path
    task_ids: list[str]


def start_scenario(
    launcher: BackendLauncher,
    llm: FakeLlmServer,
    tmp_path: Path,
    *,
    mode: str,
    storage_backend: StorageBackend,
    tasks: int,
) -> Scenario:
    """Start the Backend with one spawning MCP_Server and `tasks` Tasks that stay `working`."""
    gate = threading.Event()
    llm.script(*(StubTurn(chunks=("par", "tial"), hold_after=1, gate=gate) for _ in range(tasks)))
    pid_file = tmp_path / "pids.json"
    backend = launcher.launch(
        storage_backend=storage_backend,
        llm_base=llm.base_url,
        mcp_servers=[spawning_entry(mode, pid_file)],
    )
    pids = wait_for_tree(pid_file)
    task_ids: list[str] = []
    with backend.client() as client:
        for number in range(tasks):
            params = message_params(f"task {number}")
            params["configuration"] = {"returnImmediately": True}  # do not wait for the held LLM
            result = jsonrpc_call(client, "SendMessage", params)["result"]
            task_ids.append(str(result["task"]["id"]))
        llm.wait_for_chunks(tasks)  # every Task has its first chunk, then the stub holds
        for task_id in task_ids:
            wait_for_task_state(client, task_id, [WORKING], timeout_s=10.0)
    return Scenario(backend, llm, gate, pids, pid_file, task_ids)


def assert_log_shows_one_shutdown(backend: BackendProcess) -> None:
    events = log_events(backend)
    assert events.count("shutdown_started") == 1
    assert events.count("shutdown_finished") == 1
    assert events.index("shutdown_started") < events.index("shutdown_finished")


# --- Tests --------------------------------------------------------------------------------------


@pytest.mark.parametrize("interrupt", SIGNAL_CASES)
def test_shutdown_exit_code(
    backend_launcher: BackendLauncher,
    llm_stub: FakeLlmServer,
    tmp_path: Path,
    storage_backend: StorageBackend,
    interrupt: bool,
) -> None:
    scenario = start_scenario(
        backend_launcher,
        llm_stub,
        tmp_path,
        mode="cooperative",
        storage_backend=storage_backend,
        tasks=2,
    )
    backend = scenario.backend
    try:
        assert all(pid_alive(pid) for pid in scenario.pids)

        backend.send_shutdown_signal(interrupt=interrupt)
        started_after = wait_for_log_event(backend, "shutdown_started", SHUTDOWN_START_LIMIT_S)
        assert started_after is not None, "shutdown did not start within 1 s of the signal"
        assert backend.wait(EXIT_LIMIT_S) == 0
    finally:
        scenario.gate.set()

    assert_log_shows_one_shutdown(backend)
    assert wait_until_gone(scenario.pids, TREE_MARGIN_S) == []
    # The stub never released a Task (the gate is still closed): the Backend itself ended them.
    assert not any(
        record.get("event") == "shutdown_cancel_tasks_failed" for record in backend.log_records()
    )

    if storage_backend == "sqlite":
        # The canceled state is stored: Restart_Recovery would turn a `working` Task into `failed`.
        backend.restart().wait_healthy()
        with backend.client() as client:
            for task_id in scenario.task_ids:
                stored = jsonrpc_call(client, "GetTask", {"id": task_id})["result"]
                assert stored["status"]["state"] == CANCELED
        assert backend.stop(EXIT_LIMIT_S) == 0


@pytest.mark.parametrize("interrupt", SIGNAL_CASES)
def test_second_signal_during_shutdown_still_exits_zero(
    backend_launcher: BackendLauncher,
    llm_stub: FakeLlmServer,
    tmp_path: Path,
    interrupt: bool,
) -> None:
    run_double_signal(backend_launcher, llm_stub, tmp_path, interrupt)


@pytest.mark.parametrize("interrupt", SIGNAL_CASES)
def test_second_signal_cuts_the_grace_period(
    backend_launcher: BackendLauncher,
    llm_stub: FakeLlmServer,
    tmp_path: Path,
    interrupt: bool,
) -> None:
    elapsed = run_double_signal(backend_launcher, llm_stub, tmp_path, interrupt)
    assert elapsed < DEFAULT_GRACE_S - 1.0


def run_double_signal(
    backend_launcher: BackendLauncher,
    llm_stub: FakeLlmServer,
    tmp_path: Path,
    interrupt: bool,
) -> float:
    """Signal a Backend whose MCP server ignores the graceful stop, then signal it again.

    Asserts exit status 0, no leftover process, and one shutdown sequence. Returns the seconds
    from the second signal to the exit. Without the second signal the exit waits out the 5 s grace.
    """
    scenario = start_scenario(
        backend_launcher,
        llm_stub,
        tmp_path,
        mode="ignore-graceful",
        storage_backend="memory",
        tasks=1,
    )
    backend = scenario.backend
    try:
        backend.send_shutdown_signal(interrupt=interrupt)
        assert wait_for_log_event(backend, "shutdown_started", SHUTDOWN_START_LIMIT_S) is not None
        time.sleep(0.5)
        assert backend.running, "shutdown ended before the second signal; the test proves nothing"

        second_sent = time.monotonic()
        backend.send_shutdown_signal(interrupt=interrupt)
        assert backend.wait(EXIT_LIMIT_S) == 0
        elapsed = time.monotonic() - second_sent
    finally:
        scenario.gate.set()

    assert wait_until_gone(scenario.pids, TREE_MARGIN_S) == []
    assert_log_shows_one_shutdown(backend)
    return elapsed


@pytest.mark.parametrize("interrupt", SIGNAL_CASES)
def test_shutdown_ends_a_live_stream_with_canceled(
    backend_launcher: BackendLauncher,
    llm_stub: FakeLlmServer,
    tmp_path: Path,
    interrupt: bool,
) -> None:
    gate = threading.Event()
    llm_stub.script(StubTurn(chunks=("par", "tial"), hold_after=1, gate=gate))
    pid_file = tmp_path / "pids.json"
    backend = backend_launcher.launch(
        llm_base=llm_stub.base_url, mcp_servers=[spawning_entry("cooperative", pid_file)]
    )
    pids = wait_for_tree(pid_file)
    stream = SseCollector(backend)
    try:
        stream.start()
        llm_stub.wait_for_chunks(1)
        assert stream.wait_for(lambda: WORKING in stream.states)

        backend.send_shutdown_signal(interrupt=interrupt)
        assert wait_for_log_event(backend, "shutdown_started", SHUTDOWN_START_LIMIT_S) is not None
        assert backend.wait(EXIT_LIMIT_S) == 0
        stream.join(timeout=10)
    finally:
        gate.set()

    assert not stream.is_alive(), "the SSE stream stayed open after the Backend exited"
    assert stream.error is None, f"the SSE stream ended abnormally: {stream.error}"
    assert stream.states[-1] == CANCELED
    assert CANCELED not in stream.states[:-1]
    assert wait_until_gone(pids, TREE_MARGIN_S) == []


class SseCollector(threading.Thread):
    """Sends one `SendStreamingMessage` and records the Task states of the events it receives."""

    def __init__(self, backend: BackendProcess) -> None:
        super().__init__(name="sse-collector", daemon=True)
        self._backend = backend
        self.states: list[str] = []
        self.error: BaseException | None = None
        self._changed = threading.Condition()

    @property
    def last_state(self) -> str | None:
        return self.states[-1] if self.states else None

    def wait_for(self, predicate: Callable[[], bool], timeout_s: float = 10.0) -> bool:
        with self._changed:
            return self._changed.wait_for(predicate, timeout_s)

    def run(self) -> None:
        envelope = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendStreamingMessage",
            "params": message_params("stream it"),
        }
        try:
            with (
                httpx.Client(base_url=self._backend.url, timeout=60.0) as client,
                client.stream("POST", "/", json=envelope, headers=SSE_HEADERS) as response,
            ):
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        self._record(json.loads(line[len("data:") :])["result"])
        except BaseException as exc:
            self.error = exc

    def _record(self, result: dict[str, Any]) -> None:
        state: str | None = None
        if "task" in result:
            state = result["task"]["status"]["state"]
        elif "statusUpdate" in result:
            state = result["statusUpdate"]["status"]["state"]
        if state is not None:
            with self._changed:
                self.states.append(str(state))
                self._changed.notify_all()


@windows_only
@pytest.mark.manual
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="needs a person to close a console window")
def test_console_close(
    tmp_path: Path, isolated_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Closing the console window ends the Backend and its MCP trees within the 2 s grace (17.15).

    The Backend runs in a console window of its own. The person running the test closes it.
    """
    pid_file = tmp_path / "pids.json"
    port = free_port()
    config = config_dict(
        storage_dir=tmp_path / "storage",
        port=port,
        mcp_servers=[spawning_entry("cooperative", pid_file)],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = {**isolated_env(isolated_home), "AGENT_CONFIG": str(config_path)}
    stderr_path = tmp_path / "stderr.log"
    with stderr_path.open("wb") as err:
        popen = subprocess.Popen(
            [sys.executable, "-m", "local_agent"],
            cwd=tmp_path,
            env=env,
            stdout=err,
            stderr=err,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    PIDS.track_process(popen)
    pids = wait_for_tree(pid_file, timeout_s=30.0)

    with capsys.disabled():
        sys.stderr.write(
            f"\nMANUAL STEP: close the console window of process {popen.pid} "
            "(the new window titled with python.exe). Waiting up to 120 s.\n"
        )
    closed_at = time.monotonic()
    try:
        popen.wait(timeout=120)
    except subprocess.TimeoutExpired:
        popen.kill()
        pytest.fail("the console window was not closed within 120 s")
    assert time.monotonic() - closed_at < 120
    assert wait_until_gone(pids, CONSOLE_CLOSE_GRACE_S + TREE_MARGIN_S) == []
