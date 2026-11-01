"""Crash and restart against a real `python -m local_agent` (task 22.17).

`test_crash_restart` runs 20 seeded iterations (one pytest case per seed). Each one:

1. Completes one Task, so a terminal Task exists before the crash.
2. Starts one or two streaming Tasks (each in its own Context) against the OpenAI-compatible LLM
   stub and the echo MCP fixture, and drives each to one of three points: mid-token-stream (the stub
   holds the stream after two deltas), mid-tool-call (a slow tool call is in flight next to a fast
   one), or `input-required` (a Protected_Tool waits for approval next to a fast tool call).
3. Kills the Backend hard (`Popen.kill()`: `SIGKILL` on POSIX, `TerminateProcess` on Windows) and
   waits for the process handle, so the OS has released the Storage_Lock (18.27).
4. Restarts on the same Storage_Directory and checks: startup needs no cleanup, interrupted Tasks
   are `failed` with the recovery message, the earlier terminal Task is unchanged, each persisted
   artifact is a prefix of the text the client received, cancel on a recovered Task is refused,
   no LLM request and no Tool_Call happen before the first client request, and a new message with
   the old `context_id` sends the persisted history plus the new message to the LLM.

`test_crash_removes_mcp_tree` checks that the Process_Tree of an MCP_Server does not outlive a
hard-killed Backend. On POSIX the server sees EOF on its stdin; on Windows the job object closes.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from integration_harness import (
    FIXTURES_DIR,
    PIDS,
    BackendLauncher,
    BackendProcess,
    FakeLlmServer,
    StubToolCall,
    StubTurn,
    echo_server_entry,
    jsonrpc_call,
    llm_tool_name,
    message_params,
    read_request_log,
    wait_for_task_state,
    wait_until_gone,
)
from local_agent.storage.recovery import RECOVERY_STATUS, RECOVERY_TOOL_RESULT

# Requirements: 18.6, 18.7, 18.9, 18.12, 18.15, 18.16, 18.17, 18.18, 18.27

SEEDS: Final = range(20)
POINTS: Final = ("stream", "tool", "approval")
COMPLETED: Final = "TASK_STATE_COMPLETED"
FAILED: Final = "TASK_STATE_FAILED"
WORKING: Final = "TASK_STATE_WORKING"
INPUT_REQUIRED: Final = "TASK_STATE_INPUT_REQUIRED"
TASK_NOT_CANCELABLE: Final = -32002
SSE_HEADERS: Final = {"A2A-Version": "1.0", "Accept": "text/event-stream"}
WAIT_S: Final = 20.0
ORPHAN_WAIT_S: Final = 10.0
SLOW_TOOL_S: Final = 30
"""How long the in-flight tool call would run. A leaked MCP_Server ends on its own after that."""
SPAWNING_SERVER: Final = FIXTURES_DIR / "mcp_spawning_server.py"

STREAM_CHUNKS: Final = ("alpha ", "beta ", "gamma ")
STREAM_RECEIVED: Final = "alpha beta "
TOOL_TEXT: Final = "Checking. "
APPROVAL_TEXT: Final = "Need to write. "


# --- Reading a stream and the stored Tasks -------------------------------------------------------


class StreamCollector(threading.Thread):
    """One `SendStreamingMessage` call; records the Task, its states, and the text received."""

    def __init__(self, backend: BackendProcess, text: str) -> None:
        super().__init__(name="crash-sse", daemon=True)
        self._backend = backend
        self._text = text
        self.task_id: str | None = None
        self.context_id: str | None = None
        self.states: list[str] = []
        self.artifacts: dict[str, str] = {}
        self.error: BaseException | None = None
        self._changed = threading.Condition()

    @property
    def received(self) -> str:
        """The artifact text received so far, artifacts in the order they started."""
        with self._changed:
            return "".join(self.artifacts.values())

    def wait_for(self, predicate: Callable[[], bool], what: str, timeout_s: float = WAIT_S) -> None:
        with self._changed:
            if not self._changed.wait_for(predicate, timeout_s):
                raise TimeoutError(
                    f"timed out waiting for {what}; states={self.states}, "
                    f"artifacts={self.artifacts}, error={self.error!r}"
                )

    def run(self) -> None:
        envelope = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendStreamingMessage",
            "params": message_params(self._text),
        }
        try:
            with (
                httpx.Client(base_url=self._backend.url, timeout=60.0) as client,
                client.stream("POST", "/", json=envelope, headers=SSE_HEADERS) as response,
            ):
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        document = json.loads(line[len("data:") :])
                        if "result" in document:
                            self._record(document["result"])
        except BaseException as exc:  # the kill ends the connection abruptly; that is expected
            self.error = exc

    def _record(self, result: Mapping[str, Any]) -> None:
        with self._changed:
            if "task" in result:
                task = result["task"]
                self.task_id = str(task["id"])
                self.context_id = str(task["contextId"])
                self.states.append(str(task["status"]["state"]))
            elif "statusUpdate" in result:
                self.states.append(str(result["statusUpdate"]["status"]["state"]))
            elif "artifactUpdate" in result:
                update = result["artifactUpdate"]
                artifact = update["artifact"]
                chunk = "".join(part.get("text", "") for part in artifact["parts"])
                key = str(artifact["artifactId"])
                self.artifacts[key] = (
                    self.artifacts.get(key, "") + chunk if update.get("append") else chunk
                )
            self._changed.notify_all()


def stored_text(task: Mapping[str, Any]) -> str:
    """The text of all artifacts of a stored Task, in order."""
    return "".join(
        part.get("text", "") for artifact in task.get("artifacts", []) for part in artifact["parts"]
    )


def status_text(task: Mapping[str, Any]) -> str:
    message = task["status"].get("message") or {}
    return "".join(part.get("text", "") for part in message.get("parts", []))


def get_task(client: httpx.Client, task_id: str) -> dict[str, Any]:
    document = jsonrpc_call(client, "GetTask", {"id": task_id})
    assert "error" not in document, document
    result: dict[str, Any] = document["result"]
    return result


def call_tool_entries(log: Path) -> list[dict[str, Any]]:
    return [entry for entry in read_request_log(log) if entry["event"] == "call_tool"]


# --- The scenario --------------------------------------------------------------------------------


@dataclass
class Driven:
    """A Task that the test drove to a point before the crash."""

    point: str
    text: str
    collector: StreamCollector
    task_id: str
    context_id: str
    received: str
    unanswered: tuple[str, ...]
    """Tool_Call IDs that cannot have a real Tool_Result when the Backend dies."""


@dataclass
class Rig:
    backend: BackendProcess
    llm: FakeLlmServer
    log: Path
    gates: list[threading.Event] = field(default_factory=list)
    driven: list[Driven] = field(default_factory=list)


def turn_for(rig: Rig, point: str, tag: str) -> tuple[StubTurn, tuple[str, ...]]:
    """The LLM turn that leads a Task to `point`, and the Tool_Call IDs it leaves unanswered."""
    echo_call = StubToolCall(
        llm_tool_name("echo", "echo"), json.dumps({"text": f"early-{tag}"}), f"call-echo-{tag}"
    )
    if point == "stream":
        gate = threading.Event()  # never set: only the crash (or teardown) ends the stream
        rig.gates.append(gate)
        return StubTurn(chunks=STREAM_CHUNKS, hold_after=2, gate=gate), ()
    if point == "tool":
        slow_args = json.dumps({"seconds": SLOW_TOOL_S, "text": f"late-{tag}"})
        slow_call = StubToolCall(llm_tool_name("echo", "slow"), slow_args, f"call-slow-{tag}")
        return StubTurn(chunks=(TOOL_TEXT,), tool_calls=(slow_call, echo_call)), (
            slow_call.call_id,
        )
    write_args = json.dumps({"path": f"{tag}.txt", "content": "data"})
    write_call = StubToolCall(llm_tool_name("echo", "write_file"), write_args, f"call-write-{tag}")
    return StubTurn(chunks=(APPROVAL_TEXT,), tool_calls=(echo_call, write_call)), (
        write_call.call_id,
    )


def reach_point(rig: Rig, driven: Driven, tag: str) -> None:
    """Wait until the Task is at its point and the client holds all the text it will ever get."""
    collector = driven.collector
    if driven.point == "stream":
        collector.wait_for(lambda: collector.received == STREAM_RECEIVED, "two streamed deltas")
        return
    expected = TOOL_TEXT if driven.point == "tool" else APPROVAL_TEXT
    collector.wait_for(lambda: collector.received == expected, "the assistant text")
    if driven.point == "tool":
        deadline = time.monotonic() + WAIT_S
        while not any(
            entry["name"] == "slow" and entry["arguments"].get("text") == f"late-{tag}"
            for entry in call_tool_entries(rig.log)
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("the slow tool call never reached the MCP server")
            time.sleep(0.05)
        return
    with rig.backend.client() as client:
        wait_for_task_state(client, driven.task_id, [INPUT_REQUIRED], timeout_s=WAIT_S)


def start_driven(rig: Rig, point: str, tag: str) -> Driven:
    """Send one streaming message whose LLM turn leads to `point`, and wait for the point."""
    turn, unanswered = turn_for(rig, point, tag)
    wanted = len(rig.llm.requests) + 1
    rig.llm.turns.append(turn)
    text = f"user text {tag}"
    collector = StreamCollector(rig.backend, text)
    collector.start()
    rig.llm.wait_for_requests(wanted, WAIT_S)
    collector.wait_for(lambda: collector.task_id is not None, "the Task snapshot")
    assert collector.task_id is not None
    assert collector.context_id is not None
    driven = Driven(point, text, collector, collector.task_id, collector.context_id, "", unanswered)
    reach_point(rig, driven, tag)
    driven.received = collector.received
    rig.driven.append(driven)
    return driven


def message_texts(messages: Sequence[Mapping[str, Any]], role: str) -> list[str]:
    def flat(content: Any) -> str:
        if isinstance(content, str):
            return content
        return "".join(str(part.get("text", "")) for part in content or [])

    return [flat(message.get("content")) for message in messages if message["role"] == role]


def assert_history_integrity(messages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Every Tool_Call has a later Tool_Result (11.5). Returns the result text by Tool_Call ID."""
    results: dict[str, str] = {}
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            later = [
                m
                for m in messages[index + 1 :]
                if m["role"] == "tool" and m["tool_call_id"] == call["id"]
            ]
            assert len(later) == 1, f"tool call {call['id']} has {len(later)} results: {messages}"
            results[str(call["id"])] = "".join(message_texts(later, "tool"))
    return results


# --- test_crash_restart --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_crash_restart(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer, tmp_path: Path, seed: int
) -> None:
    rng = random.Random(seed)
    # Seeds 0..19 cover each point as the first Task, and one and two Tasks, in every combination.
    points = [POINTS[seed % 3]]
    if (seed // 3) % 2:
        points.append(rng.choice(POINTS))

    log = tmp_path / "echo-requests.jsonl"
    backend = backend_launcher.launch(
        llm_base=llm_stub.base_url,
        mcp_servers=[echo_server_entry(request_log=log)],
        permissions={"echo": ["write_file"]},
    )
    rig = Rig(backend, llm_stub, log)
    try:
        # A Task that finished before the crash.
        llm_stub.turns.append(StubTurn(chunks=("fin", "ished")))
        with backend.client() as client:
            sent = jsonrpc_call(client, "SendMessage", message_params(f"finished {seed}"))
            finished_id = str(sent["result"]["task"]["id"])
            finished_before = wait_for_task_state(client, finished_id, [COMPLETED])
        assert stored_text(finished_before) == "finished"

        for number, point in enumerate(points):
            start_driven(rig, point, f"{seed}-{number}")

        mcp_pids = PIDS.record_from_log(log)
        assert mcp_pids, "the echo server never logged its start"
        requests_before = len(llm_stub.requests)
        tool_calls_before = len(call_tool_entries(log))

        backend.kill()
        assert backend.returncode is not None
        # The MCP_Server ends with its Backend: stdin closes (POSIX) or the job closes (Windows).
        survivors = wait_until_gone(mcp_pids, ORPHAN_WAIT_S)
        assert survivors == [], f"MCP server processes outlived the killed Backend: {survivors}"

        # Restart on the same Storage_Directory, with no cleanup in between (18.27).
        # The new process takes a new port: on macOS the Backend binds without SO_REUSEADDR (17.32),
        # so the killed process's connections in TIME_WAIT keep the old port busy.
        backend = backend_launcher.launch(
            llm_base=llm_stub.base_url,
            mcp_servers=[echo_server_entry(request_log=log)],
            permissions={"echo": ["write_file"]},
        )
        assert backend.running

        # Before the first client request: no LLM request, no Tool_Call (18.17).
        assert len(llm_stub.requests) == requests_before
        assert len(call_tool_entries(log)) == tool_calls_before
        recovered = [r for r in backend.log_records() if r.get("event") == "restart_recovery"]
        assert len(recovered) == 1, recovered
        assert recovered[0]["tasks_recovered"] == len(rig.driven)

        with backend.client() as client:
            for driven in rig.driven:
                stored = get_task(client, driven.task_id)
                assert stored["contextId"] == driven.context_id
                assert stored["status"]["state"] == FAILED, stored
                assert status_text(stored) == RECOVERY_STATUS
                persisted = stored_text(stored)
                assert driven.received.startswith(persisted), (
                    f"persisted artifact text {persisted!r} is not a prefix of "
                    f"the received text {driven.received!r}"
                )

                canceled = jsonrpc_call(client, "CancelTask", {"id": driven.task_id})
                assert canceled["error"]["code"] == TASK_NOT_CANCELABLE, canceled
                assert get_task(client, driven.task_id) == stored  # the refusal changed nothing

            assert get_task(client, finished_id) == finished_before

            # Recovery and these reads sent nothing to the LLM or an MCP_Server (18.17).
            assert len(llm_stub.requests) == requests_before
            assert len(call_tool_entries(log)) == tool_calls_before

            check_resumed_history(client, rig.driven[0], llm_stub, seed)

        assert backend.stop() == 0
    finally:
        for gate in rig.gates:
            gate.set()
        backend.close()


def check_resumed_history(
    client: httpx.Client, driven: Driven, llm: FakeLlmServer, seed: int
) -> None:
    """A message with the old `context_id` sends the persisted history plus the new message."""
    llm.script(StubTurn(chunks=("resumed",)))
    new_text = f"new message {seed}"
    sent = jsonrpc_call(
        client, "SendMessage", message_params(new_text, context_id=driven.context_id)
    )
    assert "error" not in sent, sent
    task = sent["result"]["task"]
    assert task["contextId"] == driven.context_id
    assert task["id"] != driven.task_id
    assert wait_for_task_state(client, str(task["id"]), [COMPLETED])["contextId"] == (
        driven.context_id
    )

    first = llm.wait_for_requests(1, WAIT_S)[0]["messages"]
    users = message_texts(first, "user")
    assert users[-1] == new_text
    assert first[-1]["role"] == "user"
    assert users.count(driven.text) == 1, users

    results = assert_history_integrity(first)
    for call_id in driven.unanswered:
        assert results[call_id] == RECOVERY_TOOL_RESULT
    answered_early = [call_id for call_id in results if call_id.startswith("call-echo-")]
    for call_id in answered_early:  # the fast call either finished or was answered by recovery
        assert results[call_id].startswith("early-") or results[call_id] == RECOVERY_TOOL_RESULT
    assert set(results) == set(driven.unanswered) | set(answered_early)


# --- test_crash_removes_mcp_tree -----------------------------------------------------------------


def wait_for_tree(pid_file: Path, timeout_s: float = WAIT_S) -> list[int]:
    """Wait until the spawning server has published its server, child, and grandchild pids."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            data = json.loads(pid_file.read_text(encoding="utf-8"))
        except OSError, ValueError:
            data = {}
        if "server" in data and len(data.get("descendants", [])) == 2:
            pids = [int(data["server"]), *(int(pid) for pid in data["descendants"])]
            PIDS.record(*pids)
            return pids
        time.sleep(0.05)
    raise TimeoutError("the MCP server did not publish its process tree")


def test_crash_removes_mcp_tree(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer, tmp_path: Path
) -> None:
    pid_file = tmp_path / "tree.json"
    entry = {
        "name": "spawn",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SPAWNING_SERVER), "--cooperative", "--pid-file", str(pid_file)],
    }
    backend = backend_launcher.launch(llm_base=llm_stub.base_url, mcp_servers=[entry])
    pids = wait_for_tree(pid_file)
    assert all(pid != backend.pid for pid in pids)

    backend.kill()
    survivors = wait_until_gone(pids, ORPHAN_WAIT_S)
    assert survivors == [], f"MCP process tree outlived the killed Backend: {survivors}"

    restarted = backend_launcher.launch(llm_base=llm_stub.base_url, mcp_servers=[entry])
    assert restarted.stop() == 0
