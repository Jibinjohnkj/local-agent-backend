"""Restart without a crash against a real `python -m local_agent` (task 22.20).

`tests/integration/test_crash_restart.py` covers a hard kill. These tests stop the Backend with
Graceful_Shutdown (exit status 0) and start it again on the same Storage_Directory:

* `test_restart_keeps_tasks_and_context`: a completed Task (with a tool call, so two artifacts), a
  Task canceled by the client, and a Task that waited for approval (Graceful_Shutdown cancels it)
  read back with the state, status message, and artifacts they had; Restart_Recovery reports that
  it recovered nothing; nothing reaches the LLM or an MCP_Server before the first client request;
  a new message with the old `context_id` and no `task_id` sends the persisted history plus the new
  message as the first LLM request.
* `test_restart_retention_sweep`: with `session_idle_timeout_s` set, the sweep at startup deletes a
  Context whose Last_Activity was moved into the past, keeps a Context with a recent Last_Activity,
  and keeps an old Context that holds an Active_Task. `GetTask` for a Task of the deleted Context
  returns `TaskNotFoundError`, and a new message with its `context_id` starts an empty history.

The tests change the stored rows only while the Backend is stopped.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, Final

import httpx

from integration_harness import (
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
)
from local_agent.storage.recovery import RECOVERY_STATUS

# Requirements: 2.13, 3.6, 11.11, 11.12, 11.14, 18.6, 18.7

COMPLETED: Final = "TASK_STATE_COMPLETED"
CANCELED: Final = "TASK_STATE_CANCELED"
FAILED: Final = "TASK_STATE_FAILED"
INPUT_REQUIRED: Final = "TASK_STATE_INPUT_REQUIRED"
TASK_NOT_FOUND: Final = -32001
WAIT_S: Final = 20.0
DATABASE_FILE: Final = "local-agent.db"
IDLE_TIMEOUT_S: Final = 3600
"""`orchestration.session_idle_timeout_s` of the retention test."""
ONE_DAY_MS: Final = 24 * 3600 * 1000


# --- Helpers -------------------------------------------------------------------------------------


def get_task(client: httpx.Client, task_id: str) -> dict[str, Any]:
    document = jsonrpc_call(client, "GetTask", {"id": task_id})
    assert "error" not in document, document
    result: dict[str, Any] = document["result"]
    return result


def send(
    client: httpx.Client, text: str, *, context_id: str | None = None, immediate: bool = False
) -> dict[str, Any]:
    """`SendMessage` (no `task_id`); returns the Task of the reply.

    With `immediate` the call returns the new Task at once instead of waiting for the turn.
    """
    params = message_params(text, context_id=context_id)
    if immediate:
        params["configuration"] = {"returnImmediately": True}
    document = jsonrpc_call(client, "SendMessage", params)
    assert "error" not in document, document
    task: dict[str, Any] = document["result"]["task"]
    return task


def stored_text(task: Mapping[str, Any]) -> str:
    return "".join(
        part.get("text", "") for artifact in task.get("artifacts", []) for part in artifact["parts"]
    )


def flat(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(part.get("text", "")) for part in content or [])


def conversation(messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """`(role, text)` of the non-system messages of an LLM request."""
    return [(str(m["role"]), flat(m.get("content"))) for m in messages if m["role"] != "system"]


def first_llm_messages(llm: FakeLlmServer) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = llm.wait_for_requests(1, WAIT_S)[0]["messages"]
    return messages


def restart(launcher: BackendLauncher, backend: BackendProcess, **launch: Any) -> BackendProcess:
    """Graceful_Shutdown, then a new Backend on the same Storage_Directory.

    The new process takes a new port: the Backend binds without `SO_REUSEADDR` (17.32), so on macOS
    the connections of the stopped process in TIME_WAIT keep its old port busy.
    """
    assert backend.stop() == 0
    return launcher.launch(**launch)


def log_events(backend: BackendProcess, event: str) -> list[dict[str, Any]]:
    return [record for record in backend.log_records() if record.get("event") == event]


def edit_database(storage_dir: Path, *statements: tuple[str, tuple[Any, ...]]) -> None:
    """Run SQL against the Storage_Database while no Backend uses it."""
    with closing(sqlite3.connect(storage_dir / DATABASE_FILE)) as conn:
        for sql, parameters in statements:
            conn.execute(sql, parameters)
        conn.commit()


# --- test_restart_keeps_tasks_and_context --------------------------------------------------------


def test_restart_keeps_tasks_and_context(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer, tmp_path: Path
) -> None:
    log = tmp_path / "echo-requests.jsonl"
    launch: dict[str, Any] = {
        "llm_base": llm_stub.base_url,
        "mcp_servers": [echo_server_entry(request_log=log)],
        "permissions": {"echo": ["write_file"]},
    }
    gate = threading.Event()  # never set by the test body: only teardown releases the stream
    backend = backend_launcher.launch(**launch)
    try:
        with backend.client() as client:
            # A completed Task whose answer has two artifacts (before and after a tool call).
            echo_call = StubToolCall(
                llm_tool_name("echo", "echo"), json.dumps({"text": "early-A"}), "call-echo-A"
            )
            llm_stub.script(
                StubTurn(chunks=("Checking. ",), tool_calls=(echo_call,)),
                StubTurn(chunks=("done ", "ok")),
            )
            text_a = "user text A"
            sent_a = send(client, text_a)
            task_a, context_a = str(sent_a["id"]), str(sent_a["contextId"])
            completed_before = wait_for_task_state(client, task_a, [COMPLETED])
            assert len(completed_before["artifacts"]) == 2, completed_before
            assert stored_text(completed_before) == "Checking. done ok"

            # A Task the client canceled while it streamed.
            llm_stub.script(StubTurn(chunks=("alpha ", "beta ", "gamma "), hold_after=2, gate=gate))
            text_b = "user text B"
            chunks_before = llm_stub.chunks_sent
            sent_b = send(client, text_b, immediate=True)
            task_b, context_b = str(sent_b["id"]), str(sent_b["contextId"])
            llm_stub.wait_for_chunks(chunks_before + 2, WAIT_S)
            canceled = jsonrpc_call(client, "CancelTask", {"id": task_b})
            assert "error" not in canceled, canceled
            canceled_before = wait_for_task_state(client, task_b, [CANCELED])

            # A Task that waits for approval of a Protected_Tool when the Backend is stopped.
            write_args = json.dumps({"path": "c.txt", "content": "data"})
            write_call = StubToolCall(llm_tool_name("echo", "write_file"), write_args, "call-write")
            llm_stub.script(StubTurn(chunks=("Need to write. ",), tool_calls=(write_call,)))
            sent_c = send(client, "user text C", immediate=True)
            task_c = str(sent_c["id"])
            waiting_before = wait_for_task_state(client, task_c, [INPUT_REQUIRED])
            assert stored_text(waiting_before) == "Need to write. "

        # The Backend does not call the LLM or an MCP_Server on its own after the restart (18.17).
        llm_requests = len(llm_stub.requests)
        tool_calls = len([e for e in read_request_log(log) if e["event"] == "call_tool"])
        backend = restart(backend_launcher, backend, **launch)
        assert backend.running
        assert len(log_events(backend, "restart_recovery")) == 1
        assert log_events(backend, "restart_recovery")[0]["tasks_recovered"] == 0
        assert log_events(backend, "task_recovered") == []
        assert len(llm_stub.requests) == llm_requests

        with backend.client() as client:
            # Terminal Tasks read back exactly: state, status message, and artifacts in order.
            assert get_task(client, task_a) == completed_before
            assert get_task(client, task_b) == canceled_before
            # The Task that waited for approval was canceled by Graceful_Shutdown, not failed by
            # recovery, and kept the text it had produced.
            waiting_after = get_task(client, task_c)
            assert waiting_after["status"]["state"] == CANCELED, waiting_after
            assert stored_text(waiting_after) == stored_text(waiting_before)
            assert RECOVERY_STATUS not in json.dumps(waiting_after)

            assert len(llm_stub.requests) == llm_requests
            assert len([e for e in read_request_log(log) if e["event"] == "call_tool"]) == (
                tool_calls
            )

            # A new message with the old context_id and no task_id continues it (18.6).
            llm_stub.script(StubTurn(chunks=("resumed",)))
            new_a = "new message A"
            continued = send(client, new_a, context_id=context_a)
            assert continued["contextId"] == context_a
            assert continued["id"] != task_a
            wait_for_task_state(client, str(continued["id"]), [COMPLETED])
            messages = first_llm_messages(llm_stub)
            roles = [role for role, _ in conversation(messages)]
            assert roles == ["user", "assistant", "tool", "assistant", "user"], messages
            history = conversation(messages)
            assert history[0] == ("user", text_a)
            assert history[2][1].startswith("early-A")
            assert history[-1] == ("user", new_a)
            plain = [m for m in messages if m["role"] != "system"]
            assert plain[1]["tool_calls"][0]["id"] == "call-echo-A"
            assert plain[2]["tool_call_id"] == "call-echo-A"

            # The same for the canceled Task's Context: its user message is still in the history.
            llm_stub.script(StubTurn(chunks=("resumed again",)))
            new_b = "new message B"
            again = send(client, new_b, context_id=context_b)
            assert again["contextId"] == context_b
            wait_for_task_state(client, str(again["id"]), [COMPLETED])
            users = [
                text for role, text in conversation(first_llm_messages(llm_stub)) if role == "user"
            ]
            assert users == [text_b, new_b]

        assert backend.stop() == 0
    finally:
        gate.set()
        backend.close()


# --- test_restart_retention_sweep ----------------------------------------------------------------


def test_restart_retention_sweep(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer
) -> None:
    launch: dict[str, Any] = {
        "llm_base": llm_stub.base_url,
        "overrides": {"orchestration": {"session_idle_timeout_s": IDLE_TIMEOUT_S}},
    }
    backend = backend_launcher.launch(**launch)
    try:
        texts = {"old": "old context text", "fresh": "fresh context text", "live": "live text"}
        tasks: dict[str, str] = {}
        contexts: dict[str, str] = {}
        before: dict[str, dict[str, Any]] = {}
        with backend.client() as client:
            for name, text in texts.items():
                llm_stub.script(StubTurn(chunks=(f"answer {name}",)))
                sent = send(client, text)
                tasks[name], contexts[name] = str(sent["id"]), str(sent["contextId"])
                before[name] = wait_for_task_state(client, tasks[name], [COMPLETED])

        assert backend.stop() == 0

        # While the Backend is stopped: "old" and "live" were last active a day ago, and "live"
        # holds an Active_Task (a Task a stopped process left running).
        long_ago = int(time.time() * 1000) - ONE_DAY_MS
        edit_database(
            backend_launcher.storage_dir,
            (
                "UPDATE contexts SET last_activity_ms = ? WHERE context_id = ?",
                (long_ago, contexts["old"]),
            ),
            (
                "UPDATE contexts SET last_activity_ms = ? WHERE context_id = ?",
                (long_ago, contexts["live"]),
            ),
            ("UPDATE tasks SET state = 'input-required' WHERE task_id = ?", (tasks["live"],)),
        )

        backend = backend_launcher.launch(**launch)
        requests_before = len(llm_stub.requests)

        # Recovery runs first and finalizes the Active_Task; the sweep then deletes only "old".
        assert log_events(backend, "restart_recovery")[0]["tasks_recovered"] == 1
        sweeps = log_events(backend, "retention_sweep")
        assert [record["deleted_contexts"] for record in sweeps] == [1], sweeps

        with backend.client() as client:
            gone = jsonrpc_call(client, "GetTask", {"id": tasks["old"]})
            assert gone["error"]["code"] == TASK_NOT_FOUND, gone
            assert get_task(client, tasks["fresh"]) == before["fresh"]
            live = get_task(client, tasks["live"])
            assert live["contextId"] == contexts["live"]
            assert live["status"]["state"] == FAILED, live
            assert RECOVERY_STATUS in json.dumps(live)
            assert len(llm_stub.requests) == requests_before

            # The deleted Context starts over: its old text is not in the history.
            llm_stub.script(StubTurn(chunks=("fresh start",)))
            reborn = send(client, "hello again", context_id=contexts["old"])
            assert reborn["contextId"] == contexts["old"]
            wait_for_task_state(client, str(reborn["id"]), [COMPLETED])
            assert conversation(first_llm_messages(llm_stub)) == [("user", "hello again")]

            # The kept Context still has its history.
            llm_stub.script(StubTurn(chunks=("still here",)))
            kept = send(client, "hello live", context_id=contexts["live"])
            wait_for_task_state(client, str(kept["id"]), [COMPLETED])
            users = [
                text for role, text in conversation(first_llm_messages(llm_stub)) if role == "user"
            ]
            assert users == [texts["live"], "hello live"]

        assert backend.stop() == 0
    finally:
        backend.close()
