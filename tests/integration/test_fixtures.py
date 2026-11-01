"""Smoke tests for the integration fixtures of task 22.1.

Each test uses one fixture end to end, so the later integration tests (22.3 onward) can rely on
them: the echo MCP server, the scripted LLM in process, the app factory over both storage backends
and over a real uvicorn socket, the HTTP LLM stub, and the Backend subprocess launcher.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from fakes import LlmTurn, text_delta
from integration_harness import (
    A2A_HEADERS,
    ECHO_SERVER,
    PIDS,
    AppHandle,
    BackendLauncher,
    FakeLlmServer,
    ScriptedCompletion,
    StubTurn,
    echo_server_entry,
    jsonrpc_call,
    kill_pid,
    message_params,
    pid_alive,
    read_request_log,
    wait_for_task_state,
    wait_until_gone,
)

pytestmark = pytest.mark.integration

COMPLETED = "TASK_STATE_COMPLETED"


async def run_to_completion(client: httpx.AsyncClient, text: str) -> dict[str, Any]:
    """Send one message over JSON-RPC and poll until its Task is terminal. Returns the Task."""
    envelope = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": message_params(text)}
    response = await client.post("/", json=envelope, headers=A2A_HEADERS)
    assert response.status_code == 200, response.text
    task: dict[str, Any] = response.json()["result"]["task"]
    async with asyncio.timeout(20):
        while task["status"]["state"] not in {COMPLETED, "TASK_STATE_FAILED"}:
            await asyncio.sleep(0.05)
            poll = {"jsonrpc": "2.0", "id": 2, "method": "GetTask", "params": {"id": task["id"]}}
            task = (await client.post("/", json=poll, headers=A2A_HEADERS)).json()["result"]
    return task


async def test_echo_server_over_stdio(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    params = StdioServerParameters(
        command=echo_command()[0], args=[*echo_command()[1:], "--request-log", str(log)]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names: list[str] = []
        pages = 0
        cursor = None
        while True:
            page = await session.list_tools(params=None if cursor is None else _page_params(cursor))
            pages += 1
            names += [tool.name for tool in page.tools]
            cursor = page.next_cursor
            if cursor is None:
                break
        assert pages > 1
        assert {"echo", "slow", "error", "hang", "write_file", "crash"} <= set(names)

        text = "naïve 日本語 🚀"
        echoed = await session.call_tool("echo", {"text": text})
        assert echoed.content[0].text == text  # type: ignore[union-attr]
        failed = await session.call_tool("error", {"message": "boom"})
        assert failed.is_error is True

    entries = read_request_log(log)
    assert entries[0]["event"] == "start"
    assert {entry["name"] for entry in entries if entry["event"] == "call_tool"} == {
        "echo",
        "error",
    }


def echo_command() -> list[str]:
    import sys

    return [sys.executable, str(ECHO_SERVER)]


def _page_params(cursor: str) -> Any:
    from mcp_types import PaginatedRequestParams

    return PaginatedRequestParams(cursor=cursor)


async def test_app_runs_a_task_over_both_stores(
    app: AppHandle, fake_llm: ScriptedCompletion
) -> None:
    fake_llm.script(LlmTurn(deltas=(text_delta("hel"), text_delta("lo"))))

    task = await run_to_completion(app.client, "hi")

    assert task["status"]["state"] == COMPLETED
    assert len(fake_llm.calls) == 1
    assert app.config.storage.path is not None


async def test_app_with_echo_server_lists_it_as_available(app_factory: Any, tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with app_factory(
        mcp_servers=[echo_server_entry(request_log=log)], permissions={"echo": ["write_file"]}
    ) as app:
        health = await app.client.get("/health")
        assert health.status_code == 200
        assert "echo" in health.text
    PIDS.record_from_log(log)
    pids = [entry["pid"] for entry in read_request_log(log)]
    assert wait_until_gone(pids, 5.0) == []


async def test_real_uvicorn_serves_health(app_factory: Any) -> None:
    async with app_factory(storage_backend="sqlite", transport="uvicorn") as app:
        assert app.base_url.startswith("http://127.0.0.1:")
        assert (await app.client.get("/health")).status_code == 200


def test_isolated_home_points_user_dirs_into_tmp(isolated_home: Path) -> None:
    assert os.environ["HOME"] == str(isolated_home)
    assert os.environ["LOCALAPPDATA"].startswith(str(isolated_home))
    assert os.environ["XDG_DATA_HOME"].startswith(str(isolated_home))


def test_pid_helpers() -> None:
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    PIDS.track_process(child)
    assert pid_alive(child.pid)
    kill_pid(child.pid)
    child.wait(timeout=10)
    assert not pid_alive(child.pid)


def test_llm_stub_streams_scripted_chunks(llm_stub: FakeLlmServer) -> None:
    llm_stub.script(StubTurn(chunks=("a", "b"), chunk_delay_s=0.01))
    body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]}

    with httpx.stream("POST", f"{llm_stub.base_url}/chat/completions", json=body) as response:
        lines = [line for line in response.iter_lines() if line.startswith("data:")]

    assert response.status_code == 200
    assert lines[-1] == "data: [DONE]"
    assert len(lines) == 4  # two text chunks, the finish chunk, [DONE]
    assert llm_stub.requests[0]["messages"][0]["content"] == "x"


def test_backend_process_serves_stops_and_restarts_on_the_same_storage(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer
) -> None:
    llm_stub.script(StubTurn(chunks=("hel", "lo")))
    backend = backend_launcher.launch(llm_base=llm_stub.base_url)

    with backend.client() as client:
        result = jsonrpc_call(client, "SendMessage", message_params("hi"))["result"]
        task = wait_for_task_state(client, result["task"]["id"])
    assert task["status"]["state"] == COMPLETED
    assert len(llm_stub.requests) == 1

    assert backend.stop() == 0  # Graceful_Shutdown ends with status 0 (17.14)
    database = backend_launcher.storage_dir / "local-agent.db"
    assert database.exists()

    backend.restart().wait_healthy()
    with backend.client() as client:
        assert wait_for_task_state(client, result["task"]["id"])["status"]["state"] == COMPLETED
    backend.kill()
    assert backend.returncode not in (None, 0)

    with sqlite3.connect(database) as connection:  # the file is readable after a hard kill
        assert connection.execute("select count(*) from sqlite_master").fetchone()[0] > 0
