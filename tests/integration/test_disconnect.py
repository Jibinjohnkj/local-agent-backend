"""SSE disconnect over a real socket (task 22.7, Requirements 3.6 and 18.29).

The production app is served by uvicorn on a free port and the LLM is the HTTP stub of
`FakeLlmServer`. The stub streams two text deltas, then holds the response until the test sets a
gate. The test reads the SSE stream until the first Token_Event arrives, drops the connection, and
checks that the Backend keeps the Task alive: get-task shows it still `working`, then, once the gate
opens, `completed` with the full assistant text as the artifact. On SQLite the Task and its artifact
also survive a Backend restart on the same Storage_Directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import httpx
import pytest

from integration_harness import (
    A2A_HEADERS,
    AppHandle,
    FakeLlmServer,
    StubTurn,
    message_params,
)

type StorageBackend = Literal["memory", "sqlite"]
type AppOpener = Callable[..., contextlib.AbstractAsyncContextManager[AppHandle]]

pytestmark = pytest.mark.integration

CHUNKS: Final = ("alpha ", "beta ", "gamma ", "delta")
HELD_AFTER: Final = 2
TIMEOUT_S: Final = 20.0
WORKING: Final = "TASK_STATE_WORKING"
COMPLETED: Final = "TASK_STATE_COMPLETED"


async def get_task(client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
    envelope = {"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {"id": task_id}}
    response = await client.post("/", json=envelope, headers=A2A_HEADERS)
    assert response.status_code == 200, response.text
    document: dict[str, Any] = response.json()
    assert "error" not in document, document
    task: dict[str, Any] = document["result"]
    return task


async def wait_for_state(client: httpx.AsyncClient, task_id: str, state: str) -> dict[str, Any]:
    async with asyncio.timeout(TIMEOUT_S):
        while True:
            task = await get_task(client, task_id)
            if task["status"]["state"] == state:
                return task
            await asyncio.sleep(0.05)


def artifact_text(task: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "") for artifact in task.get("artifacts", []) for part in artifact["parts"]
    )


async def drop_stream_after_first_token(handle: AppHandle) -> str:
    """Open a streaming message, read until the first Token_Event, then close the socket.

    Returns the Task ID from the first event of the stream.
    """
    envelope = {
        "jsonrpc": "2.0",
        "id": "stream-1",
        "method": "SendStreamingMessage",
        "params": message_params("say something"),
    }
    task_id = ""
    async with asyncio.timeout(TIMEOUT_S):
        async with handle.client.stream("POST", "/", json=envelope, headers=A2A_HEADERS) as stream:
            assert stream.status_code == 200
            async for line in stream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                result = json.loads(line[len("data:") :])["result"]
                if "task" in result and not task_id:
                    task_id = result["task"]["id"]
                if "artifactUpdate" in result:
                    break  # leaving the block closes the connection mid-response
    assert task_id, "the first event of the stream did not carry the Task"
    return task_id


def stub_turns(gate: threading.Event) -> Sequence[StubTurn]:
    return (StubTurn(chunks=CHUNKS, hold_after=HELD_AFTER, gate=gate),)


async def test_disconnect_continues(
    app_factory: AppOpener, llm_stub: FakeLlmServer, storage_backend: StorageBackend
) -> None:
    # Feature: local-agent-backend, SSE disconnect; Validates: Requirements 3.6, 18.29
    gate = threading.Event()
    llm_stub.script(*stub_turns(gate))
    overrides = {"llm": {"api_base": llm_stub.base_url}}
    try:
        async with app_factory(
            storage_backend=storage_backend, transport="uvicorn", overrides=overrides
        ) as handle:
            task_id = await drop_stream_after_first_token(handle)

            # The client is gone and the LLM response is still held: the Task keeps working.
            await asyncio.sleep(0.3)
            held = await get_task(handle.client, task_id)
            assert held["status"]["state"] == WORKING
            assert llm_stub.disconnects == 0

            gate.set()
            done = await wait_for_state(handle.client, task_id, COMPLETED)
            assert artifact_text(done) == "".join(CHUNKS)
            assert len(llm_stub.requests) == 1

        if storage_backend == "sqlite":
            async with app_factory(
                storage_backend="sqlite", transport="uvicorn", overrides=overrides
            ) as restarted:
                again = await get_task(restarted.client, task_id)
                assert again["status"]["state"] == COMPLETED
                assert artifact_text(again) == "".join(CHUNKS)
    finally:
        gate.set()  # never leave the stub handler waiting if an assertion failed
