"""Opt-in test of the prompt-based function-calling fallback (task 22.9, requirement 10.8).

These tests talk to a real, local Ollama, so they carry `@pytest.mark.llm` and CI skips them
(`-m "not manual and not llm"`). Run them with
`pytest -m llm tests/integration/test_llm_fallback.py`.

`llm.native_function_calling: false` makes the Backend set `litellm.add_function_to_prompt`, so
LiteLLM writes the tool definitions into the prompt and parses the model's reply back into tool
calls. The tests check that such a reply reaches the Orchestrator as an assembled Tool_Call (an
id, the tool name, and JSON-text arguments), the same shape a model with native support gives.

Settings (environment variables):

* `LOCAL_AGENT_TEST_OLLAMA_URL`: Ollama address, default `http://localhost:11434`.
* `LOCAL_AGENT_TEST_OLLAMA_MODEL`: model name as `ollama list` shows it, default `gemma3n:latest`
  (a model without native tool support). A test skips when Ollama or the model is not there.

A small model can ignore the instruction, so the prompts are explicit and the temperature is 0.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Final

import httpx
import litellm
import pytest

from integration_harness import (
    A2A_HEADERS,
    PIDS,
    AppHandle,
    config_dict,
    echo_server_entry,
    message_params,
    open_app,
    read_request_log,
    wait_until_gone,
)
from local_agent.config.models import BackendConfig, LLMConfig
from local_agent.llm.gateway import LlmGateway, apply_function_calling_mode
from local_agent.models import ToolMessage

pytestmark = [pytest.mark.integration, pytest.mark.llm]

OLLAMA_URL: Final = os.environ.get("LOCAL_AGENT_TEST_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: Final = os.environ.get("LOCAL_AGENT_TEST_OLLAMA_MODEL", "gemma3n:latest")
REPLY_TIMEOUT_S: Final = 240.0
TERMINAL: Final = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED"}
MARKER: Final = "ping-42"

ECHO_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Return the text unchanged.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


@pytest.fixture(autouse=True)
def ollama_model() -> None:
    """Skip the test unless the local Ollama answers and has the model."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        response.raise_for_status()
        names = {str(entry.get("name")) for entry in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        pytest.skip(f"no Ollama at {OLLAMA_URL}: {exc}")
    if OLLAMA_MODEL not in names:
        pytest.skip(f"Ollama at {OLLAMA_URL} has no model {OLLAMA_MODEL!r}; it has {sorted(names)}")


@pytest.fixture(autouse=True)
def restore_litellm_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`apply_function_calling_mode` sets a process-global flag; put it back after the test."""
    monkeypatch.setattr(litellm, "add_function_to_prompt", False)


def llm_overrides() -> dict[str, Any]:
    return {
        "llm": {
            "model": f"ollama/{OLLAMA_MODEL}",
            "api_base": OLLAMA_URL,
            "temperature": 0,
            "max_output_tokens": 512,
            "request_timeout_s": REPLY_TIMEOUT_S,
            "native_function_calling": False,
        }
    }


async def test_prompt_based_tool_call_is_assembled_like_a_native_one() -> None:
    cfg = LLMConfig.model_validate(llm_overrides()["llm"])
    apply_function_calling_mode(cfg)
    assert litellm.add_function_to_prompt is True
    prompt = f"Call the echo tool with the text {MARKER!r}. Reply with the tool call only."
    messages = [{"role": "user", "content": prompt}]

    stream = await LlmGateway(cfg).stream(messages, [ECHO_TOOL])
    async for _text in stream:
        pass
    response = stream.result()

    assert response.tool_calls, f"the model made no tool call; it said {response.text!r}"
    call = response.tool_calls[0]
    assert call.id
    assert call.name == "echo"
    arguments = json.loads(call.arguments)
    assert isinstance(arguments, dict)
    assert MARKER in str(arguments.get("text"))


@pytest.fixture
def echo_log(tmp_path: Path, storage_dir: Path) -> Path:
    """The echo server's request log. The memory backend never creates the Storage_Directory,
    which is the MCP_Server's working directory, so it is made here."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path / "echo-requests.jsonl"


async def test_task_runs_an_echo_tool_through_the_fallback(
    storage_dir: Path, echo_log: Path
) -> None:
    log = echo_log
    config = BackendConfig.model_validate(
        config_dict(
            storage_dir=storage_dir,
            storage_backend="memory",
            mcp_servers=[echo_server_entry(request_log=log)],
            overrides=llm_overrides(),
        )
    )
    apply_function_calling_mode(config.llm)

    async with open_app(config, storage_dir) as app:
        task = await run_task(
            app,
            f"Use the echo tool with the text {MARKER!r}, then tell me what it returned.",
        )
        assert task["status"]["state"] == "TASK_STATE_COMPLETED", task
        _, history = await app.deps.session_store.get_history(task["contextId"])

    pids = PIDS.record_from_log(log)
    assert await asyncio.to_thread(wait_until_gone, pids, 5.0) == []

    calls = [entry for entry in read_request_log(log) if entry["event"] == "call_tool"]
    assert calls, "the model never called a tool"
    assert calls[0]["name"] == "echo"
    results = [message for message in history if isinstance(message, ToolMessage)]
    assert results, "the Message_History holds no Tool_Result"
    assert MARKER in results[0].content


async def run_task(app: AppHandle, text: str) -> dict[str, Any]:
    """Send one message over JSON-RPC and poll until its Task is terminal. Returns the Task."""
    envelope = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": message_params(text)}
    response = await app.client.post("/", json=envelope, headers=A2A_HEADERS)
    assert response.status_code == 200, response.text
    task: dict[str, Any] = response.json()["result"]["task"]
    async with asyncio.timeout(REPLY_TIMEOUT_S):
        while task["status"]["state"] not in TERMINAL:
            await asyncio.sleep(0.25)
            poll = {"jsonrpc": "2.0", "id": 2, "method": "GetTask", "params": {"id": task["id"]}}
            reply = await app.client.post("/", json=poll, headers=A2A_HEADERS)
            task = reply.json()["result"]
    return task
