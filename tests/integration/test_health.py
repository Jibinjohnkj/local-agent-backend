"""Health endpoint integration test (task 22.8, Requirements 15.8 and 15.9).

The production app runs over a real Session_Store with real stdio MCP_Servers: the echo fixture
(available), a child that exits before the handshake (unavailable), and a disabled echo entry. The
LLM is unreachable throughout: `llm.api_base` points at a closed port, and an in-process scripted
`litellm.acompletion` that fails every call stands behind it. `/health` must answer within 1 second
with every configured server listed in Backend_Config order, and must never contact the LLM.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal

import pytest

from fakes import LlmFailure, LlmTurn
from integration_harness import (
    PIDS,
    AppHandle,
    ScriptedCompletion,
    echo_server_entry,
    free_port,
    read_request_log,
)

type StorageBackend = Literal["memory", "sqlite"]
type AppOpener = Callable[..., contextlib.AbstractAsyncContextManager[AppHandle]]

pytestmark = pytest.mark.integration

HEALTH_BUDGET_S: Final = 1.0
UNREACHABLE_TURN: Final = LlmTurn(failure=LlmFailure.UNREACHABLE)


@pytest.fixture(autouse=True)
def _created_storage_dir(storage_dir: Path) -> None:
    """The memory backend never creates the Storage_Directory, but `open_app` runs MCP there."""
    storage_dir.mkdir(parents=True, exist_ok=True)


def broken_server_entry(name: str = "broken") -> dict[str, Any]:
    """A stdio server whose process exits before the handshake, so it never becomes available."""
    return {
        "name": name,
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-c", "import sys; sys.exit(3)"],
    }


def unreachable_llm_overrides() -> dict[str, Any]:
    """Config overrides that point the LLM at a TCP port nothing listens on."""
    return {"llm": {"api_base": f"http://127.0.0.1:{free_port()}/v1"}}


async def timed_health(handle: AppHandle) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    response = await handle.client.get("/health")
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body, elapsed


async def test_health_lists_every_server_within_a_second_while_llm_is_unreachable(
    app_factory: AppOpener,
    storage_backend: StorageBackend,
    tmp_path: Path,
    fake_llm: ScriptedCompletion,
) -> None:
    # Feature: local-agent-backend, health; Validates: Requirements 15.8, 15.9
    fake_llm.script(UNREACHABLE_TURN, UNREACHABLE_TURN, UNREACHABLE_TURN)
    log = tmp_path / "echo-requests.jsonl"
    servers = [
        echo_server_entry("echo", request_log=log),
        broken_server_entry("broken"),
        {**echo_server_entry("off"), "enabled": False},
    ]
    async with app_factory(
        storage_backend=storage_backend,
        mcp_servers=servers,
        overrides=unreachable_llm_overrides(),
    ) as app:
        PIDS.record_from_log(log)
        body, elapsed = await timed_health(app)

        assert elapsed < HEALTH_BUDGET_S
        assert body == {
            "status": "degraded",  # `broken` is enabled and unavailable (15.9)
            "mcp_servers": [
                {"name": "echo", "status": "available"},
                {"name": "broken", "status": "unavailable"},
                {"name": "off", "status": "unavailable"},  # disabled servers are listed too
            ],
        }
        assert fake_llm.calls == []  # answering a health request never contacts the LLM (15.8)

        # Repeated requests stay fast and unchanged, and still leave the LLM alone.
        for _ in range(3):
            again, elapsed = await timed_health(app)
            assert again == body
            assert elapsed < HEALTH_BUDGET_S
        assert fake_llm.calls == []
    PIDS.record_from_log(log)


async def test_health_is_ok_when_only_a_disabled_server_is_unavailable(
    app_factory: AppOpener,
    storage_backend: StorageBackend,
    tmp_path: Path,
    fake_llm: ScriptedCompletion,
) -> None:
    # Feature: local-agent-backend, health; Validates: Requirements 15.8, 15.9
    fake_llm.script(UNREACHABLE_TURN)
    log = tmp_path / "echo-requests.jsonl"
    servers = [
        echo_server_entry("echo", request_log=log),
        {**echo_server_entry("off"), "enabled": False},
    ]
    async with app_factory(
        storage_backend=storage_backend,
        mcp_servers=servers,
        overrides=unreachable_llm_overrides(),
    ) as app:
        PIDS.record_from_log(log)
        body, elapsed = await timed_health(app)

        assert elapsed < HEALTH_BUDGET_S
        assert body["status"] == "ok"  # a disabled server does not degrade the Backend (15.9)
        assert [(s["name"], s["status"]) for s in body["mcp_servers"]] == [
            ("echo", "available"),
            ("off", "unavailable"),
        ]
        assert fake_llm.calls == []
    PIDS.record_from_log(log)
    assert read_request_log(log), "the echo fixture never started"


async def test_health_with_no_servers_is_ok_over_a_real_socket(
    app_factory: AppOpener, fake_llm: ScriptedCompletion
) -> None:
    # Feature: local-agent-backend, health; Validates: Requirements 15.8, 15.9
    fake_llm.script(UNREACHABLE_TURN)
    async with app_factory(
        storage_backend="sqlite", transport="uvicorn", overrides=unreachable_llm_overrides()
    ) as app:
        body, elapsed = await timed_health(app)

        assert elapsed < HEALTH_BUDGET_S
        assert body == {"status": "ok", "mcp_servers": []}
        assert fake_llm.calls == []
