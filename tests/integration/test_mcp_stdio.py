"""MCP `stdio` integration tests (task 22.5).

The real SDK `ClientSession` runs over the Backend's `managed_stdio_client` against the echo
fixture (`tests/fixtures/mcp_echo_server.py`), and the real `McpManager` runs its supervisors over
the same transport. Nothing is faked: every test starts real child processes.

Covered: handshake (6.5), pagination (7.1), `call_tool`, concurrent calls from two Tasks (14.3),
child kill and reconnection (6.8), hanging servers settle within 17 s (6.6, 6.7), and
`"naïve 日本語 🚀"` round-trips under a non-UTF-8 locale (17.27).
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp_types import InitializeResult, TextContent

from integration_harness import (
    PIDS,
    echo_server_entry,
    kill_pid,
    pid_alive,
    read_request_log,
    wait_until_gone,
)
from local_agent.config.models import StdioServerConfig
from local_agent.mcp_client.manager import McpManager, McpSessionAdapter
from local_agent.mcp_client.mcp_compat import ClientSession, page_params
from local_agent.mcp_client.registry import ToolRegistry
from local_agent.mcp_client.stdio_transport import managed_stdio_client
from local_agent.models import TextItem
from local_agent.oscompat.process import build_spawn_spec

pytestmark = pytest.mark.integration

UNICODE_TEXT = "naïve 日本語 🚀"
ALL_TOOLS = ["echo", "slow", "error", "hang", "write_file", "crash", "get_pid"]
GRACE_S = 2.0

# A child that never answers `initialize`. It appends its pid to the file in `argv[1]`.
HANG_SCRIPT = (
    "import os, sys, time\n"
    "with open(sys.argv[1], 'a') as stream:\n"
    "    stream.write(str(os.getpid()) + '\\n')\n"
    "time.sleep(600)\n"
)


# --- helpers ------------------------------------------------------------------------------------


def non_utf8_locale_env() -> dict[str, str]:
    """Child environment with a non-UTF-8 locale: `LC_ALL=C` on POSIX, code page 1252 on Windows.

    The Python switches that would turn a C locale back into UTF-8 are turned off, so the child's
    preferred encoding really is not UTF-8.
    """
    if sys.platform == "win32":
        return {"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"}
    return {
        "LC_ALL": "C",
        "LANG": "C",
        "LC_CTYPE": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }


def stdio_config(
    entry: Mapping[str, Any], env: Mapping[str, str] | None = None
) -> StdioServerConfig:
    return StdioServerConfig.model_validate({**entry, "env": dict(env or {})})


@asynccontextmanager
async def sdk_session(
    tmp_path: Path, entry: Mapping[str, Any], env: Mapping[str, str] | None = None
) -> AsyncIterator[tuple[ClientSession, InitializeResult]]:
    """A real SDK `ClientSession` over `managed_stdio_client`, initialized."""
    cfg = stdio_config(entry, env)
    spec = await build_spawn_spec(cfg, str(tmp_path), sys.platform)
    async with (
        managed_stdio_client(spec, grace=lambda: GRACE_S, server=cfg.name) as (read, write),
        ClientSession(read, write) as session,
    ):
        initialized = await session.initialize()
        yield session, initialized


@asynccontextmanager
async def running_manager(
    tmp_path: Path,
    entries: Sequence[Mapping[str, Any]],
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[tuple[McpManager, ToolRegistry]]:
    """A started `McpManager` over real stdio servers; leaving the block stops it."""
    configs = [stdio_config(entry, env) for entry in entries]
    registry = ToolRegistry([cfg.name for cfg in configs])
    manager = McpManager(configs, registry, config_dir=str(tmp_path))
    await manager.start()
    try:
        yield manager, registry
    finally:
        await manager.stop(GRACE_S)


async def eventually(
    check: Callable[[], bool | Awaitable[bool]], timeout_s: float, what: str
) -> None:
    """Poll `check` every 50 ms until it is true, or fail after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while True:
        outcome = check()
        if await outcome if isinstance(outcome, Awaitable) else outcome:
            return
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout_s} s waiting for {what}")
        await asyncio.sleep(0.05)


def text_of(result: Any) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def registered_tools(registry: ToolRegistry, server: str) -> list[str]:
    return [tool.tool for tool in registry.snapshot().tools if tool.server == server]


def call_entries(log: Path, name: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in read_request_log(log)
        if entry["event"] == "call_tool" and entry["name"] == name
    ]


# --- handshake, pagination, call_tool -----------------------------------------------------------


async def test_handshake_and_clean_shutdown(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(tmp_path, echo_server_entry(request_log=log)) as (_, initialized):
        assert initialized.server_info.name == "echo-fixture"
        pids = PIDS.record_from_log(log)
        assert len(pids) == 1
        assert pid_alive(pids[0])

    assert wait_until_gone(pids, 5.0) == []  # leaving the context stops the child (17.11)


async def test_tools_list_follows_cursors_through_every_page(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(tmp_path, echo_server_entry(page_size=2, request_log=log)) as (
        session,
        _,
    ):
        names: list[str] = []
        cursor: str | None = None
        while True:
            page = await session.list_tools(params=page_params(cursor))
            names += [tool.name for tool in page.tools]
            cursor = page.next_cursor
            if cursor is None:
                break

    PIDS.record_from_log(log)
    assert names == ALL_TOOLS
    listings = [e for e in read_request_log(log) if e["event"] == "list_tools"]
    assert [e["cursor"] for e in listings] == [None, "2", "4", "6"]


async def test_adapter_walks_all_pages(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(tmp_path, echo_server_entry(page_size=3, request_log=log)) as (
        session,
        _,
    ):
        tools = await McpSessionAdapter(session).list_all_tools(10.0)

    PIDS.record_from_log(log)
    assert [tool.name for tool in tools] == ALL_TOOLS
    assert all(tool.input_schema["type"] == "object" for tool in tools if tool.input_schema)
    listings = [e for e in read_request_log(log) if e["event"] == "list_tools"]
    assert [e["cursor"] for e in listings] == [None, "3", "6"]


async def test_call_tool_returns_text_and_error_flag(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(tmp_path, echo_server_entry(request_log=log)) as (session, _):
        echoed = await session.call_tool("echo", {"text": "hello"})
        failed = await session.call_tool("error", {"message": "boom"})
        wrote = await session.call_tool("write_file", {"path": "a.txt", "content": "x"})
        pid = await session.call_tool("get_pid", {})

    pids = PIDS.record_from_log(log)
    assert (text_of(echoed), echoed.is_error) == ("hello", False)
    assert (text_of(failed), failed.is_error) == ("boom", True)
    assert text_of(wrote) == "wrote a.txt"
    assert [int(text_of(pid))] == pids


async def test_hanging_tool_times_out_and_the_session_stays_usable(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(tmp_path, echo_server_entry(request_log=log)) as (session, _):
        adapter = McpSessionAdapter(session)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await adapter.call_tool("hang", {}, 1.0)
        assert time.monotonic() - started < 4.0

        outcome = await adapter.call_tool("echo", {"text": "after"}, 5.0)
        assert outcome.content == (TextItem(text="after"),)

    PIDS.record_from_log(log)


# --- concurrency (14.3) -------------------------------------------------------------------------


async def test_two_tasks_call_one_session_concurrently(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(tmp_path, echo_server_entry(request_log=log)) as (session, _):

        async def slow(text: str) -> str:
            return text_of(await session.call_tool("slow", {"seconds": 1.5, "text": text}))

        started = time.monotonic()
        first = asyncio.create_task(slow("one"))
        second = asyncio.create_task(slow("two"))
        results = await asyncio.gather(first, second)
        elapsed = time.monotonic() - started

    PIDS.record_from_log(log)
    assert list(results) == ["one", "two"]  # each Task gets its own reply
    assert elapsed < 2.6  # one after the other would take 3.0 s or more
    assert len(call_entries(log, "slow")) == 2


async def test_manager_runs_calls_from_two_tasks_concurrently(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    async with running_manager(tmp_path, [echo_server_entry(request_log=log)]) as (manager, _):

        async def slow(text: str) -> str:
            outcome = await manager.call_tool(
                "echo", "slow", {"seconds": 1.5, "text": text}, timeout=10.0
            )
            return outcome.content[0].text  # type: ignore[union-attr]

        started = time.monotonic()
        results = await asyncio.gather(
            asyncio.create_task(slow("left")), asyncio.create_task(slow("right"))
        )
        elapsed = time.monotonic() - started

    PIDS.record_from_log(log)
    assert list(results) == ["left", "right"]
    assert elapsed < 2.6


# --- manager: startup, kill, reconnection (6.5, 6.8) --------------------------------------------


async def test_manager_connects_and_registers_every_page_of_tools(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    entry = echo_server_entry(page_size=2, request_log=log)
    async with running_manager(tmp_path, [entry]) as (manager, registry):
        assert manager.availability() == {"echo": True}
        assert registered_tools(registry, "echo") == ALL_TOOLS
        pids = PIDS.record_from_log(log)

    assert manager.availability() == {"echo": False}
    assert registered_tools(registry, "echo") == []
    assert wait_until_gone(pids, 5.0) == []


@pytest.mark.parametrize("how", ["sigkill", "crash_tool"])
async def test_manager_reconnects_after_the_child_dies(tmp_path: Path, how: str) -> None:
    log = tmp_path / "requests.jsonl"
    async with running_manager(tmp_path, [echo_server_entry(request_log=log)]) as (
        manager,
        registry,
    ):
        first = await manager.call_tool("echo", "get_pid", {}, timeout=5.0)
        first_pid = int(first.content[0].text)  # type: ignore[union-attr]
        PIDS.record(first_pid)

        pending = asyncio.create_task(
            manager.call_tool("echo", "slow", {"seconds": 30, "text": "x"}, timeout=30.0)
        )
        await asyncio.sleep(0.3)  # the slow call is on the server
        if how == "sigkill":
            kill_pid(first_pid)
        else:
            with pytest.raises(Exception):  # noqa: B017, PT011 - the SDK's connection-closed error
                await manager.call_tool("echo", "crash", {}, timeout=10.0)

        # A call that was in flight fails instead of waiting for its 30 s timeout.
        async with asyncio.timeout(5.0):
            with pytest.raises(Exception):  # noqa: B017, PT011
                await pending

        await eventually(
            lambda: manager.availability() == {"echo": False}, 5.0, "echo to turn unavailable"
        )
        assert registered_tools(registry, "echo") == []
        assert wait_until_gone([first_pid], 5.0) == []

        # The supervisor backs off 1 s, then starts a new child and registers the tools again.
        await eventually(
            lambda: manager.availability() == {"echo": True}, 15.0, "echo to reconnect"
        )
        assert registered_tools(registry, "echo") == ALL_TOOLS
        second = await manager.call_tool("echo", "get_pid", {}, timeout=5.0)
        second_pid = int(second.content[0].text)  # type: ignore[union-attr]
        PIDS.record(second_pid)
        assert second_pid != first_pid
        echoed = await manager.call_tool("echo", "echo", {"text": "again"}, timeout=5.0)
        assert echoed.content[0].text == "again"  # type: ignore[union-attr]

    PIDS.record_from_log(log)
    assert wait_until_gone([first_pid, second_pid], 5.0) == []


async def test_manager_reconnects_after_crash_after_the_nth_call(tmp_path: Path) -> None:
    log = tmp_path / "requests.jsonl"
    entry = echo_server_entry(crash_after=2, request_log=log)
    async with running_manager(tmp_path, [entry]) as (manager, _):
        ok = await manager.call_tool("echo", "echo", {"text": "one"}, timeout=5.0)
        assert ok.content[0].text == "one"  # type: ignore[union-attr]
        with pytest.raises(Exception):  # noqa: B017, PT011 - the second call ends the server
            await manager.call_tool("echo", "echo", {"text": "two"}, timeout=10.0)
        await eventually(
            lambda: manager.availability() == {"echo": False}, 5.0, "echo to turn unavailable"
        )
        await eventually(
            lambda: manager.availability() == {"echo": True}, 15.0, "echo to reconnect"
        )

    pids = PIDS.record_from_log(log)
    assert len(pids) == 2  # the first child and its replacement
    assert wait_until_gone(pids, 5.0) == []


# --- hanging servers settle within 17 s (6.6, 6.7) ----------------------------------------------


async def test_a_server_that_never_answers_initialize_settles_startup_within_17_seconds(
    tmp_path: Path,
) -> None:
    script = tmp_path / "hang_server.py"
    script.write_text(HANG_SCRIPT, encoding="utf-8")
    pid_file = tmp_path / "hang.pids"
    hang_entry = {
        "name": "stuck",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(script), str(pid_file)],
    }
    log = tmp_path / "requests.jsonl"

    started = time.monotonic()
    async with running_manager(tmp_path, [hang_entry, echo_server_entry(request_log=log)]) as (
        manager,
        registry,
    ):
        elapsed = time.monotonic() - started
        assert elapsed < 17.0  # the cap on startup (6.6)
        assert manager.availability() == {"stuck": False, "echo": True}  # startup went on (6.7)
        assert registered_tools(registry, "stuck") == []
        assert registered_tools(registry, "echo") == ALL_TOOLS
        statuses = {status.name: status.available for status in manager.statuses()}
        assert statuses == {"stuck": False, "echo": True}
        echoed = await manager.call_tool("echo", "echo", {"text": "alive"}, timeout=5.0)
        assert echoed.content[0].text == "alive"  # type: ignore[union-attr]

    stop_started = time.monotonic()
    pids = {int(line) for line in pid_file.read_text().split()}
    PIDS.record(*pids)
    PIDS.record_from_log(log)
    assert wait_until_gone(sorted(pids), 5.0) == []  # every hung child was terminated
    assert time.monotonic() - stop_started < 6.0


async def test_a_calling_task_never_waits_longer_than_the_tool_timeout_for_a_hung_tool(
    tmp_path: Path,
) -> None:
    log = tmp_path / "requests.jsonl"
    async with running_manager(tmp_path, [echo_server_entry(request_log=log)]) as (manager, _):
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await manager.call_tool("echo", "hang", {}, timeout=1.0)
        assert time.monotonic() - started < 4.0
        assert manager.availability() == {"echo": True}  # a timeout does not drop the server
        echoed = await manager.call_tool("echo", "echo", {"text": "next"}, timeout=5.0)
        assert echoed.content[0].text == "next"  # type: ignore[union-attr]

    PIDS.record_from_log(log)


# --- non-UTF-8 locale (17.27) -------------------------------------------------------------------


async def test_unicode_round_trips_under_a_non_utf8_locale_over_the_sdk_session(
    tmp_path: Path,
) -> None:
    log = tmp_path / "requests.jsonl"
    async with sdk_session(
        tmp_path, echo_server_entry(request_log=log), env=non_utf8_locale_env()
    ) as (session, _):
        echoed = await session.call_tool("echo", {"text": UNICODE_TEXT})
        assert text_of(echoed) == UNICODE_TEXT
        assert echoed.is_error is False
        failed = await session.call_tool("error", {"message": UNICODE_TEXT})
        assert (text_of(failed), failed.is_error) == (UNICODE_TEXT, True)

    PIDS.record_from_log(log)
    # The server saw the same characters, not a locale-damaged copy.
    assert [e["arguments"]["text"] for e in call_entries(log, "echo")] == [UNICODE_TEXT]


async def test_unicode_round_trips_under_a_non_utf8_locale_through_the_manager(
    tmp_path: Path,
) -> None:
    log = tmp_path / "requests.jsonl"
    async with running_manager(
        tmp_path, [echo_server_entry(request_log=log)], env=non_utf8_locale_env()
    ) as (manager, _):
        outcome = await manager.call_tool("echo", "echo", {"text": UNICODE_TEXT}, timeout=5.0)
        assert outcome.content[0].text == UNICODE_TEXT  # type: ignore[union-attr]
        assert outcome.is_error is False

    PIDS.record_from_log(log)
    assert [e["arguments"]["text"] for e in call_entries(log, "echo")] == [UNICODE_TEXT]
