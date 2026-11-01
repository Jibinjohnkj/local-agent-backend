"""Process-tree cleanup integration tests (task 22.10).

The real `McpManager` runs the real `managed_stdio_client` against `tests/fixtures/
mcp_spawning_server.py`, which owns a three-level tree (server, grandchild, great-grandchild). Each
test records every pid of the tree, ends the server in one of three ways, and checks that no
process of the tree is left within `1 + grace + 1.5` seconds (17.12) and that
`TerminationReport.forced` matches what the server did (17.11).

The server behaves in one of three ways when its stdin closes:

* `cooperative`: stops its descendants and exits, so no force is needed.
* `ignore-graceful`: ignores EOF and `SIGTERM`, so only the force step ends it.
* `orphan`: exits at once and leaves its descendants running, so the graceful signal to the group
  ends them and no force is needed.

It is ended by `McpManager.stop()`, by a reconnection after only the server pid was killed (6.8),
or by a failed handshake (`--no-initialize`, 6.7). The assertions on `forced` for Windows are only
made where the outcome does not depend on how `CTRL_BREAK_EVENT` reaches descendants of a dead
server; the pid checks (`OpenProcess` + `WaitForSingleObject`, in the harness) run everywhere.

The termination reports are observed by wrapping `terminate_tree` in the stdio transport; the
wrapper calls the real function and records what it returns.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pytest

from integration_harness import FIXTURES_DIR, PIDS, kill_pid, pid_alive, wait_until_gone
from local_agent.config.models import StdioServerConfig
from local_agent.mcp_client import stdio_transport
from local_agent.mcp_client.manager import McpManager
from local_agent.mcp_client.registry import ToolRegistry
from local_agent.models import TextItem
from local_agent.oscompat.process import DEFAULT_GRACE_S, TerminationReport, terminate_tree

SPAWNING_SERVER: Final = FIXTURES_DIR / "mcp_spawning_server.py"
SERVER_NAME: Final = "spawn"
MODES: Final = ("cooperative", "ignore-graceful", "orphan")
STOP_GRACE_S: Final = 1.0
STDIN_WAIT_S: Final = 1.0
REAP_WAIT_S: Final = 1.5
HANDSHAKE_TIMEOUT_S: Final = 2.0
IS_WINDOWS: Final = sys.platform == "win32"


@dataclass(frozen=True)
class Tree:
    """The pids of one Process_Tree."""

    server: int
    descendants: tuple[int, ...]

    @property
    def pids(self) -> list[int]:
        return [self.server, *self.descendants]


def bound_s(grace_s: float) -> float:
    """The longest a Process_Tree may take to be gone (17.12)."""
    return STDIN_WAIT_S + grace_s + REAP_WAIT_S


def expected_forced(mode: str, *, server_killed: bool) -> bool | None:
    """Whether the force step must have run; `None` where Windows leaves it open.

    A server that ignores the graceful signals needs the force step, unless it is already dead
    (then its descendants, which keep the default signal behaviour, end at the graceful step).
    """
    if IS_WINDOWS and (mode == "orphan" or server_killed):
        return None  # depends on how CTRL_BREAK_EVENT reaches the descendants of a dead server
    return mode == "ignore-graceful" and not server_killed


def spawning_entry(mode: str, pid_file: Path, *, no_initialize: bool = False) -> dict[str, Any]:
    args = [str(SPAWNING_SERVER), f"--{mode}", "--pid-file", str(pid_file)]
    if no_initialize:
        args.append("--no-initialize")
    return {
        "name": SERVER_NAME,
        "transport": "stdio",
        "command": sys.executable,
        "args": args,
    }


def make_manager(tmp_path: Path, entry: Mapping[str, Any], **options: Any) -> McpManager:
    cfg = StdioServerConfig.model_validate({**entry, "env": {}})
    registry = ToolRegistry([cfg.name])
    return McpManager([cfg], registry, config_dir=str(tmp_path), **options)


def read_tree(pid_file: Path) -> Tree | None:
    """The tree in the pid file once all three pids are known, otherwise `None`."""
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    descendants = tuple(int(pid) for pid in data.get("descendants", ()))
    if "server" not in data or len(descendants) != 2:
        return None
    return Tree(int(data["server"]), descendants)


async def eventually(
    check: Callable[[], bool | Awaitable[bool]], timeout_s: float, what: str
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        outcome = check()
        if await outcome if isinstance(outcome, Awaitable) else outcome:
            return
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout_s} s waiting for {what}")
        await asyncio.sleep(0.05)


async def tree_from_tool(manager: McpManager) -> Tree:
    """The tree as the server reports it through its `get_pids` tool. Records every pid."""
    outcome = await manager.call_tool(SERVER_NAME, "get_pids", {}, 20.0)
    assert not outcome.is_error
    item = outcome.content[0]
    assert isinstance(item, TextItem)
    data = json.loads(item.text)
    tree = Tree(int(data["server"]), tuple(int(pid) for pid in data["descendants"]))
    assert len(tree.descendants) == 2
    PIDS.record(*tree.pids)
    return tree


async def tree_from_file(pid_file: Path, timeout_s: float = 10.0) -> Tree:
    """The tree in the pid file (the handshake never completes, so there is no tool). Records it."""
    found: list[Tree] = []

    def ready() -> bool:
        tree = read_tree(pid_file)
        if tree is not None:
            found.append(tree)
        return tree is not None

    await eventually(ready, timeout_s, "the server to publish its process tree")
    PIDS.record(*found[-1].pids)
    return found[-1]


def report_for(reports: list[TerminationReport], tree: Tree) -> TerminationReport:
    mine = [report for report in reports if report.pid == tree.server]
    assert len(mine) == 1, f"expected one termination report for pid {tree.server}, got {mine}"
    return mine[0]


async def assert_tree_gone(tree: Tree, grace_s: float) -> None:
    """Wait off the event loop, so the supervisor and the transport keep running meanwhile."""
    leftovers = await asyncio.to_thread(wait_until_gone, tree.pids, bound_s(grace_s))
    assert leftovers == [], f"processes of the tree are still alive: {leftovers}"


def assert_forced(report: TerminationReport, mode: str, *, server_killed: bool = False) -> None:
    assert report.survivors == ()
    expected = expected_forced(mode, server_killed=server_killed)
    if expected is not None:
        assert report.forced is expected


@pytest.fixture
def reports(monkeypatch: pytest.MonkeyPatch) -> list[TerminationReport]:
    """Every `TerminationReport` the stdio transport got from the real `terminate_tree`."""
    recorded: list[TerminationReport] = []

    async def recording(*args: Any, **kwargs: Any) -> TerminationReport:
        report = await terminate_tree(*args, **kwargs)
        recorded.append(report)
        return report

    monkeypatch.setattr(stdio_transport, "terminate_tree", recording)
    return recorded


# --- McpManager.stop() --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_stop_leaves_no_process_of_the_tree(
    tmp_path: Path, reports: list[TerminationReport], mode: str
) -> None:
    manager = make_manager(tmp_path, spawning_entry(mode, tmp_path / "pids.json"))
    await manager.start()
    try:
        assert manager.availability() == {SERVER_NAME: True}
        tree = await tree_from_tool(manager)
        assert all(pid_alive(pid) for pid in tree.pids)

        started = time.monotonic()
        await manager.stop(STOP_GRACE_S)
        elapsed = time.monotonic() - started
    finally:
        await manager.stop(0)

    assert elapsed <= bound_s(STOP_GRACE_S)
    await assert_tree_gone(tree, STOP_GRACE_S)
    assert_forced(report_for(reports, tree), mode)


# --- reconnection after only the server pid was killed ------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_reconnection_after_killing_only_the_server_leaves_no_old_process(
    tmp_path: Path, reports: list[TerminationReport], mode: str
) -> None:
    pid_file = tmp_path / "pids.json"
    manager = make_manager(tmp_path, spawning_entry(mode, pid_file))
    await manager.start()
    try:
        first = await tree_from_tool(manager)
        assert all(pid_alive(pid) for pid in first.pids)

        kill_pid(first.server)  # the grandchild and great-grandchild are left behind
        await assert_tree_gone(first, DEFAULT_GRACE_S)
        await eventually(
            lambda: any(report.pid == first.server for report in reports),
            bound_s(DEFAULT_GRACE_S),
            "the termination report of the killed server's tree",
        )
        assert_forced(report_for(reports, first), mode, server_killed=True)

        def reconnected() -> bool:
            tree = read_tree(pid_file)
            return (
                manager.availability()[SERVER_NAME]
                and tree is not None
                and (tree.server != first.server)
            )

        await eventually(reconnected, 20.0, "the supervisor to reconnect to a new server")
        second = await tree_from_tool(manager)
        assert second.server != first.server
        assert set(second.pids).isdisjoint(first.pids)
        assert all(pid_alive(pid) for pid in second.pids)

        await manager.stop(STOP_GRACE_S)
    finally:
        await manager.stop(0)

    await assert_tree_gone(second, STOP_GRACE_S)
    assert_forced(report_for(reports, second), mode)


# --- failed handshake ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_failed_handshake_leaves_no_process_of_the_tree(
    tmp_path: Path, reports: list[TerminationReport], mode: str
) -> None:
    pid_file = tmp_path / "pids.json"
    # No reconnection: the pid file then holds the only tree this test ever started.
    manager = make_manager(
        tmp_path,
        spawning_entry(mode, pid_file, no_initialize=True),
        handshake_timeout_s=HANDSHAKE_TIMEOUT_S,
        max_reconnects=0,
    )
    try:
        # `start()` returns after the failed attempt, which includes the tree termination.
        await manager.start()
        tree = await tree_from_file(pid_file)
        assert manager.availability() == {SERVER_NAME: False}
    finally:
        await manager.stop(0)

    await assert_tree_gone(tree, DEFAULT_GRACE_S)
    assert_forced(report_for(reports, tree), mode)
