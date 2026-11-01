"""Property test for Graceful_Shutdown sequencing (Requirements 17.13, 17.14, 2.7)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.oscompat.signals import (
    CONSOLE_CLOSE_GRACE_S,
    DEFAULT_GRACE_S,
    ShutdownController,
    ShutdownReason,
)

TaskState = Literal["working", "input-required", "canceled"]
ServerKind = Literal["stdio", "sse"]

_TIMEOUT_S = 10.0
_ACTIVE_STATES: list[TaskState] = ["working", "input-required"]


class _FakeServer:
    """Records the moment uvicorn is told to stop accepting connections."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._should_exit = False
        self.writes = 0

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self.writes += 1
        self._should_exit = value
        if value:
            self._log.append("accept_stop")


class _FakeRunners:
    """Cancels every active Task, recording start, each Task, and finish."""

    def __init__(
        self, log: list[str], states: list[TaskState], yields: list[int], *, is_async: bool
    ) -> None:
        self._log = log
        self.states = states
        self._yields = yields
        self._is_async = is_async
        self.calls = 0
        self.finished = asyncio.Event()

    def cancel_all(self) -> object:
        self.calls += 1
        self._log.append("cancel_start")
        if not self._is_async:
            self._cancel_sync()
            return None
        return self._cancel_async()

    def _cancel_sync(self) -> None:
        for index in range(len(self.states)):
            self.states[index] = "canceled"
            self._log.append(f"cancel_task:{index}")
        self._log.append("cancel_end")
        self.finished.set()

    async def _cancel_async(self) -> None:
        for index, count in enumerate(self._yields):
            for _ in range(count):
                await asyncio.sleep(0)
            self.states[index] = "canceled"
            self._log.append(f"cancel_task:{index}")
        self._log.append("cancel_end")
        self.finished.set()


class _FakeMcpManager:
    """Stops stdio trees and SSE sessions one after another, recording each step."""

    def __init__(self, log: list[str], servers: list[ServerKind], yields: int) -> None:
        self._log = log
        self._servers = servers
        self._yields = yields
        self.calls = 0
        self.graces: list[float] = []

    async def stop(self, grace: float) -> None:
        self.calls += 1
        self.graces.append(grace)
        self._log.append("mcp_stop_start")
        for index, kind in enumerate(self._servers):
            for _ in range(self._yields):
                await asyncio.sleep(0)
            self._log.append(f"stop_tree:{index}" if kind == "stdio" else f"close_session:{index}")
        self._log.append("mcp_stop_end")


class _RecordingLog:
    """Collects the controller's own log events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append(event)

    def warning(self, event: str, **fields: object) -> None:
        self.events.append(event)

    def error(self, event: str, **fields: object) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class _Scenario:
    states: list[TaskState]
    cancel_yields: list[int]
    cancel_is_async: bool
    servers: list[ServerKind]
    mcp_yields: int
    pre_delay: int
    signals: list[tuple[ShutdownReason, int]]
    """(reason, event loop iterations to wait before delivering it)"""
    lifespan_extra: int


@st.composite
def _scenarios(draw: st.DrawFn) -> _Scenario:
    states = draw(st.lists(st.sampled_from(_ACTIVE_STATES), max_size=5))
    return _Scenario(
        states=states,
        cancel_yields=draw(st.lists(st.integers(0, 3), min_size=len(states), max_size=len(states))),
        cancel_is_async=draw(st.booleans()),
        servers=draw(st.lists(st.sampled_from(["stdio", "sse"]), max_size=4)),
        mcp_yields=draw(st.integers(0, 3)),
        pre_delay=draw(st.integers(0, 4)),
        signals=draw(
            st.lists(
                st.tuples(st.sampled_from(list(ShutdownReason)), st.integers(0, 6)),
                min_size=1,
                max_size=6,
            )
        ),
        lifespan_extra=draw(st.integers(0, 4)),
    )


def _expected_log(scenario: _Scenario) -> list[str]:
    return [
        "accept_stop",
        "cancel_start",
        *[f"cancel_task:{i}" for i in range(len(scenario.states))],
        "cancel_end",
        "mcp_stop_start",
        *[
            f"stop_tree:{i}" if kind == "stdio" else f"close_session:{i}"
            for i, kind in enumerate(scenario.servers)
        ],
        "mcp_stop_end",
        "lifespan_finished",
    ]


# Feature: local-agent-backend, Property 44: Graceful_Shutdown sequencing
# **Validates: Requirements 17.13, 17.14, 2.7**
@settings(max_examples=100, deadline=None)
@given(_scenarios())
async def test_graceful_shutdown_sequencing(scenario: _Scenario) -> None:
    log: list[str] = []
    server = _FakeServer(log)
    runners = _FakeRunners(
        log, list(scenario.states), scenario.cancel_yields, is_async=scenario.cancel_is_async
    )
    mcp = _FakeMcpManager(log, scenario.servers, scenario.mcp_yields)
    controller_log = _RecordingLog()
    ctl = ShutdownController(server, runners, log=controller_log)

    delivered = 0
    expected_graces: list[float] = []
    lifespan_saw_run_pending = False

    run_task = asyncio.ensure_future(ctl.run())

    async def deliver_signals() -> None:
        nonlocal delivered
        for _ in range(scenario.pre_delay):
            await asyncio.sleep(0)
        # Nothing may happen before the first Shutdown_Signal.
        assert log == []
        assert not run_task.done()
        assert not ctl.requested

        for position, (reason, gap) in enumerate(scenario.signals):
            for _ in range(gap):
                await asyncio.sleep(0)
            ctl.request(reason)
            delivered += 1
            if position == 0:
                # Shutdown starts in the loop iteration that receives the first signal: after one
                # yield the stop-accepting step has run and the cancel step has begun.
                await asyncio.sleep(0)
                assert log[:2] == ["accept_stop", "cancel_start"]

    async def lifespan() -> None:
        nonlocal lifespan_saw_run_pending
        # The app lifespan shutdown (McpManager.stop, then Database.close) begins after the Task
        # cancellation step has finished.
        await runners.finished.wait()
        for _ in range(scenario.lifespan_extra):
            await asyncio.sleep(0)
        # grace_s is read at the moment the stop starts, so a signal that arrived earlier counts.
        if delivered >= 2:
            expected_graces.append(0.0)
        elif scenario.signals[0][0] is ShutdownReason.CONSOLE_CLOSE:
            expected_graces.append(CONSOLE_CLOSE_GRACE_S)
        else:
            expected_graces.append(DEFAULT_GRACE_S)
        await mcp.stop(grace=ctl.grace_s)
        for _ in range(scenario.lifespan_extra):
            await asyncio.sleep(0)
        # run() must still be waiting: only lifespan_finished() lets it return.
        lifespan_saw_run_pending = not run_task.done() and not ctl.done.is_set()
        log.append("lifespan_finished")
        ctl.lifespan_finished()

    exit_status, *_ = await asyncio.wait_for(
        asyncio.gather(run_task, deliver_signals(), lifespan()), timeout=_TIMEOUT_S
    )

    # Exit status 0, and run() returned only after the lifespan finished.
    assert exit_status == 0
    assert lifespan_saw_run_pending
    assert ctl.done.is_set()

    # Step order: stop accepting, cancel every Task, MCP stop; each starts after the previous ended.
    assert log == _expected_log(scenario)

    # Each step ran exactly once however many signals arrived.
    assert server.writes == 1
    assert runners.calls == 1
    assert mcp.calls == 1
    assert controller_log.events.count("shutdown_started") == 1
    assert controller_log.events.count("shutdown_finished") == 1

    # Every active Task ended as canceled.
    assert all(state == "canceled" for state in runners.states)

    # Repeated signals only shorten the grace period handed to the MCP stop.
    assert mcp.graces == expected_graces
    assert ctl.forced == (len(scenario.signals) >= 2)
    assert ctl.reason is scenario.signals[0][0]
    assert ctl.grace_s == (0.0 if len(scenario.signals) >= 2 else expected_graces[0])
