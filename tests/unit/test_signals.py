"""The Windows console control handler and its installation (17.15).

`make_console_ctrl_handler` has no Windows-only names, so the callback is called directly here on
every OS, with a fake loop or a real loop. The `install(..., "win32")` tests need the real Windows
`signal` module and run only on Windows.
"""

import asyncio
import contextlib
import ctypes
import signal
import sys
import threading
import time
from asyncio import AbstractEventLoop
from collections.abc import Callable
from typing import cast

import pytest

from conftest import windows_only
from local_agent.oscompat import _win32, signals
from local_agent.oscompat.signals import (
    CONSOLE_CLOSE_GRACE_S,
    CONSOLE_CLOSE_WAIT_S,
    DEFAULT_GRACE_S,
    ShutdownController,
    ShutdownReason,
    install,
    make_console_ctrl_handler,
)

_FALSE = 0
_TRUE = 1


class _FakeServer:
    should_exit = False


class _FakeRunners:
    def __init__(self) -> None:
        self.cancelled = 0

    async def cancel_all(self) -> None:
        self.cancelled += 1


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append(("info", event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append(("warning", event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append(("error", event, fields))


class _FakeLoop:
    """Records `call_soon_threadsafe`; optionally runs the callback at once."""

    def __init__(self, *, run_immediately: bool) -> None:
        self.run_immediately = run_immediately
        self.calls: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def call_soon_threadsafe(self, callback: Callable[..., object], *args: object) -> None:
        self.calls.append((callback, args))
        if self.run_immediately:
            callback(*args)


class _RecordingEvent(threading.Event):
    """A done event that records the timeout of every `wait` and never blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_timeouts.append(timeout)
        return super().is_set()


class _RecordingController(ShutdownController):
    """A controller whose `done` event records waits, to check the 4.5 s limit without sleeping."""

    def __init__(self) -> None:
        super().__init__(_FakeServer(), _FakeRunners(), log=_FakeLogger())
        self.recording_done = _RecordingEvent()

    @property
    def done(self) -> threading.Event:
        return self.recording_done


def _controller() -> ShutdownController:
    return ShutdownController(_FakeServer(), _FakeRunners(), log=_FakeLogger())


def _handler_for(
    loop: _FakeLoop, ctl: ShutdownController, *, wait_s: float = 0.01
) -> Callable[[int], int]:
    return make_console_ctrl_handler(cast(AbstractEventLoop, loop), ctl, wait_s=wait_s)


async def _until(predicate: Callable[[], bool], *, limit_s: float = 5.0) -> None:
    deadline = time.monotonic() + limit_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


# --- Events other than CTRL_CLOSE_EVENT -----------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        _win32.CTRL_C_EVENT,
        _win32.CTRL_BREAK_EVENT,
        _win32.CTRL_LOGOFF_EVENT,
        _win32.CTRL_SHUTDOWN_EVENT,
        3,
        4,
        99,
    ],
)
def test_other_events_return_false_without_side_effects(event: int) -> None:
    loop = _FakeLoop(run_immediately=True)
    ctl = _controller()
    handler = _handler_for(loop, ctl, wait_s=30.0)

    started = time.monotonic()
    result = handler(event)

    assert result == _FALSE
    assert time.monotonic() - started < 1.0  # no blocking wait
    assert loop.calls == []
    assert not ctl.requested
    assert ctl.reason is None
    assert not ctl.done.is_set()
    assert ctl.grace_s == DEFAULT_GRACE_S


# --- CTRL_CLOSE_EVENT -----------------------------------------------------------------------------


def test_close_event_schedules_console_close_on_the_loop() -> None:
    loop = _FakeLoop(run_immediately=False)
    ctl = _controller()
    handler = _handler_for(loop, ctl)

    result = handler(_win32.CTRL_CLOSE_EVENT)

    assert result == _TRUE
    # The callback runs on a system thread, so it only schedules; it must not call request itself.
    assert loop.calls == [(ctl.request, (ShutdownReason.CONSOLE_CLOSE,))]
    assert not ctl.requested

    loop.calls[0][0](*loop.calls[0][1])
    assert ctl.reason is ShutdownReason.CONSOLE_CLOSE


def test_close_event_makes_grace_two_seconds() -> None:
    loop = _FakeLoop(run_immediately=True)
    ctl = _controller()
    assert ctl.grace_s == DEFAULT_GRACE_S

    handler_result = _handler_for(loop, ctl)(_win32.CTRL_CLOSE_EVENT)

    assert handler_result == _TRUE
    assert ctl.reason is ShutdownReason.CONSOLE_CLOSE
    assert ctl.grace_s == 2.0
    assert ctl.grace_s == CONSOLE_CLOSE_GRACE_S
    assert not ctl.forced


def test_close_event_after_a_signal_request_forces_zero_grace() -> None:
    loop = _FakeLoop(run_immediately=True)
    ctl = _controller()
    ctl.request(ShutdownReason.SIGNAL)

    result = _handler_for(loop, ctl)(_win32.CTRL_CLOSE_EVENT)

    assert result == _TRUE
    assert ctl.reason is ShutdownReason.SIGNAL  # the first reason stays
    assert ctl.forced
    assert ctl.grace_s == 0.0


def test_close_event_waits_at_most_4_5_seconds_by_default() -> None:
    assert CONSOLE_CLOSE_WAIT_S == 4.5
    loop = _FakeLoop(run_immediately=True)
    ctl = _RecordingController()
    handler = make_console_ctrl_handler(cast(AbstractEventLoop, loop), ctl)

    result = handler(_win32.CTRL_CLOSE_EVENT)

    assert result == _TRUE
    assert ctl.recording_done.wait_timeouts == [4.5]


def test_close_event_blocks_until_the_wait_limit_when_shutdown_never_finishes() -> None:
    loop = _FakeLoop(run_immediately=True)
    ctl = _controller()
    handler = _handler_for(loop, ctl, wait_s=0.2)

    started = time.monotonic()
    result = handler(_win32.CTRL_CLOSE_EVENT)
    elapsed = time.monotonic() - started

    assert result == _TRUE
    assert 0.15 <= elapsed < 2.0
    assert not ctl.done.is_set()


def test_close_event_returns_as_soon_as_the_done_event_is_set() -> None:
    loop = _FakeLoop(run_immediately=True)
    ctl = _controller()
    handler = _handler_for(loop, ctl, wait_s=30.0)
    timer = threading.Timer(0.1, ctl.done.set)

    timer.start()
    started = time.monotonic()
    try:
        result = handler(_win32.CTRL_CLOSE_EVENT)
    finally:
        timer.cancel()
        timer.join()
    elapsed = time.monotonic() - started

    assert result == _TRUE
    assert 0.05 <= elapsed < 5.0  # woke on the event, long before the 30 s limit
    assert ctl.done.is_set()


def test_close_event_returns_false_when_the_loop_is_closed() -> None:
    loop = asyncio.new_event_loop()
    loop.close()
    ctl = _controller()
    handler = make_console_ctrl_handler(loop, ctl, wait_s=30.0)

    started = time.monotonic()
    result = handler(_win32.CTRL_CLOSE_EVENT)

    assert result == _FALSE
    assert time.monotonic() - started < 1.0  # nothing to wait for
    assert not ctl.requested


# --- Through ctypes and a running controller ------------------------------------------------------


def test_handler_works_through_the_ctypes_callback_type() -> None:
    loop = _FakeLoop(run_immediately=True)
    ctl = _controller()
    routine = _win32.HANDLER_ROUTINE(_handler_for(loop, ctl))

    assert routine(_win32.CTRL_C_EVENT) == _FALSE
    assert routine(_win32.CTRL_BREAK_EVENT) == _FALSE
    assert not ctl.requested

    assert routine(_win32.CTRL_CLOSE_EVENT) == _TRUE
    assert ctl.reason is ShutdownReason.CONSOLE_CLOSE


async def test_close_event_blocks_until_graceful_shutdown_finishes() -> None:
    server = _FakeServer()
    runners = _FakeRunners()
    log = _FakeLogger()
    ctl = ShutdownController(server, runners, log=log)
    loop = asyncio.get_running_loop()
    handler = make_console_ctrl_handler(loop, ctl)  # default 4.5 s limit
    run_task = asyncio.create_task(ctl.run())

    # The handler runs on another thread, as the Windows console thread would.
    handler_task = asyncio.create_task(asyncio.to_thread(handler, _win32.CTRL_CLOSE_EVENT))
    await _until(lambda: ctl.requested)
    await _until(lambda: runners.cancelled == 1)

    assert ctl.reason is ShutdownReason.CONSOLE_CLOSE
    assert ctl.grace_s == 2.0
    assert server.should_exit is True
    # Shutdown is still waiting for the lifespan, so the callback must still be blocked.
    await asyncio.sleep(0.1)
    assert not handler_task.done()
    assert not ctl.done.is_set()

    ctl.lifespan_finished()

    assert await asyncio.wait_for(handler_task, timeout=5.0) == _TRUE
    assert await asyncio.wait_for(run_task, timeout=5.0) == 0
    assert ctl.done.is_set()
    started = [fields for _lvl, event, fields in log.events if event == "shutdown_started"]
    assert started == [{"reason": "console_close"}]


# --- install(..., "win32"): Windows only ----------------------------------------------------------


@windows_only
async def test_install_windows_registers_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "win32":
        registered: list[tuple[object, int]] = []

        def fake_set_console_ctrl_handler(routine: object, add: int) -> int:
            registered.append((routine, add))
            return 1

        monkeypatch.setattr(_win32, "SetConsoleCtrlHandler", fake_set_console_ctrl_handler)
        loop = asyncio.get_running_loop()
        ctl = _controller()
        old_int = signal.getsignal(signal.SIGINT)
        old_break = signal.getsignal(signal.SIGBREAK)
        live_before = len(signals._live_callbacks)

        uninstall = install(loop, ctl, "win32")
        try:
            assert signal.getsignal(signal.SIGINT) is not old_int
            assert callable(signal.getsignal(signal.SIGINT))
            assert callable(signal.getsignal(signal.SIGBREAK))
            assert len(registered) == 1
            routine, add = registered[0]
            assert add == 1
            assert isinstance(routine, _win32.HANDLER_ROUTINE)
            # The callback object stays referenced while installed.
            assert len(signals._live_callbacks) == live_before + 1
            assert any(routine is live for live in signals._live_callbacks)
        finally:
            uninstall()

        assert signal.getsignal(signal.SIGINT) is old_int
        assert signal.getsignal(signal.SIGBREAK) is old_break
        assert len(registered) == 2
        assert registered[1] == (routine, 0)
        assert len(signals._live_callbacks) == live_before
        uninstall()  # a second call changes nothing
        assert len(registered) == 2


@windows_only
async def test_install_windows_signal_handlers_request_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "win32":
        monkeypatch.setattr(_win32, "SetConsoleCtrlHandler", lambda _routine, _add: 1)
        for sig in (signal.SIGINT, signal.SIGBREAK):
            ctl = _controller()
            uninstall = install(asyncio.get_running_loop(), ctl, "win32")
            try:
                python_handler = signal.getsignal(sig)
                assert callable(python_handler)
                python_handler(sig, None)  # what CPython does on Ctrl+C / Ctrl+Break
                await _until(lambda ctl=ctl: ctl.requested)
                assert ctl.reason is ShutdownReason.SIGNAL
                assert ctl.grace_s == DEFAULT_GRACE_S
            finally:
                uninstall()


@windows_only
async def test_install_windows_console_callback_is_the_close_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "win32":
        registered: list[object] = []

        def fake_set_console_ctrl_handler(routine: object, add: int) -> int:
            if add:
                registered.append(routine)
            return 1

        monkeypatch.setattr(_win32, "SetConsoleCtrlHandler", fake_set_console_ctrl_handler)
        ctl = _controller()
        ctl.done.set()  # shutdown "already finished", so the close callback returns at once
        uninstall = install(asyncio.get_running_loop(), ctl, "win32")
        try:
            routine = cast(Callable[[int], int], registered[0])
            assert routine(_win32.CTRL_C_EVENT) == _FALSE
            assert not ctl.requested
            assert routine(_win32.CTRL_CLOSE_EVENT) == _TRUE
            await _until(lambda: ctl.requested)
            assert ctl.reason is ShutdownReason.CONSOLE_CLOSE
            assert ctl.grace_s == 2.0
        finally:
            uninstall()


@windows_only
async def test_install_windows_survives_a_failed_console_handler_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "win32":
        monkeypatch.setattr(_win32, "SetConsoleCtrlHandler", lambda _routine, _add: 0)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 0)
        old_int = signal.getsignal(signal.SIGINT)
        log = _FakeLogger()
        monkeypatch.setattr(signals, "_log", log)

        uninstall = install(asyncio.get_running_loop(), _controller(), "win32")
        try:
            # The signal handlers still work; only the console close event is lost.
            assert signal.getsignal(signal.SIGINT) is not old_int
            assert [e for _lvl, e, _f in log.events] == ["console_ctrl_handler_failed"]
        finally:
            with contextlib.suppress(Exception):
                uninstall()
        assert signal.getsignal(signal.SIGINT) is old_int
