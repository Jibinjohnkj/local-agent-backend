"""Shutdown_Signal handling and the Graceful_Shutdown sequence (2.7, 6.11, 17.13-17.15).

`__main__` creates one `ShutdownController`, calls `install(loop, ctl, sys.platform)` before
`runner.run(...)`, and runs `ctl.run()` next to uvicorn's `serve()`. The OS-specific code only turns
a signal into `ctl.request(...)` on the event loop. The ordered sequence is the same everywhere:

1. `server.should_exit = True`, so uvicorn stops accepting connections.
2. `runners.cancel_all()`, so every `working` / `input-required` Task ends as `canceled` and its
   SSE stream closes.
3. Wait until the app lifespan shutdown has finished. That is where `McpManager.stop(grace=
   ctl.grace_s)` and `Database.close()` run. The lifespan calls `ctl.lifespan_finished()` as its
   last step.
4. Return exit status 0.

Windows delivers three kinds of event:
- `CTRL_C_EVENT` and `CTRL_BREAK_EVENT` reach CPython as `SIGINT` and `SIGBREAK`.
- `CTRL_CLOSE_EVENT` is not mapped by CPython, so a ctypes `SetConsoleCtrlHandler` callback takes
  it. The callback runs on a system thread, and Windows ends the process as soon as it returns, so
  it blocks (at most 4.5 s) until shutdown has finished.
- Every other control event is declined (`FALSE`), so CPython's own handler still runs.

The Windows calls sit behind `sys.platform == "win32"` checks, so type checkers on other hosts do
not see Windows-only names. `make_console_ctrl_handler` has no Windows-only names and runs on any
OS, which lets tests call the callback directly.
"""

import asyncio
import contextlib
import inspect
import signal
import sys
import threading
from asyncio import AbstractEventLoop
from collections.abc import Callable
from enum import Enum
from types import FrameType
from typing import Final, Protocol

from local_agent.logging_setup import get_logger
from local_agent.models import Clock, SystemClock
from local_agent.oscompat import _win32

if sys.platform == "win32":
    import ctypes

__all__ = [
    "CONSOLE_CLOSE_GRACE_S",
    "CONSOLE_CLOSE_WAIT_S",
    "DEFAULT_GRACE_S",
    "RunnerRegistry",
    "ShutdownController",
    "ShutdownLogger",
    "ShutdownReason",
    "ShutdownServer",
    "install",
    "make_console_ctrl_handler",
]

DEFAULT_GRACE_S: Final = 5.0
"""Process_Tree grace period after a Shutdown_Signal (6.11, 17.11)."""

CONSOLE_CLOSE_GRACE_S: Final = 2.0
"""Grace period after a Windows console close event (17.15)."""

CONSOLE_CLOSE_WAIT_S: Final = 4.5
"""How long the `CTRL_CLOSE_EVENT` callback blocks. Windows ends the process about 5 s after the
event, so 0.5 s of margin remains (17.15)."""

_FALSE: Final = 0
_TRUE: Final = 1


class ShutdownReason(Enum):
    """Why shutdown was requested."""

    SIGNAL = "signal"
    CONSOLE_CLOSE = "console_close"


class ShutdownServer(Protocol):
    """The slice of `uvicorn.Server` the controller uses."""

    should_exit: bool


class RunnerRegistry(Protocol):
    """The slice of `orchestrator.loop.TaskRunnerRegistry` the controller uses.

    `cancel_all` moves every `working` / `input-required` Task to `canceled` (2.7). It may be a
    coroutine function or a plain function; a returned awaitable is awaited.
    """

    def cancel_all(self) -> object: ...


class ShutdownLogger(Protocol):
    """The slice of a structlog logger this module uses: `event` plus keyword fields."""

    def info(self, event: str, **fields: object) -> None: ...

    def warning(self, event: str, **fields: object) -> None: ...

    def error(self, event: str, **fields: object) -> None: ...


_log: Final[ShutdownLogger] = get_logger("oscompat.signals")


class ShutdownController:
    """Runs Graceful_Shutdown once, after the first request (17.13, 17.14).

    All methods except `done` and the properties are meant for the event loop thread. Code on
    another thread must go through `loop.call_soon_threadsafe(ctl.request, ...)`.
    """

    def __init__(
        self,
        server: ShutdownServer,
        runners: RunnerRegistry,
        *,
        clock: Clock | None = None,
        log: ShutdownLogger | None = None,
    ) -> None:
        self._server = server
        self._runners = runners
        self._clock: Clock = SystemClock() if clock is None else clock
        self._log: ShutdownLogger = _log if log is None else log
        self._requested = asyncio.Event()
        self._lifespan_done = asyncio.Event()
        self._done = threading.Event()
        self._reason: ShutdownReason | None = None
        self._forced = False
        self._running = False

    @property
    def reason(self) -> ShutdownReason | None:
        """The reason of the first request, or `None` before any request."""
        return self._reason

    @property
    def requested(self) -> bool:
        return self._reason is not None

    @property
    def forced(self) -> bool:
        """True once a second request arrived."""
        return self._forced

    @property
    def done(self) -> threading.Event:
        """Set when `run()` has finished. The console close callback blocks on it (17.15)."""
        return self._done

    @property
    def grace_s(self) -> float:
        """Process_Tree grace period: 5.0, 2.0 for a console close, 0.0 after a second request.

        `McpManager.stop(grace=...)` and the stdio transports read this when they terminate a
        tree, so a request that arrives mid-shutdown still shortens the wait (17.14).
        """
        if self._forced:
            return 0.0
        if self._reason is ShutdownReason.CONSOLE_CLOSE:
            return CONSOLE_CLOSE_GRACE_S
        return DEFAULT_GRACE_S

    def request(self, reason: ShutdownReason) -> None:
        """Ask for shutdown. The first call starts it; every later call only forces `grace_s` 0."""
        if self._reason is None:
            self._reason = reason
            self._requested.set()
        else:
            self._forced = True

    def lifespan_finished(self) -> None:
        """Tell the controller the app lifespan shutdown is over.

        The lifespan calls this after `McpManager.stop(...)` and `Database.close()`.
        """
        self._lifespan_done.set()

    async def run(self) -> int:
        """Wait for the first request, run Graceful_Shutdown, and return exit status 0.

        Call it once, as a task started at startup. It does not return before a request arrives.
        A failure in the Task cancellation step is logged and does not stop the sequence (17.14).
        """
        if self._running:
            raise RuntimeError("ShutdownController.run() may only be called once")
        self._running = True
        try:
            await self._requested.wait()
            started = self._clock.monotonic()
            reason = self._reason
            self._log.info("shutdown_started", reason=None if reason is None else reason.value)
            self._server.should_exit = True
            await self._cancel_tasks()
            await self._lifespan_done.wait()
            self._log.info(
                "shutdown_finished", duration_ms=round((self._clock.monotonic() - started) * 1000)
            )
            return 0
        finally:
            self._done.set()

    async def _cancel_tasks(self) -> None:
        try:
            outcome = self._runners.cancel_all()
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:
            self._log.error("shutdown_cancel_tasks_failed", error=repr(exc))


def make_console_ctrl_handler(
    loop: AbstractEventLoop,
    ctl: ShutdownController,
    *,
    wait_s: float = CONSOLE_CLOSE_WAIT_S,
) -> Callable[[int], int]:
    """Build the `SetConsoleCtrlHandler` callback as a plain Python function.

    It runs on a system thread. For `CTRL_CLOSE_EVENT` it schedules `ctl.request(CONSOLE_CLOSE)`
    on the loop, then blocks until shutdown has finished or `wait_s` has passed, and returns
    `TRUE`. For any other event it returns `FALSE` without side effects, so CPython's handler
    still runs (17.15). If the loop is already closed there is nothing to shut down, so it
    returns `FALSE`.
    """

    def handler(ctrl_type: int) -> int:
        if ctrl_type != _win32.CTRL_CLOSE_EVENT:
            return _FALSE
        try:
            loop.call_soon_threadsafe(ctl.request, ShutdownReason.CONSOLE_CLOSE)
        except RuntimeError:
            return _FALSE
        ctl.done.wait(wait_s)
        return _TRUE

    return handler


# Keeps the ctypes callback objects alive while installed. A garbage-collected callback would make
# Windows call freed memory.
_live_callbacks: list[object] = []


def install(loop: AbstractEventLoop, ctl: ShutdownController, platform: str) -> Callable[[], None]:
    """Route the OS shutdown signals of `platform` to `ctl.request`; return the uninstall function.

    Call it from the main thread, on the loop obtained from `runner.get_loop()`, before
    `runner.run(...)`. `asyncio.Runner` then finds a non-default `SIGINT` handler and leaves it.
    """
    if platform == "win32":
        return _install_windows(loop, ctl)
    return _install_posix(loop, ctl)


def _install_posix(loop: AbstractEventLoop, ctl: ShutdownController) -> Callable[[], None]:
    """`SIGINT` and `SIGTERM` through `loop.add_signal_handler` (asyncio and uvloop)."""
    if sys.platform == "win32":
        raise RuntimeError("POSIX signal handlers are not available on Windows")
    else:
        signals = (signal.SIGINT, signal.SIGTERM)
        for sig in signals:
            loop.add_signal_handler(sig, ctl.request, ShutdownReason.SIGNAL)
        installed = True

        def uninstall() -> None:
            nonlocal installed
            if not installed:
                return
            installed = False
            for sig in signals:
                loop.remove_signal_handler(sig)

        return uninstall


def _install_windows(loop: AbstractEventLoop, ctl: ShutdownController) -> Callable[[], None]:
    """`SIGINT` / `SIGBREAK` through `signal.signal`, `CTRL_CLOSE_EVENT` through ctypes."""
    if sys.platform == "win32":

        def on_signal(_signum: int, _frame: FrameType | None) -> None:
            # Runs in the main thread between bytecodes, possibly while the loop is idle. The
            # Proactor loop's wakeup fd makes `call_soon_threadsafe` take effect promptly.
            # A closed loop has nothing left to shut down.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(ctl.request, ShutdownReason.SIGNAL)

        previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGBREAK)}

        routine = _win32.HANDLER_ROUTINE(make_console_ctrl_handler(loop, ctl))
        _live_callbacks.append(routine)
        if not _win32.SetConsoleCtrlHandler(routine, True):
            # The signal handlers above still work; only the console close event is lost.
            code = ctypes.get_last_error()
            reason = ctypes.FormatError(code).strip() if code else "no error code"
            _log.warning("console_ctrl_handler_failed", error=f"{reason} (error {code})")
        installed = True

        def uninstall() -> None:
            nonlocal installed
            if not installed:
                return
            installed = False
            _win32.SetConsoleCtrlHandler(routine, False)
            for sig, old in previous.items():
                signal.signal(sig, signal.SIG_DFL if old is None else old)
            _live_callbacks.remove(routine)

        return uninstall
    else:
        raise RuntimeError("Windows console handlers are only available on Windows")
