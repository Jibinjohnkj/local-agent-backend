"""Event loop factory selection (17.28, 17.29).

The Backend owns its event loop. `__main__` passes the factory returned here to
`asyncio.Runner(loop_factory=...)`. Nothing in this module touches the deprecated asyncio
event-loop policy API, which Python 3.14 deprecates and 3.16 removes.
"""

import asyncio
import importlib
import sys
from asyncio import AbstractEventLoop
from collections.abc import Callable


def select_loop_factory(
    platform: str, uvloop_new_event_loop: Callable[[], AbstractEventLoop] | None
) -> Callable[[], AbstractEventLoop]:
    """Pick the loop factory for `platform`; a pure function with no imports or I/O.

    macOS and Linux use uvloop when the caller found it importable (17.29). Everything else
    gets `asyncio.EventLoop`, which is the Proactor loop on Windows (needed for asyncio
    subprocesses) and the selector loop on POSIX.
    """
    if platform != "win32" and uvloop_new_event_loop is not None:
        return uvloop_new_event_loop
    return asyncio.EventLoop


def _import_uvloop_new_event_loop() -> Callable[[], AbstractEventLoop] | None:
    """Return `uvloop.new_event_loop`, or None when uvloop is not installed.

    Never called on Windows: uvloop has no Windows build, so the import is skipped there.
    """
    try:
        uvloop = importlib.import_module("uvloop")
    except ImportError:
        return None
    factory: Callable[[], AbstractEventLoop] = uvloop.new_event_loop
    return factory


def default_loop_factory() -> Callable[[], AbstractEventLoop]:
    """Return the loop factory for the running interpreter.

    Imports `uvloop` only when `sys.platform != "win32"`, then delegates to
    `select_loop_factory`.
    """
    platform = sys.platform
    uvloop_new_event_loop = None if platform == "win32" else _import_uvloop_new_event_loop()
    return select_loop_factory(platform, uvloop_new_event_loop)
