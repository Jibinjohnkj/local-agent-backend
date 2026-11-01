"""`BackendServer`: uvicorn on a loop and signals that the Backend owns (17.13, 17.14, 17.28).

`uvicorn.Server.serve()` wraps its work in `capture_signals()`. That context manager installs
`signal.signal` handlers and, after a signal-driven shutdown, raises the captured signal again, so
the process would end with a signal status instead of 0. The Backend installs its own handlers
(`oscompat/signals.py`) and drives `should_exit` through the `ShutdownController`, so
`BackendServer` turns `capture_signals()` into a no-op.

`capture_signals` is not documented uvicorn API. The pinned version is covered by
`test_shutdown_exit_code`.
"""

import contextlib
from collections.abc import Iterator
from typing import Final

import uvicorn
from starlette.types import ASGIApp

__all__ = ["GRACEFUL_SHUTDOWN_TIMEOUT_S", "BackendServer", "build_server"]

GRACEFUL_SHUTDOWN_TIMEOUT_S: Final = 2
"""Seconds uvicorn waits for open connections before the lifespan shutdown (17.14)."""


class BackendServer(uvicorn.Server):
    """A `uvicorn.Server` that never installs or re-raises OS signal handlers."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def build_server(app: ASGIApp, host: str, port: int) -> BackendServer:
    """Build the `BackendServer` for `app`.

    `loop="none"` because `__main__` creates the loop. The sockets are bound beforehand and passed
    to `serve(sockets=...)`, so `host` and `port` only describe them. uvicorn's own log
    configuration and access log are off: the Backend writes its records through structlog.
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        loop="none",
        lifespan="on",
        ws="none",
        log_config=None,
        access_log=False,
        server_header=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_S,
    )
    return BackendServer(config)
