"""Run the real Backend entry point and record which event loop it serves on.

Usage: ``python loop_probe_backend.py OUT_FILE`` with ``AGENT_CONFIG`` set, like ``python -m
local_agent``. Everything is production code except one wrapper around ``StartupSteps.serve``.
When the wrapper runs, the Backend's loop is already running, and the wrapper writes the loop's
class (module and qualified name) to ``OUT_FILE`` as JSON before it hands control to the real
``serve``. The exit status is the Backend's own.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

from local_agent import __main__ as entry
from local_agent.oscompat.signals import ShutdownController
from local_agent.server import BackendServer


def record_loop(out_file: Path, loop: asyncio.AbstractEventLoop) -> None:
    """Write the class of `loop` to `out_file`. Called once, before the server starts."""
    loop_class = type(loop)
    record = {"module": loop_class.__module__, "name": loop_class.__qualname__}
    out_file.write_text(json.dumps(record), encoding="utf-8")


def main(out_file: Path) -> int:
    steps = entry.StartupSteps()
    real_serve = steps.serve

    async def probing_serve(
        server: BackendServer, controller: ShutdownController, sockets: Sequence[socket.socket]
    ) -> int:
        record_loop(out_file, asyncio.get_running_loop())
        return await real_serve(server, controller, sockets)

    return entry.main(steps=dataclasses.replace(steps, serve=probing_serve))


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
