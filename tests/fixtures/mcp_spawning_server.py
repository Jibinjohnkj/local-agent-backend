"""Stdio MCP server that owns a three-level process tree, for process-tree cleanup tests.

The server spawns a grandchild. The grandchild spawns a great-grandchild, and both sleep for
``SLEEP_S`` seconds, so a leaked tree expires on its own. The ``get_pids`` tool returns
``{"server": pid, "descendants": [grandchild_pid, great_grandchild_pid]}``.

Behaviour on termination (at most one flag; the default is ``--cooperative``):

``--cooperative``
    Exits on stdin EOF after stopping its own descendants.
``--ignore-graceful``
    Ignores stdin EOF and ``SIGTERM`` / ``CTRL_BREAK_EVENT``. Only a forced kill ends it.
    The descendants keep the default signal behaviour.
``--orphan``
    Exits immediately on stdin EOF and leaves its descendants running.

Other flags:

``--no-initialize``
    Spawns the tree, then reads stdin without ever answering ``initialize`` (failed handshake).
    The termination flags above still apply once stdin closes.
``--spawn-first``
    Spawns the grandchild as the first instruction, before the MCP SDK is imported.
``--pid-file PATH``
    Writes ``{"server": ..., "descendants": [...]}`` to PATH as soon as each pid is known, so a
    test can record every pid even when the handshake never completes. Rewritten atomically.

Only the standard library is imported at module level. Nothing but the MCP protocol may reach
stdout, and the children get ``DEVNULL`` for stdout and stderr so they never hold that pipe open.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.server.mcpserver import MCPServer

SLEEP_S: Final = 600
_PID_WAIT_S: Final = 10.0
_CLEANUP_WAIT_S: Final = 2.0

# One line, so the argument survives Windows command-line quoting. Prints the pid of its own child.
_GRANDCHILD_CODE: Final = (
    "import subprocess, sys, time; "
    "p = subprocess.Popen("
    f"[sys.executable, '-c', 'import time; time.sleep({SLEEP_S})'], "
    "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
    "print(p.pid, flush=True); "
    f"time.sleep({SLEEP_S})"
)


class ProcessTree:
    """The grandchild and great-grandchild, with their pids published as they become known."""

    def __init__(self, pid_file: Path | None) -> None:
        self._pid_file = pid_file
        self._grandchild: subprocess.Popen[str] | None = None
        self._great_grandchild: int | None = None
        self._settled = threading.Event()
        self._write_lock = threading.Lock()

    def spawn(self) -> None:
        """Start the grandchild. The children stay in the server's group, session, and job."""
        self._grandchild = subprocess.Popen(
            [sys.executable, "-c", _GRANDCHILD_CODE],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._write_pid_file()
        threading.Thread(target=self._read_great_grandchild, daemon=True).start()

    def snapshot(self) -> dict[str, int | list[int]]:
        """Wait for the great-grandchild pid, then return the tool payload."""
        self._settled.wait(_PID_WAIT_S)
        return self._payload()

    def stop_descendants(self) -> None:
        """Stop both descendants, as a cooperative server does before it exits."""
        self._settled.wait(_CLEANUP_WAIT_S)
        if self._great_grandchild is not None:
            _terminate(self._great_grandchild)
        if self._grandchild is not None:
            _terminate(self._grandchild.pid)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._grandchild.wait(_CLEANUP_WAIT_S)

    def _payload(self) -> dict[str, int | list[int]]:
        descendants: list[int] = []
        if self._grandchild is not None:
            descendants.append(self._grandchild.pid)
        if self._great_grandchild is not None:
            descendants.append(self._great_grandchild)
        return {"server": os.getpid(), "descendants": descendants}

    def _read_great_grandchild(self) -> None:
        assert self._grandchild is not None
        assert self._grandchild.stdout is not None
        try:
            line = self._grandchild.stdout.readline()
            self._great_grandchild = int(line.strip())
        except OSError, ValueError:
            self._great_grandchild = None
        self._write_pid_file()
        self._settled.set()

    def _write_pid_file(self) -> None:
        if self._pid_file is None:
            return
        text = json.dumps(self._payload())
        with self._write_lock:
            temp = self._pid_file.with_name(self._pid_file.name + ".tmp")
            try:
                temp.write_text(text, encoding="utf-8")
                os.replace(temp, self._pid_file)
            except OSError:
                # A reader may hold the file open (Windows). A direct write is good enough.
                self._pid_file.write_text(text, encoding="utf-8")


def _terminate(pid: int) -> None:
    """`SIGTERM` on POSIX, `TerminateProcess` on Windows. A missing process is fine."""
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mcp_spawning_server")
    behaviour = parser.add_mutually_exclusive_group()
    behaviour.add_argument("--cooperative", action="store_true", help="exit on stdin EOF (default)")
    behaviour.add_argument("--ignore-graceful", action="store_true")
    behaviour.add_argument("--orphan", action="store_true")
    parser.add_argument("--no-initialize", action="store_true")
    parser.add_argument("--spawn-first", action="store_true")
    parser.add_argument("--pid-file", type=Path, default=None)
    return parser.parse_args(argv)


def _ignore_graceful_signals() -> None:
    """Ignore the graceful stop signals. Call after spawning, because `SIG_IGN` is inherited."""
    for name in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.SIG_IGN)


def _build_server(tree: ProcessTree) -> MCPServer:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("spawning-fixture")

    @server.tool()
    def get_pids() -> dict[str, int | list[int]]:
        """Return the server pid and the pids of its descendants."""
        return tree.snapshot()

    return server


def _drain_stdin() -> None:
    """Read stdin to EOF without writing anything: the handshake never completes."""
    stream = sys.stdin.buffer
    while stream.read(65536):
        pass


def _exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    tree = ProcessTree(args.pid_file)
    if args.spawn_first:
        tree.spawn()

    server = None if args.no_initialize else _build_server(tree)
    if not args.spawn_first:
        tree.spawn()
    if args.ignore_graceful:
        _ignore_graceful_signals()

    if server is None:
        _drain_stdin()
    else:
        server.run("stdio")

    # stdin is closed here.
    if args.ignore_graceful:
        while True:
            time.sleep(3600)
    if not args.orphan:
        tree.stop_descendants()
    _exit(0)


if __name__ == "__main__":
    main()
