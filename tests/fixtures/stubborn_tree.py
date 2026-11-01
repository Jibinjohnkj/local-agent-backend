"""A process tree that ignores the graceful stop signal, for the Windows `taskkill` test (22.11).

Usage: ``stubborn_tree.py PID_FILE DEPTH``.

Every process ignores ``SIGBREAK`` (the signal behind ``CTRL_BREAK_EVENT``), appends its pid as one
line to PID_FILE, and, while DEPTH is above zero, starts a child of itself with DEPTH - 1. Then it
sleeps for ``SLEEP_S`` seconds, so a leaked tree expires on its own. It never reads stdin, so
closing stdin does nothing either. Only a forced kill of the whole tree ends it.

The pid is written after the signal is ignored, so a pid in the file belongs to a process that no
longer reacts to ``CTRL_BREAK_EVENT``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

SLEEP_S: Final = 600


def main(argv: list[str]) -> None:
    pid_file, depth = Path(argv[0]), int(argv[1])
    sigbreak = getattr(signal, "SIGBREAK", None)  # Windows only
    if sigbreak is not None:
        signal.signal(sigbreak, signal.SIG_IGN)
    with pid_file.open("a", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
    if depth > 0:
        subprocess.Popen(
            [sys.executable, __file__, str(pid_file), str(depth - 1)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(SLEEP_S)


if __name__ == "__main__":
    main(sys.argv[1:])
