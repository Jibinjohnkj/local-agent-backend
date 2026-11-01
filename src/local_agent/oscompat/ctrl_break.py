"""Send `CTRL_BREAK_EVENT` to the console of a child process tree (Requirements 17.10, 17.11).

`GenerateConsoleCtrlEvent` only reaches processes attached to the caller's console, and an MCP
stdio child runs on its own hidden console. The Backend cannot detach from its own console without
losing its stderr logging. So `terminate_tree` runs this module in a short-lived helper process:

    sys.executable -m local_agent.oscompat.ctrl_break <pid>

The helper takes four steps, in this order:

1. `FreeConsole()` detaches the helper from the console it inherited.
2. `AttachConsole(pid)` joins the console of the child.
3. `SetConsoleCtrlHandler(NULL, TRUE)` makes the helper ignore Ctrl+C, so the event does not kill
   the helper itself.
4. `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0)` signals every process on that console, which is
   the whole Process_Tree.

The helper exits with `0` after the event was generated. Any other status means the event was not
sent, and the caller skips this step and goes on to forced termination. A message with the failing
call and the Windows error text goes to stderr. The caller applies the 1 s timeout.

Exit statuses:
- `0`: the event was generated.
- `1`: a Windows call failed.
- `2`: bad command line (not exactly one positive integer).
- `3`: not running on Windows.

The Windows calls go through the module (`_win32.FreeConsole`), so a test can replace one binding.
They sit behind a `sys.platform == "win32"` check, so type checkers on other hosts do not see the
Windows-only names.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Final

if sys.platform == "win32":
    import ctypes

    from local_agent.oscompat import _win32

__all__ = ["EXIT_NOT_WINDOWS", "EXIT_OK", "EXIT_USAGE", "EXIT_WIN32_FAILURE", "main"]

EXIT_OK: Final = 0
EXIT_WIN32_FAILURE: Final = 1
EXIT_USAGE: Final = 2
EXIT_NOT_WINDOWS: Final = 3

_USAGE: Final = "usage: python -m local_agent.oscompat.ctrl_break <pid>"


def _parse_pid(argv: Sequence[str]) -> int | None:
    """Return the pid in `argv`, or `None` unless it is exactly one positive integer."""
    if len(argv) != 1:
        return None
    text = argv[0]
    if not (text.isascii() and text.isdigit()):
        return None
    pid = int(text)
    return pid if pid > 0 else None


if sys.platform == "win32":

    def _fail(call: str) -> int:
        """Report the failed Windows call with its error text on stderr; return the status."""
        error = ctypes.get_last_error()
        reason = ctypes.FormatError(error).strip() if error else "no error code"
        sys.stderr.write(f"ctrl_break: {call} failed: {reason} (error {error})\n")
        return EXIT_WIN32_FAILURE

    def _send_ctrl_break(pid: int) -> int:
        """Run the four steps against the console of `pid`; return the exit status."""
        # A helper started without a console makes `FreeConsole` fail. It has nothing to detach
        # from then, so that failure is not an error.
        _win32.FreeConsole()
        if not _win32.AttachConsole(pid):
            return _fail("AttachConsole")
        if not _win32.SetConsoleCtrlHandler(None, True):
            return _fail("SetConsoleCtrlHandler")
        if not _win32.GenerateConsoleCtrlEvent(_win32.CTRL_BREAK_EVENT, 0):
            return _fail("GenerateConsoleCtrlEvent")
        return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the four-step sequence for the pid in `argv` (default `sys.argv[1:]`)."""
    args = sys.argv[1:] if argv is None else argv
    pid = _parse_pid(args)
    if pid is None:
        sys.stderr.write(f"{_USAGE}\n")
        return EXIT_USAGE
    if sys.platform == "win32":
        return _send_ctrl_break(pid)
    else:
        sys.stderr.write("ctrl_break: only supported on Windows\n")
        return EXIT_NOT_WINDOWS


if __name__ == "__main__":
    sys.exit(main())
