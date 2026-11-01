"""Subprocess support for `stdio` MCP_Servers (Requirements 17.4 to 17.12).

This module grows across several tasks. Each section below is self-contained, and the pure
decision cores come first so property tests can reach them without any I/O:

1. Executable resolution (17.4 to 17.6): `resolve_command`, `Resolved`, `NotFound`.
2. Windows command lines (17.7 to 17.9): `build_command_line`, `quote_batch_args`,
   `BatchArgRejected`.
3. Spawn specification (17.10, 17.19, 17.20, 17.22): `SpawnSpec`, `SpawnRejected`,
   `build_spawn_spec`. This is the only section that reads the environment and the file system,
   and it does both off the event loop (14.1).
4. Termination plan (17.11): `Step`, `TerminationReport`, `termination_plan`.
5. Spawning and tree termination (6.11, 14.1, 17.10 to 17.12, 17.15): `TreeHandle`, `spawn`,
   `terminate_tree`, and the injectable OS-call layer `TreeOps`. This is the only section that
   starts processes and sends signals. Its schedule lives in `terminate_tree`, which takes the
   OS-call layer, a clock, and a sleep function as arguments, so a fake tree can drive it.

Every function in the pure sections takes the platform as an argument and touches neither the file
system nor the environment. File existence arrives through an injected `is_file` callable, so the
Windows rules are testable on macOS and Linux (and the reverse).

`platform` takes `sys.platform` values: `"win32"` selects the Windows rules, and every other value
selects the POSIX rules.
"""

import asyncio
import contextlib
import ctypes
import ntpath
import os
import posixpath
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, Protocol, cast

import anyio
import anyio.abc
from anyio.streams.file import FileReadStream, FileWriteStream

from local_agent.logging_setup import get_logger
from local_agent.models import Clock, SystemClock
from local_agent.oscompat import _win32
from local_agent.oscompat.env import inherited_env, lookup_env, merge_env
from local_agent.oscompat.paths import PathFieldError, has_separator, resolve_path_field

if TYPE_CHECKING:
    from local_agent.config.models import StdioServerConfig

__all__ = [
    "CREATE_NEW_PROCESS_GROUP",
    "CREATE_NO_WINDOW",
    "CREATE_SUSPENDED",
    "CTRL_BREAK_TIMEOUT_S",
    "DEFAULT_GRACE_S",
    "DEFAULT_PATHEXT",
    "REAP_TIMEOUT_S",
    "STDIN_WAIT_S",
    "WINDOWS_CREATION_FLAGS",
    "BatchArgRejected",
    "NotFound",
    "ProcessLogger",
    "Resolved",
    "SpawnRejected",
    "SpawnSpec",
    "Step",
    "TerminationReport",
    "TreeHandle",
    "TreeOps",
    "build_command_line",
    "build_spawn_spec",
    "quote_batch_args",
    "resolve_command",
    "spawn",
    "system_tree_ops",
    "terminate_tree",
    "termination_plan",
]

_WIN32: Final = "win32"


@dataclass(frozen=True, slots=True)
class _PathOps:
    """The path operations and `PATH` separator of one platform family."""

    join: Callable[[str, str], str]
    splitext: Callable[[str], tuple[str, str]]
    dirname: Callable[[str], str]
    path_separator: str


_WINDOWS_OPS: Final = _PathOps(
    join=ntpath.join,
    splitext=ntpath.splitext,
    dirname=ntpath.dirname,
    path_separator=";",
)
_POSIX_OPS: Final = _PathOps(
    join=posixpath.join,
    splitext=posixpath.splitext,
    dirname=posixpath.dirname,
    path_separator=":",
)


def _ops(platform: str) -> _PathOps:
    return _WINDOWS_OPS if platform == _WIN32 else _POSIX_OPS


# ---------------------------------------------------------------------------------------------
# 1. Executable resolution (17.4 to 17.6)
# ---------------------------------------------------------------------------------------------

# Windows default when the child's environment has no `PATHEXT` (design: "Executable resolution").
DEFAULT_PATHEXT: Final = ".COM;.EXE;.BAT;.CMD"

_NATIVE_SUFFIXES: Final = frozenset({".exe", ".com"})
_BATCH_SUFFIXES: Final = frozenset({".cmd", ".bat"})


@dataclass(frozen=True, slots=True)
class Resolved:
    """An executable the resolver found.

    `kind` decides how the command line is built: `native` goes to `CreateProcess` (or `exec`)
    directly, and `batch` (Windows `.cmd`/`.bat` only) goes through `cmd.exe` with quoting.
    """

    path: str
    kind: Literal["native", "batch"]


@dataclass(frozen=True, slots=True)
class NotFound:
    """No launchable executable exists for `command` (17.6).

    `searched` lists the directories that were probed, in order. `found` is set only when a file
    matched but its suffix is not launchable (for example `tool.ps1`): the message then says so
    instead of claiming that nothing exists.
    """

    command: str
    searched: tuple[str, ...]
    found: str | None = None

    @property
    def message(self) -> str:
        """Human-readable reason; it names the command and every searched directory."""
        searched = ", ".join(self.searched) if self.searched else "(none)"
        if self.found is None:
            return f"no executable found for command '{self.command}' (searched: {searched})"
        return (
            f"'{self.found}' is not a launchable executable for command '{self.command}' "
            f"(searched: {searched})"
        )


def _parse_pathext(pathext_value: str | None) -> tuple[str, ...]:
    """Split a `PATHEXT` value into its extensions, keeping order and spelling.

    A missing value (`None`) means `DEFAULT_PATHEXT`. Blank pieces and pieces that do not start
    with a dot followed by a name cannot be appended to a file name meaningfully, so they are
    dropped.
    """
    value = DEFAULT_PATHEXT if pathext_value is None else pathext_value
    pieces = (piece.strip() for piece in value.split(";"))
    return tuple(piece for piece in pieces if len(piece) > 1 and piece.startswith("."))


def _path_directories(
    path_value: str, *, platform: str, base_dir: str, ops: _PathOps
) -> tuple[str, ...]:
    """The `PATH` directories to search, in order, as absolute normalized paths (17.4).

    - Empty entries are skipped: an empty entry would mean "the current directory" to some
      shells, and the resolver never searches it implicitly.
    - On Windows, surrounding double quotes are stripped first.
    - A relative entry is joined to `base_dir`, the spawn working directory. A leading `~` is an
      ordinary directory name here, because `exec` does not expand it.
    - A Windows drive-relative entry such as `C:bin` has no single meaning, so it is skipped.
    """
    directories: list[str] = []
    for raw in path_value.split(ops.path_separator):
        entry = raw.strip('"') if platform == _WIN32 else raw
        if not entry:
            continue
        if entry.startswith("~"):
            entry = "./" + entry
        try:
            directories.append(
                resolve_path_field(entry, base_dir=base_dir, home=None, platform=platform)
            )
        except PathFieldError:
            continue
    return tuple(directories)


def _classify(
    path: str, *, command: str, searched: tuple[str, ...], platform: str, ops: _PathOps
) -> Resolved | NotFound:
    """Decide how a found file is launched.

    POSIX launches any executable file as it is. On Windows the suffix decides: `.cmd`/`.bat` are
    `batch`, `.exe`/`.com` are `native`, and everything else (`.ps1`, `.vbs`, no suffix) is not
    something `CreateProcess` can run, so it is `NotFound`.
    """
    if platform != _WIN32:
        return Resolved(path, "native")
    suffix = ops.splitext(path)[1].lower()
    if suffix in _BATCH_SUFFIXES:
        return Resolved(path, "batch")
    if suffix in _NATIVE_SUFFIXES:
        return Resolved(path, "native")
    return NotFound(command, searched, found=path)


def _resolve_with_separator(
    command: str,
    *,
    platform: str,
    extensions: tuple[str, ...],
    base_dir: str,
    home: str | None,
    is_file: Callable[[str], bool],
    ops: _PathOps,
) -> Resolved | NotFound:
    """Resolve a command that contains a path separator, without searching `PATH` (17.5).

    The command is a Path_Field. The exact path is probed first. Only on Windows, and only when
    the path has no extension, are the `PATHEXT` extensions then tried in order.
    """
    path = resolve_path_field(command, base_dir=base_dir, home=home, platform=platform)
    candidates = [path]
    if platform == _WIN32 and not ops.splitext(path)[1]:
        candidates.extend(path + extension for extension in extensions)
    searched = (ops.dirname(path),)
    for candidate in candidates:
        if is_file(candidate):
            return _classify(
                candidate, command=command, searched=searched, platform=platform, ops=ops
            )
    return NotFound(command, searched)


def _resolve_on_path(
    command: str,
    *,
    platform: str,
    path_value: str,
    extensions: tuple[str, ...],
    base_dir: str,
    is_file: Callable[[str], bool],
    ops: _PathOps,
) -> Resolved | NotFound:
    """Resolve a bare command name against the `PATH` directories in order (17.4).

    On Windows, when the name's suffix is not in `PATHEXT` (compared case-insensitively), each
    directory is probed with `name + extension` for every extension in order: directory order
    outer, extension order inner, like cmd.exe. A suffix that is in `PATHEXT` probes only the
    exact name. The bare name is never probed on its own when it has no `PATHEXT` suffix, so the
    extensionless shell script that ships next to `npx.cmd` cannot shadow it.
    """
    if platform == _WIN32 and ops.splitext(command)[1].lower() not in {
        extension.lower() for extension in extensions
    }:
        names = tuple(command + extension for extension in extensions)
    else:
        names = (command,)
    directories = _path_directories(path_value, platform=platform, base_dir=base_dir, ops=ops)
    for directory in directories:
        for name in names:
            candidate = ops.join(directory, name)
            if is_file(candidate):
                return _classify(
                    candidate, command=command, searched=directories, platform=platform, ops=ops
                )
    return NotFound(command, directories)


def resolve_command(
    command: str,
    *,
    platform: str,
    path_value: str,
    pathext_value: str | None,
    base_dir: str,
    home: str | None,
    is_file: Callable[[str], bool],
) -> Resolved | NotFound:
    """Find the executable for a `stdio` MCP_Server `command` (17.4 to 17.6).

    Pure: the only file system access is the `is_file` callable. `path_value` and `pathext_value`
    belong to the **child's** environment, not the Backend's. `base_dir` is the spawn working
    directory: it anchors relative `PATH` entries and, for a command with a separator, a relative
    Path_Field (the config loader has already made that absolute).

    - A command with a separator (`has_separator`) is a Path_Field and `PATH` is not searched.
    - Any other command is looked up through `path_value` in order.
    - A missing `pathext_value` means `DEFAULT_PATHEXT`; it is read on Windows only.
    - The first existing candidate decides the answer, even if its suffix is not launchable.

    Raises `PathFieldError` only for a separator command that cannot be resolved (`~` without a
    `home`, or a drive-relative Windows path); the config loader rejects those first.
    """
    ops = _ops(platform)
    extensions = _parse_pathext(pathext_value) if platform == _WIN32 else ()
    if has_separator(command, platform):
        return _resolve_with_separator(
            command,
            platform=platform,
            extensions=extensions,
            base_dir=base_dir,
            home=home,
            is_file=is_file,
            ops=ops,
        )
    return _resolve_on_path(
        command,
        platform=platform,
        path_value=path_value,
        extensions=extensions,
        base_dir=base_dir,
        is_file=is_file,
        ops=ops,
    )


# ---------------------------------------------------------------------------------------------
# 2. Windows command lines (17.7 to 17.9)
# ---------------------------------------------------------------------------------------------

# `cmd.exe` ends a quoted region at `"` and expands `%VAR%` even inside quotes, and a line break
# ends the command. None of the four can be made literal, so a batch target refuses them (17.9).
_BATCH_FORBIDDEN: Final = frozenset('"%\r\n')

# Everything between `cmd.exe` and the quoted line: skip AutoRun (`/d`), keep command extensions
# (`/e:ON`), make `!` literal (`/v:OFF`), and run one command then exit (`/c`).
_CMD_SWITCHES: Final = "/d /e:ON /v:OFF /c"


class BatchArgRejected(ValueError):
    """A batch-file path or argument holds a character `cmd.exe` cannot take literally (17.9).

    `position` counts arguments from 1. The batch path itself is not an argument, so a bad
    character in the path reports `position` 0 (the `argv[0]` slot). `char` is the first
    offending character found in that string, scanning left to right.
    """

    def __init__(self, position: int, char: str) -> None:
        self.position = position
        self.char = char
        where = "the batch file path" if position == 0 else f"argument {position}"
        super().__init__(f"{where} contains the disallowed character {char!r}")


def _first_forbidden(text: str) -> str | None:
    return next((char for char in text if char in _BATCH_FORBIDDEN), None)


def _quote_one(text: str) -> str:
    """Wrap `text` in double quotes; trailing backslashes are doubled so the closing quote lives.

    Without the doubling, `C:\\dir\\` would end in `\\"`, and a C runtime parser reads that as an
    escaped quote. Other backslashes stay as they are because the text holds no `"`. An empty
    string comes out as `""`.
    """
    trailing = len(text) - len(text.rstrip("\\"))
    return f'"{text}{"\\" * trailing}"'


def quote_batch_args(batch_path: str, args: Sequence[str]) -> str:
    """Build the inner command line for a `.cmd`/`.bat` target (17.8, 17.9).

    Pure. The path and every argument are checked first, path before arguments, so the error
    names the first offender in order (`BatchArgRejected`). Then each piece is enclosed in
    double quotes and the pieces are joined with single spaces. Inside those quotes `& | < > ^`
    and spaces are literal to `cmd.exe`, and a C runtime program behind `%*` parses the original
    values back.
    """
    pieces = (batch_path, *args)
    for position, piece in enumerate(pieces):
        char = _first_forbidden(piece)
        if char is not None:
            raise BatchArgRejected(position, char)
    return " ".join(_quote_one(piece) for piece in pieces)


def build_command_line(resolved: Resolved, args: Sequence[str], system_root: str) -> str:
    """The Windows `CreateProcess` command line for a resolved executable (17.7, 17.8).

    Pure, and Windows-only in meaning (`system_root` is the `%SystemRoot%` value).

    - `native`: `subprocess.list2cmdline([path, *args])`. The Microsoft C runtime and
      `CommandLineToArgvW` parse that back into the same list for any argument without NUL.
    - `batch`: `<system_root>\\System32\\cmd.exe /d /e:ON /v:OFF /c "<quoted>"`, where
      `<quoted>` comes from `quote_batch_args`. `/c` strips the outer quotes and leaves the
      inner line intact. No `shell=True` and no `COMSPEC` are involved.

    Raises `BatchArgRejected` for a batch target whose path or arguments are disallowed.
    """
    if resolved.kind == "native":
        return subprocess.list2cmdline([resolved.path, *args])
    quoted = quote_batch_args(resolved.path, args)
    cmd_exe = ntpath.join(system_root, "System32", "cmd.exe")
    return f'{subprocess.list2cmdline([cmd_exe])} {_CMD_SWITCHES} "{quoted}"'


# ---------------------------------------------------------------------------------------------
# 3. Spawn specification (17.4, 17.6, 17.9, 17.10, 17.19, 17.20, 17.22, 14.1)
# ---------------------------------------------------------------------------------------------

# Windows process-creation flags. `subprocess.CREATE_*` exists on Windows only, so the Win32
# values are spelled out here and the spec can be built (and tested) on any host.
CREATE_NEW_PROCESS_GROUP: Final = 0x00000200
CREATE_SUSPENDED: Final = 0x00000004
CREATE_NO_WINDOW: Final = 0x08000000
WINDOWS_CREATION_FLAGS: Final = CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_NO_WINDOW

# Seconds the graceful step waits for the tree to exit before force (design: 5 s; 2 s on console
# close). The shutdown code overrides this at termination time.
DEFAULT_GRACE_S: Final = 5.0

# Used for the batch-file `cmd.exe` path when the child environment has neither variable.
_FALLBACK_SYSTEM_ROOT: Final = "C:\\Windows"


class SpawnRejected(Exception):
    """A `stdio` server cannot be launched, and the reason is already in `reason`.

    Raised for a command that resolves to nothing launchable (17.6), a batch target whose path or
    argument `cmd.exe` cannot take literally (17.9), and a Path_Field that cannot be resolved. The
    supervisor treats it as a failed connection attempt (6.7) and logs `reason`.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SpawnSpec:
    """Everything needed to start one `stdio` server, with no shell involved.

    POSIX fills `argv` (`[exe, *args]`, for `exec`) and leaves `command_line` and `application`
    `None`. Windows fills `application` (`lpApplicationName`: the executable, or `cmd.exe` for a
    batch target) and `command_line` (the full `CreateProcess` string) and leaves `argv` `None`.

    `env` holds the child's whole environment, secrets included, so it is left out of `repr`.
    `grace_s` is a default; the shutdown code sets the real value at termination time.
    """

    platform: str
    argv: tuple[str, ...] | None
    command_line: str | None
    application: str | None
    env: Mapping[str, str] = field(repr=False)
    cwd: str
    start_new_session: bool
    creationflags: int
    grace_s: float = DEFAULT_GRACE_S


def _default_is_file(platform: str) -> Callable[[str], bool]:
    """The real file probe: a regular file, and on POSIX one the process may execute."""
    if platform == "win32":
        return os.path.isfile

    def is_executable_file(path: str) -> bool:
        return os.path.isfile(path) and os.access(path, os.X_OK)

    return is_executable_file


def _home_dir(environ: Mapping[str, str], platform: str) -> str | None:
    """The user's home directory from the Backend's own environment, for a leading `~`."""
    name = "USERPROFILE" if platform == "win32" else "HOME"
    return lookup_env(environ, name, platform) or None


def _resolve_or_reject(
    cfg: StdioServerConfig,
    *,
    config_dir: str,
    cwd: str,
    home: str | None,
    env: Mapping[str, str],
    platform: str,
    is_file: Callable[[str], bool],
) -> Resolved:
    """Find the executable (blocking file probes) or raise `SpawnRejected` (17.4 to 17.6).

    `PATH` and `PATHEXT` come from the merged child environment, not the Backend's. A command with
    a separator is a Path_Field and resolves against the config directory; any other command is
    searched through `PATH`, with relative `PATH` entries anchored at `cwd`.
    """
    command = cfg.command
    try:
        if has_separator(command, platform):
            command = resolve_path_field(command, base_dir=config_dir, home=home, platform=platform)
        outcome = resolve_command(
            command,
            platform=platform,
            path_value=lookup_env(env, "PATH", platform) or "",
            pathext_value=lookup_env(env, "PATHEXT", platform),
            base_dir=cwd,
            home=home,
            is_file=is_file,
        )
    except PathFieldError as exc:
        raise SpawnRejected(f"command '{cfg.command}' cannot be resolved: {exc}") from exc
    if isinstance(outcome, NotFound):
        raise SpawnRejected(outcome.message)
    return outcome


async def build_spawn_spec(
    cfg: StdioServerConfig,
    config_dir: str,
    platform: str,
    *,
    environ: Mapping[str, str] | None = None,
    is_file: Callable[[str], bool] | None = None,
) -> SpawnSpec:
    """Turn a `stdio` server configuration into a `SpawnSpec` (17.10, 17.19, 17.20, 17.22).

    - `cwd` is the configured directory resolved against `config_dir`, or `config_dir` itself
      when none is configured (17.19).
    - `env` is `merge_env(inherited_env(environ), cfg.env)` (17.22). `PATH` and `PATHEXT` for
      the executable search come from that merged environment.
    - `args` reach `argv` or the command line character for character: no `~` expansion, no
      separator conversion, no relative resolution (17.20).
    - POSIX gets `argv` and `start_new_session=True`. Windows gets `application` and
      `command_line` with `CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_NO_WINDOW`
      (17.10).

    The file probes run in `asyncio.to_thread`, so the event loop never blocks (14.1). `environ`
    (default `os.environ`) and `is_file` (default: a real file probe) are injection points for
    tests; production callers leave both out.

    Raises `SpawnRejected` when the executable cannot be found or launched (17.6), a batch
    argument is disallowed (17.9), or a Path_Field cannot be resolved. The message is the
    resolver's own.
    """
    source = os.environ if environ is None else environ
    probe = _default_is_file(platform) if is_file is None else is_file
    home = _home_dir(source, platform)
    try:
        cwd = (
            config_dir
            if cfg.cwd is None
            else resolve_path_field(cfg.cwd, base_dir=config_dir, home=home, platform=platform)
        )
    except PathFieldError as exc:
        raise SpawnRejected(f"working directory '{cfg.cwd}' cannot be resolved: {exc}") from exc
    configured_env = {name: value.get_secret_value() for name, value in cfg.env.items()}
    env = merge_env(inherited_env(source, platform), configured_env, platform)

    resolved = await asyncio.to_thread(
        _resolve_or_reject,
        cfg,
        config_dir=config_dir,
        cwd=cwd,
        home=home,
        env=env,
        platform=platform,
        is_file=probe,
    )
    args = tuple(cfg.args)

    if platform != "win32":
        return SpawnSpec(
            platform=platform,
            argv=(resolved.path, *args),
            command_line=None,
            application=None,
            env=env,
            cwd=cwd,
            start_new_session=True,
            creationflags=0,
        )

    system_root = (
        lookup_env(env, "SYSTEMROOT", platform)
        or lookup_env(env, "WINDIR", platform)
        or _FALLBACK_SYSTEM_ROOT
    )
    try:
        command_line = build_command_line(resolved, args, system_root)
    except BatchArgRejected as exc:
        raise SpawnRejected(str(exc)) from exc
    application = (
        ntpath.join(system_root, "System32", "cmd.exe")
        if resolved.kind == "batch"
        else resolved.path
    )
    return SpawnSpec(
        platform=platform,
        argv=None,
        command_line=command_line,
        application=application,
        env=env,
        cwd=cwd,
        start_new_session=False,
        creationflags=WINDOWS_CREATION_FLAGS,
    )


# ---------------------------------------------------------------------------------------------
# 4. Termination plan (17.11)
# ---------------------------------------------------------------------------------------------


class Step(Enum):
    """One stage of Process_Tree termination, in the order `terminate_tree` runs them."""

    CLOSE_STDIN = "close_stdin"
    GRACEFUL = "graceful"
    WAIT_GRACE = "wait_grace"
    FORCE_GROUP = "force_group"
    FORCE_JOB = "force_job"
    FORCE_TASKKILL = "force_taskkill"
    REAP = "reap"


@dataclass(frozen=True, slots=True)
class TerminationReport:
    """What `terminate_tree` did to one Process_Tree.

    `steps` lists the steps that actually ran (a subset of the plan once the tree was empty),
    `forced` says whether the force step ran, and `survivors` holds any pid still alive at the end
    (it must be empty, 17.12).
    """

    pid: int
    steps: tuple[Step, ...]
    forced: bool
    duration_ms: int
    survivors: tuple[int, ...]


def termination_plan(platform: str, job_assigned: bool) -> tuple[Step, ...]:
    """The ordered step template for stopping a Process_Tree on `platform` (17.11). Pure.

    POSIX kills the process group. Windows kills the job object, or falls back to `taskkill /T`
    when the child could not be assigned to a job (`job_assigned` is false). `job_assigned` is
    ignored on POSIX.
    """
    if platform != "win32":
        force = Step.FORCE_GROUP
    elif job_assigned:
        force = Step.FORCE_JOB
    else:
        force = Step.FORCE_TASKKILL
    return (Step.CLOSE_STDIN, Step.GRACEFUL, Step.WAIT_GRACE, force, Step.REAP)


# ---------------------------------------------------------------------------------------------
# 5. Spawning and tree termination (6.11, 14.1, 17.10, 17.11, 17.12, 17.15)
# ---------------------------------------------------------------------------------------------

# Seconds `terminate_tree` waits for the tree to empty after stdin closes (0.5 on console close).
STDIN_WAIT_S: Final = 1.0
# Seconds the Windows CTRL_BREAK helper process may run (design: 1 s).
CTRL_BREAK_TIMEOUT_S: Final = 1.0
# Seconds the reap step waits for the tree to be gone after the force step (design: 1.5 s).
REAP_TIMEOUT_S: Final = 1.5

# `taskkill.exe` must finish inside this time; it runs only when the child is not in a job object.
_TASKKILL_TIMEOUT_S: Final = 1.0
# Exit status `taskkill.exe` reports when the process no longer exists, which is the goal.
_TASKKILL_NOT_FOUND: Final = 128
# Exit code the job object gives the processes it terminates.
_JOB_EXIT_CODE: Final = 1
# How often the wait loops look at the tree.
_POLL_INTERVAL_S: Final = 0.05
# How long reap lets the stderr reader finish by itself, and then how long it waits after cancel.
_STDERR_DRAIN_S: Final = 0.1
# A stderr line longer than this is logged in pieces, so a server cannot grow the buffer forever.
_STDERR_LINE_LIMIT: Final = 64 * 1024
# `ps` is only a diagnostic for the survivor list, so it gets a short bound.
_PS_TIMEOUT_S: Final = 1.0

_FORCE_STEPS: Final = frozenset({Step.FORCE_GROUP, Step.FORCE_JOB, Step.FORCE_TASKKILL})

Sleep = Callable[[float], Awaitable[None]]


class ProcessLogger(Protocol):
    """The slice of a structlog logger this module uses: `event` plus keyword fields."""

    def debug(self, event: str, **fields: object) -> None: ...

    def info(self, event: str, **fields: object) -> None: ...

    def warning(self, event: str, **fields: object) -> None: ...

    def error(self, event: str, **fields: object) -> None: ...


_log: Final[ProcessLogger] = get_logger("oscompat.process")


def _no_exit_code() -> int | None:
    return None


@dataclass(eq=False)
class TreeHandle:
    """A running `stdio` server and the means to stop its whole Process_Tree.

    `pid` is the child's pid. On POSIX it is also the process group id, because the child was
    started with `start_new_session=True`; the group id is never read back with `getpgid`, which
    fails once the leader has been reaped. `stdin` and `stdout` are the pipes the MCP transport
    uses. `job_assigned` says whether the Windows child sits inside a job object (always false on
    POSIX), which selects the force step in `termination_plan`.

    `spawn` builds the handle and `terminate_tree` consumes it. The underscore fields belong to
    the real OS-call layer; a fake `TreeOps` ignores them.
    """

    platform: str
    pid: int
    stdin: anyio.abc.ByteSendStream
    stdout: anyio.abc.ByteReceiveStream
    job_assigned: bool = False
    server: str | None = None
    _stderr: anyio.abc.ByteReceiveStream | None = field(default=None, init=False, repr=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _process: anyio.abc.Process | None = field(default=None, init=False, repr=False)
    _job: int | None = field(default=None, init=False, repr=False)
    _exit_code: Callable[[], int | None] = field(default=_no_exit_code, init=False, repr=False)


class TreeOps(Protocol):
    """The OS-call layer `terminate_tree` drives.

    Every method is one OS action or one question; none of them waits for the tree. The schedule
    (what runs when, and how long to wait) belongs to `terminate_tree`. Tests substitute a fake
    tree that records the calls and empties on a script. A method that sends a signal returns
    `None` on success and a reason string on failure, and never raises for an ordinary OS error.
    """

    async def close_stdin(self, handle: TreeHandle) -> None:
        """Close the child's stdin. An already closed pipe is not an error."""
        ...

    def group_empty(self, handle: TreeHandle) -> bool:
        """True when no process of the tree is alive. Quick and non-blocking."""
        ...

    async def graceful(self, handle: TreeHandle) -> str | None:
        """`SIGTERM` to the group (POSIX) or `CTRL_BREAK_EVENT` through the helper (Windows)."""
        ...

    async def force(self, handle: TreeHandle, step: Step) -> str | None:
        """The force action named by `step`: `FORCE_GROUP`, `FORCE_JOB`, or `FORCE_TASKKILL`."""
        ...

    async def close_streams(self, handle: TreeHandle) -> None:
        """Stop the stderr reader and close every pipe and transport. Bounded."""
        ...

    async def survivors(self, handle: TreeHandle) -> tuple[int, ...]:
        """The pids still alive in the tree. Called only when `group_empty` is false after reap."""
        ...

    async def release(self, handle: TreeHandle) -> None:
        """Release OS resources that outlive the pipes (the Windows job handle)."""
        ...


async def _close_quietly(stream: anyio.abc.AsyncResource | None) -> None:
    """Close a pipe, ignoring the errors a closed or broken pipe raises."""
    if stream is None:
        return
    with contextlib.suppress(
        OSError, ValueError, anyio.ClosedResourceError, anyio.BrokenResourceError
    ):
        await stream.aclose()


async def _close_handle_streams(handle: TreeHandle) -> None:
    """Stop the stderr reader, then close stdin, stdout, stderr, and the POSIX transport.

    The tree is already stopped, so the stderr pipe has normally reached end of file and the
    reader finishes by itself. A survivor that still holds the pipe makes the reader time out and
    get cancelled instead. Every wait here is bounded.
    """
    task = handle._stderr_task
    if task is not None:
        await asyncio.wait({task}, timeout=_STDERR_DRAIN_S)
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=_STDERR_DRAIN_S)
    await _close_quietly(handle.stdin)
    await _close_quietly(handle.stdout)
    await _close_quietly(handle._stderr)
    process = handle._process
    # `Process.aclose` waits for the exit, so call it only once the exit is known.
    if process is not None and process.returncode is not None:
        with contextlib.suppress(Exception):
            await process.aclose()


# --- stderr logging ------------------------------------------------------------------------------


def _log_stderr_line(log: ProcessLogger, server: str | None, raw: bytes) -> None:
    line = raw.decode("utf-8", errors="replace").rstrip("\r")
    if line.strip():
        log.debug("mcp_server_stderr", server=server, line=line)


async def _drain_stderr(
    stream: anyio.abc.ByteReceiveStream, log: ProcessLogger, server: str | None
) -> None:
    """Read the child's stderr until it closes and log it line by line at `debug`.

    Reading it, whether or not debug logging is on, keeps the pipe from filling and blocking the
    child. Bytes are split on `\\n` before decoding, so a multi-byte character split across
    reads is never decoded early; invalid UTF-8 becomes U+FFFD.
    """
    buffer = bytearray()
    try:
        async for chunk in stream:
            buffer.extend(chunk)
            while (index := buffer.find(b"\n")) >= 0:
                _log_stderr_line(log, server, bytes(buffer[:index]))
                del buffer[: index + 1]
            if len(buffer) > _STDERR_LINE_LIMIT:
                _log_stderr_line(log, server, bytes(buffer))
                buffer.clear()
    except OSError, ValueError, anyio.ClosedResourceError, anyio.BrokenResourceError:
        pass
    finally:
        if buffer:
            _log_stderr_line(log, server, bytes(buffer))


def _attach(
    handle: TreeHandle, stderr: anyio.abc.ByteReceiveStream, log: ProcessLogger
) -> TreeHandle:
    """Start the stderr reader for `handle` and keep a reference to its task."""
    handle._stderr = stderr
    handle._stderr_task = asyncio.create_task(
        _drain_stderr(stderr, log, handle.server), name=f"stderr:{handle.server or handle.pid}"
    )
    return handle


# --- spawn ---------------------------------------------------------------------------------------


async def _spawn_posix(spec: SpawnSpec, server: str | None, log: ProcessLogger) -> TreeHandle:
    """Start the child in its own session and process group (17.10)."""
    if spec.argv is None:
        raise ValueError("a POSIX SpawnSpec needs argv")
    process = await anyio.open_process(
        list(spec.argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(spec.env),
        cwd=spec.cwd,
        start_new_session=True,
    )
    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr
    if stdin is None or stdout is None or stderr is None:
        process.kill()
        raise RuntimeError("the child was started without its three pipes")
    handle = TreeHandle(
        platform=spec.platform, pid=process.pid, stdin=stdin, stdout=stdout, server=server
    )
    handle._process = process
    handle._exit_code = lambda: process.returncode
    return _attach(handle, stderr, log)


def _last_error_text() -> str:
    """The calling thread's last Windows error as `text (error N)`. Windows only."""
    if sys.platform == "win32":
        error = ctypes.get_last_error()
        return f"{ctypes.FormatError(error).strip()} (error {error})"
    else:
        return "not Windows"


if sys.platform == "win32":

    def _create_job() -> tuple[int | None, str | None]:
        """Create a job object that kills its processes when the last handle closes.

        Returns `(handle, None)` or `(None, reason)`.
        """
        job = _win32.CreateJobObjectW(None, None)
        if not job:
            return None, f"CreateJobObjectW failed: {_last_error_text()}"
        info = _win32.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _win32.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _win32.SetInformationJobObject(
            job,
            _win32.JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            reason = f"SetInformationJobObject failed: {_last_error_text()}"
            _win32.CloseHandle(job)
            return None, reason
        return int(job), None

    def _assign_to_job(job: int, popen: subprocess.Popen[bytes]) -> str | None:
        """Put the suspended child into `job`. Returns `None` or the failure reason."""
        process_handle = getattr(popen, "_handle", None)
        if process_handle is None:
            return "the process handle is not available"
        if not _win32.AssignProcessToJobObject(job, int(process_handle)):
            return f"AssignProcessToJobObject failed: {_last_error_text()}"
        return None

    def _resume_thread(thread_id: int) -> None:
        thread = _win32.OpenThread(_win32.THREAD_SUSPEND_RESUME, False, thread_id)
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if _win32.ResumeThread(thread) == _win32.RESUME_THREAD_FAILED:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _win32.CloseHandle(thread)

    def _resume_primary_thread(pid: int) -> None:
        """Resume the one thread of the suspended process `pid`.

        `Popen` does not expose the thread handle, so the thread is found through a snapshot.
        """
        snapshot = _win32.CreateToolhelp32Snapshot(_win32.TH32CS_SNAPTHREAD, 0)
        if not snapshot or snapshot == _win32.INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = _win32.THREADENTRY32()
            entry.dwSize = ctypes.sizeof(entry)
            more = _win32.Thread32First(snapshot, ctypes.byref(entry))
            while more:
                if entry.th32OwnerProcessID == pid:
                    _resume_thread(entry.th32ThreadID)
                    return
                more = _win32.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError(f"no thread found for the suspended process {pid}")
        finally:
            _win32.CloseHandle(snapshot)

    def _start_suspended(
        spec: SpawnSpec,
    ) -> tuple[subprocess.Popen[bytes], int | None, str | None]:
        """Create the job, start the child suspended, assign it, and resume it (blocking).

        Returns `(popen, job, problem)`. `job` is `None` and `problem` holds the reason when no
        job could be used; the child runs anyway and termination falls back to `taskkill` (17.11).
        Raises `OSError` when the child cannot start or cannot be resumed.
        """
        if spec.command_line is None:
            raise ValueError("a Windows SpawnSpec needs command_line")
        job, problem = _create_job()
        try:
            # `bufsize=0` gives raw pipes: a read returns what is available instead of waiting
            # for a full buffer, and a write is never held back in a buffer.
            popen = subprocess.Popen(
                spec.command_line,
                executable=spec.application,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(spec.env),
                cwd=spec.cwd,
                creationflags=spec.creationflags,
                bufsize=0,
            )
        except BaseException:
            if job is not None:
                _win32.CloseHandle(job)
            raise
        try:
            if job is not None:
                problem = _assign_to_job(job, popen)
                if problem is not None:
                    _win32.CloseHandle(job)
                    job = None
            _resume_primary_thread(popen.pid)
        except BaseException:
            popen.kill()
            for pipe in (popen.stdin, popen.stdout, popen.stderr):
                if pipe is not None:
                    pipe.close()
            if job is not None:
                _win32.CloseHandle(job)
            raise
        return popen, job, problem

    async def _spawn_windows(spec: SpawnSpec, server: str | None, log: ProcessLogger) -> TreeHandle:
        """Start the child suspended inside a job object, then resume it (17.10, 17.11)."""
        popen, job, problem = await asyncio.to_thread(_start_suspended, spec)
        if problem is not None:
            log.warning("job_object_unavailable", server=server, reason=problem)
        if popen.stdin is None or popen.stdout is None or popen.stderr is None:
            popen.kill()
            raise RuntimeError("the child was started without its three pipes")
        handle = TreeHandle(
            platform=spec.platform,
            pid=popen.pid,
            stdin=FileWriteStream(cast(BinaryIO, popen.stdin)),
            stdout=FileReadStream(cast(BinaryIO, popen.stdout)),
            job_assigned=job is not None,
            server=server,
        )
        handle._job = job
        handle._exit_code = popen.poll
        return _attach(handle, FileReadStream(cast(BinaryIO, popen.stderr)), log)


async def spawn(
    spec: SpawnSpec, *, server: str | None = None, log: ProcessLogger | None = None
) -> TreeHandle:
    """Start the `stdio` server described by `spec` and return a handle to its Process_Tree.

    POSIX starts the child with `anyio.open_process(..., start_new_session=True)`, so it leads its
    own session and process group. Windows creates a job object, starts the child suspended with
    the string command line, assigns it to the job, and resumes its primary thread, so no
    descendant can start outside the job. When the job cannot be created or assigned, a
    `job_object_unavailable` warning is logged, the child is resumed anyway, and `job_assigned` is
    false. Every blocking Windows call runs in a worker thread (14.1), and the pipes are read and
    written by worker threads through anyio file streams.

    The child's stderr is read line by line and logged at `debug`; `server` names it in the log.

    Raises `OSError` when the child cannot start (a missing working directory, for example) and
    `ValueError` when `spec` was built for another operating system than this one.
    """
    logger = _log if log is None else log
    if spec.platform == _WIN32:
        if sys.platform == "win32":
            return await _spawn_windows(spec, server, logger)
        raise ValueError("a Windows SpawnSpec cannot be spawned on this operating system")
    if sys.platform != "win32":
        return await _spawn_posix(spec, server, logger)
    raise ValueError("a POSIX SpawnSpec cannot be spawned on this operating system")


# --- the real OS-call layer ----------------------------------------------------------------------


def _posix_group_members(pgid: int) -> tuple[int, ...]:
    """The live pids of process group `pgid`, from `ps`; `(pgid,)` when `ps` cannot answer."""
    try:
        done = subprocess.run(
            ["ps", "-A", "-o", "pid=,pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=_PS_TIMEOUT_S,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return (pgid,)
    if done.returncode != 0:
        return (pgid,)
    members: list[int] = []
    for line in done.stdout.splitlines():
        fields = line.split()
        # A zombie has exited; it only waits for its parent, so it is not a survivor.
        if len(fields) == 3 and fields[1] == str(pgid) and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return tuple(members)


if sys.platform != "win32":

    class _PosixOps:
        """`TreeOps` for macOS and Linux: signals go to the process group `handle.pid`."""

        async def close_stdin(self, handle: TreeHandle) -> None:
            await _close_quietly(handle.stdin)

        def group_empty(self, handle: TreeHandle) -> bool:
            try:
                os.killpg(handle.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                # The group exists but belongs to someone else's view of it (macOS reports this
                # for a group of zombies). Treat it as alive; reap re-checks and lists survivors.
                return False
            return False

        def _signal(self, handle: TreeHandle, sig: signal.Signals) -> str | None:
            try:
                os.killpg(handle.pid, sig)
            except ProcessLookupError:
                return None
            except OSError as exc:
                return f"killpg({handle.pid}, {sig.name}) failed: {exc}"
            return None

        async def graceful(self, handle: TreeHandle) -> str | None:
            return self._signal(handle, signal.SIGTERM)

        async def force(self, handle: TreeHandle, step: Step) -> str | None:
            return self._signal(handle, signal.SIGKILL)

        async def close_streams(self, handle: TreeHandle) -> None:
            await _close_handle_streams(handle)

        async def survivors(self, handle: TreeHandle) -> tuple[int, ...]:
            return await asyncio.to_thread(_posix_group_members, handle.pid)

        async def release(self, handle: TreeHandle) -> None:
            return None


if sys.platform == "win32":

    def _run_ctrl_break(pid: int) -> str | None:
        """Run the `ctrl_break` helper against `pid` with a 1 s limit (blocking)."""
        try:
            done = subprocess.run(
                [sys.executable, "-m", "local_agent.oscompat.ctrl_break", str(pid)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=CTRL_BREAK_TIMEOUT_S,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"the ctrl_break helper did not finish within {CTRL_BREAK_TIMEOUT_S} s"
        except OSError as exc:
            return f"the ctrl_break helper could not start: {exc}"
        if done.returncode == 0:
            return None
        detail = done.stderr.decode("utf-8", errors="replace").strip()
        return f"the ctrl_break helper exited with status {done.returncode}: {detail}"

    def _terminate_job(job: int) -> str | None:
        if _win32.TerminateJobObject(job, _JOB_EXIT_CODE):
            return None
        return f"TerminateJobObject failed: {_last_error_text()}"

    def _taskkill(pid: int) -> str | None:
        """Kill `pid` and its children with `taskkill.exe /T /F` (blocking, bounded)."""
        system_root = (
            os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or _FALLBACK_SYSTEM_ROOT
        )
        executable = ntpath.join(system_root, "System32", "taskkill.exe")
        try:
            done = subprocess.run(
                [executable, "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_TASKKILL_TIMEOUT_S,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"taskkill did not finish within {_TASKKILL_TIMEOUT_S} s"
        except OSError as exc:
            return f"taskkill could not start: {exc}"
        if done.returncode in (0, _TASKKILL_NOT_FOUND):
            return None
        detail = done.stderr.decode("utf-8", errors="replace").strip()
        return f"taskkill exited with status {done.returncode}: {detail}"

    def _job_active_processes(job: int) -> int | None:
        info = _win32.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        ok = _win32.QueryInformationJobObject(
            job,
            _win32.JobObjectBasicAccountingInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        )
        return int(info.ActiveProcesses) if ok else None

    def _job_process_ids(job: int) -> tuple[int, ...] | None:
        info = _win32.JOBOBJECT_BASIC_PROCESS_ID_LIST()
        ok = _win32.QueryInformationJobObject(
            job,
            _win32.JobObjectBasicProcessIdList,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        )
        # A job with more processes than the list holds fails with ERROR_MORE_DATA but still
        # fills the list.
        if not ok and ctypes.get_last_error() != _win32.ERROR_MORE_DATA:
            return None
        return tuple(int(info.ProcessIdList[i]) for i in range(info.NumberOfProcessIdsInList))

    def _windows_survivors(handle: TreeHandle) -> tuple[int, ...]:
        if handle._job is not None:
            pids = _job_process_ids(handle._job)
            if pids is not None:
                return pids
        # Without a job only the leader can be checked; its descendants are not tracked.
        return (handle.pid,) if handle._exit_code() is None else ()

    class _WindowsOps:
        """`TreeOps` for Windows: the job object is the tree, with `taskkill /T` as fallback."""

        async def close_stdin(self, handle: TreeHandle) -> None:
            await _close_quietly(handle.stdin)

        def group_empty(self, handle: TreeHandle) -> bool:
            if handle._job is not None:
                active = _job_active_processes(handle._job)
                if active is not None:
                    return active == 0
            return handle._exit_code() is not None

        async def graceful(self, handle: TreeHandle) -> str | None:
            return await asyncio.to_thread(_run_ctrl_break, handle.pid)

        async def force(self, handle: TreeHandle, step: Step) -> str | None:
            if step is Step.FORCE_JOB and handle._job is not None:
                return _terminate_job(handle._job)
            return await asyncio.to_thread(_taskkill, handle.pid)

        async def close_streams(self, handle: TreeHandle) -> None:
            await _close_handle_streams(handle)

        async def survivors(self, handle: TreeHandle) -> tuple[int, ...]:
            return await asyncio.to_thread(_windows_survivors, handle)

        async def release(self, handle: TreeHandle) -> None:
            job, handle._job = handle._job, None
            if job is not None:
                await asyncio.to_thread(_win32.CloseHandle, job)


def system_tree_ops() -> TreeOps:
    """The real OS-call layer for the running operating system."""
    if sys.platform == "win32":
        return _WindowsOps()
    else:
        return _PosixOps()


# --- terminate_tree ------------------------------------------------------------------------------


async def _wait_empty(
    ops: TreeOps,
    handle: TreeHandle,
    timeout_s: float | Callable[[], float],
    clock: Clock,
    sleep: Sleep,
) -> bool:
    """Poll until the tree is empty or `timeout_s` passes on `clock`; True when it is empty.

    The tree is always looked at once before the deadline is checked, so a zero timeout still
    reports an empty tree. A callable `timeout_s` is read on every poll, so lowering it during the
    wait shortens the wait (the deadline is the start time plus the current value).
    """
    started = clock.monotonic()
    while True:
        if ops.group_empty(handle):
            return True
        limit = timeout_s() if callable(timeout_s) else timeout_s
        remaining = started + limit - clock.monotonic()
        if remaining <= 0:
            return False
        await sleep(min(_POLL_INTERVAL_S, remaining))


def _report_signal_failure(log: ProcessLogger, handle: TreeHandle, step: Step, reason: str) -> None:
    """Log a failed signal. The Windows CTRL_BREAK helper has its own event name (17.11)."""
    if step is Step.GRACEFUL and handle.platform == _WIN32:
        log.warning("ctrl_break_failed", server=handle.server, pid=handle.pid, reason=reason)
    else:
        log.warning(
            "mcp_process_signal_failed",
            server=handle.server,
            pid=handle.pid,
            step=step.value,
            reason=reason,
        )


async def _run_termination(
    handle: TreeHandle,
    grace_s: float | Callable[[], float],
    stdin_wait_s: float,
    ops: TreeOps,
    clock: Clock,
    sleep: Sleep,
    log: ProcessLogger,
) -> TerminationReport:
    started = clock.monotonic()
    steps: list[Step] = []
    forced = False
    survivors: tuple[int, ...] = ()
    empty = False
    for step in termination_plan(handle.platform, handle.job_assigned):
        # Once the tree is empty only the reap step still has work to do.
        if empty and step is not Step.REAP:
            continue
        steps.append(step)
        if step is Step.CLOSE_STDIN:
            await ops.close_stdin(handle)
            empty = await _wait_empty(ops, handle, stdin_wait_s, clock, sleep)
        elif step is Step.GRACEFUL:
            reason = await ops.graceful(handle)
            if reason is not None:
                _report_signal_failure(log, handle, step, reason)
        elif step is Step.WAIT_GRACE:
            empty = await _wait_empty(ops, handle, grace_s, clock, sleep)
        elif step in _FORCE_STEPS:
            forced = True
            reason = await ops.force(handle, step)
            if reason is not None:
                _report_signal_failure(log, handle, step, reason)
        else:
            empty = await _wait_empty(ops, handle, REAP_TIMEOUT_S, clock, sleep)
            await ops.close_streams(handle)
            survivors = () if empty else await ops.survivors(handle)
            await ops.release(handle)
    report = TerminationReport(
        pid=handle.pid,
        steps=tuple(steps),
        forced=forced,
        duration_ms=int((clock.monotonic() - started) * 1000),
        survivors=survivors,
    )
    fields = {
        "server": handle.server,
        "pid": handle.pid,
        "forced": forced,
        "duration_ms": report.duration_ms,
        "survivors": list(survivors),
    }
    if survivors:
        log.error("mcp_process_tree_stopped", **fields)
    else:
        log.info("mcp_process_tree_stopped", **fields)
    return report


async def terminate_tree(
    handle: TreeHandle,
    grace_s: float | Callable[[], float],
    stdin_wait_s: float = STDIN_WAIT_S,
    *,
    ops: TreeOps | None = None,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
    log: ProcessLogger | None = None,
) -> TerminationReport:
    """Stop the whole Process_Tree of `handle` and report what it took (6.11, 17.11, 17.12, 17.15).

    The steps follow `termination_plan(handle.platform, handle.job_assigned)` and run under a
    shielded cancel scope, so cancelling the caller cannot leave a process behind. Every wait is
    bounded:

    1. close stdin, then wait up to `stdin_wait_s` for the tree to empty;
    2. graceful: `killpg(pid, SIGTERM)`, or the `ctrl_break` helper on Windows (1 s limit); a
       failure logs `ctrl_break_failed` (or `mcp_process_signal_failed`) and the schedule goes on;
    3. wait up to `grace_s` for the tree to empty. A callable `grace_s` is read on every poll, so
       lowering it mid-wait (a second Shutdown_Signal sets it to 0) ends the wait early;
    4. force: `killpg(pid, SIGKILL)`, `TerminateJobObject`, or `taskkill.exe /PID <pid> /T /F`;
    5. reap: wait up to 1.5 s for the tree to be gone, close the pipes, release the job handle,
       and log `mcp_process_tree_stopped` with the survivors (an error record when there are any).

    A step is skipped once the tree is empty, so `steps` lists only the steps that ran. The
    graceful step counts as run when it was attempted, even if the helper failed. The call
    returns normally when processes survive; the caller checks `TerminationReport.survivors`.

    `ops`, `clock`, and `sleep` default to the real OS-call layer, `SystemClock`, and
    `anyio.sleep`. Tests pass a fake tree, a fake clock, and a fake sleep that advances the clock.
    """
    with anyio.CancelScope(shield=True):
        report = await _run_termination(
            handle,
            grace_s,
            stdin_wait_s,
            system_tree_ops() if ops is None else ops,
            SystemClock() if clock is None else clock,
            anyio.sleep if sleep is None else sleep,
            _log if log is None else log,
        )
    return report
