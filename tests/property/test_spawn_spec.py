"""Property test for the spawn specification (Requirements 17.10, 17.19, 17.20).

`build_spawn_spec` is driven with a fake file system in which every launchable file exists and a
fake Backend environment, so the result depends only on the
configuration, the config directory, and the platform. Each example checks:

- no shell: a `SpawnSpec` has no `shell` field, POSIX `argv` is exactly the resolved executable
  plus the arguments, and the Windows native command line parses back (through the independent
  `winargv_ref` port of the C runtime rules) to the executable named by `application`. A batch
  target goes through a fixed `cmd.exe` invocation with no `COMSPEC` involved;
- POSIX: `start_new_session` is true;
- Windows: `creationflags` holds `CREATE_NEW_PROCESS_GROUP`, `CREATE_SUSPENDED`, and
  `CREATE_NO_WINDOW` (the Win32 values are spelled out here, not imported);
- `cwd` equals the configured directory resolved against the config directory (reference built
  from `ntpath` / `posixpath`), or the config directory itself when none is configured;
- the arguments reach the process character for character: `~`, `/` and `\\`, relative
  segments, `$VAR` and empty strings are never touched.

Windows and POSIX cases run on every host because the platform is an argument. NUL is excluded
from arguments (a process argument cannot hold it). Arguments for a `.cmd`/`.bat` target also
exclude the four characters `cmd.exe` cannot take literally (`"`, `%`, CR, LF), because those make
the configuration invalid for a batch target (17.9).
"""

import asyncio
import dataclasses
import ntpath
import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from hypothesis import given, settings
from hypothesis import strategies as st
from winargv_ref import parse_arguments, parse_command_line

from local_agent.config.models import StdioServerConfig
from local_agent.oscompat.process import SpawnSpec, build_spawn_spec

# Win32 process-creation flags, spelled out so a wrong constant in the implementation is caught.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000

_PLATFORMS = ["win32", "linux", "darwin"]

_WIN_CONFIG_DIRS = ["C:\\cfg\\app", "D:\\Users\\me\\.agent config", "C:\\"]
_POSIX_CONFIG_DIRS = ["/etc/app", "/home/u/cfg dir", "/", "/opt/a b/c"]
_WIN_HOME = "C:\\Users\\u"
_POSIX_HOME = "/home/u"

_WIN_PATH_DIRS = ["C:\\bin", "D:\\tools", "C:\\Program Files\\Node"]
_POSIX_PATH_DIRS = ["/usr/bin", "/opt/tools", "/usr/local/bin"]

_WIN_COMMANDS = [
    "node",
    "tool.exe",
    "npx.cmd",
    "run.BAT",
    "C:\\tools\\srv.exe",
    "tools\\srv.cmd",
    "./bin/srv.exe",
    "..\\x\\srv",
]
_POSIX_COMMANDS = [
    "node",
    "python3",
    "my-server",
    "tool.sh",
    "/opt/tools/srv",
    "./bin/srv",
    "../x/srv",
    "bin/srv",
]

_WIN_CWDS = [
    "work",
    ".\\a\\..\\b",
    "..\\up",
    "C:\\abs\\dir",
    "\\rootrel",
    "~\\proj",
    "~",
    "sub/dir",
]
_POSIX_CWDS = ["work", "./a/../b", "../up", "/abs/dir", "~/proj", "~", "a b/c", "/abs/../x"]

# Arguments that a path or shell layer would change if it touched them.
_ARG_FIXED = [
    "",
    "~",
    "~/x",
    "~user/y",
    "./rel",
    "../rel",
    "dir/sub",
    "C:/a/b",
    "a\\b\\",
    "$HOME",
    "${PATH}",
    "--flag=~/v",
    "-y",
    "two words",
    "tab\there",
    "*.txt",
    "x && y",
    "é€\U0001f600",
]
_BATCH_FORBIDDEN = '"%\r\n'


@dataclass(frozen=True, slots=True)
class _Case:
    platform: str
    config_dir: str
    command: str
    args: list[str]
    cwd: str | None
    path_dirs: list[str]


def _module(platform: str) -> ModuleType:
    return ntpath if platform == "win32" else posixpath


def _is_batch(case: _Case) -> bool:
    return case.platform == "win32" and case.command.lower().endswith((".cmd", ".bat"))


@st.composite
def _cases(draw: st.DrawFn) -> _Case:
    platform = draw(st.sampled_from(_PLATFORMS))
    windows = platform == "win32"
    command = draw(st.sampled_from(_WIN_COMMANDS if windows else _POSIX_COMMANDS))
    batch = windows and command.lower().endswith((".cmd", ".bat"))

    random_text = st.text(
        alphabet=st.characters(
            exclude_characters="\x00" + (_BATCH_FORBIDDEN if batch else ""),
            exclude_categories=["Cs"],
        ),
        max_size=10,
    )
    fixed = st.sampled_from(_ARG_FIXED)
    if batch:
        fixed = fixed.filter(lambda arg: not any(ch in arg for ch in _BATCH_FORBIDDEN))
    args = draw(st.lists(st.one_of(fixed, random_text), max_size=6))

    return _Case(
        platform=platform,
        config_dir=draw(st.sampled_from(_WIN_CONFIG_DIRS if windows else _POSIX_CONFIG_DIRS)),
        command=command,
        args=args,
        cwd=draw(st.one_of(st.none(), st.sampled_from(_WIN_CWDS if windows else _POSIX_CWDS))),
        path_dirs=draw(
            st.lists(
                st.sampled_from(_WIN_PATH_DIRS if windows else _POSIX_PATH_DIRS),
                min_size=1,
                max_size=3,
                unique=True,
            )
        ),
    )


def _environ(case: _Case) -> dict[str, str]:
    if case.platform == "win32":
        return {
            "PATH": ";".join(case.path_dirs),
            "SYSTEMROOT": "C:\\Windows",
            "USERPROFILE": _WIN_HOME,
        }
    return {"PATH": ":".join(case.path_dirs), "HOME": _POSIX_HOME}


def _expected_cwd(case: _Case) -> str:
    """Reference for 17.19: the configured directory against the config directory, or the latter."""
    if case.cwd is None:
        return case.config_dir
    mod = _module(case.platform)
    home = _WIN_HOME if case.platform == "win32" else _POSIX_HOME
    value = case.cwd
    if value == "~" or value.startswith(("~/", "~\\" if case.platform == "win32" else "~/")):
        value = mod.join(home, value[2:]) if len(value) > 1 else home
    resolved: str = mod.normpath(mod.join(case.config_dir, value))
    return resolved


def _resolving_is_file(platform: str) -> Callable[[str], bool]:
    """A fake file system in which every launchable file exists.

    On POSIX any path is an executable file. On Windows only paths with a launchable suffix exist,
    so an extensionless command resolves through `PATHEXT` instead of matching a stray file.
    """
    if platform != "win32":
        return lambda _path: True
    return lambda path: ntpath.splitext(path)[1].lower() in {".exe", ".com", ".cmd", ".bat"}


def _build(case: _Case) -> SpawnSpec:
    cfg = StdioServerConfig(
        name="srv", transport="stdio", command=case.command, args=case.args, cwd=case.cwd
    )
    return asyncio.run(
        build_spawn_spec(
            cfg,
            case.config_dir,
            case.platform,
            environ=_environ(case),
            is_file=_resolving_is_file(case.platform),
        )
    )


def _check_posix(case: _Case, spec: SpawnSpec) -> None:
    assert spec.start_new_session is True
    assert spec.command_line is None
    assert spec.application is None
    assert spec.argv is not None
    exe, *rest = spec.argv
    # Arguments are untouched, in order, and nothing is inserted before or after them.
    assert rest == case.args
    # The executable is the command itself (a path) or the command found in a PATH directory.
    assert posixpath.basename(exe) == posixpath.basename(case.command)
    if "/" in case.command:
        assert exe == posixpath.normpath(posixpath.join(case.config_dir, case.command))
    else:
        assert exe == posixpath.join(case.path_dirs[0], case.command)


def _check_windows(case: _Case, spec: SpawnSpec) -> None:
    flags = spec.creationflags
    for flag in (_CREATE_NEW_PROCESS_GROUP, _CREATE_SUSPENDED, _CREATE_NO_WINDOW):
        assert flags & flag == flag
    assert spec.argv is None
    assert spec.application is not None
    assert spec.command_line is not None

    if not _is_batch(case):
        # `application` is the program itself and the line parses back to `[program, *args]`.
        assert parse_command_line(spec.command_line) == [spec.application, *case.args]
        assert (
            ntpath.basename(spec.application)
            .lower()
            .startswith(ntpath.basename(case.command).lower())
        )
        return

    # Batch target: a fixed `cmd.exe` with literal switches runs one quoted line.
    cmd_exe = "C:\\Windows\\System32\\cmd.exe"
    assert spec.application == cmd_exe
    prefix = f'{cmd_exe} /d /e:ON /v:OFF /c "'
    assert spec.command_line.startswith(prefix)
    assert spec.command_line.endswith('"')
    inner = spec.command_line[len(prefix) : -1]
    parsed = parse_arguments(inner)
    assert parsed[1:] == case.args
    assert ntpath.basename(parsed[0]).lower() == ntpath.basename(case.command).lower()


# Feature: local-agent-backend, Property 42: Spawn specification
# **Validates: Requirements 17.10, 17.19, 17.20**
@settings(max_examples=200)
@given(_cases())
def test_spawn_spec_flags_cwd_and_untouched_arguments(case: _Case) -> None:
    spec = _build(case)

    # No shell: the spec has no way to request one.
    assert "shell" not in {f.name for f in dataclasses.fields(spec)}
    assert spec.platform == case.platform
    assert spec.cwd == _expected_cwd(case)

    if case.platform == "win32":
        _check_windows(case, spec)
    else:
        _check_posix(case, spec)
