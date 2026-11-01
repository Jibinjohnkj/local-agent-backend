"""Property test for executable resolution (Requirements 17.4, 17.5, 17.6).

`resolve_command` is pure: its only file system access is the injected `is_file` callable. The
test supplies an in-memory set of paths as that oracle and records every probe. A small reference
resolver, written here from the design's rules and using only `ntpath` / `posixpath`, lists the
candidate paths in the order they must be tried. The test then checks three things:

- the result equals the reference result (first existing candidate, classified by platform);
- the probes are exactly the reference candidates up to the first hit, so nothing outside the
  `PATH` directories or the resolved path is ever probed, and the order is the specified order;
- a missing executable gives `NotFound` that names the command.

Windows and POSIX cases both run on every host, because the platform is an argument.
"""

import ntpath
import posixpath
from dataclasses import dataclass
from types import ModuleType

from hypothesis import given
from hypothesis import strategies as st

from local_agent.oscompat.process import DEFAULT_PATHEXT, NotFound, Resolved, resolve_command

_PLATFORMS = ["win32", "linux", "darwin"]

# --- Platform vocabulary -----------------------------------------------------------------------

_WIN_BASE = "C:\\work\\proj"
_POSIX_BASE = "/work/proj"
_WIN_HOME = "C:\\Users\\u"
_POSIX_HOME = "/home/u"

_WIN_ABSOLUTE_DIRS = [
    "C:\\bin",
    "C:\\Program Files\\Tool",
    "D:\\tools",
    "C:\\Windows\\System32",
    "\\\\server\\share\\bin",
    "\\tools",  # root-relative: takes the drive of the base directory
]
_WIN_RELATIVE_DIRS = ["bin", ".\\bin", "..\\shared", "~bin", "sub/dir"]
_POSIX_ABSOLUTE_DIRS = ["/usr/bin", "/opt/tools", "/usr/local/bin", "/opt/my tools"]
_POSIX_RELATIVE_DIRS = ["bin", "./bin", "../shared", "~bin", "sub/dir"]

_WIN_NAMES = ["tool", "tool.exe", "TOOL.CMD", "tool.ps1", "node.js", "tool.Bat", "tool.com"]
_POSIX_NAMES = ["tool", "tool.sh", "node", "a.b", "we\\ird"]  # `\` is no separator on POSIX

_WIN_PATHS = [
    "C:\\bin\\tool",
    "C:\\bin\\tool.exe",
    "tools\\tool",
    "tools/tool.cmd",
    ".\\tool.ps1",
    "~\\bin\\tool",
    "\\tools\\tool",
    "D:/tools/Tool.BAT",
    "C:\\bin\\tool.v2",
    "C:\\a\\..\\bin\\tool",
]
_POSIX_PATHS = [
    "/opt/tools/tool",
    "bin/tool",
    "./tool",
    "../x/tool",
    "~/bin/tool",
    "/opt/a/../tool",
    "/opt/tools/tool.sh",
]

# `None` means "variable absent", which selects DEFAULT_PATHEXT. The empty string yields no
# extensions at all, and the messy values hold blanks, junk, and mixed case.
_PATHEXT_VALUES = st.sampled_from(
    [
        None,
        DEFAULT_PATHEXT,
        "",
        ".CMD;.EXE",
        ".exe;.cmd",
        ".ps1;.EXE;.BAT",
        ".EXE;;  .cmd ;junk;.;",
        ".COM;.EXE;.EXE",
    ]
)


@dataclass(frozen=True, slots=True)
class _Case:
    platform: str
    command: str
    path_value: str
    pathext_value: str | None
    base_dir: str
    home: str
    files: frozenset[str]


# --- Reference resolver ------------------------------------------------------------------------


def _module(platform: str) -> ModuleType:
    return ntpath if platform == "win32" else posixpath


def _has_separator(command: str, platform: str) -> bool:
    return "/" in command or (platform == "win32" and "\\" in command)


def _reference_extensions(pathext_value: str | None) -> list[str]:
    value = DEFAULT_PATHEXT if pathext_value is None else pathext_value
    pieces = [piece.strip() for piece in value.split(";")]
    return [piece for piece in pieces if len(piece) > 1 and piece.startswith(".")]


def _reference_directories(case: _Case) -> list[str]:
    """`PATH` entries in order: empty skipped, quotes stripped (Windows), relative to the cwd."""
    windows = case.platform == "win32"
    mod = _module(case.platform)
    directories: list[str] = []
    for raw in case.path_value.split(";" if windows else ":"):
        entry = raw.strip('"') if windows else raw
        if not entry:
            continue
        if windows:
            drive, root, _ = ntpath.splitroot(entry)
            if drive and not root:
                continue  # `C:bin` has no single meaning
        directories.append(mod.normpath(mod.join(case.base_dir, entry)))
    return directories


def _reference_candidates(case: _Case) -> tuple[list[str], list[str]]:
    """The ordered probe list and the directories that count as searched."""
    windows = case.platform == "win32"
    mod = _module(case.platform)
    extensions = _reference_extensions(case.pathext_value) if windows else []

    if _has_separator(case.command, case.platform):
        command = case.command
        if command[0] == "~" and len(command) > 1 and command[1] in ("/\\" if windows else "/"):
            rest = command[2:]
            command = mod.join(case.home, rest) if rest else case.home
        path = mod.normpath(mod.join(case.base_dir, command))
        candidates = [path]
        if windows and not mod.splitext(path)[1]:
            candidates += [path + extension for extension in extensions]
        return candidates, [mod.dirname(path)]

    directories = _reference_directories(case)
    if windows and mod.splitext(case.command)[1].lower() not in {e.lower() for e in extensions}:
        names = [case.command + extension for extension in extensions]
    else:
        names = [case.command]
    return [mod.join(directory, name) for directory in directories for name in names], directories


def _reference_result(case: _Case) -> tuple[Resolved | NotFound, list[str]]:
    """The expected result and the expected probe sequence for `case`."""
    mod = _module(case.platform)
    candidates, searched = _reference_candidates(case)
    probes: list[str] = []
    for candidate in candidates:
        probes.append(candidate)
        if candidate not in case.files:
            continue
        if case.platform != "win32":
            return Resolved(candidate, "native"), probes
        suffix = mod.splitext(candidate)[1].lower()
        if suffix in (".cmd", ".bat"):
            return Resolved(candidate, "batch"), probes
        if suffix in (".exe", ".com"):
            return Resolved(candidate, "native"), probes
        return NotFound(case.command, tuple(searched), found=candidate), probes
    return NotFound(case.command, tuple(searched)), probes


# --- Strategies --------------------------------------------------------------------------------


@st.composite
def _cases(draw: st.DrawFn) -> _Case:
    platform = draw(st.sampled_from(_PLATFORMS))
    windows = platform == "win32"
    base_dir = _WIN_BASE if windows else _POSIX_BASE
    home = _WIN_HOME if windows else _POSIX_HOME

    if windows:
        entries = st.one_of(
            st.sampled_from(_WIN_ABSOLUTE_DIRS),
            st.sampled_from(_WIN_ABSOLUTE_DIRS).map(lambda d: f'"{d}"'),
            st.sampled_from(_WIN_RELATIVE_DIRS),
            st.sampled_from(["", '""', "C:bin"]),
        )
        separator = ";"
    else:
        entries = st.one_of(
            st.sampled_from(_POSIX_ABSOLUTE_DIRS),
            st.sampled_from(_POSIX_RELATIVE_DIRS),
            st.just(""),
        )
        separator = ":"
    path_value = separator.join(draw(st.lists(entries, max_size=5)))

    if draw(st.booleans()):
        command = draw(st.sampled_from(_WIN_PATHS if windows else _POSIX_PATHS))
    else:
        command = draw(st.sampled_from(_WIN_NAMES if windows else _POSIX_NAMES))

    case = _Case(platform, command, path_value, draw(_PATHEXT_VALUES), base_dir, home, frozenset())

    # Files come from the candidates the resolver may probe, plus decoys it must never probe:
    # the bare name in every searched directory (the shell script next to `npx.cmd`), a `.ps1`
    # sibling, and the bare name in the working directory (no implicit current directory).
    candidates, searched = _reference_candidates(case)
    mod = _module(platform)
    decoys = [mod.join(d, name) for d in searched for name in (command, command + ".ps1")]
    if not _has_separator(command, platform):
        decoys.append(mod.join(base_dir, command))
        decoys.append(mod.join(base_dir, command + ".exe"))
    universe = sorted({*candidates, *decoys})
    files = draw(st.frozensets(st.sampled_from(universe))) if universe else frozenset[str]()
    return _Case(platform, command, path_value, case.pathext_value, base_dir, home, files)


# Feature: local-agent-backend, Property 39: Executable resolution
# Validates: Requirements 17.4, 17.5, 17.6
@given(_cases())
def test_resolve_command_matches_reference_and_probes_only_candidates(case: _Case) -> None:
    probes: list[str] = []

    def is_file(path: str) -> bool:
        probes.append(path)
        return path in case.files

    result = resolve_command(
        case.command,
        platform=case.platform,
        path_value=case.path_value,
        pathext_value=case.pathext_value,
        base_dir=case.base_dir,
        home=case.home,
        is_file=is_file,
    )

    expected, expected_probes = _reference_result(case)
    assert result == expected
    # Same candidates, same order, stopping at the first existing file: nothing else is probed.
    assert probes == expected_probes
    if isinstance(result, NotFound):
        assert result.command == case.command
        assert case.command in result.message
