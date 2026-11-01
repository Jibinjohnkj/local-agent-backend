"""Property test for Windows argument fidelity (Requirement 17.7).

`build_command_line` turns a resolved native executable and its arguments into one
`CreateProcess` string. The test parses that string with an independent port of the Microsoft C
runtime / `CommandLineToArgvW` rules (`winargv_ref`) and requires exactly `[exe, *args]` back.

The arguments are built from the characters that break naive quoting: spaces, tabs, quotes, runs
of backslashes (alone and next to quotes), `cmd.exe` operators, line breaks, and non-ASCII text.
NUL is excluded because a process argument cannot hold it.

The executable path is a Windows file path, so it never holds a `"`: Win32 forbids that character
in file names, and the program-name rule for `argv[0]` cannot express one. It also does not end
in a backslash, because a path ending that way names a directory.

The same example also checks the POSIX half of the property: `build_spawn_spec` on macOS and
Linux puts `(exe, *args)` in `argv` unchanged and fills neither `command_line` nor
`application`. A spec has no field through which a shell could be requested, and the Windows
native command line must parse back to the program itself rather than to `cmd.exe`.
"""

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st
from winargv_ref import parse_arguments, parse_command_line

from local_agent.config.models import StdioServerConfig
from local_agent.oscompat.process import Resolved, build_command_line, build_spawn_spec

_SYSTEM_ROOT = "C:\\Windows"

_TROUBLE_CHARS = ' \t"\\&|<>^%!()\r\n\v\f\u00a0\u3000\u00e9\u20ac\U0001f600'

_text_chars = st.one_of(
    st.sampled_from(_TROUBLE_CHARS),
    st.sampled_from('\\\\\\"" '),  # weight toward backslash and quote runs
    st.characters(exclude_characters="\x00", exclude_categories=["Cs"]),
)
_arg = st.text(alphabet=_text_chars, max_size=14)
_args = st.lists(_arg, max_size=6)

_exe = st.text(
    alphabet=st.one_of(
        st.sampled_from(" \t\\:.-_é€\U0001f600"),
        st.characters(exclude_characters='\x00"', exclude_categories=["Cs"]),
    ),
    max_size=24,
).filter(lambda path: not path.endswith("\\"))

_POSIX_EXECUTABLES = ["/usr/bin/node", "/opt/my tools/server", "/opt/tools/ünï/srv.py", "/bin/x"]


def _posix_argv(platform: str, exe: str, args: list[str]) -> tuple[str, ...] | None:
    """The `argv` that `build_spawn_spec` produces for `exe` on a POSIX platform."""
    cfg = StdioServerConfig(name="srv", transport="stdio", command=exe, args=args)
    spec = asyncio.run(
        build_spawn_spec(
            cfg, "/cfg", platform, environ={"PATH": "/usr/bin"}, is_file=lambda _: True
        )
    )
    assert spec.command_line is None
    assert spec.application is None
    return spec.argv


# Feature: local-agent-backend, Property 40: Windows argument fidelity for native executables
@given(
    exe=_exe,
    args=_args,
    posix_exe=st.sampled_from(_POSIX_EXECUTABLES),
    posix_platform=st.sampled_from(["linux", "darwin"]),
)
def test_windows_argument_fidelity_for_native_executables(
    exe: str, args: list[str], posix_exe: str, posix_platform: str
) -> None:
    """Validates: Requirements 17.7."""
    line = build_command_line(Resolved(exe, "native"), args, _SYSTEM_ROOT)
    assert parse_command_line(line) == [exe, *args]

    assert _posix_argv(posix_platform, posix_exe, args) == (posix_exe, *args)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('"abc" d e', ["abc", "d", "e"]),
        ('a\\\\b d"e f"g h', ["a\\\\b", "de fg", "h"]),
        ('a\\\\\\"b c d', ['a\\"b', "c", "d"]),
        ('a\\\\\\\\"b c" d e', ["a\\\\b c", "d", "e"]),
        ('a"b"" c d', ['ab" c d']),
        ('"" x  "" ', ["", "x", ""]),
        ("a\tb\u00a0c", ["a", "b\u00a0c"]),
    ],
)
def test_reference_parser_matches_microsoft_documented_examples(
    line: str, expected: list[str]
) -> None:
    """The oracle itself agrees with the examples in Microsoft's argument-parsing documentation."""
    assert parse_arguments(line) == expected


def test_reference_parser_treats_the_program_name_without_escapes() -> None:
    assert parse_command_line('"C:\\Program Files\\a\\" b') == ["C:\\Program Files\\a\\", "b"]
    assert parse_command_line("") == []
