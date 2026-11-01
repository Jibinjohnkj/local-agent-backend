"""Environment-variable name rules (Requirements 17.21, 17.22, 13.2).

Windows treats variable names as case-insensitive (CPython's ``os.environ`` upper-cases them
with ``str.upper()``); macOS and Linux match names exactly. Every function takes the platform
as a parameter and reads no global state, so the Windows rules are testable on any host.
"""

import sys
from collections.abc import Mapping

from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS as _SDK_INHERITED_ENV_VARS

_WIN32 = "win32"

# Copies of the SDK lists for the platform family the SDK did not build its own list for.
# The SDK picks its list from ``sys.platform`` at import time, so on a Linux runner it cannot
# supply the Windows list that the Windows rules need under test.
_SDK_WINDOWS_NAMES: tuple[str, ...] = (
    "APPDATA",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "USERNAME",
    "USERPROFILE",
)
_SDK_POSIX_NAMES: tuple[str, ...] = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")

# Extra names that ``npx.cmd`` and Node need on Windows (17.22 design note).
WINDOWS_EXTRA_INHERITED_ENV_VARS: tuple[str, ...] = (
    "COMSPEC",
    "WINDIR",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
)


def _base_names(platform: str) -> tuple[str, ...]:
    """SDK default names for ``platform``; the live SDK constant when it matches the host."""
    wants_windows = platform == _WIN32
    if wants_windows == (sys.platform == _WIN32):
        return tuple(_SDK_INHERITED_ENV_VARS)
    return _SDK_WINDOWS_NAMES if wants_windows else _SDK_POSIX_NAMES


def _build_inherited_names(platform: str) -> tuple[str, ...]:
    names = _base_names(platform)
    if platform != _WIN32:
        return names
    extras = tuple(n for n in WINDOWS_EXTRA_INHERITED_ENV_VARS if n not in names)
    return names + extras


# Allow-lists of variable names a child process inherits, per platform family (17.22).
INHERITED_ENV_NAMES_WINDOWS: tuple[str, ...] = _build_inherited_names(_WIN32)
INHERITED_ENV_NAMES_POSIX: tuple[str, ...] = _build_inherited_names("linux")


def inherited_env_names(platform: str) -> tuple[str, ...]:
    """The allow-list of inherited variable names for ``platform``."""
    return INHERITED_ENV_NAMES_WINDOWS if platform == _WIN32 else INHERITED_ENV_NAMES_POSIX


def env_key(name: str, platform: str) -> str:
    """The comparison key of an environment-variable name: upper-case on Windows, else as-is."""
    return name.upper() if platform == _WIN32 else name


def lookup_env(env: Mapping[str, str], name: str, platform: str) -> str | None:
    """Value of the variable called ``name``, or ``None`` when no entry matches (17.21).

    Windows matches by ordinal ``str.upper()``; macOS and Linux match the exact name. When a
    Windows mapping holds several spellings of one name, an exact spelling wins, otherwise the
    first entry in iteration order.
    """
    if platform != _WIN32:
        return env.get(name)
    if name in env:
        return env[name]
    wanted = name.upper()
    for key, value in env.items():
        if key.upper() == wanted:
            return value
    return None


def merge_env(
    inherited: Mapping[str, str], configured: Mapping[str, str], platform: str
) -> dict[str, str]:
    """Inherited variables overlaid with configured ones (17.22).

    Each configured name removes every inherited entry with the same ``env_key`` and is then
    inserted with its own spelling, so on Windows ``Path`` configured over an inherited ``PATH``
    leaves exactly one entry. Inherited entries that no configured name replaces are unchanged.
    """
    replaced = {env_key(name, platform) for name in configured}
    merged = {k: v for k, v in inherited.items() if env_key(k, platform) not in replaced}
    merged.update(configured)
    return merged


def inherited_env(environ: Mapping[str, str], platform: str) -> dict[str, str]:
    """The allow-listed subset of ``environ`` a child process inherits.

    Names are looked up with ``lookup_env`` and stored under the allow-list spelling. Values
    starting with ``()`` are skipped, as the SDK does, because they are exported shell functions.
    """
    result: dict[str, str] = {}
    for name in inherited_env_names(platform):
        value = lookup_env(environ, name, platform)
        if value is None or value.startswith("()"):
            continue
        result[name] = value
    return result
