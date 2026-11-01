"""Path_Field resolution (Requirements 13.1, 17.16 to 17.18).

Every function here is pure, except `ensure_storage_dir`: no file system access, no environment
reads, and no use of the host's `os.path`. The path rules come from `ntpath` or `posixpath`, chosen
by the `platform` argument, so the Windows rules can be tested on macOS and Linux (and the
reverse).

`platform` takes `sys.platform` values: `"win32"` selects the Windows rules, and every other value
(`"darwin"`, `"linux"`) selects the POSIX rules.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from local_agent.oscompat.env import lookup_env
from local_agent.storage import StorageError

__all__ = [
    "APP_DIR",
    "PathFieldError",
    "default_storage_dir",
    "ensure_storage_dir",
    "has_separator",
    "resolve_path_field",
]

APP_DIR: Final = "local-agent"


class PathFieldError(ValueError):
    """A Path_Field value cannot be resolved to an absolute path."""


@dataclass(frozen=True, slots=True)
class _PathRules:
    """The path operations and separator set of one platform family."""

    join: Callable[[str, str], str]
    normpath: Callable[[str], str]
    splitroot: Callable[[str], tuple[str, str, str]]
    separators: str
    home_variable: str
    windows: bool


_WINDOWS: Final = _PathRules(
    join=ntpath.join,
    normpath=ntpath.normpath,
    splitroot=ntpath.splitroot,
    separators="/\\",
    home_variable="USERPROFILE",
    windows=True,
)
_POSIX: Final = _PathRules(
    join=posixpath.join,
    normpath=posixpath.normpath,
    splitroot=posixpath.splitroot,
    separators="/",
    home_variable="HOME",
    windows=False,
)


def _rules(platform: str) -> _PathRules:
    return _WINDOWS if platform == "win32" else _POSIX


def has_separator(value: str, platform: str) -> bool:
    """Report whether `value` contains a path separator on `platform`.

    `/` counts everywhere. `\\` counts on Windows only; on POSIX it is an ordinary filename
    character (17.16).
    """
    return any(sep in value for sep in _rules(platform).separators)


def _expand_home(value: str, rules: _PathRules, home: str | None) -> str:
    """Replace a leading `~`, `~/` (or on Windows `~\\`) with `home`; `~user` stays (17.17)."""
    if value != "~" and not (len(value) > 1 and value[0] == "~" and value[1] in rules.separators):
        return value
    if not home:
        raise PathFieldError(
            f"cannot expand '~': the {rules.home_variable} environment variable is not set"
        )
    # Extra leading separators (`~//x`, Windows `~\\x`) would make `join` treat the rest as
    # rooted and drop `home`, so strip them before joining.
    rest = value[2:].lstrip(rules.separators)
    return rules.join(home, rest) if rest else home


def _is_unc_drive(drive: str) -> bool:
    return drive.replace("/", "\\").startswith("\\\\")


def _absolutize(value: str, rules: _PathRules, base_dir: str) -> str:
    """Turn a home-expanded value into an absolute path by applying the platform's rules (17.18)."""
    drive, root, _ = rules.splitroot(value)
    if rules.windows:
        if drive and not root and not _is_unc_drive(drive):
            raise PathFieldError("drive-relative paths are not supported")
        if drive:
            return value  # `C:\...` or UNC: already absolute
        if root:
            # Root-relative `\tools\x` takes the drive of the base directory.
            base_drive, _, _ = rules.splitroot(base_dir)
            return base_drive + value
        return rules.join(base_dir, value)
    if root:
        return value
    return rules.join(base_dir, value)


def resolve_path_field(value: str, *, base_dir: str, home: str | None, platform: str) -> str:
    """Resolve a Path_Field `value` to a lexically normalized path.

    Steps, in order:
    1. A leading `~` (`~`, `~/...`, and on Windows `~\\...`) becomes `home`. A missing or empty
       `home` raises `PathFieldError` (17.17). `~user` forms are ordinary relative names.
    2. A relative result is joined to `base_dir`: the process working directory for `AGENT_CONFIG`,
       the config file's directory for every other Path_Field (17.18). On Windows a root-relative
       `\\tools\\x` takes the drive of `base_dir`, and a drive-relative `C:tools\\x` raises
       `PathFieldError`. UNC and drive-absolute paths are accepted as they are.
    3. The result goes through `normpath`: no symlink resolution and no file system access, so the
       function returns its own output unchanged when called again on it.
    """
    rules = _rules(platform)
    expanded = _expand_home(value, rules, home)
    return rules.normpath(_absolutize(expanded, rules, base_dir))


def _required_env(env: Mapping[str, str], name: str, platform: str) -> str:
    """Value of a home variable; a missing or empty one raises `PathFieldError` (13.10, 18.24)."""
    value = lookup_env(env, name, platform)
    if not value:
        raise PathFieldError(
            f"cannot determine the default storage directory: "
            f"the {name} environment variable is not set"
        )
    return value


def default_storage_dir(platform: str, env: Mapping[str, str]) -> str:
    """The default Storage_Directory for `platform`, from the variables in `env` (17.35).

    Variables are read with `lookup_env`, so names match case-insensitively on Windows.
    - `win32`: a non-empty `LOCALAPPDATA` gives `<LOCALAPPDATA>\\local-agent`; otherwise
      `<USERPROFILE>\\AppData\\Local\\local-agent`.
    - `darwin`: `<HOME>/Library/Application Support/local-agent`.
    - `linux` and every other platform: an absolute `XDG_DATA_HOME` gives
      `<XDG_DATA_HOME>/local-agent`; a relative or empty value is ignored (XDG Base Directory
      spec); otherwise `<HOME>/.local/share/local-agent`.

    A missing home variable raises `PathFieldError`. The result is normalized lexically with the
    platform's path module and nothing touches the file system.
    """
    rules = _rules(platform)
    if platform == "win32":
        local_app_data = lookup_env(env, "LOCALAPPDATA", platform)
        if local_app_data:
            base = rules.join(local_app_data, APP_DIR)
        else:
            profile = _required_env(env, rules.home_variable, platform)
            base = rules.join(rules.join(rules.join(profile, "AppData"), "Local"), APP_DIR)
    elif platform == "darwin":
        home = _required_env(env, rules.home_variable, platform)
        base = rules.join(rules.join(rules.join(home, "Library"), "Application Support"), APP_DIR)
    else:
        xdg_data_home = lookup_env(env, "XDG_DATA_HOME", platform)
        if xdg_data_home and posixpath.isabs(xdg_data_home):
            base = rules.join(xdg_data_home, APP_DIR)
        else:
            home = _required_env(env, rules.home_variable, platform)
            base = rules.join(rules.join(rules.join(home, ".local"), "share"), APP_DIR)
    return rules.normpath(base)


def ensure_storage_dir(path: str) -> None:
    """Create the Storage_Directory and every missing parent (17.38, 18.23).

    An existing non-directory raises `StorageError("not a directory")`. Otherwise this calls
    `os.makedirs(path, exist_ok=True)`, which gives every created directory the default
    permissions; restricting the Storage_Directory itself is `oscompat/permissions.py`'s job.
    An `OSError` becomes a `StorageError` carrying the OS reason (18.24). The reason is the
    `strerror` text only, so it holds no file name that a caller would have to redact.
    """
    try:
        if os.path.exists(path) and not os.path.isdir(path):
            raise StorageError(path, "not a directory")
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise StorageError(path, exc.strerror or type(exc).__name__) from exc
