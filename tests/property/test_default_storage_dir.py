"""Property test for the default Storage_Directory (Requirements 17.35, 13.10).

`default_storage_dir(platform, env)` picks a per-user directory from `LOCALAPPDATA`,
`USERPROFILE`, `HOME` and `XDG_DATA_HOME`. The test builds environments where each of the four
variables is independently absent, empty, relative or absolute, and compares the result with a
reference written straight from the wording of 17.35 using plain string concatenation. Windows
environments spell the variable names in random letter case.
"""

import ntpath
import posixpath

import pytest
from hypothesis import given
from hypothesis import strategies as st

from local_agent.oscompat.paths import PathFieldError, default_storage_dir

_VARIABLES = ("LOCALAPPDATA", "USERPROFILE", "HOME", "XDG_DATA_HOME")
_PLATFORMS = ("win32", "darwin", "linux")
_KINDS = ("absent", "empty", "relative", "absolute")

# Components include `.` and `..` so that lexical normalization is exercised, and a name with a
# space (as in "Application Support"). The POSIX pool adds a backslash, which is an ordinary
# filename character there.
_COMPONENTS = st.sampled_from(["a", "Users", "jo hn", "x.y", ".", "..", "local-agent", "share"])
_POSIX_COMPONENTS = st.one_of(_COMPONENTS, st.just("a\\b"))


@st.composite
def _value(draw: st.DrawFn, kind: str, windows: bool) -> str | None:
    """A variable value of the given kind; `None` means the variable is absent."""
    if kind == "absent":
        return None
    if kind == "empty":
        return ""
    if windows:
        sep = draw(st.sampled_from(["\\", "/"]))
        parts = draw(st.lists(_COMPONENTS, min_size=0 if kind == "absolute" else 1, max_size=4))
        body = sep.join(parts)
        if kind == "relative":
            return body
        letter = draw(st.sampled_from("CcDdZz"))
        if draw(st.booleans()):
            return f"\\\\server\\share\\{body}" if body else "\\\\server\\share"
        return f"{letter}:{sep}{body}"
    parts = draw(st.lists(_POSIX_COMPONENTS, min_size=1, max_size=4))
    body = "/".join(parts)
    if kind == "relative":
        return body
    return "/" + body


@st.composite
def _cases(draw: st.DrawFn) -> tuple[str, dict[str, str], dict[str, str | None]]:
    """A platform, the environment mapping, and the canonical-name view the reference reads."""
    platform = draw(st.sampled_from(_PLATFORMS))
    windows = platform == "win32"
    model: dict[str, str | None] = {}
    env: dict[str, str] = {}
    for name in _VARIABLES:
        value = draw(_value(draw(st.sampled_from(_KINDS)), windows))
        model[name] = value
        if value is None:
            continue
        if windows:
            # Random case per letter; "_" and digits are unchanged.
            flips = draw(st.lists(st.booleans(), min_size=len(name), max_size=len(name)))
            cased = [c.lower() if flip else c for c, flip in zip(name, flips, strict=True)]
            env["".join(cased)] = value
        else:
            env[name] = value
    if not windows:
        # Names are exact on macOS and Linux, so a differently cased spelling must be ignored.
        for name in _VARIABLES:
            if draw(st.booleans()):
                env.setdefault(name.lower(), "/decoy/path")
    if draw(st.booleans()):
        env["PATH"] = "/usr/bin"
    return platform, env, model


def _reference(platform: str, model: dict[str, str | None]) -> str | None:
    """The 17.35 rule as text. `None` means the required home variable is missing."""
    if platform == "win32":
        local_app_data = model["LOCALAPPDATA"]
        if local_app_data:
            return ntpath.normpath(local_app_data + "\\local-agent")
        profile = model["USERPROFILE"]
        if not profile:
            return None
        return ntpath.normpath(profile + "\\AppData\\Local\\local-agent")
    if platform == "darwin":
        home = model["HOME"]
        if not home:
            return None
        return posixpath.normpath(home + "/Library/Application Support/local-agent")
    xdg = model["XDG_DATA_HOME"]
    if xdg and xdg.startswith("/"):
        return posixpath.normpath(xdg + "/local-agent")
    home = model["HOME"]
    if not home:
        return None
    return posixpath.normpath(home + "/.local/share/local-agent")


# Feature: local-agent-backend, Property 56: Default storage directory
# Validates: Requirements 17.35, 13.10
@given(_cases())
def test_default_storage_dir_follows_reference_rule(
    case: tuple[str, dict[str, str], dict[str, str | None]],
) -> None:
    platform, env, model = case
    expected = _reference(platform, model)

    if expected is None:
        with pytest.raises(PathFieldError):
            default_storage_dir(platform, env)
        return

    result = default_storage_dir(platform, env)
    assert result == expected
    path_module = ntpath if platform == "win32" else posixpath
    assert path_module.basename(result) == "local-agent"
