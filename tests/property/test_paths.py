"""Property test for Path_Field resolution (Requirements 17.16, 17.17, 17.18, 13.1, 13.10, 13.11).

`resolve_path_field` is exercised directly and through the loader for `storage.path`, with the
config directory as `base_dir`. The `platform` argument selects the Windows or POSIX rules, so
every example runs on every host.

Each case is a platform, a base directory, a home value (absent, empty, or absolute), and a path
string built from a prefix (none, root, drive, UNC, `~` forms) and name components joined with
random separators. The expected result comes from a reference that reads the wording of the
property: replace a leading `~` textually, then normalize either the value itself (absolute) or
the base directory joined with it (relative). It uses `ntpath` and `posixpath` directly.
"""

import ntpath
import posixpath
import re
from dataclasses import dataclass

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.config.loader import ConfigError, load_config_from_text
from local_agent.oscompat.paths import PathFieldError, has_separator, resolve_path_field

# `.` and `..` exercise lexical normalization. `~` and `~user` as components check that only a
# leading `~` followed by a separator (or nothing) is expanded.
_COMMON = ["a", "b c", "x.y", "tools", ".", "..", "~", "~user"]
_WIN_COMPONENTS = st.sampled_from(_COMMON)
_POSIX_COMPONENTS = st.sampled_from([*_COMMON, "a\\b", "\\"])
# Home and base directories hold plain names so that their own normalization is trivial.
_PLAIN = st.sampled_from(["Users", "jo hn", "work", "x.y", "proj"])

_WIN_PREFIXES = [
    "",
    "\\",
    "/",
    "C:\\",
    "d:/",
    "C:",
    "z:",
    "\\\\srv\\share\\",
    "//srv/share/",
    "~",
    "~\\",
    "~/",
    "~user\\",
    "~//",
    "~\\\\",
    "~/\\",
]
_POSIX_PREFIXES = ["", "/", "\\", "//", "~", "~/", "~user/", "~\\", "~//"]


@dataclass(frozen=True)
class _Case:
    platform: str
    value: str
    base_dir: str
    home: str | None


@st.composite
def _directory(draw: st.DrawFn, windows: bool) -> str:
    """An absolute directory in the style of `windows` with one to three plain components."""
    parts = draw(st.lists(_PLAIN, min_size=1, max_size=3))
    if not windows:
        return "/" + "/".join(parts)
    sep = draw(st.sampled_from(["\\", "/"]))
    if draw(st.booleans()):
        return draw(st.sampled_from(["C", "d", "Z"])) + ":" + sep + sep.join(parts)
    return "\\\\server\\share\\" + "\\".join(parts)


@st.composite
def _cases(draw: st.DrawFn) -> _Case:
    platform = draw(st.sampled_from(["win32", "darwin", "linux"]))
    windows = platform == "win32"
    separators = ["/", "\\", "//", "\\\\", "/\\"] if windows else ["/", "//"]
    prefix = draw(st.sampled_from(_WIN_PREFIXES if windows else _POSIX_PREFIXES))
    parts = draw(st.lists(_WIN_COMPONENTS if windows else _POSIX_COMPONENTS, max_size=4))
    body = parts[0] if parts else ""
    for part in parts[1:]:
        body += draw(st.sampled_from(separators)) + part
    home = draw(st.one_of(st.none(), st.just(""), _directory(windows), _directory(windows)))
    return _Case(platform, prefix + body, draw(_directory(windows)), home)


def _reference(case: _Case) -> str | None:
    """The expected result; `None` means `PathFieldError`."""
    value, base, home = case.value, case.base_dir, case.home
    windows = case.platform == "win32"
    separators = "/\\" if windows else "/"
    if value == "~" or (value[:1] == "~" and value[1:2] != "" and value[1] in separators):
        if not home:
            return None
        value = home + value[1:]
    if not windows:
        return posixpath.normpath(value if value.startswith("/") else posixpath.join(base, value))
    if re.match(r"[A-Za-z]:(?![/\\])", value):
        return None  # drive-relative
    if re.match(r"[A-Za-z]:[/\\]", value) or re.match(r"[/\\]{2}", value):
        return ntpath.normpath(value)  # drive-absolute or UNC
    if value.startswith(("/", "\\")):
        return ntpath.normpath(ntpath.splitdrive(base)[0] + value)  # root-relative
    return ntpath.normpath(ntpath.join(base, value))


def _is_absolute(path: str, platform: str) -> bool:
    if platform == "win32":
        return bool(re.match(r"[A-Za-z]:[/\\]", path) or re.match(r"[/\\]{2}", path))
    return path.startswith("/")


def _env(case: _Case) -> dict[str, str]:
    if case.home is None:
        return {}
    return {"USERPROFILE" if case.platform == "win32" else "HOME": case.home}


def _config_text(backend: str, value: str) -> str:
    tree = {"llm": {"model": "m"}, "storage": {"backend": backend, "path": value}}
    return yaml.safe_dump(tree)


# Feature: local-agent-backend, Property 34: Path_Field resolution
# Validates: Requirements 17.16, 17.17, 17.18, 13.1, 13.10, 13.11
@settings(max_examples=300)
@given(_cases())
def test_path_field_resolution_follows_reference_rules(case: _Case) -> None:
    expected = _reference(case)
    windows = case.platform == "win32"

    # `/` separates everywhere; `\` separates on Windows only.
    assert has_separator(case.value, case.platform) == (
        "/" in case.value or (windows and "\\" in case.value)
    )

    # Directly.
    try:
        result = resolve_path_field(
            case.value, base_dir=case.base_dir, home=case.home, platform=case.platform
        )
    except PathFieldError:
        assert expected is None
    else:
        assert result == expected
        assert _is_absolute(result, case.platform)
        again = resolve_path_field(
            result, base_dir=case.base_dir, home=case.home, platform=case.platform
        )
        assert again == result
        if not windows and ".." not in case.value.split("/"):
            # A backslash is an ordinary filename character on POSIX and survives.
            assert result.count("\\") == case.value.count("\\")

    # Through the loader for `storage.path`, with the config directory as the base.
    env = _env(case)
    text = _config_text("sqlite", case.value)
    if expected is None:
        with pytest.raises(ConfigError) as caught:
            load_config_from_text(
                text, env, "config.yaml", base_dir=case.base_dir, platform=case.platform
            )
        assert [issue.path for issue in caught.value.issues] == ["storage.path"]
    else:
        config = load_config_from_text(
            text, env, "config.yaml", base_dir=case.base_dir, platform=case.platform
        )
        assert config.storage.path == expected

    # With the memory backend `storage.path` stays as written and raises no issue.
    memory = load_config_from_text(
        _config_text("memory", case.value),
        env,
        "config.yaml",
        base_dir=case.base_dir,
        platform=case.platform,
    )
    assert memory.storage.path == case.value
