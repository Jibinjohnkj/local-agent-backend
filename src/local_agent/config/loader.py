"""Config_Loader: turn `config.yaml` into a validated `BackendConfig` (Requirement 13).

The pipeline for one load, in order:

1. Read the file as bytes and decode it as strict UTF-8, dropping one leading BOM (17.23, 17.24).
2. Normalize CRLF to LF, then parse with `yaml.safe_load` (17.25, 13.7).
3. Substitute `${VAR}` in every string value (13.2, 13.8).
4. Resolve Path_Fields against the config directory (13.1, 13.10, 17.16 to 17.18).
5. Validate with pydantic (pass 1), then run the cross-field checks on the raw tree (pass 2).

Every problem found in steps 3 to 5 lands in one `ConfigError`, so the user sees all of them at
once (13.3). No message contains a secret or a substituted value: pydantic's `input` is never
copied, and a bad value is echoed only for non-secret fields when it was written literally in the
file.

Functions take `platform` (a `sys.platform` value) so the Windows rules can be tested on any host.
`None` means the running platform.
"""

import ipaddress
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple

import yaml
from pydantic import AnyUrl, SecretStr, ValidationError

from local_agent.config.models import BackendConfig
from local_agent.oscompat.env import lookup_env
from local_agent.oscompat.paths import PathFieldError, has_separator, resolve_path_field

__all__ = [
    "ConfigError",
    "ConfigIssue",
    "decode_config_bytes",
    "dump_config",
    "load_config",
    "load_config_from_text",
    "resolve_config_path",
    "substitute_env",
]

_UTF8_BOM: Final = b"\xef\xbb\xbf"
_MAX_PATTERNS: Final = 100  # 5.3
_MAX_PATTERN_LENGTH: Final = 128  # 5.3
_MIN_BEARER_TOKEN_LENGTH: Final = 32  # 16.5
_ECHO_LIMIT: Final = 60
_MISSING: Final = object()

# `$${` is the escape for a literal `${`; it must be tried before the reference form.
_REFERENCE: Final = re.compile(r"\$\$\{|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Pydantic error types whose offending value is safe and useful to echo (the port example, 1.7).
_ECHOED_ERROR_TYPES: Final = frozenset(
    {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "literal_error",
        "int_parsing",
        "float_parsing",
        "bool_parsing",
        "string_pattern_mismatch",
        "string_too_short",
        "string_too_long",
    }
)


class ConfigIssue(NamedTuple):
    path: str  # dotted path, for example "mcp_servers.1.command"
    message: str  # never contains substituted env values


class ConfigError(Exception):
    """The configuration cannot be loaded. `issues` lists every problem found."""

    def __init__(self, issues: Iterable[ConfigIssue]) -> None:
        self.issues: list[ConfigIssue] = list(issues)
        super().__init__(self.issues)

    def __str__(self) -> str:
        lines = ["invalid configuration:"]
        lines.extend(f"  {issue.path}: {issue.message}" for issue in self.issues)
        return "\n".join(lines)


def _platform(platform: str | None) -> str:
    return sys.platform if platform is None else platform


def _join(path: str, key: object) -> str:
    return f"{path}.{key}" if path else str(key)


def _home(env: Mapping[str, str], platform: str) -> str | None:
    name = "USERPROFILE" if platform == "win32" else "HOME"
    return lookup_env(env, name, platform) or None


# --- Locating the file -----------------------------------------------------------------------


def resolve_config_path(env: Mapping[str, str], cwd: Path, *, platform: str | None = None) -> Path:
    """The config file to load: `AGENT_CONFIG` when set and non-empty, else `cwd/config.yaml`.

    `AGENT_CONFIG` is a Path_Field resolved against `cwd` (13.1). There is no fallback to another
    file (13.6).
    """
    plat = _platform(platform)
    value = lookup_env(env, "AGENT_CONFIG", plat)
    if not value:
        return cwd / "config.yaml"
    try:
        return Path(
            resolve_path_field(value, base_dir=str(cwd), home=_home(env, plat), platform=plat)
        )
    except PathFieldError as exc:
        raise ConfigError([ConfigIssue("AGENT_CONFIG", str(exc))]) from exc


# --- Bytes to text ---------------------------------------------------------------------------


def decode_config_bytes(data: bytes, source: str = "<config>") -> str:
    """Decode config file bytes as UTF-8, independent of locale or code page (17.23, 17.24).

    One leading BOM is dropped. Invalid UTF-8 raises `ConfigError` naming `source` and the byte
    offset counted from the start of the file, BOM included.
    """
    bom_length = len(_UTF8_BOM) if data.startswith(_UTF8_BOM) else 0
    try:
        return data[bom_length:].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        offset = exc.start + bom_length
        raise ConfigError([ConfigIssue(source, f"invalid UTF-8 at byte offset {offset}")]) from exc


def _parse_yaml(text: str, source: str) -> Any:
    """Parse `text`; a failure reports `source` with a 1-based line and column (13.7)."""
    try:
        return yaml.safe_load(text)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        reason = exc.problem or exc.context or "parse error"
        raise ConfigError([ConfigIssue(source, f"malformed YAML{where}: {reason}")]) from exc
    except yaml.reader.ReaderError as exc:
        # Not a MarkedYAMLError: it carries a character index into the text, not a Mark.
        index = exc.position if isinstance(exc.position, int) else 0
        line = text.count("\n", 0, index) + 1
        column = index - (text.rfind("\n", 0, index) + 1) + 1
        raise ConfigError(
            [ConfigIssue(source, f"malformed YAML at line {line}, column {column}: {exc.reason}")]
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError([ConfigIssue(source, "malformed YAML")]) from exc


# --- ${VAR} substitution ---------------------------------------------------------------------


def _substitute_string(
    value: str, env: Mapping[str, str], path: str, platform: str, issues: list[ConfigIssue]
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name is None:  # the `$${` escape
            return "${"
        found = lookup_env(env, name, platform)
        if found is None:
            issues.append(ConfigIssue(path, f"environment variable {name} is not set"))
            return match.group(0)
        return found

    return _REFERENCE.sub(replace, value)


def _substitute_tree(
    raw: Any, env: Mapping[str, str], path: str, platform: str, issues: list[ConfigIssue]
) -> Any:
    """Return a copy of `raw` with every string value substituted. Keys are left alone."""
    if isinstance(raw, str):
        return _substitute_string(raw, env, path, platform, issues)
    if isinstance(raw, list):
        return [
            _substitute_tree(item, env, _join(path, index), platform, issues)
            for index, item in enumerate(raw)
        ]
    if isinstance(raw, dict):
        return {
            key: _substitute_tree(item, env, _join(path, key), platform, issues)
            for key, item in raw.items()
        }
    return raw


def substitute_env(
    raw: Any, env: Mapping[str, str], path: str = "", *, platform: str | None = None
) -> Any:
    """Replace `${NAME}` in every string value of `raw` (pure, 13.2).

    `$${` becomes a literal `${`. Non-string values and mapping keys are unchanged. Names match
    per 17.21. Every missing variable is collected and raised together as one `ConfigError`
    whose issues name the variable and the dotted field path (13.8).
    """
    issues: list[ConfigIssue] = []
    result = _substitute_tree(raw, env, path, _platform(platform), issues)
    if issues:
        raise ConfigError(issues)
    return result


# --- Path_Fields -----------------------------------------------------------------------------


def _resolve_path_fields(
    tree: dict[str, Any], *, base_dir: str, home: str | None, platform: str, skip: set[str]
) -> list[ConfigIssue]:
    """Resolve the Path_Fields of `tree` in place and return one issue per failure.

    The Path_Fields are `mcp_servers[*].cwd`, `mcp_servers[*].command` when it contains a
    separator (17.5), and `storage.path` unless `storage.backend` is `memory` (13.11). Bare
    command names and `args` are left alone (17.20). Fields whose text had a missing `${VAR}`
    are skipped because their value is incomplete.
    """
    issues: list[ConfigIssue] = []

    def resolve(container: dict[str, Any], key: str, path: str, *, need_separator: bool) -> None:
        value = container.get(key)
        if not isinstance(value, str) or path in skip:
            return
        if need_separator and not has_separator(value, platform):
            return
        try:
            container[key] = resolve_path_field(
                value, base_dir=base_dir, home=home, platform=platform
            )
        except PathFieldError as exc:
            issues.append(ConfigIssue(path, str(exc)))

    servers = tree.get("mcp_servers")
    if isinstance(servers, list):
        for index, server in enumerate(servers):
            if isinstance(server, dict) and server.get("transport") == "stdio":
                prefix = f"mcp_servers.{index}"
                resolve(server, "cwd", f"{prefix}.cwd", need_separator=False)
                resolve(server, "command", f"{prefix}.command", need_separator=True)

    storage = tree.get("storage")
    if isinstance(storage, dict) and storage.get("backend") != "memory":
        resolve(storage, "path", "storage.path", need_separator=False)
    return issues


# --- Pass 1: pydantic ------------------------------------------------------------------------


def _lookup(tree: Any, loc: Iterable[int | str]) -> Any:
    node = tree
    for part in loc:
        if isinstance(node, dict):
            if part not in node:
                return _MISSING
        elif not (isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node)):
            return _MISSING
        node = node[part]
    return node


def _is_secret_loc(loc: list[int | str]) -> bool:
    if any(part in ("api_key", "bearer_token") for part in loc):
        return True
    return len(loc) >= 3 and loc[0] == "mcp_servers" and loc[2] in ("env", "headers")


def _echo(loc: list[int | str], error_type: str, tree: Any, raw_tree: Any) -> str:
    """` (got X)` for a non-secret scalar the user wrote literally in the file, else nothing.

    The value is shown only when the validated tree and the raw parsed tree agree on it, so a
    value that came from `${VAR}` substitution or Path_Field resolution is never repeated.
    """
    if error_type not in _ECHOED_ERROR_TYPES or _is_secret_loc(loc):
        return ""
    value = _lookup(tree, loc)
    if value is _MISSING or not isinstance(value, (int, float, str)):
        return ""
    if _lookup(raw_tree, loc) != value:
        return ""
    text = repr(value)
    if len(text) > _ECHO_LIMIT:
        text = text[:_ECHO_LIMIT] + "..."
    return f" (got {text})"


def _validate(
    tree: dict[str, Any], raw_tree: Any
) -> tuple[BackendConfig | None, list[ConfigIssue]]:
    try:
        return BackendConfig.model_validate(tree), []
    except ValidationError as exc:
        issues: list[ConfigIssue] = []
        for error in exc.errors():
            loc: list[int | str] = list(error["loc"])
            error_type = error["type"]
            message = error["msg"]
            is_server = len(loc) >= 2 and loc[0] == "mcp_servers" and isinstance(loc[1], int)
            # The discriminated union adds its tag ("stdio"/"sse") to the location.
            if is_server and len(loc) >= 3 and loc[2] in ("stdio", "sse"):
                del loc[2]
            elif is_server and len(loc) == 2 and isinstance(_lookup(tree, loc), dict):
                # Pydantic's own text repeats the transport value, which could be substituted.
                if error_type == "union_tag_invalid":
                    loc.append("transport")
                    message = "unsupported transport; expected 'stdio' or 'sse'"
                elif error_type == "union_tag_not_found":
                    loc.append("transport")
                    message = "transport is required; expected 'stdio' or 'sse'"
            path = ".".join(str(part) for part in loc)
            issues.append(ConfigIssue(path, message + _echo(loc, error_type, tree, raw_tree)))
        return None, issues


# --- Pass 2: cross-field checks on the raw tree ----------------------------------------------


def _is_loopback_host(host: str) -> bool:
    """`localhost`, an address in 127.0.0.0/8, or `::1` (16.1)."""
    name = host.strip("[]").lower()
    if name == "localhost":
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.ip_network("127.0.0.0/8")
    return address == ipaddress.IPv6Address("::1")


def _cross_check(tree: dict[str, Any]) -> list[ConfigIssue]:
    """Checks that need more than one field. They run even when pass 1 failed (13.3)."""
    issues: list[ConfigIssue] = []

    # Duplicate MCP_Server names (6.4): every colliding entry is reported.
    indexes_by_name: dict[str, list[int]] = {}
    servers = tree.get("mcp_servers")
    if isinstance(servers, list):
        for index, server in enumerate(servers):
            if isinstance(server, dict) and isinstance(server.get("name"), str):
                indexes_by_name.setdefault(server["name"], []).append(index)
    for name, indexes in indexes_by_name.items():
        if len(indexes) > 1:
            where = ", ".join(f"mcp_servers.{i}" for i in indexes)
            for index in indexes:
                issues.append(
                    ConfigIssue(
                        f"mcp_servers.{index}.name",
                        f"duplicate MCP_Server name {name!r} (defined at {where})",
                    )
                )

    # Permission_Policy: known servers, at most 100 patterns, each 1 to 128 characters (5.3).
    permissions = tree.get("permissions")
    if isinstance(permissions, dict):
        for server_name, patterns in permissions.items():
            base = f"permissions.{server_name}"
            if server_name not in indexes_by_name:
                issues.append(
                    ConfigIssue(
                        base,
                        f"references MCP_Server {server_name!r}, which is not defined in "
                        "mcp_servers",
                    )
                )
            if not isinstance(patterns, list):
                continue
            if len(patterns) > _MAX_PATTERNS:
                issues.append(
                    ConfigIssue(
                        base,
                        f"MCP_Server {server_name!r} has {len(patterns)} patterns; "
                        f"at most {_MAX_PATTERNS} are allowed",
                    )
                )
            for index, pattern in enumerate(patterns):
                if not isinstance(pattern, str):
                    continue
                if not pattern:
                    issues.append(
                        ConfigIssue(f"{base}.{index}", f"empty pattern for {server_name!r}")
                    )
                elif len(pattern) > _MAX_PATTERN_LENGTH:
                    issues.append(
                        ConfigIssue(
                            f"{base}.{index}",
                            f"pattern for {server_name!r} is {len(pattern)} characters; at "
                            f"most {_MAX_PATTERN_LENGTH} are allowed (starts with "
                            f"{pattern[:24]!r})",
                        )
                    )

    # A bind that is not loopback needs a bearer token of at least 32 characters (16.5).
    server_section = tree.get("server")
    if isinstance(server_section, dict):
        host = server_section.get("host", "127.0.0.1")
        token = server_section.get("bearer_token")
        if (
            isinstance(host, str)
            and not _is_loopback_host(host)
            and not (isinstance(token, str) and len(token) >= _MIN_BEARER_TOKEN_LENGTH)
        ):
            issues.append(
                ConfigIssue(
                    "server.bearer_token",
                    f"a bearer token of at least {_MIN_BEARER_TOKEN_LENGTH} characters is "
                    "required when server.host is not a loopback address",
                )
            )
    return issues


# --- Loading ---------------------------------------------------------------------------------


def _under(path: str, roots: set[str]) -> bool:
    return any(path == root or path.startswith(root + ".") for root in roots)


def load_config_from_text(
    text: str,
    env: Mapping[str, str],
    source: str,
    *,
    base_dir: str | None = None,
    platform: str | None = None,
) -> BackendConfig:
    """Parse, substitute, resolve, and validate `text`; `source` names it in error reports.

    `base_dir` is the directory relative Path_Fields resolve against: the config file's
    directory. It defaults to the process working directory.
    """
    plat = _platform(platform)
    base = os.getcwd() if base_dir is None else base_dir

    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    data = _parse_yaml(text, source)
    if not isinstance(data, dict):
        raise ConfigError([ConfigIssue(source, "top-level value must be a mapping")])

    substitution_issues: list[ConfigIssue] = []
    tree: dict[str, Any] = _substitute_tree(data, env, "", plat, substitution_issues)
    unresolved = {issue.path for issue in substitution_issues}

    path_issues = _resolve_path_fields(
        tree, base_dir=base, home=_home(env, plat), platform=plat, skip=unresolved
    )
    config, pass_one = _validate(tree, data)
    pass_two = _cross_check(tree)

    # Pass 2 words the Permission_Policy errors with the server and pattern (5.3), so its text
    # replaces pydantic's at the same path. A value with a missing `${VAR}` is incomplete, so
    # errors about it would only add noise.
    pass_two_paths = {issue.path for issue in pass_two}
    pass_one = [
        issue
        for issue in pass_one
        if not (issue.path.startswith("permissions.") and issue.path in pass_two_paths)
    ]
    issues = list(
        dict.fromkeys(
            [
                *substitution_issues,
                *path_issues,
                *(i for i in [*pass_one, *pass_two] if not _under(i.path, unresolved)),
            ]
        )
    )
    if issues:
        raise ConfigError(issues)
    if config is None:  # pass 1 failed without reporting; keep the error contract regardless
        raise ConfigError([ConfigIssue(source, "configuration could not be validated")])
    return config


def load_config(
    path: Path, env: Mapping[str, str], *, platform: str | None = None
) -> BackendConfig:
    """Read, decode, and validate the config file at `path`.

    Reads bytes, so it needs no event loop and no locale (17.23). A missing or unreadable file
    raises `ConfigError` naming the path and the OS reason (13.6).
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ConfigError([ConfigIssue(str(path), f"cannot read config file: {reason}")]) from exc
    text = decode_config_bytes(data, str(path))
    return load_config_from_text(
        text,
        env,
        str(path),
        base_dir=os.path.dirname(os.path.abspath(path)),
        platform=platform,
    )


# --- Dumping ---------------------------------------------------------------------------------


def _plain(value: object) -> object:
    """Convert a `model_dump` tree to YAML-ready data: reveal secrets, stringify URLs.

    Every string value gets `${` escaped as `$${` so loading it again gives the same text.
    Mapping keys are never substituted on load, so they are copied as they are.
    """
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    elif isinstance(value, AnyUrl):
        value = str(value)
    if isinstance(value, str):
        return value.replace("${", "$${")
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


# Unicode line breaks that PyYAML's emitter treats as printable (so `allow_unicode=True` writes
# them raw) but its reader folds in a flow scalar: NEL is read back as a space. They must be
# written as escapes, which only the double-quoted style has.
_UNSAFE_RAW_CHARS: Final = frozenset("\x85\u2028\u2029")


class _ConfigDumper(yaml.SafeDumper):
    """A `SafeDumper` that double-quotes strings holding characters YAML cannot round-trip raw.

    A subclass, so the global `yaml.SafeDumper` representers stay untouched.
    """


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = '"' if not _UNSAFE_RAW_CHARS.isdisjoint(data) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_ConfigDumper.add_representer(str, _represent_str)


def dump_config(cfg: BackendConfig) -> str:
    """Serialize `cfg` to YAML with secrets revealed and literal `${` escaped (13.5).

    `load_config_from_text(dump_config(cfg), {})` returns a config equal to `cfg`. Path_Fields
    in `cfg` are already absolute, and resolving an absolute path returns it unchanged.
    """
    data = _plain(cfg.model_dump(mode="python"))
    # A huge width stops PyYAML from folding long strings across lines.
    return yaml.dump(
        data,
        Dumper=_ConfigDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=2**31 - 1,
    )
