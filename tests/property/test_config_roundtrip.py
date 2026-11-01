"""Property test for the configuration round-trip (Requirement 13.5).

`dump_config` writes a `BackendConfig` as YAML with secrets revealed and a literal `${` escaped as
`$${`. Loading that text again with an empty environment must give back an equal config. The
generator builds a raw config tree and validates it, so every example is a valid config. It
covers secrets, strings holding `${` and Unicode, optional fields set and unset, both server
transports, and `storage` sections for both backends with `path` set and unset.

Path_Fields in a validated config are absolute, so the generator makes them absolute for the host
platform: the loader resolves an absolute path to itself.
"""

import operator
import string
import sys
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from local_agent.config.loader import dump_config, load_config_from_text
from local_agent.config.models import BackendConfig

_FRAGMENTS = [
    "${",
    "$${",
    "${X}",
    "${",
    "}",
    "$",
    "é",
    "日本",
    "🙂",
    "\t",
    "\n",
    "\r\n",
    " ",
    "#",
    ": ",
]
_NAME_CHARS = string.ascii_letters + string.digits + "_-"
_SEGMENT_CHARS = string.ascii_letters + string.digits + "_-é"
_LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "::1", "127.0.0.2"]
_OTHER_HOSTS = ["0.0.0.0", "example.com", "192.168.1.5"]

# Free text that often contains `${`, `$${`, whitespace and non-ASCII characters.
_text = st.lists(st.one_of(st.text(max_size=8), st.sampled_from(_FRAGMENTS)), max_size=4).map(
    "".join
)
_nonempty_text = st.builds(operator.add, st.text(min_size=1, max_size=6), _text)


@st.composite
def _abs_path(draw: st.DrawFn) -> str:
    """An absolute, normalized path for the host platform (what the loader leaves unchanged)."""
    segments = draw(
        st.lists(st.text(_SEGMENT_CHARS, min_size=1, max_size=8), min_size=1, max_size=4)
    )
    if sys.platform == "win32":
        return "C:\\" + "\\".join(segments)
    return "/" + "/".join(segments)


_bare_command = st.text(_NAME_CHARS, min_size=1, max_size=10)
_secret_map = st.dictionaries(_text, _text, max_size=3)
_url = st.builds(
    "http{}://{}{}/{}".format,
    st.sampled_from(["", "s"]),
    st.text(string.ascii_lowercase, min_size=1, max_size=10),
    st.one_of(st.just(""), st.integers(1, 65535).map(lambda p: f":{p}")),
    st.text(string.ascii_lowercase + string.digits + "/", max_size=10),
)


@st.composite
def _stdio_server(draw: st.DrawFn, name: str) -> dict[str, Any]:
    server: dict[str, Any] = {
        "name": name,
        "transport": "stdio",
        "enabled": draw(st.booleans()),
        "command": draw(st.one_of(_bare_command, _abs_path())),
        "args": draw(st.lists(_text, max_size=4)),
        "env": draw(_secret_map),
    }
    cwd = draw(st.one_of(st.none(), _abs_path()))
    if cwd is not None:
        server["cwd"] = cwd
    return server


@st.composite
def _sse_server(draw: st.DrawFn, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "transport": "sse",
        "enabled": draw(st.booleans()),
        "url": draw(_url),
        "headers": draw(_secret_map),
    }


@st.composite
def _server_section(draw: st.DrawFn) -> dict[str, Any]:
    loopback = draw(st.booleans())
    host = draw(st.sampled_from(_LOOPBACK_HOSTS if loopback else _OTHER_HOSTS))
    # A non-loopback bind needs a bearer token of at least 32 characters (16.5).
    token = draw(
        st.builds(operator.add, st.text(min_size=32, max_size=40), _text)
        if not loopback
        else st.one_of(st.none(), _nonempty_text)
    )
    origin = st.builds(
        "http{}://{}{}".format,
        st.sampled_from(["", "s"]),
        st.text(string.ascii_lowercase, min_size=1, max_size=10),
        st.one_of(st.just(""), st.integers(1, 65535).map(lambda p: f":{p}")),
    )
    section: dict[str, Any] = {
        "host": host,
        "port": draw(st.integers(1, 65535)),
        "cors_allowed_origins": draw(st.lists(origin, max_size=3)),
    }
    if token is not None:
        section["bearer_token"] = token
    return section


@st.composite
def _llm_section(draw: st.DrawFn) -> dict[str, Any]:
    section: dict[str, Any] = {
        "model": draw(_nonempty_text),
        "temperature": draw(st.floats(0, 2)),
        "max_output_tokens": draw(st.integers(1, 1_000_000)),
        "request_timeout_s": draw(
            st.floats(0, 1e6, exclude_min=True, allow_nan=False, allow_infinity=False)
        ),
        "native_function_calling": draw(st.booleans()),
    }
    api_base = draw(st.one_of(st.none(), _url))
    if api_base is not None:
        section["api_base"] = api_base
    api_key = draw(st.one_of(st.none(), _text))
    if api_key is not None:
        section["api_key"] = api_key
    return section


@st.composite
def _orchestration_section(draw: st.DrawFn) -> dict[str, Any]:
    # Prefixes keep the two keyword sets non-empty and disjoint after casefold (the validator).
    approve = draw(st.lists(st.builds(operator.add, st.just("y"), _text), min_size=1, max_size=3))
    deny = draw(st.lists(st.builds(operator.add, st.just("n"), _text), min_size=1, max_size=3))
    return {
        "max_tool_iterations": draw(st.integers(1, 100)),
        "tool_timeout_s": draw(st.floats(0, 3600, exclude_min=True)),
        "max_tool_output_chars": draw(st.integers(1_000, 1_000_000)),
        "context_window_budget": draw(st.integers(1_024, 1_000_000)),
        "approval_timeout_s": draw(st.one_of(st.none(), st.integers(1, 86_400))),
        "approval_keywords": {"approve": approve, "deny": deny},
        "session_idle_timeout_s": draw(st.one_of(st.none(), st.integers(60, 604_800))),
        "per_session_discovery": draw(st.booleans()),
    }


@st.composite
def _storage_section(draw: st.DrawFn) -> dict[str, Any]:
    backend = draw(st.sampled_from(["sqlite", "memory"]))
    section: dict[str, Any] = {"backend": backend}
    # Path set or unset. For `memory` the loader leaves `path` unresolved (13.11), so any text
    # must survive; for `sqlite` the validated value is an absolute path.
    path = draw(st.one_of(st.none(), _nonempty_text if backend == "memory" else _abs_path()))
    if path is not None:
        section["path"] = path
    return section


@st.composite
def _configs(draw: st.DrawFn) -> BackendConfig:
    names = draw(st.lists(st.text(_NAME_CHARS, min_size=1, max_size=32), unique=True, max_size=4))
    servers = [draw(st.one_of(_stdio_server(name), _sse_server(name))) for name in names]
    permissions: dict[str, list[str]] = {}
    for name in names:
        if draw(st.booleans()):
            permissions[name] = draw(st.lists(st.text(min_size=1, max_size=20), max_size=4))
    raw: dict[str, Any] = {
        "server": draw(_server_section()),
        "llm": draw(_llm_section()),
        "agent": {
            "name": draw(_nonempty_text),
            "description": draw(_nonempty_text),
            "version": draw(_nonempty_text),
            "base_instructions": draw(st.one_of(st.none(), _text)),
        },
        "orchestration": draw(_orchestration_section()),
        "mcp_servers": servers,
        "permissions": permissions,
        "logging": {"level": draw(st.sampled_from(["debug", "info", "warning", "error"]))},
        "storage": draw(_storage_section()),
    }
    return BackendConfig.model_validate(raw)


# Feature: local-agent-backend, Property 26: Configuration round-trip
# **Validates: Requirements 13.5**
@given(cfg=_configs())
def test_dump_then_load_returns_an_equal_config(cfg: BackendConfig) -> None:
    text = dump_config(cfg)
    loaded = load_config_from_text(text, {}, "<roundtrip>")
    assert loaded == cfg
