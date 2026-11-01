"""Property test for the complete validation report.

Validates Requirements 1.7, 5.3, 6.1, 6.4, 13.3, 13.9, 16.5.

The test builds a valid Backend_Config, injects a random set of independent defects, renders the
result to YAML, and loads it through `load_config_from_text`. Each defect touches its own field or
its own MCP_Server entry, so the expected dotted path of one defect never depends on another.
Loading must raise one `ConfigError` whose issues include every expected path. The same test also
covers the empty set: a config with no defects must load.
"""

import string
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from local_agent.config.loader import ConfigError, load_config_from_text
from local_agent.config.models import BackendConfig

_DEFECTS = (
    "port",
    "invalid_name",
    "duplicate_name",
    "missing_command",
    "unsupported_transport",
    "undefined_permission",
    "bad_pattern",
    "no_bearer_token",
    "bad_backend",
)

_NAME_CHARS = string.ascii_letters + string.digits + "_-"
_WORD = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8)

# A name the schema rejects: empty, longer than 32 characters, or holding a character outside
# `[A-Za-z0-9_-]`. The alphabet has no `$`, so substitution never touches it.
_INVALID_NAME = st.one_of(
    st.just(""),
    st.text(alphabet=_NAME_CHARS, min_size=33, max_size=60),
    st.tuples(
        st.text(alphabet=_NAME_CHARS, max_size=6),
        st.sampled_from(" .!/@:"),
        st.text(alphabet=_NAME_CHARS, max_size=6),
    ).map("".join),
)
_BAD_PORT = st.one_of(
    st.integers(min_value=-(10**9), max_value=0),
    st.integers(min_value=65_536, max_value=10**9),
)
_LOOPBACK_HOST = st.sampled_from(["127.0.0.1", "localhost", "::1", "127.0.0.2"])
_REMOTE_HOST = st.sampled_from(["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "example.com"])
_LONG_TOKEN = st.text(alphabet=_NAME_CHARS, min_size=32, max_size=64)
_SHORT_TOKEN = st.one_of(st.none(), st.text(alphabet=_NAME_CHARS, max_size=31))
_BAD_TRANSPORT = st.sampled_from(["ftp", "http", "grpc", "websocket", "STDIO", ""])
_BAD_BACKEND = st.text(alphabet=_NAME_CHARS, min_size=1, max_size=12).filter(
    lambda value: value not in ("sqlite", "memory")
)
_WORD_PORT = st.integers(min_value=1, max_value=65_535).map(str)
_PATTERN = st.text(alphabet=string.ascii_lowercase + "*?_", min_size=1, max_size=128)


@st.composite
def _scenario(draw: st.DrawFn) -> tuple[dict[str, Any], set[str], dict[str, int]]:
    """A config tree, the dotted paths its defects must produce, and the injected port values."""
    defects = draw(st.sets(st.sampled_from(_DEFECTS)))
    expected: set[str] = set()

    # Servers: five fill the roles below at random positions, up to three more are plain.
    count = draw(st.integers(min_value=5, max_value=8))
    anchor, renamed, duplicate, no_command, bad_transport = draw(st.permutations(range(count)))[:5]
    servers: list[dict[str, Any]] = []
    for index in range(count):
        is_stdio = index == no_command or draw(st.booleans())
        server: dict[str, Any] = {"name": f"srv{index}", "enabled": draw(st.booleans())}
        if is_stdio:
            server |= {"transport": "stdio", "command": draw(_WORD), "args": [draw(_WORD)]}
        else:
            server |= {"transport": "sse", "url": f"http://localhost:{draw(_WORD_PORT)}/sse"}
        servers.append(server)

    if "invalid_name" in defects:
        servers[renamed]["name"] = draw(_INVALID_NAME)
        expected.add(f"mcp_servers.{renamed}.name")
    if "duplicate_name" in defects:
        servers[duplicate]["name"] = servers[anchor]["name"]
        expected.add(f"mcp_servers.{duplicate}.name")
    if "missing_command" in defects:
        if draw(st.booleans()):
            del servers[no_command]["command"]
        else:
            servers[no_command]["command"] = ""
        expected.add(f"mcp_servers.{no_command}.command")
    if "unsupported_transport" in defects:
        servers[bad_transport]["transport"] = draw(_BAD_TRANSPORT)
        expected.add(f"mcp_servers.{bad_transport}.transport")

    # Permissions only name the anchor server, which no other defect renames.
    anchor_name = servers[anchor]["name"]
    patterns = draw(st.lists(_PATTERN, max_size=5))
    permissions: dict[str, list[str]] = {}
    if "bad_pattern" in defects:
        bad = draw(st.one_of(st.just(""), st.text(alphabet="ab*?", min_size=129, max_size=200)))
        position = draw(st.integers(min_value=0, max_value=len(patterns)))
        patterns.insert(position, bad)
        expected.add(f"permissions.{anchor_name}.{position}")
    if patterns:
        permissions[anchor_name] = patterns
    if "undefined_permission" in defects:
        ghost = f"ghost-{draw(_WORD)}"
        permissions[ghost] = draw(st.lists(_PATTERN, max_size=3))
        expected.add(f"permissions.{ghost}")

    # Server section: a valid bind is loopback, or remote with a token of 32 or more characters.
    server_section: dict[str, Any] = {"port": draw(st.integers(min_value=1, max_value=65_535))}
    if draw(st.booleans()):
        server_section["host"] = draw(_LOOPBACK_HOST)
        if draw(st.booleans()):
            server_section["bearer_token"] = draw(_LONG_TOKEN)
    else:
        server_section["host"] = draw(_REMOTE_HOST)
        server_section["bearer_token"] = draw(_LONG_TOKEN)
    ports: dict[str, int] = {}
    if "port" in defects:
        server_section["port"] = draw(_BAD_PORT)
        ports["server.port"] = server_section["port"]
        expected.add("server.port")
    if "no_bearer_token" in defects:
        server_section["host"] = draw(_REMOTE_HOST)
        token = draw(_SHORT_TOKEN)
        if token is None:
            server_section.pop("bearer_token", None)
        else:
            server_section["bearer_token"] = token
        expected.add("server.bearer_token")

    storage = {"backend": draw(st.sampled_from(["sqlite", "memory"]))}
    if "bad_backend" in defects:
        storage["backend"] = draw(_BAD_BACKEND)
        expected.add("storage.backend")

    tree: dict[str, Any] = {
        "server": server_section,
        "llm": {"model": "ollama_chat/llama3.1"},
        "mcp_servers": servers,
        "permissions": permissions,
        "storage": storage,
    }
    return tree, expected, ports


# Feature: local-agent-backend, Property 27: Complete validation report
# **Validates: Requirements 1.7, 5.3, 6.1, 6.4, 13.3, 13.9, 16.5**
@given(scenario=_scenario())
def test_complete_validation_report(
    scenario: tuple[dict[str, Any], set[str], dict[str, int]],
) -> None:
    tree, expected, ports = scenario
    text = yaml.safe_dump(tree, sort_keys=False, allow_unicode=True)

    if not expected:
        assert isinstance(load_config_from_text(text, {}, "config.yaml"), BackendConfig)
        return

    with pytest.raises(ConfigError) as caught:
        load_config_from_text(text, {}, "config.yaml")

    issues = caught.value.issues
    reported = {issue.path for issue in issues}
    assert expected <= reported, f"missing paths: {sorted(expected - reported)}"

    # The out-of-range port is named in its issue (1.7).
    for path, value in ports.items():
        assert any(str(value) in issue.message for issue in issues if issue.path == path)
