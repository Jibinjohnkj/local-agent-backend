"""Unit tests for configuration loading (Requirements 6.2, 6.3, 13.1, 13.4, 13.6, 13.7, 13.9,
13.10, 13.11, 16.1, 16.6, 17.17, 17.18, 17.19).

Every load passes an explicit `platform` and `env`, so the tests do not depend on the host OS or on
the real environment. Only the tests that read a file use `tmp_path`.
"""

from __future__ import annotations

import ntpath
import posixpath
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from local_agent.config.loader import (
    ConfigError,
    load_config,
    load_config_from_text,
    resolve_config_path,
)
from local_agent.config.models import (
    BackendConfig,
    SseServerConfig,
    StdioServerConfig,
    storage_directory,
)
from local_agent.oscompat.paths import default_storage_dir

POSIX = "linux"
WINDOWS = "win32"
BASE = "/srv/agent"
MINIMAL = "llm:\n  model: ollama_chat/llama3.1\n"


def load(
    text: str,
    env: dict[str, str] | None = None,
    *,
    base_dir: str = BASE,
    platform: str = POSIX,
) -> BackendConfig:
    return load_config_from_text(
        text, env or {}, "config.yaml", base_dir=base_dir, platform=platform
    )


def issues_of(
    text: str,
    env: dict[str, str] | None = None,
    *,
    base_dir: str = BASE,
    platform: str = POSIX,
) -> dict[str, str]:
    """Load `text`, expect failure, and return the issues as `{path: message}`."""
    with pytest.raises(ConfigError) as info:
        load(text, env, base_dir=base_dir, platform=platform)
    return dict(info.value.issues)


# --- Locating the config file (13.1, 13.6) ---------------------------------------------------


class TestResolveConfigPath:
    def test_default_is_config_yaml_in_cwd(self) -> None:
        cwd = Path("/work/project")
        assert resolve_config_path({}, cwd, platform=POSIX) == cwd / "config.yaml"

    def test_empty_agent_config_uses_the_default(self) -> None:
        cwd = Path("/work/project")
        assert resolve_config_path({"AGENT_CONFIG": ""}, cwd, platform=POSIX) == (
            cwd / "config.yaml"
        )

    def test_absolute_agent_config_is_used_as_is(self) -> None:
        result = resolve_config_path(
            {"AGENT_CONFIG": "/etc/agent/my.yaml"}, Path("/work/project"), platform=POSIX
        )
        assert result == Path("/etc/agent/my.yaml")

    def test_relative_agent_config_is_resolved_against_cwd(self) -> None:
        result = resolve_config_path(
            {"AGENT_CONFIG": "conf/../conf/my.yaml"}, Path("/work/project"), platform=POSIX
        )
        assert result == Path("/work/project/conf/my.yaml")

    def test_relative_agent_config_uses_the_process_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = resolve_config_path({"AGENT_CONFIG": "my.yaml"}, Path.cwd(), platform=POSIX)
        assert result == Path.cwd() / "my.yaml"

    def test_agent_config_expands_home(self) -> None:
        env = {"AGENT_CONFIG": "~/agent.yaml", "HOME": "/home/ann"}
        assert resolve_config_path(env, Path("/work"), platform=POSIX) == Path(
            "/home/ann/agent.yaml"
        )

    def test_agent_config_tilde_without_home_is_reported_at_agent_config(self) -> None:
        with pytest.raises(ConfigError) as info:
            resolve_config_path({"AGENT_CONFIG": "~/agent.yaml"}, Path("/work"), platform=POSIX)
        [issue] = info.value.issues
        assert issue.path == "AGENT_CONFIG"
        assert "HOME" in issue.message


class TestLoadConfigFile:
    def test_missing_file_names_the_path_and_the_reason(self, tmp_path: Path) -> None:
        missing = tmp_path / "config.yaml"
        with pytest.raises(ConfigError) as info:
            load_config(missing, {}, platform=POSIX)
        [issue] = info.value.issues
        assert issue.path == str(missing)
        assert "cannot read config file" in issue.message

    def test_missing_agent_config_target_does_not_fall_back_to_cwd_file(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.yaml").write_text(MINIMAL, encoding="utf-8")
        chosen = resolve_config_path({"AGENT_CONFIG": "other.yaml"}, tmp_path, platform=POSIX)
        assert chosen == tmp_path / "other.yaml"
        with pytest.raises(ConfigError) as info:
            load_config(chosen, {}, platform=POSIX)
        assert info.value.issues[0].path == str(tmp_path / "other.yaml")

    def test_loads_a_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL, encoding="utf-8")
        assert load_config(path, {}, platform=POSIX).llm.model == "ollama_chat/llama3.1"

    def test_relative_storage_path_resolves_against_the_config_directory(
        self, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "conf"
        config_dir.mkdir()
        path = config_dir / "config.yaml"
        path.write_text(MINIMAL + "storage:\n  path: data/../state\n", encoding="utf-8")
        cfg = load_config(path, {}, platform=POSIX)
        assert cfg.storage.path == str(config_dir / "state")


# --- Defaults (6.2, 6.3, 13.4, 13.9, 16.1, 16.6) ---------------------------------------------


class TestDefaults:
    def test_server_defaults(self) -> None:
        cfg = load(MINIMAL)
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.port == 10000
        assert cfg.server.cors_allowed_origins == []
        assert cfg.server.bearer_token is None

    def test_orchestration_defaults(self) -> None:
        orchestration = load(MINIMAL).orchestration
        assert orchestration.max_tool_iterations == 10
        assert orchestration.tool_timeout_s == 60
        assert orchestration.session_idle_timeout_s is None

    def test_storage_defaults(self) -> None:
        storage = load(MINIMAL).storage
        assert storage.backend == "sqlite"
        assert storage.path is None

    def test_no_mcp_servers_or_permissions_by_default(self) -> None:
        cfg = load(MINIMAL)
        assert cfg.mcp_servers == []
        assert cfg.permissions == {}

    def test_llm_model_is_required(self) -> None:
        assert "llm" in issues_of("server:\n  port: 8080\n")

    def test_unknown_top_level_key_is_rejected(self) -> None:
        assert "surprise" in issues_of(MINIMAL + "surprise: 1\n")

    def test_config_is_frozen(self) -> None:
        cfg = load(MINIMAL)
        with pytest.raises(ValidationError):
            cfg.storage = cfg.storage.model_copy()  # type: ignore[misc]

    def test_session_idle_timeout_accepts_the_range_bounds(self) -> None:
        for value in (60, 604_800):
            cfg = load(MINIMAL + f"orchestration:\n  session_idle_timeout_s: {value}\n")
            assert cfg.orchestration.session_idle_timeout_s == value

    @pytest.mark.parametrize("value", [59, 604_801])
    def test_session_idle_timeout_outside_the_range_is_rejected(self, value: int) -> None:
        text = MINIMAL + f"orchestration:\n  session_idle_timeout_s: {value}\n"
        assert "orchestration.session_idle_timeout_s" in issues_of(text)


# --- CORS origins (16.4, 16.6) ---------------------------------------------------------------


class TestCorsOrigins:
    def test_origins_are_kept_exactly_as_written(self) -> None:
        text = MINIMAL + (
            "server:\n  cors_allowed_origins:\n    - http://localhost:3000\n    - https://app.test\n"
        )
        assert load(text).server.cors_allowed_origins == [
            "http://localhost:3000",
            "https://app.test",
        ]

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost:3000/", "ftp://host", "localhost:3000", "http://h/path", "*"],
    )
    def test_malformed_origin_is_reported_at_its_index(self, origin: str) -> None:
        text = MINIMAL + f"server:\n  cors_allowed_origins:\n    - '{origin}'\n"
        assert "server.cors_allowed_origins.0" in issues_of(text)


# --- YAML errors (13.7) ----------------------------------------------------------------------


class TestYamlErrors:
    def test_reports_one_based_line_and_column(self) -> None:
        # A tab cannot start a token; it is the first character of line 3.
        text = "llm:\n  model: x\n\tbad: 1\n"
        issues = issues_of(text)
        assert list(issues) == ["config.yaml"]
        assert "line 3, column 1" in issues["config.yaml"]

    def test_unclosed_flow_sequence_reports_a_position(self) -> None:
        issues = issues_of("llm:\n  model: [a, b\nserver: {}\n")
        assert "malformed YAML at line" in issues["config.yaml"]

    @pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n", ""])
    def test_top_level_must_be_a_mapping(self, text: str) -> None:
        issues = issues_of(text)
        assert issues == {"config.yaml": "top-level value must be a mapping"}

    def test_crlf_line_endings_give_the_same_line_number(self) -> None:
        issues = issues_of("llm:\r\n  model: x\r\n\tbad: 1\r\n")
        assert "line 3, column 1" in issues["config.yaml"]


# --- stdio and SSE servers (6.2, 6.3, 17.19, 17.20) ------------------------------------------


STDIO = """
mcp_servers:
  - name: files
    transport: stdio
    command: npx
    args: ["-y", "server", "./relative/arg", "~/also-not-a-path"]
    env:
      TOKEN: abc
"""

SSE = """
mcp_servers:
  - name: remote
    transport: sse
    url: https://tools.example.com/sse
    headers:
      Authorization: Bearer secret-value
"""


class TestStdioServer:
    def test_fields_and_defaults(self) -> None:
        [server] = load(MINIMAL + STDIO).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.name == "files"
        assert server.enabled is True
        assert server.command == "npx"
        assert server.env == {"TOKEN": SecretStr("abc")}

    def test_args_are_never_path_processed(self) -> None:
        [server] = load(MINIMAL + STDIO).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.args == ["-y", "server", "./relative/arg", "~/also-not-a-path"]

    def test_omitted_cwd_stays_none(self) -> None:
        [server] = load(MINIMAL + STDIO).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.cwd is None

    def test_bare_command_is_left_untouched(self) -> None:
        [server] = load(MINIMAL + STDIO).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.command == "npx"

    def test_command_with_a_separator_is_resolved_against_the_config_directory(self) -> None:
        text = MINIMAL + (
            "mcp_servers:\n  - {name: a, transport: stdio, command: ./bin/../bin/run}\n"
        )
        [server] = load(text).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.command == "/srv/agent/bin/run"

    def test_relative_cwd_is_resolved_against_the_config_directory(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: stdio, command: x, cwd: work}\n"
        [server] = load(text).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.cwd == "/srv/agent/work"

    def test_tilde_cwd_expands_to_home(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: stdio, command: x, cwd: '~/w'}\n"
        [server] = load(text, {"HOME": "/home/ann"}).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.cwd == "/home/ann/w"

    def test_tilde_without_home_is_reported_at_the_field_path(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: stdio, command: x, cwd: '~/w'}\n"
        issues = issues_of(text, {})
        assert set(issues) == {"mcp_servers.0.cwd"}
        assert "HOME" in issues["mcp_servers.0.cwd"]

    def test_tilde_with_empty_home_is_reported(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: stdio, command: x, cwd: '~'}\n"
        assert "mcp_servers.0.cwd" in issues_of(text, {"HOME": ""})

    def test_tilde_user_form_is_an_ordinary_relative_name(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: stdio, command: x, cwd: '~bob'}\n"
        [server] = load(text, {}).mcp_servers
        assert isinstance(server, StdioServerConfig)
        assert server.cwd == "/srv/agent/~bob"

    def test_empty_command_is_rejected(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: stdio, command: ''}\n"
        assert "mcp_servers.0.command" in issues_of(text)

    def test_invalid_server_name_is_rejected(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: 'bad name', transport: stdio, command: x}\n"
        assert "mcp_servers.0.name" in issues_of(text)

    def test_unknown_transport_is_reported_at_the_transport_field(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: carrier-pigeon, command: x}\n"
        issues = issues_of(text)
        assert "mcp_servers.0.transport" in issues
        assert "carrier-pigeon" not in issues["mcp_servers.0.transport"]

    def test_duplicate_names_are_reported_for_every_entry(self) -> None:
        text = MINIMAL + (
            "mcp_servers:\n"
            "  - {name: a, transport: stdio, command: x}\n"
            "  - {name: b, transport: stdio, command: x}\n"
            "  - {name: a, transport: stdio, command: y}\n"
        )
        issues = issues_of(text)
        assert set(issues) == {"mcp_servers.0.name", "mcp_servers.2.name"}


class TestSseServer:
    def test_fields_and_defaults(self) -> None:
        [server] = load(MINIMAL + SSE).mcp_servers
        assert isinstance(server, SseServerConfig)
        assert server.name == "remote"
        assert server.enabled is True
        assert str(server.url) == "https://tools.example.com/sse"
        assert server.headers == {"Authorization": SecretStr("Bearer secret-value")}

    def test_http_url_is_accepted(self) -> None:
        text = (
            MINIMAL + "mcp_servers:\n  - {name: a, transport: sse, url: 'http://localhost:9/s'}\n"
        )
        [server] = load(text).mcp_servers
        assert isinstance(server, SseServerConfig)
        assert server.url.host == "localhost"

    @pytest.mark.parametrize("url", ["ftp://tools.example.com/sse", "file:///tmp/x", "not a url"])
    def test_non_http_url_is_rejected_at_the_url_field(self, url: str) -> None:
        text = MINIMAL + f"mcp_servers:\n  - {{name: a, transport: sse, url: '{url}'}}\n"
        assert "mcp_servers.0.url" in issues_of(text)

    def test_url_is_required(self) -> None:
        text = MINIMAL + "mcp_servers:\n  - {name: a, transport: sse}\n"
        assert "mcp_servers.0.url" in issues_of(text)


class TestSecrets:
    def test_secret_values_are_masked_in_repr(self) -> None:
        text = (
            "llm:\n  model: m\n  api_key: sk-top-secret\n"
            "server:\n  bearer_token: tok-top-secret\n" + STDIO + SSE.replace("mcp_servers:\n", "")
        )
        shown = repr(load(text))
        for secret in ("sk-top-secret", "tok-top-secret", "abc", "Bearer secret-value"):
            assert secret not in shown

    def test_error_text_never_includes_the_offending_secret(self) -> None:
        text = "llm:\n  model: m\n  api_key: [sk-top-secret]\n"
        with pytest.raises(ConfigError) as info:
            load(text)
        assert "sk-top-secret" not in str(info.value)


# --- Path_Fields on Windows (17.17, 17.18) ---------------------------------------------------


class TestWindowsPathFields:
    def test_drive_relative_command_is_reported_at_the_field_path(self) -> None:
        text = MINIMAL + (
            "mcp_servers:\n  - name: a\n    transport: stdio\n    command: 'C:tools\\x.exe'\n"
        )
        issues = issues_of(text, base_dir="D:\\cfg", platform=WINDOWS)
        assert set(issues) == {"mcp_servers.0.command"}
        assert "drive-relative paths are not supported" in issues["mcp_servers.0.command"]

    def test_drive_relative_cwd_is_reported_at_the_field_path(self) -> None:
        text = MINIMAL + (
            "mcp_servers:\n  - name: a\n    transport: stdio\n    command: x\n    cwd: 'C:work'\n"
        )
        issues = issues_of(text, base_dir="D:\\cfg", platform=WINDOWS)
        assert set(issues) == {"mcp_servers.0.cwd"}

    def test_drive_relative_storage_path_is_reported_at_storage_path(self) -> None:
        issues = issues_of(
            MINIMAL + "storage:\n  path: 'C:data'\n", base_dir="D:\\cfg", platform=WINDOWS
        )
        assert set(issues) == {"storage.path"}

    def test_root_relative_path_takes_the_drive_of_the_config_directory(self) -> None:
        cfg = load(MINIMAL + "storage:\n  path: '\\data'\n", base_dir="D:\\cfg", platform=WINDOWS)
        assert cfg.storage.path == "D:\\data"

    def test_absolute_and_unc_paths_are_kept(self) -> None:
        for value in ("C:\\state", "\\\\host\\share\\state"):
            cfg = load(
                MINIMAL + f"storage:\n  path: '{value}'\n", base_dir="D:\\cfg", platform=WINDOWS
            )
            assert cfg.storage.path == ntpath.normpath(value)

    def test_tilde_without_userprofile_names_userprofile(self) -> None:
        issues = issues_of(
            MINIMAL + "storage:\n  path: '~\\state'\n",
            {"HOME": "/ignored"},
            base_dir="D:\\cfg",
            platform=WINDOWS,
        )
        assert "USERPROFILE" in issues["storage.path"]


# --- Storage (13.9, 13.10, 13.11) ------------------------------------------------------------


class TestStorage:
    def test_relative_path_resolves_against_the_config_directory(self) -> None:
        cfg = load(MINIMAL + "storage:\n  path: data/agent\n")
        assert cfg.storage.path == "/srv/agent/data/agent"

    def test_absolute_path_is_normalized_but_kept(self) -> None:
        cfg = load(MINIMAL + "storage:\n  path: /var/lib/./agent//db\n")
        assert cfg.storage.path == "/var/lib/agent/db"

    def test_tilde_path_expands_to_home(self) -> None:
        cfg = load(MINIMAL + "storage:\n  path: '~/agent'\n", {"HOME": "/home/ann"})
        assert cfg.storage.path == "/home/ann/agent"

    def test_tilde_path_without_home_is_reported_at_storage_path(self) -> None:
        issues = issues_of(MINIMAL + "storage:\n  path: '~/agent'\n", {})
        assert set(issues) == {"storage.path"}
        assert "HOME" in issues["storage.path"]

    def test_unset_path_stays_none(self) -> None:
        assert load(MINIMAL + "storage:\n  backend: sqlite\n").storage.path is None

    def test_memory_backend_leaves_the_path_unresolved_and_reports_no_issue(self) -> None:
        text = MINIMAL + "storage:\n  backend: memory\n  path: '~/not/expanded'\n"
        cfg = load(text, {})
        assert cfg.storage.backend == "memory"
        assert cfg.storage.path == "~/not/expanded"

    def test_memory_backend_accepts_a_drive_relative_path_on_windows(self) -> None:
        text = MINIMAL + "storage:\n  backend: memory\n  path: 'C:data'\n"
        assert load(text, base_dir="D:\\cfg", platform=WINDOWS).storage.path == "C:data"

    def test_unknown_backend_is_reported_at_storage_backend(self) -> None:
        issues = issues_of(MINIMAL + "storage:\n  backend: postgres\n")
        assert set(issues) == {"storage.backend"}

    def test_unknown_backend_and_other_defects_are_reported_together(self) -> None:
        text = MINIMAL + "storage:\n  backend: postgres\nserver:\n  port: 0\n"
        assert set(issues_of(text)) == {"storage.backend", "server.port"}

    def test_unknown_storage_key_is_not_silently_accepted_as_a_path(self) -> None:
        cfg = load(MINIMAL + "storage:\n  backend: sqlite\n  path: db\n")
        assert cfg.storage.path == "/srv/agent/db"


class TestStorageDirectory:
    def test_memory_backend_has_no_directory(self) -> None:
        cfg = load(MINIMAL + "storage:\n  backend: memory\n")
        assert storage_directory(cfg, {"HOME": "/home/ann"}, POSIX) is None

    def test_memory_backend_has_no_directory_even_with_a_path(self) -> None:
        cfg = load(MINIMAL + "storage:\n  backend: memory\n  path: /x\n")
        assert storage_directory(cfg, {"HOME": "/home/ann"}, POSIX) is None

    def test_configured_path_is_returned_as_resolved(self) -> None:
        cfg = load(MINIMAL + "storage:\n  path: data\n")
        assert storage_directory(cfg, {}, POSIX) == "/srv/agent/data"

    def test_unset_path_uses_the_platform_default(self) -> None:
        cfg = load(MINIMAL)
        env = {"HOME": "/home/ann"}
        assert storage_directory(cfg, env, POSIX) == default_storage_dir(POSIX, env)
        assert storage_directory(cfg, env, POSIX) is not None
        assert posixpath.isabs(storage_directory(cfg, env, POSIX) or "")


# --- Substitution and the error report (13.2, 13.3, 13.8) ------------------------------------


class TestErrorReport:
    def test_all_issues_are_reported_in_one_error(self) -> None:
        text = (
            "llm:\n  model: ''\n"
            "server:\n  port: 70000\n"
            "storage:\n  backend: postgres\n"
            "orchestration:\n  max_tool_iterations: 0\n"
        )
        assert set(issues_of(text)) == {
            "llm.model",
            "server.port",
            "storage.backend",
            "orchestration.max_tool_iterations",
        }

    def test_out_of_range_literal_is_echoed_for_non_secret_fields(self) -> None:
        assert "70000" in issues_of(MINIMAL + "server:\n  port: 70000\n")["server.port"]

    def test_substituted_values_are_not_echoed(self) -> None:
        issues = issues_of(MINIMAL + "server:\n  port: '${PORT}'\n", {"PORT": "99999999"})
        assert "99999999" not in issues["server.port"]

    def test_missing_variable_is_reported_with_name_and_path(self) -> None:
        issues = issues_of("llm:\n  model: '${MODEL_NAME}'\n", {})
        assert issues == {"llm.model": "environment variable MODEL_NAME is not set"}

    def test_substitution_fills_values_and_dollar_escape_is_literal(self) -> None:
        text = "llm:\n  model: '${MODEL}'\nagent:\n  name: 'cost $${5}'\n"
        cfg = load(text, {"MODEL": "gpt-x"})
        assert cfg.llm.model == "gpt-x"
        assert cfg.agent.name == "cost ${5}"

    def test_non_loopback_host_requires_a_long_bearer_token(self) -> None:
        text = MINIMAL + "server:\n  host: 0.0.0.0\n  bearer_token: short\n"
        assert "server.bearer_token" in issues_of(text)

    def test_non_loopback_host_with_long_token_is_accepted(self) -> None:
        token = "t" * 32
        cfg = load(MINIMAL + f"server:\n  host: 0.0.0.0\n  bearer_token: {token}\n")
        assert cfg.server.bearer_token == SecretStr(token)

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.9.9.9", "localhost", "::1", "[::1]"])
    def test_loopback_hosts_need_no_token(self, host: str) -> None:
        assert load(MINIMAL + f"server:\n  host: '{host}'\n").server.bearer_token is None

    def test_permission_for_an_undefined_server_is_reported(self) -> None:
        issues = issues_of(MINIMAL + "permissions:\n  ghost: ['*']\n")
        assert "permissions.ghost" in issues
