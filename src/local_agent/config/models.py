"""Backend_Config: the pydantic models that describe `config.yaml`.

The loader (`config/loader.py`) parses YAML, substitutes `${VAR}`, resolves Path_Fields, and then
hands the raw tree to `BackendConfig.model_validate`. This module only describes the shape,
defaults, and ranges. It never touches the file system.
"""

from collections.abc import Mapping
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    StringConstraints,
    model_validator,
)

from local_agent.oscompat.paths import default_storage_dir

__all__ = [
    "AgentConfig",
    "ApprovalKeywords",
    "BackendConfig",
    "HttpOrigin",
    "LLMConfig",
    "LoggingConfig",
    "MCPServerConfig",
    "OrchestrationConfig",
    "PermissionPattern",
    "ServerConfig",
    "ServerName",
    "SseServerConfig",
    "StdioServerConfig",
    "StorageConfig",
    "storage_directory",
]


def _check_http_origin(value: str) -> str:
    """Accept exactly `http(s)://host[:port]` and return the string unchanged.

    CORSMiddleware compares allowed origins with the `Origin` header as plain strings. A parsed
    URL type would add a trailing slash and break that comparison, so we validate the shape and
    keep the original text.
    """
    try:
        parts = urlsplit(value)
        _ = parts.port  # raises ValueError for a non-numeric or out-of-range port
    except ValueError as exc:
        msg = "must be an origin like http://host[:port]"
        raise ValueError(msg) from exc
    well_formed = (
        parts.scheme in ("http", "https")
        and bool(parts.hostname)
        and parts.path == ""
        and not parts.query
        and not parts.fragment
        and parts.username is None
        and parts.password is None
        and not value.endswith(("?", "#"))
    )
    if not well_formed:
        msg = "must be an origin like http://host[:port] with no path, query, or credentials"
        raise ValueError(msg)
    return value


HttpOrigin = Annotated[str, AfterValidator(_check_http_origin)]
"""A CORS origin: scheme `http` or `https`, a host, and an optional port (16.4, 16.6)."""


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"  # 13.4, 16.1; "localhost" -> 127.0.0.1 + optional ::1 (17.30)
    port: int = Field(10000, ge=1, le=65535)  # 1.7
    bearer_token: SecretStr | None = None  # >=32 characters for non-loopback binds, 16.5
    cors_allowed_origins: list[HttpOrigin] = []  # exact scheme://host[:port], 16.4/16.6


class LLMConfig(BaseModel):
    model: str = Field(min_length=1)  # for example "ollama_chat/llama3.1"
    api_base: AnyHttpUrl | None = None
    api_key: SecretStr | None = None
    temperature: float = Field(0.7, ge=0, le=2)
    max_output_tokens: int = Field(2048, ge=1)
    request_timeout_s: float = Field(120, gt=0)
    native_function_calling: bool = True


class AgentConfig(BaseModel):
    name: str = Field("Local Agent", min_length=1)
    description: str = Field("Local AI assistant with MCP tools", min_length=1)
    version: str = Field("0.1.0", min_length=1)
    base_instructions: str | None = None


class ApprovalKeywords(BaseModel):
    """Words the user can reply with to approve or deny a Protected_Tool Tool_Call (5.1).

    Satisfies the `ApprovalKeywordSet` protocol in `orchestrator/approval_gate.py`, which only
    reads `approve` and `deny`.
    """

    approve: list[str] = ["approve", "yes"]
    deny: list[str] = ["deny", "no"]

    @model_validator(mode="after")
    def _non_empty_and_disjoint(self) -> ApprovalKeywords:
        if not self.approve:
            msg = "approve must contain at least one keyword"
            raise ValueError(msg)
        if not self.deny:
            msg = "deny must contain at least one keyword"
            raise ValueError(msg)
        # The parser compares trimmed, casefolded text, so overlap is judged the same way.
        approve = {word.strip().casefold() for word in self.approve}
        deny = {word.strip().casefold() for word in self.deny}
        if "" in approve or "" in deny:
            msg = "keywords must not be empty or whitespace"
            raise ValueError(msg)
        if approve & deny:
            msg = "approve and deny must not share a keyword (compared case-insensitively)"
            raise ValueError(msg)
        return self


class OrchestrationConfig(BaseModel):
    max_tool_iterations: int = Field(10, ge=1, le=100)
    tool_timeout_s: float = Field(60, gt=0, le=3600)
    max_tool_output_chars: int = Field(32_000, ge=1_000, le=1_000_000)
    context_window_budget: int = Field(8_192, ge=1_024, le=1_000_000)
    approval_timeout_s: int | None = Field(None, ge=1, le=86_400)
    approval_keywords: ApprovalKeywords = ApprovalKeywords()
    session_idle_timeout_s: int | None = Field(None, ge=60, le=604_800)
    per_session_discovery: bool = False


ServerName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,32}$")]


class StdioServerConfig(BaseModel):
    name: ServerName
    transport: Literal["stdio"]
    enabled: bool = True
    command: str = Field(min_length=1)  # Path_Field iff it contains a separator (17.5)
    args: list[str] = []  # never path-processed (17.20)
    cwd: str | None = None  # Path_Field; None -> config file directory (17.19)
    env: dict[str, SecretStr] = {}  # merged into the child environment per 17.22


class SseServerConfig(BaseModel):
    name: ServerName
    transport: Literal["sse"]
    enabled: bool = True
    url: HttpUrl  # http/https only
    headers: dict[str, SecretStr] = {}


MCPServerConfig = Annotated[StdioServerConfig | SseServerConfig, Field(discriminator="transport")]

PermissionPattern = Annotated[str, StringConstraints(min_length=1, max_length=128)]


class LoggingConfig(BaseModel):
    level: Literal["debug", "info", "warning", "error"] = "info"  # 15.7


class StorageConfig(BaseModel):
    backend: Literal["sqlite", "memory"] = "sqlite"  # 13.9
    # Path_Field (13.10). None -> default_storage_dir (17.35). Accepted and ignored for "memory"
    # (13.11).
    path: str | None = None


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server: ServerConfig = ServerConfig()
    llm: LLMConfig
    agent: AgentConfig = AgentConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    mcp_servers: list[MCPServerConfig] = Field([], max_length=50)
    permissions: dict[str, list[PermissionPattern]] = Field({})  # each list <=100 entries (5.3)
    logging: LoggingConfig = LoggingConfig()
    storage: StorageConfig = StorageConfig()


def storage_directory(cfg: BackendConfig, env: Mapping[str, str], platform: str) -> str | None:
    """Return the Storage_Directory: `None` for `memory`, else `storage.path` or the OS default.

    `cfg.storage.path` was already resolved by the loader, so we return it as is. When it is
    unset, `default_storage_dir` derives the per-OS location from `env` (17.35).
    """
    if cfg.storage.backend == "memory":
        return None
    if cfg.storage.path is not None:
        return cfg.storage.path
    return default_storage_dir(platform, env)
