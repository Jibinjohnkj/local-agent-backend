# Implementation Plan: Local Agent Backend

## Overview

The Backend is built in Python 3.14+ (CPython standard GIL build) and runs on Windows, macOS, and Linux (Requirement 17). It follows the module layout in Appendix A of the requirements (`src/local_agent/...`), the stack in Appendix B, and the design: `a2a-sdk` 1.2.x, `mcp` 2.3.x, `litellm`, `pydantic>=2.12`/`pydantic-settings`, `structlog`, `jsonschema`, `PyYAML>=6.0.3`, `uvicorn[standard]>=0.36`/`starlette`, `uvloop>=0.22` on macOS and Linux, the standard-library `sqlite3` for persistent storage (no third-party database package), with `pytest`, `pytest-asyncio`, and `hypothesis` for tests. The order is incremental:

1. Project scaffolding, shared data models, and the pure `oscompat` foundations that configuration and storage depend on (platform check, Path_Field resolution, environment-name rules, loop factory, default Storage_Directory, user-only permissions).
2. Pure cores and their property tests: config (byte decoding, BOM, CRLF, Path_Fields, `storage` section), redactor and UTF-8/LF log sink, naming and schema translation, pattern matching, approval parsing, stream assembler, result formatter, context window trimming, prompt builder, transition table.
3. Pure OS decision cores: executable resolution, Windows command-line and batch quoting, spawn specs, bind planning.
4. Storage (Requirement 18): the `SessionStore` ABC and `StorageTxn` transaction model, `InMemorySessionStore`, the single-writer-thread `Database`, Storage_Lock, Schema_Version migrations, `SQLiteSessionStore`, `ProjectedTaskStore`, Restart_Recovery, and retention.
5. I/O shells: process spawning and tree termination, own stdio transport, MCP manager, LLM gateway, commit-before-emit emitter, approval gate, tool dispatcher, task runner, shutdown controller and signals, A2A error contract and dispatcher, executor and handler, middleware, health, app wiring, `BackendServer`, and the entry point with the storage startup sequence.
6. Integration tests with a fake LLM and fake/real stdio MCP servers, platform-specific process, signal, bind, permission, lock, and crash/restart tests, cross-platform transcript equivalence, and the OS-matrix CI workflow.

Each property test is one Hypothesis test with at least 100 examples, tagged `# Feature: local-agent-backend, Property N: <title>`. OS-parameterized pure functions take `platform` as an argument, so their property tests run on every runner. Storage property tests open a fresh `TemporaryDirectory` per example inside the test body (never the shared pytest `tmp_path`). Commands use `uv` and work in any shell (for example `uv run pytest -q`).

## Tasks

- [x] 1. Set up project scaffolding and shared models
  - [x] 1.1 Create project structure, dependencies, and test configuration
    - Create `pyproject.toml` with `requires-python = ">=3.14"`; runtime deps from Appendix B with the floors `pydantic>=2.12`, `pyyaml>=6.0.3`, `uvicorn[standard]>=0.36`, `uvloop>=0.22; sys_platform != 'win32'`, `packaging` (for `accepts_version`), and `litellm` pinned to a release whose Windows wheel imports offline; dev deps `pytest`, `pytest-asyncio`, `hypothesis`, `respx`, `ruff`, `mypy`; pytest markers `integration`, `llm`, and `manual`; pytest `filterwarnings` turning `DeprecationWarning` from `asyncio` and `local_agent` into errors; ruff config; `[tool.mypy] python_version = "3.14"` with `strict = true`
    - Add `[tool.uv] required-environments` listing the five Supported_Platform markers (Windows x86_64, macOS arm64, macOS x86_64, Linux x86_64 glibc, Linux aarch64 glibc), then generate `uv.lock` with pinned versions so locking fails if any package lacks a wheel
    - Create `.gitattributes` with `* text=auto eol=lf` and `tests/fixtures/** -text` (this also keeps the binary SQLite fixture databases under `tests/fixtures/db/` byte-exact)
    - Create the package skeleton under `src/local_agent/` (`config/`, `a2a_server/`, `orchestrator/`, `llm/`, `mcp_client/`, `session/`, `storage/`, `oscompat/` with `__init__.py` files) and `tests/{unit,property,integration,fixtures,fixtures/db}/`
    - Create `tests/conftest.py` registering a Hypothesis profile `ci` (`max_examples=200`, `deadline=None`, `suppress_health_check=[HealthCheck.too_slow]`), an injectable fake `Clock` (`now_ms()` epoch milliseconds and monotonic time) and `sleep`, and the platform markers `windows_only`, `posix_only`, `macos_only`, `linux_only` from the design
    - Create `config.example.yaml` matching the design's example config (including `cwd: ~/mcp`, the `npx` → `npx.cmd` note, and the `storage` section with `backend: sqlite` and a commented `path`), with a note that wildcard binds need a reachable hostname for remote clients and that network file systems are unsupported for the Storage_Directory
    - _Requirements: 1.1, 13.1, 13.4, 13.9, 14.1, 17.1, 17.25, 17.28, 17.29, 17.33_

  - [x] 1.2 Define shared runtime data models
    - Create `src/local_agent/models.py` with `ToolCall`, `AssembledResponse`, `SystemMessage`/`UserMessage`/`AssistantMessage`/`ToolMessage` (with `to_openai()`), `ChatMessage`, `ContextRecord` (`created_at_ms`, `last_activity_ms` as integer epoch milliseconds), `ArtifactRecord` (`artifact_id`, `ordinal`, `text`, `closed`), `TaskRecord` (state, redacted `status_message`, `metadata`, `created_at_ms`, `updated_at_ms`, artifacts ordered by `ordinal`), `McpToolInfo`, `RegisteredTool`, `ToolResult` (with `as_message()`), `ServerStatus`, and content-item types, as specified in the design's Runtime models; use frozen dataclasses and no `from __future__ import annotations`
    - Add the `Clock` protocol and `SystemClock` (`now_ms() = time.time_ns() // 1_000_000`, plus monotonic time for timeouts) and a pure `replace_lone_surrogates(text)` helper (each unpaired surrogate → U+FFFD) used at Message_History ingress
    - _Requirements: 8.4, 11.1, 11.2, 11.5, 11.11, 18.3, 18.4, 18.5_

- [x] 2. Implement oscompat foundations
  - [x] 2.1 Implement platform detection
    - Write `oscompat/__init__.py` with `PlatformFacts`, `PlatformVerdict`, pure `classify_platform(facts)` (CPython 3.14 GIL build; Windows build >= 19045 on native x86_64, macOS >= 14 on arm64/x86_64, Linux glibc >= 2.28 on x86_64/arm64; `aarch64` normalized to `arm64`; musl and `Py_GIL_DISABLED == 1` unsupported), and `gather_facts()` (native machine via `IsWow64Process2` on Windows)
    - _Requirements: 17.1, 17.2_

  - [x] 2.2 Write property test for platform classification
    - File: `tests/property/test_platform.py`
    - **Property 33: Platform classification**
    - **Validates: Requirements 17.1, 17.2**

  - [x] 2.3 Implement Path_Field resolution
    - Write `oscompat/paths.py` with `PathFieldError`, pure `has_separator(value, platform)` and `resolve_path_field(value, *, base_dir, home, platform)` using `ntpath`/`posixpath` by `platform` (never the host module): `/` everywhere and `\` on Windows only, `~`/`~/`/`~\` expansion (no `~user`), missing home raises, relative joined to `base_dir`, lexical normalization, UNC and drive-absolute accepted, root-relative takes the base drive, drive-relative rejected
    - _Requirements: 13.1, 17.16, 17.17, 17.18_

  - [x] 2.4 Implement environment-name rules
    - Write `oscompat/env.py` with pure `env_key`, `lookup_env` (ordinal upper-case match on Windows, exact match elsewhere), `merge_env` (configured names replace every inherited entry with the same `env_key`, keeping configured spelling), and `inherited_env` (SDK `DEFAULT_INHERITED_ENV_VARS` plus `COMSPEC`, `WINDIR`, `PROGRAMDATA`, `PROGRAMFILES`, `PROGRAMFILES(X86)` on Windows, as a module constant)
    - _Requirements: 17.21, 17.22, 13.2_

  - [x] 2.5 Write property test for environment-variable name matching
    - File: `tests/property/test_env_lookup.py`
    - **Property 37: Environment-variable name matching**
    - **Validates: Requirements 17.21, 13.2**

  - [x] 2.6 Write property test for subprocess environment merge
    - File: `tests/property/test_env_merge.py`
    - **Property 38: Subprocess environment merge**
    - **Validates: Requirements 17.22**

  - [x] 2.7 Implement event loop factory selection
    - Write `oscompat/event_loop.py` with pure `select_loop_factory(platform, uvloop_new_event_loop)` returning `uvloop.new_event_loop` on macOS/Linux when available and `asyncio.EventLoop` otherwise, plus a wrapper that imports `uvloop` only when `sys.platform != "win32"`; use no event-loop policy API
    - _Requirements: 17.28, 17.29, 14.1_

  - [x] 2.8 Write unit tests for loop selection and deprecated API usage
    - File: `tests/unit/test_event_loop.py`
    - Test `select_loop_factory` for all six platform × uvloop combinations; scan `src/` for the deprecated policy names (`set_event_loop_policy`, `get_event_loop_policy`, `WindowsSelectorEventLoopPolicy`, `WindowsProactorEventLoopPolicy`, `DefaultEventLoopPolicy`, `AbstractEventLoopPolicy`, `asyncio.iscoroutinefunction`) and assert none are present
    - _Requirements: 17.28, 17.29_

  - [x] 2.9 Implement the default Storage_Directory and directory creation
    - Extend `oscompat/paths.py` with `APP_DIR = "local-agent"`, pure `default_storage_dir(platform, env)` reading variables through `lookup_env` (`win32`: non-empty `LOCALAPPDATA` → `<LOCALAPPDATA>\local-agent`, else `<USERPROFILE>\AppData\Local\local-agent`; `darwin`: `<HOME>/Library/Application Support/local-agent`; `linux`: absolute `XDG_DATA_HOME` → `<XDG_DATA_HOME>/local-agent`, relative or empty ignored, else `<HOME>/.local/share/local-agent`; missing home variable raises `PathFieldError`; result lexically normalized with the platform's path module)
    - Add I/O `ensure_storage_dir(path)`: existing non-directory → `StorageError("not a directory")`; otherwise `os.makedirs(path, exist_ok=True)` creating every missing parent with default permissions; OS errors become `StorageError` with the OS reason
    - Define the single `StorageError(path, reason)` class (reason already redacted) and `StaleState` in `storage/__init__.py`; `storage/database.py` and `session/store.py` re-export them
    - _Requirements: 13.10, 17.35, 17.38, 18.23, 18.24_

  - [x] 2.10 Write property test for the default storage directory
    - File: `tests/property/test_default_storage_dir.py`
    - Generate environments with `LOCALAPPDATA`, `USERPROFILE`, `HOME`, `XDG_DATA_HOME` independently absent, empty, relative, or absolute, with random letter case on Windows
    - **Property 56: Default storage directory**
    - **Validates: Requirements 17.35, 13.10**

  - [x] 2.11 Implement Windows ctypes declarations
    - Write `oscompat/_win32.py`, importable on every OS with functions available only on Windows: process and job declarations (`CreateJobObjectW`, `SetInformationJobObject` with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, `AssignProcessToJobObject`, `QueryInformationJobObject` `ActiveProcesses`, `TerminateJobObject`, `CreateToolhelp32Snapshot`/`Thread32First`/`Thread32Next`, `OpenThread`, `ResumeThread`, `CloseHandle`, `IsWow64Process2`, `SetConsoleCtrlHandler`, `HANDLER_ROUTINE`) and security declarations for `oscompat/permissions.py` (`OpenProcessToken`, `GetCurrentProcess`, `GetTokenInformation(TokenUser)`, `InitializeAcl`, `AddAccessAllowedAceEx`, `SetNamedSecurityInfoW`, `GetNamedSecurityInfoW`, `GetAce`, `GetSecurityDescriptorControl`, `LocalFree`, `FormatMessageW`, and the `ACL_REVISION`, `FILE_ALL_ACCESS`, `OBJECT_INHERIT_ACE`, `CONTAINER_INHERIT_ACE`, `SE_FILE_OBJECT`, `DACL_SECURITY_INFORMATION`, `PROTECTED_DACL_SECURITY_INFORMATION` constants)
    - _Requirements: 17.10, 17.11, 17.37_

  - [x] 2.12 Implement user-only storage permissions
    - Write `oscompat/permissions.py` with `STORAGE_FILES` (`local-agent.db`, `-wal`, `-shm`, `local-agent.lock`), `PermissionSetupError(path, os_reason)`, `restrict_dir(path, platform)`, `restrict_file(path, platform)`, `restrict_storage(dir, platform)` (directory plus every existing `STORAGE_FILES` entry), and `describe_access(path, platform) -> AccessReport` (mode bits on POSIX; DACL ACEs and the protected flag on Windows)
    - POSIX: `os.chmod(dir, 0o700)` and `os.chmod(file, 0o600)`, applied even when the files already exist with broader modes
    - Windows via `_win32.py`: current user SID from the process token, a DACL with exactly one allow ACE (`FILE_ALL_ACCESS`, user SID; `OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE` on the directory, `0` on files), applied with `SetNamedSecurityInfoW(..., DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, ...)` without changing owner or group; a non-zero return becomes `PermissionSetupError(path, FormatMessageW(code))`; do not rely on `os.mkdir(path, 0o700)`
    - _Requirements: 16.8, 16.9, 17.36, 17.37_

  - [x] 2.13 Write unit tests for storage permissions
    - File: `tests/unit/test_permissions.py`
    - `posix_only`: under `umask 000`, pre-existing `0755` directory and `0644` files become `0700`/`0600` (`stat.S_IMODE`); `windows_only`: `describe_access` shows a protected DACL with exactly one ACE for the current user SID, `OI|CI` on the directory, and a file created later in the directory inherits it; both: an injected `chmod`/`SetNamedSecurityInfoW` failure raises `PermissionSetupError` naming the path and OS reason
    - _Requirements: 16.8, 16.9, 17.36, 17.37_

- [x] 3. Implement configuration loading
  - [x] 3.1 Implement Backend_Config pydantic models
    - Write `config/models.py` with `ServerConfig`, `LLMConfig`, `AgentConfig`, `ApprovalKeywords` (non-empty, disjoint after casefold), `OrchestrationConfig` (including `session_idle_timeout_s` in [60, 604,800] or `None`), `StdioServerConfig` (`command`, `args` never path-processed, optional `cwd` stored as `None` when omitted, `env`), `SseServerConfig` (discriminated on `transport`), `LoggingConfig`, `StorageConfig` (`backend: Literal["sqlite", "memory"] = "sqlite"`, `path: str | None = None`), and `BackendConfig` (`extra="forbid"`, `frozen=True`, `storage: StorageConfig = StorageConfig()`), with all defaults and ranges from the design
    - Add `storage_directory(cfg, env, platform) -> str | None`: `None` for `memory`, otherwise the resolved `cfg.storage.path` or `default_storage_dir(platform, env)`
    - Use `SecretStr` for `llm.api_key`, `server.bearer_token`, MCP `env` values, and MCP `headers` values
    - _Requirements: 1.7, 5.1, 6.1, 6.2, 6.3, 8.9, 10.5, 11.8, 11.11, 13.4, 13.9, 13.10, 13.11, 15.7, 16.1, 16.6, 17.19, 17.20, 17.35_

  - [x] 3.2 Implement config loader
    - Write `config/loader.py` with `ConfigIssue`, `ConfigError`, `resolve_config_path` (`AGENT_CONFIG` resolved as a Path_Field against the process cwd, or `./config.yaml`, no fallback), `decode_config_bytes` (strip one UTF-8 BOM, strict UTF-8 decode independent of locale, invalid UTF-8 reported with the file path and byte offset counted from the file start including the BOM), CRLF→LF normalization before parsing, `substitute_env` (`${NAME}` replacement via `oscompat.env.lookup_env`, `$${` escape, missing-variable issues with dotted paths), `load_config` (reads bytes before the loop exists), `load_config_from_text`, and `dump_config`
    - Resolve Path_Fields (`mcp_servers[*].cwd`, `mcp_servers[*].command` when it contains a separator, and `storage.path` unless `storage.backend` is `memory`, in which case `storage.path` is accepted, left unresolved, and produces no issue) after substitution and before validation with `resolve_path_field(base_dir=<config dir>)`; report `PathFieldError` as a `ConfigIssue` at the field's dotted path; leave bare command names and `args` untouched
    - Parse with `yaml.safe_load`; report 1-based line/column for `MarkedYAMLError`; reject non-mapping top level
    - Run pydantic validation and the raw-tree cross-checks (duplicate server names, permission server references and pattern lengths, non-loopback bind needing a >=32-character bearer token, `storage.backend` other than `sqlite`/`memory` reported at `storage.backend`) so all issues are reported in one `ConfigError`; never include secret or substituted values in messages
    - _Requirements: 1.7, 5.3, 6.4, 13.1, 13.2, 13.3, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.11, 13.12, 16.5, 17.16, 17.17, 17.18, 17.20, 17.21, 17.23, 17.24, 17.25_

  - [x] 3.3 Write property test for environment substitution
    - File: `tests/property/test_config_substitution.py`
    - **Property 24: Environment substitution**
    - **Validates: Requirements 13.2**

  - [x] 3.4 Write property test for missing environment variable errors
    - File: `tests/property/test_config_missing_env.py`
    - **Property 25: Missing environment variable error**
    - **Validates: Requirements 13.8**

  - [x] 3.5 Write property test for configuration round-trip
    - File: `tests/property/test_config_roundtrip.py`
    - Include `storage` sections with both backends and `path` set and unset
    - **Property 26: Configuration round-trip**
    - **Validates: Requirements 13.5**

  - [x] 3.6 Write property test for complete validation report
    - File: `tests/property/test_config_validation.py`
    - Defect kinds include an invalid `storage.backend` value, reported at `storage.backend`
    - **Property 27: Complete validation report**
    - **Validates: Requirements 1.7, 5.3, 6.1, 6.4, 13.3, 13.9, 16.5**

  - [x] 3.7 Write property test for Path_Field resolution
    - File: `tests/property/test_paths.py`
    - Cover `resolve_path_field` directly and through the loader for `storage.path` (config directory as `base_dir`; `memory` backend leaves it unresolved with no issue)
    - **Property 34: Path_Field resolution**
    - **Validates: Requirements 17.16, 17.17, 17.18, 13.1, 13.10, 13.11**

  - [x] 3.8 Write property test for line-ending and BOM invariance
    - File: `tests/property/test_config_line_endings.py`
    - Generate CRLF/LF/BOM variants in memory, not from checked-out fixtures
    - **Property 35: Line-ending and BOM invariance**
    - **Validates: Requirements 17.23, 17.25, 13.7**

  - [x] 3.9 Write property test for invalid UTF-8 reporting
    - File: `tests/property/test_config_invalid_utf8.py`
    - **Property 36: Invalid UTF-8 reporting**
    - **Validates: Requirements 17.24**

  - [x] 3.10 Write unit tests for config loading
    - File: `tests/unit/test_config.py`
    - Test path resolution (including `AGENT_CONFIG` relative to cwd), defaults (Max_Tool_Iterations 10, Tool_Timeout 60, host `127.0.0.1`, empty CORS list, `storage.backend` `sqlite`, `storage.path` `None`, `session_idle_timeout_s` unset), missing file without fallback, YAML line/column, stdio and SSE fields including `ftp://` rejection, omitted `cwd` stays `None`, drive-relative `C:tools\x.exe` error message at the field path, `~` with no home variable, relative `storage.path` resolved against the config directory, `storage.backend: postgres` issue at `storage.backend`, `storage_directory()` returns `None` for `memory`
    - _Requirements: 6.2, 6.3, 13.1, 13.4, 13.6, 13.7, 13.9, 13.10, 13.11, 16.1, 16.6, 17.17, 17.18, 17.19_

- [x] 4. Implement structured logging and secret redaction
  - [x] 4.1 Implement `logging_setup.py`
    - Write `Redactor` (4-character window masking, short secrets matched whole, collapsed runs into one placeholder), `collect_secrets(cfg)`, `render_line(event) -> bytes` (`json.dumps(..., ensure_ascii=False, separators=(",", ":"))` encoded UTF-8 with `backslashreplace` plus exactly one `b"\n"`), `configure_logging` (structlog chain: contextvars, level, UTC millisecond timestamp, recursive redact processor, `render_line`, `make_filtering_bound_logger(level)`, `BytesLoggerFactory(sys.stderr.buffer)`), and `get_logger(component)`
    - _Requirements: 15.1, 15.5, 15.7, 17.26_

  - [x] 4.2 Write property test for secret redaction
    - File: `tests/property/test_redaction.py`
    - **Property 28: Secret redaction**
    - **Validates: Requirements 15.5, 9.7, 2.6**

  - [x] 4.3 Write property test for log line encoding
    - File: `tests/property/test_log_encoding.py`
    - **Property 47: Log line encoding**
    - **Validates: Requirements 17.26, 15.1**

  - [x] 4.4 Write unit tests for log format and level filtering
    - File: `tests/unit/test_logging.py`
    - Test single-line JSON with timestamp, level, component, event, `context_id`, `task_id`; `null` IDs outside a Task; level filtering and `info` default; output is bytes with LF only
    - _Requirements: 15.1, 15.7, 17.26_

- [x] 5. Implement tool naming, schema translation, and result formatting
  - [x] 5.1 Implement schema translator
    - Write `mcp_client/schema_translator.py` `translate(server, tool, llm_name)` returning an OpenAI function definition or `SchemaRejected` (meta-schema check via `validator_for(...).check_schema`, non-object `type` rejected, missing `type` defaulted to `object` with `properties` `{}`)
    - _Requirements: 7.2, 7.5, 7.6, 7.8_

  - [x] 5.2 Implement Tool_Registry with naming and permission matching
    - Write `mcp_client/registry.py` with `sanitize`, `assign_names` (collision suffixes `_2`, `_3`, ... within 64 characters), `compile_pattern` (`*`/`?` only, everything else escaped, `fullmatch`), `is_protected`, `ToolRegistry.replace_server`/`remove_server` (rebuild names in config order, atomic snapshot swap, schema-rejection warnings), and immutable `RegistrySnapshot` (`resolve`, `openai_tools` for available servers only, `is_protected`, `tools_by_server`)
    - _Requirements: 5.1, 5.2, 7.3, 7.4, 7.7, 7.8, 7.9, 7.12_

  - [x] 5.3 Write property test for schema translation
    - File: `tests/property/test_schema_translator.py`
    - **Property 7: Schema translation preserves the input schema**
    - **Validates: Requirements 7.2, 7.5, 7.6, 7.8**

  - [x] 5.4 Write property test for LLM-facing names
    - File: `tests/property/test_registry_names.py`
    - **Property 5: LLM-facing name format, uniqueness, and determinism**
    - **Validates: Requirements 7.3, 7.7, 7.12**

  - [x] 5.5 Write property test for permission pattern matching
    - File: `tests/property/test_permission_patterns.py`
    - **Property 9: Permission pattern matching**
    - **Validates: Requirements 5.1, 5.2**

  - [x] 5.6 Write property test for available-only tool exposure
    - File: `tests/property/test_registry_availability.py`
    - **Property 8: Only available servers' tools are exposed**
    - **Validates: Requirements 7.9**

  - [x] 5.7 Implement result formatter
    - Write `mcp_client/result_formatter.py` with `content_to_text` (newline join, non-text descriptions with type, MIME type, URI; empty list text; lone surrogates replaced with U+FFFD via `replace_lone_surrogates`), `truncate` (notice with original length and limit), `timeout_text`, and `exception_text` (redacted `"<Type>: <message>"`)
    - _Requirements: 8.6, 8.9, 8.11, 9.1, 9.2, 9.7, 16.11, 18.5_

  - [x] 5.8 Write property test for MCP content conversion
    - File: `tests/property/test_content_to_text.py`
    - **Property 16: MCP content conversion**
    - **Validates: Requirements 8.6, 8.11**

  - [x] 5.9 Write property test for tool output truncation
    - File: `tests/property/test_truncate.py`
    - **Property 17: Tool output truncation**
    - **Validates: Requirements 8.9, 9.1**

- [x] 6. Implement orchestration pure cores
  - [x] 6.1 Implement approval decision parsing
    - Create `orchestrator/approval_gate.py` with `Decision` enum and `parse_approval_decision(message, keywords)` (first valid structured `decision` data part wins, else trimmed casefolded text keyword match, else `None`)
    - _Requirements: 5.12_

  - [x] 6.2 Write property test for approval decision parsing
    - File: `tests/property/test_approval_parsing.py`
    - **Property 10: Approval decision parsing**
    - **Validates: Requirements 5.12**

  - [x] 6.3 Implement stream assembler
    - Write `llm/stream_assembler.py` `StreamAssembler.feed` (returns non-empty text, groups tool-call fragments by index: first id, first name, concatenated arguments) and `finish(existing_ids)` (ascending index order, `call_<uuid4 hex>` for missing IDs, argument text unchanged apart from lone-surrogate replacement); apply `replace_lone_surrogates` to the assembled text and arguments before they reach the Message_History
    - _Requirements: 3.2, 10.3, 10.11, 18.5_

  - [x] 6.4 Write property test for tool-call fragment assembly
    - File: `tests/property/test_stream_assembler.py`
    - **Property 18: Tool-call fragment assembly**
    - **Validates: Requirements 10.3, 10.11**

  - [x] 6.5 Implement context window trimming
    - Write `orchestrator/context_window.py` with `split_turns`, `trim_to_budget` (drop oldest whole turns, keep system prompt and last turn, raise `TurnTooLarge`, never mutate input), and `estimate_tokens` (`litellm.token_counter` via `asyncio.to_thread`, `ceil(chars/4)` fallback when tokenization fails, including a missing tiktoken cache; tool definitions counted)
    - _Requirements: 11.8, 11.9, 14.1_

  - [x] 6.6 Write property test for context window trimming
    - File: `tests/property/test_context_window.py`
    - **Property 20: Context window trimming**
    - **Validates: Requirements 11.8, 11.9**

  - [x] 6.7 Implement prompt builder
    - Write `orchestrator/prompt_builder.py` with `DEFAULT_BASE_INSTRUCTIONS` and `build_system_prompt(base, tools)`: base or default first, exactly one `## Tool usage` section with the three required instructions, tools grouped under `### <server>` headings with description or "(no description provided)", or "No tools are currently available."
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.8, 12.9_

  - [x] 6.8 Write property test for system prompt structure
    - File: `tests/property/test_prompt_builder.py`
    - **Property 22: System prompt structure**
    - **Validates: Requirements 12.1, 12.2, 12.4, 12.8, 12.9**

  - [x] 6.9 Implement Tool_Call preparation and status messages
    - Create `orchestrator/tool_dispatch.py` with `Prepared`, `Rejected`, `prepare_call` (unknown name, JSON parse / non-object, `jsonschema` errors with JSON path and keyword sorted by path, unavailable server, then `Prepared(protected=...)`), `denied_result`, and `start_status_message` (first path/file/uri/url string argument, truncated to 120 characters)
    - _Requirements: 4.2, 5.5, 9.3, 9.4, 9.5, 9.6_

  - [x] 6.10 Write property test for tool start status message
    - File: `tests/property/test_start_status_message.py`
    - **Property 11: Tool start status message**
    - **Validates: Requirements 4.2**

  - [x] 6.11 Write property test for rejected Tool_Calls
    - File: `tests/property/test_prepare_call.py`
    - **Property 15: Rejected Tool_Calls are never sent**
    - **Validates: Requirements 9.3, 9.4, 9.5, 5.5**

  - [x] 6.12 Implement Task_State transition table
    - Create `a2a_server/events.py` with the internal `TaskState` mapping to the pinned `a2a-sdk` enum names (isolated here), the `ALLOWED` transition set, `RECOVERY_ONLY = {(SUBMITTED, FAILED)}`, `TERMINAL_STATES`, `STREAM_CLOSING_STATES`, `LIVE_STATES` (`submitted`, `working`, `input-required`), and pure `is_allowed(current, new, *, recovery=False)` where only `recovery=True` admits `RECOVERY_ONLY`
    - _Requirements: 2.9, 3.5, 18.15_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement OS decision cores for processes and sockets
  - [x] 8.1 Implement executable resolution
    - Write `oscompat/process.py` with `Resolved`, `NotFound`, and pure `resolve_command(command, *, platform, path_value, pathext_value, base_dir, home, is_file)`: separator commands resolved as Path_Fields without `PATH`; otherwise `PATH` entries in order (empty skipped, quotes stripped on Windows, relative joined to the spawn cwd, no implicit current directory); on Windows `PATHEXT` extensions in order within each directory when the suffix is not in `PATHEXT` (default `.COM;.EXE;.BAT;.CMD`); `.cmd`/`.bat` → `batch`, `.exe`/`.com` → `native`, other suffixes such as `.ps1` → `NotFound` "not a launchable executable"; `NotFound` names the command and searched directories
    - _Requirements: 17.4, 17.5, 17.6_

  - [x] 8.2 Write property test for executable resolution
    - File: `tests/property/test_resolve_command.py`
    - Use an in-memory file set as the `is_file` oracle and record every probe
    - **Property 39: Executable resolution**
    - **Validates: Requirements 17.4, 17.5, 17.6**

  - [x] 8.3 Implement Windows command-line construction and batch quoting
    - Extend `oscompat/process.py` with `BatchArgRejected(position, char)`, pure `quote_batch_args(batch_path, args)` (reject `"`, `%`, CR, LF in the path or any argument with the 1-based position of the first offender; double-quote the path and every argument; double trailing backslashes before the closing quote; empty → `""`), and pure `build_command_line(resolved, args, system_root)` (`list2cmdline` for native; `%SystemRoot%\System32\cmd.exe /d /e:ON /v:OFF /c "<quoted>"` for batch)
    - _Requirements: 17.7, 17.8, 17.9_

  - [x] 8.4 Write property test for Windows argument fidelity
    - Files: `tests/property/winargv_ref.py` (reference Microsoft C runtime / `CommandLineToArgvW` parser and cmd.exe quote-state scanner) and `tests/property/test_win_argv.py`
    - **Property 40: Windows argument fidelity for native executables**
    - **Validates: Requirements 17.7**

  - [x] 8.5 Write property test for batch-file quoting and rejection
    - File: `tests/property/test_batch_quoting.py` (uses `winargv_ref.py`)
    - **Property 41: Batch-file quoting and rejection**
    - **Validates: Requirements 17.8, 17.9**

  - [x] 8.6 Implement spawn specification and termination plan
    - Extend `oscompat/process.py` with `SpawnSpec`, `SpawnRejected`, async `build_spawn_spec(cfg, config_dir, platform)` (file probes via `asyncio.to_thread`; `cwd` = resolved configured directory or config directory; `env` = `merge_env(inherited_env(os.environ), cfg.env)` with `PATH`/`PATHEXT` for resolution taken from the merged environment; POSIX `argv` with `start_new_session=True`; Windows `application` + `command_line` with `CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_NO_WINDOW`; `NotFound` and `BatchArgRejected` raise `SpawnRejected` with the resolver's message; `args` untouched), `Step`, `TerminationReport`, and pure `termination_plan(platform, job_assigned)`
    - _Requirements: 17.4, 17.6, 17.9, 17.10, 17.11, 17.19, 17.20, 17.22, 14.1_

  - [x] 8.7 Write property test for spawn specification
    - File: `tests/property/test_spawn_spec.py`
    - **Property 42: Spawn specification**
    - **Validates: Requirements 17.10, 17.19, 17.20**

  - [x] 8.8 Implement bind planning
    - Write `oscompat/net.py` with `BindTarget`, `SockOpts`, pure `plan_binds(host, port)` (`localhost` → required `127.0.0.1` + optional `::1` without name resolution; literals → one target; other hostnames → first `getaddrinfo` result), pure `socket_options(platform, family)` (`SO_EXCLUSIVEADDRUSE` on Windows, `SO_REUSEADDR` on Linux only, never `SO_REUSEPORT`, `IPV6_V6ONLY` on IPv6), pure `classify_bind_error(exc)`, and `bind_all(host, port, platform)` (one warning and skip for an unavailable optional target; on any other failure close bound sockets and raise `BindError` naming host, port, and OS error) with an injectable socket layer
    - _Requirements: 1.8, 17.30, 17.31, 17.32_

  - [x] 8.9 Write property test for bind plan and socket options
    - File: `tests/property/test_bind_plan.py`
    - **Property 45: Bind plan and socket options**
    - **Validates: Requirements 17.30, 17.31, 17.32, 1.8**

- [x] 9. Implement the session store interface and in-memory backend
  - [x] 9.1 Implement the SessionStore ABC and the StorageTxn model
    - Write `session/store.py` with frozen `StateChange` (`task_id`, `expected_from: frozenset[TaskState]`, `new`, redacted `status_message`, `metadata`), `ArtifactWrite` (`task_id`, `artifact_id`, `ordinal`, `text`, `closed`), `StorageTxn` (`context_id`, `now_ms`, ordered `appends` of `(producing task_id, ChatMessage)`, `system_prompt` slot replacement with an `UNSET` sentinel, optional `state`, `artifacts`, and `merge(other)`), `TaskQuery`/`TaskPage`, re-exports of `StorageError` and `StaleState` from `storage/__init__.py`, and the async ABC: `commit(txn)` (all or nothing), `create_task(task_id, context_id, now_ms, a2a_snapshot) -> bool` (one txn: ensure Context + insert `submitted` Task; `task_id` collision fails), `get_history`, `get_task` (artifacts in emission order), `list_tasks`, `active_task` (any Task in `submitted`/`working`/`input-required`), `set_a2a_snapshot`, `delete_expired(cutoff_ms)`, plus convenience wrappers `append`, `set_system_prompt`, `set_state` built on `commit`
    - Last_Activity rule: `commit` sets `last_activity_ms = now_ms` when the txn creates the Context, creates a Task, changes a Task_State, or appends messages; same-state status messages do not count
    - Timestamps are integer epoch milliseconds from the injected `Clock.now_ms()`; no `time.monotonic()` values are stored
    - _Requirements: 2.1, 11.1, 11.2, 11.6, 11.10, 11.11, 12.7, 18.3, 18.4, 18.8, 18.10_

  - [x] 9.2 Implement InMemorySessionStore
    - Write `session/memory_store.py`: per-Context records (single system-prompt slot, history, `created_at_ms`, `last_activity_ms`), Task records with artifacts keyed by `artifact_id` and ordered by `ordinal`, A2A snapshot slot; `commit` validates the whole `StorageTxn` first (CAS on `expected_from` → `StaleState`, unknown Context/Task, `task_id` collision) and then applies it with no `await` in between, so it is atomic on the event loop; `list_tasks` ordered by `updated_at_ms DESC, task_id DESC` with an opaque cursor; `delete_expired` left as a stub filled in by task 11.7
    - _Requirements: 2.1, 11.1, 11.2, 11.6, 11.10, 12.7, 18.2_

  - [x] 9.3 Write unit tests for the in-memory session store
    - File: `tests/unit/test_memory_store.py`
    - Test empty history on creation, task-to-context mapping, single system-prompt slot replacement, `active_task` for each of the six states, CAS failure leaves the store unchanged, a failing txn applies nothing, `task_id` collision fails `create_task`, Last_Activity unchanged by a same-state status message
    - _Requirements: 2.1, 11.1, 11.2, 11.7, 12.7_

- [x] 10. Implement the Storage_Database, Storage_Lock, and migrations
  - [x] 10.1 Implement the Database writer thread
    - Write `storage/database.py` with `Database.open(path, *, create, connection_factory=None)`, `run_sync(fn)` (startup only), async `read(fn)` and `transaction(fn)` via `loop.run_in_executor` on a dedicated `ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-agent-db")`, and `close()` (`PRAGMA optimize`, close on its thread, shut the executor down)
    - Connection created inside the writer thread: `sqlite3.connect(f"file:{quote(path)}?mode=rw", uri=True, autocommit=True, timeout=5)` (`mode=rwc` when absent; on POSIX pre-create the file with `os.open(path, O_CREAT | O_EXCL | O_RDWR, 0o600)`); pragmas checked by reading back: `journal_mode=WAL` must return `wal`, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`; `PRAGMA quick_check` at open; `sqlite3.sqlite_version_info < (3, 24)` → `StorageError` naming both versions; `sqlite3.DatabaseError` ("file is not a database") → `StorageError`
    - `transaction(fn)`: `BEGIN IMMEDIATE`, `fn(conn)`, `COMMIT`; any `Exception` runs `ROLLBACK` and is re-raised as `StorageError` (redacted reason) or `StaleState`; `BaseException` skips both (simulated kill); never delete, truncate, or recreate the database file or its sidecars
    - _Requirements: 14.1, 18.13, 18.14, 18.24, 18.30, 16.11_

  - [x] 10.2 Create the storage fault-injection helper
    - File: `tests/storage_faults.py` with `SimulatedCrash(BaseException)` and `FaultInjectingConnection`, which wraps the real connection and raises a chosen exception (`sqlite3.OperationalError("disk I/O error")`, `sqlite3.OperationalError("database or disk is full")`, or `SimulatedCrash`) at the k-th `execute`/`commit`, plus a statement counter used to enumerate crash points
    - _Requirements: 18.9, 18.30_

  - [x] 10.3 Write unit tests for the Database
    - File: `tests/unit/test_database.py`
    - Test `journal_mode` reads back `wal` and `synchronous` reads back `1`; every SQL call runs on the `local-agent-db` thread, never the loop thread; exception → rollback and `StorageError`; `SimulatedCrash` → no commit visible after reopen; garbage-bytes file, read-only file, and monkeypatched SQLite version `3.23.0` each raise `StorageError` and leave the file's SHA-256 unchanged; POSIX new file mode is `0600`
    - _Requirements: 18.13, 18.14, 18.24, 18.30_

  - [x] 10.4 Implement the Storage_Lock
    - Write `storage/lock.py` with `StorageInUse(path)` and `StorageLock.acquire(storage_dir, platform)` / `release()`: `<storage_dir>/local-agent.lock` opened with `os.open(O_RDWR | O_CREAT [| O_BINARY], 0o600)`; POSIX `fchmod` to `0600` and `fcntl.flock(fd, LOCK_EX | LOCK_NB)` (`BlockingIOError` → `StorageInUse`); Windows `restrict_file` on a pre-existing lock file and `msvcrt.locking(fd, LK_NBLCK, 1)` on byte 0 (`OSError` → `StorageInUse`); never waits; the descriptor stays open for the process lifetime; the PID is written after locking for diagnostics only and never read
    - _Requirements: 17.36, 17.37, 18.25, 18.26, 18.27_

  - [x] 10.5 Write unit tests for the Storage_Lock
    - File: `tests/unit/test_lock.py`
    - Test acquire → release → acquire succeeds; a second acquire from a child `python -c` process raises `StorageInUse` immediately (no wait, under 1 s); the lock file is `0600` on POSIX; the lock file contents are never read for decisions (a lock file with garbage contents still locks)
    - _Requirements: 18.25, 18.26, 18.27, 17.36_

  - [x] 10.6 Implement Schema_Version handling and schema v1
    - Write `storage/migrations.py` with `CURRENT_SCHEMA_VERSION = 1`, `Migration(version, description, statements)`, `SCHEMA_V1` (the `schema_version`, `contexts`, `messages`, `tasks`, `artifacts` tables, CHECK constraints, `ON DELETE CASCADE` foreign keys, and the `contexts_by_activity`, `messages_by_task`, `tasks_by_context`, `tasks_live`, `tasks_by_update` indexes from the design), `MIGRATIONS` (injectable for tests), pure `classify_schema(tables, version_rows)` → `Empty` / `AtVersion(v)` / `NoVersionTable` / `BadVersionValue` (user tables from `sqlite_schema` excluding `sqlite_%`; `typeof(version)` checked so `'1'` and `1.0` are rejected), pure `plan(state, supported, migrations)` → ordered migrations or `Refusal` (path, stored version or "no readable schema version", supported version), and `apply(db, path)` (read-only inspection first; `Refusal` → `SchemaError` before any write; each migration plus its `schema_version` `INSERT`/`UPDATE` in its own `transaction`; failure → that migration rolled back and `MigrationError(path, from_version, target_version)`)
    - _Requirements: 13.12, 18.19, 18.20, 18.21, 18.22_

  - [x] 10.7 Create migration fixture databases and fixture tests
    - Create `tests/fixtures/db/build_fixtures.py` (refuses to overwrite an existing fixture) and check in `v1_sample.db` (CRLF, emoji, an empty assistant text, non-JSON Tool_Call arguments, a Task in each state, closed and unclosed artifacts) with `v1_sample.expected.json`, `newer_v999.db`, `no_version_table.db`, `bad_version_two_rows.db`, `bad_version_text.db`, and `not_sqlite.db`
    - File: `tests/unit/test_migration_fixtures.py`: every `v*_sample.db` copy migrates to the current version and loads equal to its golden JSON; each refusal fixture is opened from a copy and the copy's SHA-256 is unchanged; a zero-length file and an absent file both end at Schema_Version 1 with every table present
    - _Requirements: 18.19, 18.20, 18.22, 18.24_

  - [x] 10.8 Write property test for migrations
    - File: `tests/property/test_migrations.py`
    - Use a synthetic test-only migration list (v1 → v2 → v3 with column additions and a table rebuild) injected through `MIGRATIONS`, and failure injection inside migration m
    - **Property 54: Migrations preserve data and roll back on failure**
    - **Validates: Requirements 18.19, 18.20, 18.21**

  - [x] 10.9 Write property test for unsupported schema refusal
    - File: `tests/property/test_schema_refusal.py`
    - Generate user tables and `schema_version` contents (absent table, zero rows, several rows, text or real value, integer above, equal to, or below the supported version); compare file SHA-256 before and after
    - **Property 55: Unsupported schema refused without modification**
    - **Validates: Requirements 18.22, 13.12**

- [x] 11. Implement the SQLite store, Task_Store, Restart_Recovery, and retention
  - [x] 11.1 Implement SQLiteSessionStore
    - Write `session/sqlite_store.py` `SQLiteSessionStore(db, redactor, clock)`: each method is one `db.transaction(fn)` or `db.read(fn)` with `fn` a plain synchronous function over the connection; `commit(txn)` runs the appends (`seq = COALESCE(MAX(seq), -1) + 1` per Context, one row per message with the producing `task_id`), system-prompt slot replacement, the CAS `UPDATE tasks SET state=? … WHERE task_id=? AND state IN (…)` (rowcount ≠ 1 → `StaleState`), artifact UPSERTs, `updated_at_ms`, and Last_Activity in one `BEGIN IMMEDIATE … COMMIT`
    - Encoding: text bound as `TEXT` parameters exactly as the Python `str` (never formatted into SQL, stripped, or normalized); `None` vs `""` kept distinct; `tool_calls` as `json.dumps([{"id","name","arguments"}], ensure_ascii=False, separators=(",", ":"))` with `arguments` kept as a JSON string; metadata as a JSON object; one mapping between `TaskState` and the lower-case stored state names
    - Redaction: `status_message`, failure messages, and exception-derived Tool_Results pass through `Redactor.redact` before entering a txn; no config table and no Backend_Config secret is written anywhere
    - `create_task` inserts the Context if absent and the Task as `submitted` with its `a2a_snapshot` in one txn; a `task_id` primary-key collision fails the commit; `list_tasks` uses `tasks_by_update` with an opaque cursor; `active_task` uses `tasks_live`; `delete_expired` left as a stub filled in by task 11.7
    - _Requirements: 2.1, 11.1, 11.2, 11.6, 11.7, 11.10, 12.7, 16.10, 16.11, 18.1, 18.3, 18.4, 18.5, 18.8, 18.10, 18.11, 18.14_

  - [x] 11.2 Implement ProjectedTaskStore
    - Write `storage/task_store.py` `ProjectedTaskStore(TaskStore)` depending only on the `SessionStore` ABC and a small `LiveArtifacts` protocol (implemented later by `TaskRunnerRegistry`): `expect_creation(task_id)` future; `save(task, context)` (unknown `task_id` + `SUBMITTED` → `sessions.create_task(...)`, resolve the future with success or the `StorageError` and re-raise on failure; unknown `task_id` with any other state → debug log, not stored; known `task_id` → store only SDK `history`/`metadata` in `a2a_snapshot` when changed, never state, status, or artifacts; a terminal state the Backend did not commit → `task_store_divergent_save` warning); `get` (Task proto from the committed row, `status.timestamp` from `updated_at_ms`, `history` from `a2a_snapshot`, artifacts by `ordinal`, overlaid with `live_artifacts()` for running Tasks; `None` when absent); `list` (filters `context_id`, `status`, `status_timestamp_after`, `page_size`, `page_token`); `delete` (one Task and its artifacts); `snapshot(task_id)` for first events; tenant/owner scoping ignored
    - Wraps `InMemorySessionStore` unchanged for `storage.backend: memory`
    - _Requirements: 2.10, 2.11, 11.14, 18.1, 18.2, 18.7, 18.8, 18.9, 18.30_

  - [x] 11.3 Write property test for the persistence round trip
    - File: `tests/property/test_persistence_roundtrip.py`
    - Use `stored_texts()` and `context_task_sets()` (CR, LF, CRLF, tabs, NUL, non-BMP, empty and whitespace-only strings, `None` vs `""` assistant content, non-JSON argument text, System_Prompt present or absent, metadata, artifacts in any number and order); close and reopen the `Database`; also check `ProjectedTaskStore.get` and the first chat-completion request built by `TaskRunner._build_request` for a new Task in the reopened Context
    - **Property 49: Persistence round trip**
    - **Validates: Requirements 18.3, 18.4, 18.5, 18.6, 18.7, 2.13**

  - [x] 11.4 Write unit tests for the SQLite store and Task_Store
    - File: `tests/unit/test_sqlite_store.py`
    - Test `None` vs `""` assistant content, a lone surrogate arriving through `StreamAssembler` is stored as U+FFFD, a `task_id` collision with a persisted row fails creation, `save` of an unknown non-`SUBMITTED` Task stores nothing, a divergent terminal `save` logs `task_store_divergent_save` and changes nothing, `list` paging and ordering, `delete` keeps the Context and its history, the SDK's "already exists" log line for continuation snapshots is downgraded to debug
    - _Requirements: 2.1, 18.3, 18.5, 18.9, 18.30_

  - [x] 11.5 Implement Restart_Recovery
    - Write `storage/recovery.py` with `RECOVERY_STATUS = "The backend restarted while this task was running"`, `RECOVERY_TOOL_RESULT = "Tool call not completed: the backend restarted while this task was running"`, `TaskRow`, `StoredMessage`, `RecoveryAction`, pure `plan_recovery(tasks, histories)` (Tasks in `submitted`/`working`/`input-required` ordered by `created_at_ms` then `task_id`; for each, the unanswered Tool_Calls of its own assistant messages in history order then Tool_Call order; terminal Tasks produce no action), and `run(db, clock) -> RecoveryReport` (one transaction per action: append the synthetic `ToolMessage`s, `StateChange(expected_from={from_state}, new=FAILED, status_message=RECOVERY_STATUS)` checked with `is_allowed(from_state, FAILED, recovery=True)`, Last_Activity = now; a failing transaction rolls back that Task, stops recovery, and raises `StorageError`)
    - Sends nothing to any MCP_Server, the LLM, or any client; starts no runner
    - _Requirements: 2.9, 11.5, 18.15, 18.16, 18.17, 18.18, 18.24_

  - [x] 11.6 Write property test for Restart_Recovery
    - File: `tests/property/test_recovery.py`
    - Use databases from the kill-injection helper in `tests/property/scenarios.py` (task 17.9) and `db_states()` generated rows (all six Task_States, partly answered Tool_Calls, several Tasks per Context); recording LLM, MCP, and client fakes; compare file contents for the idempotence clauses
    - **Property 53: Restart_Recovery integrity and idempotence**
    - **Validates: Requirements 18.15, 18.16, 18.17, 18.18, 2.9**

  - [x] 11.7 Implement retention
    - Write `storage/retention.py` with `ContextActivity`, pure `select_expired(contexts, now_ms, idle_timeout_s)` (`now_ms - last_activity_ms > idle_timeout_s * 1000` and no live Task, input order), and async `sweep(store, runners, clock, idle_timeout_s)` (calls `store.delete_expired(cutoff_ms)`, then `runners.forget(context_ids)` through a `ForgetContexts` protocol)
    - Fill in `SQLiteSessionStore.delete_expired` (one txn: `SELECT` with `last_activity_ms < cutoff` and `NOT EXISTS` a live Task, then `DELETE FROM contexts WHERE context_id IN (…)`, cascading to messages, Tasks, and artifacts) and `InMemorySessionStore.delete_expired` (applies `select_expired`); nothing runs when `session_idle_timeout_s` is unset
    - _Requirements: 11.11, 11.12, 11.13, 11.14_

  - [x] 11.8 Write property test for retention
    - File: `tests/property/test_retention.py`
    - Run against both stores; on SQLite check the SQL predicate against `select_expired` (model-based), absence from `SessionStore.get_task` and `ProjectedTaskStore.get`, and an injected failure inside the sweep transaction leaving every Context present
    - **Property 21: Retention**
    - **Validates: Requirements 11.11, 11.12, 11.14**

  - [x] 11.9 Write unit tests for retention and recovery examples
    - File: `tests/unit/test_retention_recovery.py`
    - Test that an unset `session_idle_timeout_s` deletes nothing after a simulated year; a Context with a `submitted` Task is never deleted; recovery on a database with no live Task leaves the file SHA-256 unchanged; a recovery transaction failure rolls back that Task, keeps earlier recovered Tasks, and raises `StorageError`
    - _Requirements: 11.12, 11.13, 18.18, 18.24_

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement process spawning, tree termination, and the stdio transport
  - [x] 13.1 Implement the CTRL_BREAK helper
    - Write `oscompat/ctrl_break.py` (`python -m local_agent.oscompat.ctrl_break <pid>`: `FreeConsole`, `AttachConsole(pid)`, `SetConsoleCtrlHandler(NULL, TRUE)`, `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0)`) using the declarations from `oscompat/_win32.py`
    - _Requirements: 17.10, 17.11_

  - [x] 13.2 Implement process spawning and tree termination
    - Extend `oscompat/process.py` with `TreeHandle`, `spawn(spec)` (POSIX `anyio.open_process(..., start_new_session=True)`; Windows: create job, `Popen` with the string command line suspended, assign to the job, resume the primary thread, `job_object_unavailable` warning and resume on failure; all blocking calls via `to_thread`; pipe I/O via worker threads; child stderr logged line by line at `debug`) and `terminate_tree(h, grace_s, stdin_wait_s)` executing `termination_plan` under a shielded cancel scope with bounded waits: close stdin, `killpg(pid, SIGTERM)` or the `ctrl_break` helper with 1 s timeout (`ctrl_break_failed` warning on failure), poll until empty or grace elapses, `killpg(SIGKILL)` / `TerminateJobObject` / `taskkill.exe /PID <pid> /T /F`, reap ≤ 1.5 s, `CloseHandle(job)`, `mcp_process_tree_stopped` log with survivors
    - Inject the OS-call layer and clock so the schedule is testable with a fake tree
    - _Requirements: 6.11, 14.1, 17.10, 17.11, 17.12, 17.15_

  - [x] 13.3 Write property test for process-tree termination schedule
    - File: `tests/property/test_terminate_tree.py`
    - Use a `FakeTree` with scripted member exits, a fake clock, and a recording OS-call shim
    - **Property 43: Process-tree termination schedule**
    - **Validates: Requirements 17.11, 17.15, 6.11**

  - [x] 13.4 Write unit tests for process helpers
    - File: `tests/unit/test_process.py`
    - Test `termination_plan` templates per platform and job outcome, `PATHEXT` default when unset, `.ps1` rejected as not launchable, `SpawnRejected` message for an unresolved command
    - _Requirements: 17.4, 17.6, 17.11_

  - [x] 13.5 Implement the managed stdio transport
    - Write `mcp_client/mcp_compat.py` isolating `mcp` SDK import locations (`mcp.types` vs `mcp_types`, the JSON-RPC message adapter, `SessionMessage`, timeout argument types)
    - Write `mcp_client/stdio_transport.py` with pure `encode_line` (`model_dump_json(by_alias=True, exclude_unset=True)` + `"\n"`, UTF-8) and `decode_line` (strict UTF-8, SDK JSON-RPC adapter, `Exception` item on failure), and `managed_stdio_client(spec, *, grace)` yielding `(read_stream, write_stream)` of `SessionMessage`: bytes-level split on `b"\n"`, one trailing `b"\r"` stripped, writer task, and `terminate_tree` with `grace()` read at exit on normal exit, error, or cancellation
    - _Requirements: 6.7, 6.11, 17.11, 17.12, 17.27_

  - [x] 13.6 Write property test for stdio framing round-trip
    - File: `tests/property/test_stdio_framing.py`
    - **Property 48: stdio framing round-trip**
    - **Validates: Requirements 17.27**

- [x] 14. Implement MCP client manager
  - [x] 14.1 Implement McpSessionAdapter
    - Create `mcp_client/manager.py` with `McpSessionAdapter` wrapping `ClientSession` through `mcp_compat`: `list_all_tools(timeout)` following `next_cursor` until none, and `call_tool(name, args, timeout)` returning a `CallOutcome` (content items, `is_error`), isolating `mcp` SDK naming and timeout-type differences
    - _Requirements: 7.1, 7.10, 9.2_

  - [x] 14.2 Write property test for paginated tool discovery
    - File: `tests/property/test_list_all_tools.py`
    - **Property 32: Paginated tool discovery**
    - **Validates: Requirements 7.1**

  - [x] 14.3 Implement ServerSupervisor and McpManager
    - Add `backoff_delay(n) = min(2**n, 30)`, `ServerState`, `ServerSupervisor.run` (stdio: `managed_stdio_client(await build_spawn_spec(cfg, config_dir, sys.platform), grace=...)`; SSE: `sse_client(url, headers=...)`; `SpawnRejected` treated as a failed attempt with the resolver's message; 15 s handshake, tool listing, registry replace/remove, up to 5 reconnects, Process_Tree terminated after a failed handshake and before each reconnection by leaving the transport context; logging with redaction), and `McpManager` (`start` concurrently with 17 s cap, disabled servers skipped, `stop(grace=5.0)` stopping all supervisors concurrently under `asyncio.timeout(1 + grace + 1.5)`, `refresh_available` keeping previous tools on failure, `availability`, `statuses`, `call_tool` with `asyncio.timeout` cancelling late responses, concurrent calls on one session)
    - Log `mcp_server_connected`, `mcp_server_failed`, and `tool_call_finished`-related timing
    - _Requirements: 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 7.10, 7.11, 9.2, 9.9, 14.3, 17.6, 17.9, 17.11, 17.15_

  - [x] 14.4 Write unit tests for McpManager with a fake transport
    - File: `tests/unit/test_mcp_manager.py`
    - Test backoff sequence `[1, 2, 4, 8, 16]` and stop after 5, re-registration after reconnect, refresh failure keeps tools, tool-list failure registers none, unavailable server, timeout, late reply discarded, exception text, `SpawnRejected` marks the server unavailable with the resolver's reason, `stop(grace)` forwards the grace to every transport
    - _Requirements: 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 7.10, 7.11, 9.2, 9.6, 9.7, 9.9, 17.6, 17.15_

- [x] 15. Implement LLM gateway
  - [x] 15.1 Implement LlmGateway and LlmStream
    - Write `llm/gateway.py` with `LlmUnreachable`, `LlmRejected(reason)`, `LlmInterrupted`, `LlmGateway.stream(messages, tools)` (LiteLLM `acompletion` with model, api_base, api_key only when set, temperature, max_tokens, timeout, `stream=True`, `num_retries=0`, tools and `tool_choice="auto"` only when non-empty), retry before first chunk (1 s, 2 s; 3 attempts), no retry on 4xx or after first chunk, and `LlmStream` yielding text chunks immediately with `result()` from `StreamAssembler`
    - Inject `sleep`/clock for tests; write one `llm_request_finished` log record per request with `duration_ms`, outcome, and error fields
    - Add a startup helper that sets `litellm.add_function_to_prompt = True` when `native_function_calling` is false
    - _Requirements: 10.1, 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 15.1, 15.4, 15.6_

  - [x] 15.2 Write property test for LLM retry policy
    - File: `tests/property/test_llm_retry.py`
    - **Property 19: LLM retry policy**
    - **Validates: Requirements 10.6, 10.9, 10.10**

  - [x] 15.3 Write unit tests for gateway request kwargs
    - File: `tests/unit/test_llm_gateway.py`
    - Test api_key only when set, defaults, tools omitted when disabled or empty, immediate chunk yielding
    - _Requirements: 10.1, 10.2, 10.4, 10.5_

- [x] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Implement event emitter, approval gate, dispatcher, and task runner
  - [x] 17.1 Implement TaskEventEmitter with commit-before-emit
    - Extend `a2a_server/events.py` with `TaskEventEmitter(sessions, sink, redactor, clock, db_path)`: `transition(new, message=None, metadata=None, *, with_txn=None, emit=True)` (disallowed → `transition_rejected` log and `False`; otherwise one `StorageTxn` with `StateChange(expected_from={cur})`, redacted status message, metadata, `updated_at_ms`, and Last_Activity merged with `with_txn`, awaited through `SessionStore.commit` **before** the status event is enqueued; `task_state_changed` log; final flag on terminal and `input-required`; on a terminal transition the text of every artifact without a last-chunk event is added to the same txn), `status` (same-state message committed before enqueue), `emit_current_status()` (enqueue the committed state, no commit), `token` (append to an in-memory `ArtifactBuffer` and enqueue without a commit), `close_artifact` (commit the concatenated text with `closed=1`, then enqueue the empty-text `append=true`, `last_chunk=true` event), `live_artifacts()`, `tool_started`, `tool_completed` (exactly once per call ID), `approval_required`, `reprompt_approval`, seal after terminal state
    - Failure paths: a `StorageError` from `commit` means rollback; enqueue nothing dependent, log `storage_write_failed` (database path, `task_id`, redacted reason), and raise `StorageWriteFailed`; a failed `submitted → working` commit sets the logical state to `working` and the follow-up commit uses `expected_from ∈ {submitted, working}` (18.30); if the `failed` commit also fails, enqueue the final `failed` Status_Event unpersisted, seal, and leave the row at its last committed state (18.31)
    - Sinks: the attached `EventQueue` via `TaskUpdater`, and `DetachedSink`, which drops events (their state is already committed and served by `ProjectedTaskStore`)
    - _Requirements: 2.9, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9, 4.4, 4.5, 4.6, 4.9, 11.2, 14.6, 14.7, 15.2, 16.11, 18.8, 18.10, 18.11, 18.12, 18.30, 18.31_

  - [x] 17.2 Write property test for transition table and emitter seal
    - File: `tests/property/test_emitter.py`
    - Run the emitter over a recording store; assert `submitted → failed` is accepted only by `is_allowed(..., recovery=True)` and never through the emitter, and that rejected requests leave the in-memory and stored state unchanged
    - **Property 4: Transition table and emitter seal**
    - **Validates: Requirements 2.9, 3.9, 4.9, 14.6**

  - [x] 17.3 Write property test for persisted artifact prefixes
    - File: `tests/property/test_artifact_prefix.py`
    - Drive the emitter over `SQLiteSessionStore` with `FaultInjectingConnection` for any chunk sequence, interruption (none, LLM failure after chunk j, cancel after chunk j), and crash point k; compare persisted text with the `RecordingEventSink` concatenation per artifact
    - **Property 52: Persisted artifact text is a prefix of emitted text**
    - **Validates: Requirements 18.12, 3.4, 3.8**

  - [x] 17.4 Implement ApprovalGate
    - Extend `orchestrator/approval_gate.py` with `PendingApproval` and `ApprovalGate` (`request` stores the pending call in memory only, transitions to `input-required` with LLM-facing name, original name, server, call ID, arguments, and `tool_approval_required` metadata, awaits the future under the optional approval timeout, transitions back to `working`; `resolve`; `discard`; at most one pending per Task)
    - _Requirements: 5.4, 5.6, 5.7, 5.8, 5.13, 5.14, 5.15, 2.7_

  - [x] 17.5 Write unit tests for ApprovalGate
    - File: `tests/unit/test_approval_gate.py`
    - Test status contents, approve, deny, timeout denial, indefinite wait without timeout, discard on cancel
    - _Requirements: 5.6, 5.8, 5.10, 5.14, 5.15_

  - [x] 17.6 Implement ToolDispatcher
    - Extend `orchestrator/tool_dispatch.py` with `ToolDispatcher.run_turn` (rejected calls get results and `tool_completed(failed)`, non-protected calls run concurrently via `asyncio.gather`, then protected calls one at a time through the ApprovalGate in call order, results returned in Tool_Call order) and `_execute` (`tool_started`, `McpManager.call_tool` with Tool_Timeout, `ResultFormatter` conversion and truncation, `tool_completed`, `tool_call_finished` log with outcome and `duration_ms`; non-cancellation exceptions become redacted error results)
    - _Requirements: 4.2, 4.3, 4.5, 4.6, 4.7, 4.8, 5.9, 5.10, 5.13, 8.2, 8.3, 8.5, 9.1, 9.2, 9.7, 9.8, 9.10, 15.3, 15.4, 16.11_

  - [x] 17.7 Write property test for name round-trip and faithful dispatch
    - File: `tests/property/test_name_roundtrip.py`
    - **Property 6: Name round-trip and faithful dispatch**
    - **Validates: Requirements 7.4, 8.1, 8.2**

  - [x] 17.8 Implement TaskRunner, TaskRunnerRegistry, and reasoning loop
    - Write `orchestrator/loop.py`: `TaskRunner` (`attach`, `start`, `cancel` with 2 s cap and pending approval discard, `wait_stream_closing`), `_run` (single "Discovering local tools..." `working` transition, per-session discovery refresh, user message append as one commit, exception-to-`failed` mapping with redacted messages, `StorageWriteFailed` → "The conversation could not be saved"), `_reasoning_loop` (round-trip counting, tools-disabled final request at Max_Tool_Iterations, assistant message committed before dispatch, ordered Tool_Results committed after the turn and before the next request, every message row tagged with the producing `task_id`), `_build_request` (rebuild System_Prompt from live snapshot into the single slot, load persisted history, trim to budget), `_stream_response` (per-response artifact IDs, Token_Events for non-empty text, `close_artifact`, empty artifact for empty final response), `_finalize` (`missing_results()` synthetic "not completed" Tool_Results passed as `with_txn` so they, unclosed artifact text, and the terminal state commit in one txn, under `asyncio.timeout(1.5)`), and `TaskRunnerRegistry` (admission reservations per Context with a 1 s wait for the reserved `task_id`, `create`, `get`, `forget(context_ids)` for retention, `live_artifacts(task_id)` for `ProjectedTaskStore`, and `cancel_all()` for Graceful_Shutdown)
    - Bind `context_id`/`task_id` structlog contextvars inside each runner; use `asyncio.get_running_loop()` and `inspect.iscoroutinefunction()` only
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 3.2, 3.3, 3.4, 3.6, 3.8, 4.1, 6.12, 8.7, 8.8, 8.10, 8.12, 10.7, 10.10, 11.3, 11.4, 11.5, 11.8, 11.9, 11.14, 12.5, 12.6, 12.7, 14.4, 14.5, 17.14, 17.28, 18.6, 18.10, 18.11, 18.12, 18.30_

  - [x] 17.9 Build scenario fakes and strategies for orchestration property tests
    - Create `tests/fakes.py` with `FakeLlmGateway`, `FakeMcpManager`, `RecordingEventSink`, and `tests/property/scenarios.py` with the `scenario_scripts()` strategy (per-turn text chunks, tool calls of each kind, latency and outcomes, approval decisions, cancel/failure injection), a `run_scenario` helper using a fresh event loop, fake clock, and a store fixture parametrized over `InMemorySessionStore` and `SQLiteSessionStore` (fresh `TemporaryDirectory` per example), a kill-injection helper that runs a scenario with `FaultInjectingConnection` raising `SimulatedCrash` at statement k and returns the database path, and a transcript normalizer that replaces generated IDs, timestamps, and durations with placeholders
    - _Requirements: 3.4, 8.5, 11.5, 17.3, 18.29_

  - [x] 17.10 Write property test for stream reassembly and token framing
    - File: `tests/property/test_stream_framing.py`
    - **Property 1: Stream reassembly and token framing**
    - **Validates: Requirements 3.2, 3.3, 3.4**

  - [x] 17.11 Write property test for interrupted streams
    - File: `tests/property/test_stream_interrupted.py`
    - **Property 2: Interrupted streams keep the emitted prefix**
    - **Validates: Requirements 3.8, 10.10, 10.7**

  - [x] 17.12 Write property test for event ordering
    - File: `tests/property/test_event_ordering.py` (both stores)
    - **Property 3: Event ordering**
    - **Validates: Requirements 2.4, 4.1, 4.3, 4.4, 4.5, 4.6**

  - [x] 17.13 Write property test for history integrity and result ordering
    - File: `tests/property/test_history_integrity.py` (both stores)
    - **Property 12: History integrity and result ordering**
    - **Validates: Requirements 8.4, 8.5, 8.10, 9.8, 9.10, 11.4, 11.5, 14.5, 5.13**

  - [x] 17.14 Write property test for context isolation
    - File: `tests/property/test_context_isolation.py` (both stores)
    - **Property 13: Context isolation**
    - **Validates: Requirements 11.6**

  - [x] 17.15 Write property test for iteration limit
    - File: `tests/property/test_iteration_limit.py` (both stores)
    - **Property 14: Iteration limit**
    - **Validates: Requirements 8.8, 8.12**

  - [x] 17.16 Write property test for system prompt placement and single storage
    - File: `tests/property/test_system_prompt_placement.py` (both stores)
    - **Property 23: System prompt placement and single storage**
    - **Validates: Requirements 8.7, 12.5, 12.6, 12.7**

  - [x] 17.17 Write property test for store agreement under kill injection
    - File: `tests/property/test_store_agreement.py`
    - For any scenario and crash index k, reopen and compare `SessionStore.get_task` with `ProjectedTaskStore.get`, the last recorded Status_Event state, observable messages, and history integrity parts (a) and (b)
    - **Property 50: Store agreement and history integrity under kill injection**
    - **Validates: Requirements 18.8, 18.9, 18.10, 18.11, 11.2, 11.5**

  - [x] 17.18 Write property test for storage write failures
    - File: `tests/property/test_storage_write_failures.py`
    - Two or more concurrent Contexts; one or two consecutive commits of one Task fail with `OperationalError`; compare other Contexts with a fault-free run; check `storage_write_failed` records name the database path and `task_id` and contain no secret
    - **Property 51: Storage write failures**
    - **Validates: Requirements 18.30, 18.31, 14.7, 16.11**

  - [x] 17.19 Write property test for secrets in the Storage_Database
    - File: `tests/property/test_no_secrets_in_db.py`
    - Run scenarios on `SQLiteSessionStore` whose tool exceptions, LLM rejection reasons, and status messages embed configured secrets; dump every TEXT value of every table
    - **Property 61: No Backend_Config secrets in the Storage_Database**
    - **Validates: Requirements 16.10, 16.11**

  - [x] 17.20 Write unit tests for reasoning loop edge cases
    - File: `tests/unit/test_reasoning_loop.py`
    - Test empty final response artifact, failure message without traceback, user message committed before the first request, no new LLM call on approval resume, approve uses identical arguments, cancel between approval and send, a failed commit ends the Task `failed` with "The conversation could not be saved", every message row carries the producing `task_id`
    - _Requirements: 2.5, 2.6, 4.7, 4.8, 5.7, 5.9, 11.3, 18.10, 18.30_

- [x] 18. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Implement Graceful_Shutdown and signal handling
  - [x] 19.1 Implement ShutdownController and signal installation
    - Write `oscompat/signals.py` with `ShutdownReason` (`SIGNAL`, `CONSOLE_CLOSE`), `ShutdownController` (`request` idempotent, second request forces `grace_s = 0`; `grace_s` 5.0 / 2.0 for console close; `run()` executing in order: `server.should_exit = True`, `TaskRunnerRegistry.cancel_all()`, wait for lifespan shutdown to finish `McpManager.stop(grace=grace_s)` and `Database.close()`, return exit status 0; `shutdown_started`/`shutdown_finished` logs; a `threading.Event` set when done), and `install(loop, ctl, platform)` returning an uninstall callable: POSIX `loop.add_signal_handler` for `SIGINT`/`SIGTERM`; Windows `signal.signal` for `SIGINT`/`SIGBREAK` via `call_soon_threadsafe`, plus a referenced ctypes `SetConsoleCtrlHandler` callback that schedules `CONSOLE_CLOSE` and blocks ≤ 4.5 s on the done event for `CTRL_CLOSE_EVENT` and returns `FALSE` for other events
    - _Requirements: 2.7, 6.11, 17.13, 17.14, 17.15_

  - [x] 19.2 Write property test for Graceful_Shutdown sequencing
    - File: `tests/property/test_shutdown_sequence.py`
    - Use fake server, runner registry, and MCP manager recording step start/finish
    - **Property 44: Graceful_Shutdown sequencing**
    - **Validates: Requirements 17.13, 17.14, 2.7**

  - [x] 19.3 Write unit tests for the Windows console control handler
    - File: `tests/unit/test_signals.py`
    - Call the handler callback directly: returns `FALSE` for non-close events; for `CTRL_CLOSE_EVENT` requests `CONSOLE_CLOSE` and blocks until the done event or 4.5 s; grace becomes 2 s
    - _Requirements: 17.15_

- [x] 20. Implement A2A server surface, app wiring, and entry point
  - [x] 20.1 Implement Agent_Card builder
    - Write `a2a_server/agent_card.py` with `endpoint_url(host, port)` (IPv6 bracketed, configured `localhost` kept) and `build_agent_card(cfg)` (streaming true, text modes, `local-tools-chat` skill, JSON-RPC interface at the endpoint URL)
    - _Requirements: 1.3, 1.4, 1.5, 17.30_

  - [x] 20.2 Write property test for agent card content
    - File: `tests/property/test_agent_card.py`
    - **Property 31: Agent card content**
    - **Validates: Requirements 1.3, 1.4, 1.5**

  - [x] 20.3 Implement loopback and bearer-token middleware
    - Write `a2a_server/auth.py` with pure `is_loopback(host)`, `LoopbackGuardMiddleware` (403 for non-loopback clients when enabled; both `127.0.0.1` and `::1` sockets pass), and `BearerAuthMiddleware` (`hmac.compare_digest`, 401 before routing, agent card path exempt, `OPTIONS` passed through)
    - _Requirements: 1.2, 16.1, 16.2, 16.3, 16.7, 17.30_

  - [x] 20.4 Write property test for loopback classification and bearer check
    - File: `tests/property/test_auth.py`
    - **Property 30: Loopback classification and bearer check**
    - **Validates: Requirements 16.1, 16.2, 16.3**

  - [x] 20.5 Implement health endpoint
    - Write `a2a_server/health.py` with pure `health_payload(statuses)` (`ok` iff every enabled server is available) and a Starlette `health_endpoint` reading only in-memory `McpManager` state
    - _Requirements: 15.8, 15.9_

  - [x] 20.6 Write property test for health status
    - File: `tests/property/test_health.py`
    - **Property 29: Health status**
    - **Validates: Requirements 15.8, 15.9**

  - [x] 20.7 Implement the A2A error contract types and version check
    - Write `a2a_server/a2a_compat.py` isolating `a2a-sdk` import names (error classes, `A2A_ERROR_MAPPING`, `A2A_ERROR_REASONS`, `ErrorMapping`, `JsonRpcDispatcher`, `DefaultRequestHandler`/`DefaultRequestHandlerV2`, `TaskStore`, `TaskState` names)
    - Write `a2a_server/errors.py` with `HostContractError(A2AError)` (`jsonrpc_code`, `jsonrpc_error()`), `ContextBusyError(context_id, active_task_id)` (`-32050`, "Context is busy"), `TaskNotAwaitingInputError(task_id, state)` (`-32602`, "Task is not awaiting input", `state` as `TaskState.Name(...)`), `StorageWriteError` (`InternalError` subclass, `-32603`, "Storage could not be written"), idempotent `install_error_mappings()` (REST: `ContextBusyError` → 409 `ABORTED` `CONTEXT_BUSY`; `TaskNotAwaitingInputError` → 400 `INVALID_ARGUMENT` `INVALID_PARAMS`; string-valued `ErrorInfo.metadata`), and pure `accepts_version(header)` (true iff `header.strip()` parses as a `packaging.version.Version` whose release is `(1, 0)` with optional trailing zeros)
    - _Requirements: 1.10, 1.11, 1.12, 1.14, 1.15, 2.14, 11.7, 18.30_

  - [x] 20.8 Implement HostJsonRpcDispatcher
    - Write `a2a_server/dispatch.py` `HostJsonRpcDispatcher(JsonRpcDispatcher)` overriding `_generate_error_response` (streaming path: a `HostContractError` renders as `{"jsonrpc": "2.0", "id": <id>, "error": err.jsonrpc_error()}` before any SSE response starts) and `_create_response` (non-streaming path: replace the SDK's generic error dict when `call_context.state["host_contract_error"]` holds a `HostContractError`); keep the ASGI-wrapper fallback from the design documented in a module docstring
    - _Requirements: 1.14, 1.15, 2.14_

  - [x] 20.9 Implement HostRequestHandler and HostAgentExecutor
    - Write `a2a_server/executor.py` with `HostRequestHandler` (`DefaultRequestHandler` subclass): `_check_version(context)` reads `context.state["headers"]` and raises `VersionNotSupportedError` for a present header that `accepts_version` rejects, leaving an absent header to the SDK; `_admit(params, context)` under a per-Context `asyncio.Lock` reading committed state and runner reservations (busy Context, including `submitted` Tasks and unclaimed reservations → `ContextBusyError`; continuation for a `submitted`/`working` Task → `TaskNotAwaitingInputError`; unknown, retention-deleted, or persisted `context_id` admitted; unknown or terminal `task_id` left to the SDK), storing the error in `context.state["host_contract_error"]`; `on_message_send`/`on_message_send_stream` wrap `super()` in the reservation; `on_cancel_task` checks the Session_Store first (unknown → `TaskNotFoundError`, terminal including recovered → `TaskNotCancelableError`, non-terminal without a runner → error log and `TaskNotCancelableError`); `on_get_task`/`on_list_tasks`/`on_subscribe_to_task` run `_check_version` then `super()`
    - `HostAgentExecutor.execute`: new Task → create runner (claims the reservation), attach, `expect_creation`, enqueue the `SUBMITTED` Task_Snapshot first, await creation (5 s), then `runner.start()` (never started if creation failed); continuation → parse the decision, attach; unrecognized → enqueue the `INPUT_REQUIRED` snapshot then `reprompt_approval()` (final) without a history append; recognized → `transition(WORKING, emit=False)`, enqueue the `WORKING` snapshot, `emit_current_status()`, `approval_gate.resolve(decision)`; then `wait_stream_closing()`; `cancel` attaches and delegates to `runner.cancel`
    - _Requirements: 1.12, 1.13, 1.14, 1.15, 2.1, 2.2, 2.3, 2.7, 2.8, 2.11, 2.12, 2.13, 2.14, 3.1, 3.5, 3.6, 3.7, 3.10, 3.11, 3.12, 3.13, 5.8, 5.11, 11.7, 11.14, 18.6, 18.17, 18.30_

  - [x] 20.10 Write unit tests for executor admission
    - File: `tests/unit/test_executor.py`
    - Test new Task and Context IDs, Context association, unknown and retention-deleted `context_id` creation, a persisted `context_id` resumed, busy Context rejection (including a `submitted` Task and an unclaimed reservation), re-prompt on unrecognized approval without a history append, rejection of messages to `submitted`/`working` Tasks, cancel of a recovered Task returns `TaskNotCancelableError`, creation commit failure starts no runner
    - _Requirements: 2.1, 2.2, 2.3, 2.13, 2.14, 5.11, 11.7, 11.14, 18.17, 18.30_

  - [x] 20.11 Implement app wiring and lifespan
    - Write `app.py` with `Deps` dataclass (config, session store, task store, `Database` or `None` for the memory backend, clock, MCP manager, registry, LLM gateway, runner registry, logger, `ShutdownController`), `build_deps(cfg, database)` selecting `SQLiteSessionStore` + `ProjectedTaskStore` or `InMemorySessionStore` + `ProjectedTaskStore` by `storage.backend`, `build_app(cfg, deps)` (`install_error_mappings()`, `HostRequestHandler(HostAgentExecutor(deps), task_store=deps.task_store, ...)`, agent card routes, `Route("/", HostJsonRpcDispatcher(handler).handle_requests, methods=["POST"])` instead of `create_jsonrpc_routes`, REST routes, `/health`, `LoopbackGuardMiddleware`, `CORSMiddleware` with exact origins outermost relative to auth, `BearerAuthMiddleware`), and lifespan (start `McpManager`; start the retention task every 30 s when `session_idle_timeout_s` is set; on shutdown stop retention, cancel any remaining runners idempotently, call `McpManager.stop(grace=shutdown_controller.grace_s)`, then `Database.close()`)
    - _Requirements: 1.1, 1.2, 6.5, 6.11, 11.10, 11.11, 11.13, 16.2, 16.4, 16.6, 17.14, 17.15, 18.1, 18.2, 18.29_

  - [x] 20.12 Implement BackendServer and the process entry point with the storage startup sequence
    - Write `src/local_agent/server.py` with `BackendServer(uvicorn.Server)` whose `capture_signals()` is a no-op context manager, configured with `loop="none"` and `timeout_graceful_shutdown=2`
    - Write `__main__.py` in this order: load config (print report and exit 2 on `ConfigError`, including invalid UTF-8 and Path_Field errors); configure logging; `classify_platform(gather_facts())` with one `platform_unsupported` warning; for `storage.backend: sqlite` only (memory touches no file, takes no lock, runs no migration or recovery): `storage_directory()` → `ensure_storage_dir` → `restrict_storage` → `StorageLock.acquire` → `Database.open` → `restrict_file` on the database and sidecars → `migrations.apply` → `recovery.run` → one `retention.sweep` when configured, all via `Database.run_sync` before the loop exists; `StorageError`/`PathFieldError`/`PermissionSetupError`/`SchemaError`/`MigrationError` → report in the config-error format naming `storage.path`, the resolved path, and the reason, exit 2; `StorageInUse` → one `storage_in_use` error record ("storage is in use" + database path), exit 1; then `os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")` before importing litellm, apply the function-calling fallback setting, `bind_all(plan)` (log host:port and exit 1 on `BindError`), create `asyncio.Runner(loop_factory=select_loop_factory(...))`, `signals.install(runner.get_loop(), ctl, sys.platform)`, `runner.run(server.serve(sockets=...))` (uvicorn startup runs the lifespan, which starts MCP), then `sys.exit(0)`
    - Make each startup step injectable so the call order is testable with fakes
    - _Requirements: 1.6, 1.8, 10.8, 11.11, 13.3, 13.6, 13.7, 13.11, 13.12, 14.1, 16.8, 16.9, 17.2, 17.13, 17.14, 17.24, 17.28, 17.29, 17.30, 17.31, 17.32, 17.35, 17.38, 18.2, 18.15, 18.19, 18.21, 18.22, 18.23, 18.24, 18.25, 18.26, 18.28_

  - [x] 20.13 Write property test for the A2A error contract
    - File: `tests/property/test_error_contract.py`
    - Use the `httpx.ASGITransport` app with seeded stores and random `context_id`/`task_id` strings, over JSON-RPC (`SendMessage`, `SendStreamingMessage`) and REST; check exact bodies, echoed request `id`, and unchanged state
    - **Property 57: A2A error contract**
    - **Validates: Requirements 1.14, 1.15, 2.14, 11.7**

  - [x] 20.14 Write property test for the A2A-Version gate
    - File: `tests/property/test_version_gate.py`
    - Generate `version_headers()`; check `accepts_version` and that a rejected header yields `-32009` with no Session_Store write (recording store)
    - **Property 58: A2A-Version gate**
    - **Validates: Requirements 1.11, 1.12**

  - [x] 20.15 Write property test for the first stream event
    - File: `tests/property/test_first_event.py`
    - Use the ASGI app, scenario fakes, and an SSE parser for new Tasks, recognized continuations, and unrecognized continuations
    - **Property 59: First stream event is a Task_Snapshot**
    - **Validates: Requirements 3.1, 3.10, 3.11, 3.12, 3.13, 5.11**

  - [x] 20.16 Write property test for the ProtoJSON enum form
    - File: `tests/property/test_protojson.py`
    - Serialize Tasks, Status_Events, Token_Events, and get-task and cancel results built by the Backend's builders with the SDK serializer
    - **Property 60: ProtoJSON enum form**
    - **Validates: Requirements 1.10**

  - [x] 20.17 Write A2A contract tests against the pinned SDK
    - File: `tests/integration/test_a2a_contract.py`
    - `test_error_contract_jsonrpc` and `test_error_contract_rest` (both errors, `SendMessage` and `SendStreamingMessage`, request ID echoed; guards the private dispatcher hooks); `test_protojson_contract` on raw response and SSE bytes for Task (all six states), Status_Event, Token_Event, get-task, cancel, and both contract errors on both bindings; `test_first_event_snapshot`; `test_no_divergent_saves` (fails on any `task_store_divergent_save` record during the suite); `test_storage_failure_on_create` (creation commit fails → `-32603`/500 "Storage could not be written", no Task or Context row)
    - _Requirements: 1.10, 1.14, 1.15, 2.14, 3.10, 3.11, 3.12, 3.13, 18.9, 18.30_

  - [x] 20.18 Write unit tests for auth and CORS wiring
    - File: `tests/unit/test_app_security.py`
    - Test allowed vs disallowed origin headers, empty origin list, loopback without token, agent card exempt from auth, health protected on non-loopback bind
    - _Requirements: 1.2, 16.2, 16.4, 16.6, 16.7_

  - [x] 20.19 Write unit tests for BackendServer and startup ordering
    - File: `tests/unit/test_entrypoint.py`
    - Test that `BackendServer.capture_signals()` installs no signal handlers; `LITELLM_LOCAL_MODEL_COST_MAP` is set before litellm is imported; the recorded call order is dir → restrict → lock → open → migrate → recover → retention → bind → MCP; config errors exit 2 before any storage step; each storage failure (path is a file, cannot create, restrict fails, garbage database, newer schema, migration failure, recovery failure) exits 2 before `bind_all`; a held lock exits 1 before `Database.open`; bind errors exit 1 before any MCP_Server is spawned; `storage.backend: memory` creates no file under a watched temporary Storage_Directory, takes no lock, and runs no migration or recovery (all with injected fakes)
    - _Requirements: 1.8, 13.3, 13.11, 13.12, 16.9, 17.14, 17.28, 18.2, 18.21, 18.22, 18.24, 18.26, 18.28_

- [x] 21. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 22. Write integration and platform-specific tests
  - [x] 22.1 Create integration fixtures
    - Create `tests/fixtures/mcp_echo_server.py` (stdio MCP server built with the `mcp` SDK: paginated tools, echo/slow/error/hang tools, one Protected_Tool, optional crash, non-ASCII echo) and `tests/integration/conftest.py` with a scriptable fake LLM (patched `litellm.acompletion` in-process, plus a local OpenAI-compatible HTTP stub with scripted chunk delays for subprocess tests), an app factory over `httpx.ASGITransport` for both storage backends, a real-uvicorn fixture on a free port, a temporary Storage_Directory per test (`storage.path` always set, and `LOCALAPPDATA`/`HOME`/`XDG_DATA_HOME` pointed at a temporary directory), and a session-scoped finalizer that kills any leftover recorded pid
    - _Requirements: 6.5, 7.1, 10.2, 17.12_

  - [x] 22.2 Create the process-spawning MCP fixture
    - Create `tests/fixtures/mcp_spawning_server.py`: stdio MCP server that spawns a grandchild which spawns a great-grandchild; `get_pids` tool; flags `--cooperative`, `--ignore-graceful`, `--orphan`, `--no-initialize`, and `--spawn-first` (spawns the grandchild as its first instruction)
    - _Requirements: 6.11, 17.10, 17.11, 17.12_

  - [x] 22.3 Write A2A surface integration tests
    - File: `tests/integration/test_a2a_surface.py`
    - Test JSON-RPC send, SSE streaming, REST, get-task, cancel; each JSON-RPC error category; terminal-state and unknown-Task errors; first SSE event within 1 s; stream closes after final status; run on both storage backends
    - _Requirements: 1.1, 1.9, 2.8, 2.10, 2.11, 2.12, 3.1, 3.5, 3.7, 18.29_

  - [x] 22.4 Write startup process tests
    - File: `tests/integration/test_process.py`
    - Run `python -m local_agent` as a subprocess: bind failure exit code 1 naming host:port, accept within 10 s, config error exit 2 before bind, invalid UTF-8 config exit with byte offset, one `platform_unsupported` warning when `oscompat` facts are overridden through the test-only environment hook, `storage.path` pointing at a file exits 2 naming `storage.path`, a garbage database exits 2 with the file unchanged, a newer-schema database exits 2 with the file unchanged
    - _Requirements: 1.6, 1.8, 13.3, 13.12, 17.2, 17.24, 18.22, 18.24_

  - [x] 22.5 Write MCP stdio integration tests
    - File: `tests/integration/test_mcp_stdio.py`
    - Run the real SDK `ClientSession` over `managed_stdio_client` against the echo fixture on every runner: handshake, pagination, `call_tool`, concurrent calls from two Tasks, child kill and reconnection, hanging servers settle within 17 s, `"naïve 日本語 🚀"` round-trips under a non-UTF-8 locale (`LC_ALL=C` on POSIX, code page 1252 on Windows)
    - _Requirements: 6.5, 6.6, 6.7, 6.8, 7.1, 14.3, 17.27_

  - [x] 22.6 Write concurrency and cancellation integration tests
    - File: `tests/integration/test_concurrency.py`
    - Test 10 concurrent Tasks with first status within 500 ms, cancel during slow tool and during `input-required` within 2 s, failure isolation across Contexts, concurrent tool turn under slowest + 500 ms; repeat the 10-Task streaming case on the SQLite backend with no slow-callback warnings allowed; run with asyncio debug and `slow_callback_duration=0.1`
    - _Requirements: 2.7, 8.3, 14.1, 14.2, 14.4, 14.7, 18.14, 18.29_

  - [x] 22.7 Write SSE disconnect integration test
    - File: `tests/integration/test_disconnect.py`
    - Drop the SSE stream mid-response on a real uvicorn socket; get-task later shows `completed` with the full artifact (`test_disconnect_continues`)
    - _Requirements: 3.6, 18.29_

  - [x] 22.8 Write health integration test
    - File: `tests/integration/test_health.py`
    - Health responds within 1 s with all servers listed while the fake LLM is unreachable
    - _Requirements: 15.8, 15.9_

  - [x] 22.9 Write opt-in function-calling fallback test
    - File: `tests/integration/test_llm_fallback.py`, marked `@pytest.mark.llm`, against a local Ollama
    - Prompt-based tool calls come back in the assembled Tool_Call format
    - _Requirements: 10.8_

  - [x] 22.10 Write process-tree cleanup integration test
    - File: `tests/integration/test_process_tree.py` (no marker; OS-specific assertions inside)
    - Parametrize `--cooperative` / `--ignore-graceful` / `--orphan` × `McpManager.stop()` / reconnection after killing only the server pid / failed handshake (`--no-initialize`); assert every recorded pid is gone within `1 + grace + 1.5` s (POSIX `os.kill(pid, 0)`; Windows `OpenProcess` + `WaitForSingleObject`) and `TerminationReport.forced` matches the behaviour
    - _Requirements: 6.7, 6.8, 6.11, 17.11, 17.12_

  - [x] 22.11 Write Windows-only process tests
    - File: `tests/integration/test_windows_process.py`, marked `windows_only`, with a `tests/fixtures/echo_args.cmd` batch fixture that forwards `%*` to a CRT program
    - `test_ctrl_break_reaches_tree`, `test_job_assignment_before_resume` (`--spawn-first`), `test_taskkill_fallback` (job creation forced to fail), `test_batch_args_literal` (`a&b`, `x|y`, `<z>`, `c^d`, `sp ace`, `trail\` literal; `%` and `"` rejected), `test_npx_resolution` (via child `PATHEXT`), `test_env_case_merge` (configured `Path` replaces inherited `PATH`), and a check that `winargv_ref` matches `python.exe -c "import sys, json; print(json.dumps(sys.argv[1:]))"` for a fixed corpus
    - _Requirements: 17.4, 17.7, 17.8, 17.9, 17.10, 17.11, 17.12, 17.22_

  - [x] 22.12 Write POSIX-only process and loop tests
    - File: `tests/integration/test_posix_process.py`, marked `posix_only`
    - `test_new_session_isolation` (child `getsid()` differs; `SIGINT` to the Backend's group does not reach the child), `test_sigterm_ignoring_child` (`SIGKILL` after 5 s ± 0.5 s), `test_uvloop_selected` (skipped when uvloop is not installed)
    - _Requirements: 17.10, 17.11, 17.29_

  - [x] 22.13 Write socket bind integration tests
    - File: `tests/integration/test_bind.py` (per-OS expectations)
    - `test_exclusive_bind_conflict` (second listener on the same port exits 1 naming host:port), `test_localhost_dual_stack` (binds `127.0.0.1` and `::1`, or warns once; `::1` assertions skipped when `socket.has_ipv6` is false)
    - _Requirements: 1.8, 17.30, 17.31, 17.32_

  - [x] 22.14 Write shutdown and signal integration tests
    - File: `tests/integration/test_shutdown.py`
    - `test_shutdown_exit_code`: POSIX sends `SIGTERM` and `SIGINT`, Windows sends `CTRL_BREAK_EVENT` to a Backend started with `CREATE_NEW_PROCESS_GROUP`; assert shutdown starts ≤ 1 s, active Tasks end `canceled` (and are stored as `canceled`), MCP trees are gone, exit status 0, and a second signal still exits 0
    - `test_console_close`, marked `windows_only` and `@pytest.mark.manual`: trees gone within the 2 s grace
    - _Requirements: 6.11, 17.12, 17.13, 17.14, 17.15_

  - [x] 22.15 Write property test for cross-platform transcript equivalence
    - File: `tests/property/test_transcript_equivalence.py` with `@settings(derandomize=True, database=None)`, using the scenario fakes and normalizer; write the digest to `artifacts/transcripts-<os>.sha256`
    - **Property 46: Cross-platform transcript equivalence**
    - **Validates: Requirements 17.3**

  - [x] 22.16 Write the two-process Storage_Lock conflict test
    - File: `tests/integration/test_storage_lock.py` (`test_storage_lock_conflict`, no marker)
    - Start two real `python -m local_agent` processes on one Storage_Directory with different ports: the second exits 1 within 2 s with a `storage_in_use` record containing `storage is in use` and the database path, never opens the database (file unchanged, no bind), and the first keeps serving and its Tasks continue
    - _Requirements: 18.25, 18.26_

  - [x] 22.17 Write the crash and restart integration test
    - File: `tests/integration/test_crash_restart.py` (`test_crash_restart`, no marker)
    - For 20 seeded iterations: start one or two streaming Tasks against the fake LLM HTTP stub and the echo fixture, drive them mid-token-stream, mid-tool-call, or into `input-required`; kill hard (`os.kill(pid, SIGKILL)` on POSIX, `Popen.kill()` / `TerminateProcess` on Windows) and wait for the process handle; restart on the same Storage_Directory and assert startup without manual cleanup, interrupted Tasks `failed` with the exact recovery message, earlier terminal Tasks unchanged, persisted artifact text a prefix of the received text, cancel on a recovered Task returns the not-cancelable error, no LLM or MCP request before the first client request, and a new message with the old `context_id` sends the persisted history plus the new message
    - _Requirements: 18.6, 18.7, 18.9, 18.12, 18.15, 18.16, 18.17, 18.18, 18.27_

  - [x] 22.18 Write per-OS storage permission tests
    - File: `tests/integration/test_storage_permissions.py`
    - `test_storage_modes_posix` (`posix_only`; under `umask 000` and with pre-existing `0644`/`0755` files, after startup the directory is `0700` and the database, `-wal`, `-shm`, and lock files are `0600`); `test_storage_dacl_windows` (`windows_only`; protected DACL with exactly one ACE for the current user SID, `OI|CI` on the directory, `-wal`/`-shm` created later by SQLite inherit it; also against a pre-existing directory with inherited `Users` ACEs); `test_storage_restrict_failure` (per-OS injection; exit 2 naming `storage.path`, the path, and the OS reason; no MCP_Server spawned, port not bound)
    - _Requirements: 16.8, 16.9, 17.36, 17.37, 18.28_

  - [x] 22.19 Write the A2A-Version header integration test
    - File: `tests/integration/test_version_header.py` (`test_version_header`, real SDK)
    - `1.0` and `1.0.0` accepted; `1.1`, `2.0`, `abc` rejected with `-32009` and no Task; a **missing** header asserted against the recorded behaviour of the pinned SDK (observed on 1.2.x: read as `0.3` → `-32009`), written into the test as the expected value so an SDK upgrade that changes it fails loudly
    - _Requirements: 1.11, 1.12, 1.13_

  - [x] 22.20 Write the restart-without-crash integration test
    - File: `tests/integration/test_restart.py`
    - Graceful_Shutdown, then restart on the same Storage_Directory: GetTask returns completed and canceled Tasks with persisted state, status message, and artifacts in emission order; a new message without `task_id` and with the old `context_id` continues the persisted conversation (the first LLM request holds the persisted history plus the new message); retention at startup deletes a Context whose Last_Activity was set in the past and keeps a Context with an Active_Task; get-task for a deleted Context's Task returns `TaskNotFoundError`
    - _Requirements: 2.13, 3.6, 11.11, 11.12, 11.14, 18.6, 18.7_

- [x] 23. Add the OS-matrix CI workflow
  - [x] 23.1 Create `.github/workflows/ci.yml`
    - `test` job on `windows-latest`, `macos-latest`, `macos-15-intel`, `ubuntu-latest`, `ubuntu-24.04-arm` with `fail-fast: false`: checkout, `astral-sh/setup-uv`, `uv python install 3.14` (standard build, never `3.14t`), `uv sync --locked --no-build`, `uv run python -c "import sqlite3, a2a.server.request_handlers, a2a.server.tasks; print(sqlite3.sqlite_version)"` (logs the SQLite version and checks the SDK imports without SQLAlchemy), `ruff check .`, `ruff format --check .`, `mypy src`, `pytest tests/unit tests/property tests/integration -m "not manual and not llm"` with `HYPOTHESIS_PROFILE=ci` and the `DeprecationWarning` error filters, upload `artifacts/transcripts-<os>.sha256`
    - Windows-only step before tests: `uv run python -c "import litellm"` with `HTTPS_PROXY`/`HTTP_PROXY` set to `http://127.0.0.1:9`
    - `equivalence` job (`needs: test`) downloads the five digests and fails if they differ; the workflow fails if any job fails
    - _Requirements: 17.3, 17.33, 17.34, 18.24_

- [x] 24. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; integration fixtures (22.1, 22.2), scenario fakes (17.9), the storage fault-injection helper (10.2), and the migration fixture databases (10.7) are only needed by the optional tests
- Each task references specific requirements for traceability; each of the design's 61 correctness properties has exactly one property-test sub-task (P33–P48 cover Requirement 17; P49–P56 and P61 cover Requirement 18 storage and its security; P57–P60 cover the A2A interface contract)
- Storage property tests (P21, P49–P55, P61) and the scenario properties (P3, P12, P13, P14, P23) run against both `InMemorySessionStore` and `SQLiteSessionStore` where the design says so; each example uses its own `TemporaryDirectory`, and text strategies exclude lone surrogates, which ingress replaces with U+FFFD
- P49 (11.3) uses `TaskRunner._build_request` and P53 (11.6) uses the kill-injection helper from 17.9, so the dependency graph schedules them after those tasks even though they sit next to the storage code they test
- `src/local_agent/models.py`, `a2a_server/a2a_compat.py`, `a2a_server/errors.py`, `a2a_server/dispatch.py`, `mcp_client/mcp_compat.py`, and `src/local_agent/server.py` are additions to Appendix A that the design implies (shared runtime types, SDK name isolation, the A2A error contract, the dispatcher subclass, and the `BackendServer` subclass); `oscompat/env.py`, `oscompat/net.py`, `oscompat/_win32.py`, `oscompat/ctrl_break.py`, and `mcp_client/stdio_transport.py` are the design's own additions; `storage/task_store.py` is the `ProjectedTaskStore` adapter chosen in the design (not the SDK `DatabaseTaskStore`)
- At implementation time, verify the pinned `a2a-sdk` (1.2.x), `mcp` (2.3.x), and `uvicorn` (>= 0.36, `capture_signals` override) against `a2a_compat.py`, `mcp_compat.py`, `stdio_transport.py`, and `server.py`; for `a2a-sdk` also re-check `JsonRpcDispatcher._generate_error_response`/`_create_response`, `A2A_ERROR_MAPPING`/`A2A_ERROR_REASONS`, missing-header version handling, the save-before-forward order, and continuation-snapshot "already exists" handling (guarded by 20.17 and 22.19); confirm the `macos-15-intel` runner label
- Platform-specific tests use the `windows_only`, `posix_only`, `macos_only`, and `linux_only` markers from `tests/conftest.py`; pure OS-rule tests take `platform` as a parameter and run on every runner; integration tests never touch the developer's real per-user Storage_Directory
- Run tests with `uv run pytest -q`, integration tests with `uv run pytest -m integration`, and lint/type checks with `uv run ruff check` and `uv run mypy src`; these work the same in PowerShell, bash, and zsh

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "2.3", "2.4", "2.7", "2.11", "5.1"] },
    { "id": 2, "tasks": ["2.2", "2.5", "2.6", "2.8", "2.9", "2.12", "5.2", "6.1", "6.3", "6.5", "6.7", "6.12", "8.1", "8.8", "13.1"] },
    { "id": 3, "tasks": ["2.10", "2.13", "3.1", "5.3", "5.4", "5.5", "5.6", "6.2", "6.4", "6.6", "6.8", "6.9", "8.2", "8.3", "8.9", "9.1", "10.4"] },
    { "id": 4, "tasks": ["3.2", "4.1", "6.10", "6.11", "8.4", "8.6", "9.2", "10.5", "20.1", "20.3", "20.5"] },
    { "id": 5, "tasks": ["3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "4.2", "4.3", "4.4", "5.7", "8.5", "8.7", "9.3", "10.1", "13.2", "15.1", "17.1", "20.2", "20.4", "20.6", "20.7"] },
    { "id": 6, "tasks": ["5.8", "5.9", "10.2", "10.6", "13.3", "13.4", "13.5", "15.2", "15.3", "17.2", "17.4", "20.8"] },
    { "id": 7, "tasks": ["10.3", "10.9", "11.1", "11.2", "13.6", "14.1", "17.5"] },
    { "id": 8, "tasks": ["10.7", "10.8", "11.4", "11.5", "11.7", "14.2", "14.3", "17.3"] },
    { "id": 9, "tasks": ["11.8", "11.9", "14.4", "17.6"] },
    { "id": 10, "tasks": ["17.7", "17.8"] },
    { "id": 11, "tasks": ["11.3", "17.9", "17.20", "19.1", "20.9"] },
    { "id": 12, "tasks": ["11.6", "17.10", "17.11", "17.12", "17.13", "17.14", "17.15", "17.16", "17.17", "17.18", "17.19", "19.2", "19.3", "20.10", "20.11"] },
    { "id": 13, "tasks": ["20.12", "20.13", "20.14", "20.15", "20.16", "20.17", "20.18"] },
    { "id": 14, "tasks": ["20.19", "22.1", "22.2", "23.1"] },
    { "id": 15, "tasks": ["22.3", "22.4", "22.5", "22.6", "22.7", "22.8", "22.9", "22.10", "22.11", "22.12", "22.13", "22.14", "22.15", "22.16", "22.17", "22.18", "22.19", "22.20"] }
  ]
}
```
