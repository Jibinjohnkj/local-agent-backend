# Implementation Plan: Local Agent Backend

## Overview

The Backend is built in Python 3.14+ (CPython standard GIL build) and runs on Windows, macOS, and Linux (Requirement 17). It follows the module layout in Appendix A of the requirements (`src/local_agent/...`), the stack in Appendix B, and the design: `a2a-sdk` 1.2.x, `mcp` 2.3.x, `litellm`, `pydantic>=2.12`/`pydantic-settings`, `structlog`, `jsonschema`, `PyYAML>=6.0.3`, `uvicorn[standard]>=0.36`/`starlette`, `uvloop>=0.22` on macOS and Linux, with `pytest`, `pytest-asyncio`, and `hypothesis` for tests. The order is incremental:

1. Project scaffolding, shared data models, and the pure `oscompat` foundations that configuration depends on (platform check, Path_Field resolution, environment-name rules, loop factory).
2. Pure cores and their property tests: config (byte decoding, BOM, CRLF, Path_Fields), redactor and UTF-8/LF log sink, naming and schema translation, pattern matching, approval parsing, stream assembler, result formatter, context window trimming, prompt builder, transition table.
3. Pure OS decision cores: executable resolution, Windows command-line and batch quoting, spawn specs, bind planning.
4. I/O shells: session store, process spawning and tree termination, own stdio transport, MCP manager, LLM gateway, emitter, approval gate, tool dispatcher, task runner, shutdown controller and signals, A2A executor and handler, middleware, health, app wiring, `BackendServer`, entry point.
5. Integration tests with a fake LLM and fake/real stdio MCP servers, platform-specific process, signal, and bind tests, cross-platform transcript equivalence, and the OS-matrix CI workflow.

Each property test is one Hypothesis test with at least 100 examples, tagged `# Feature: local-agent-backend, Property N: <title>`. OS-parameterized pure functions take `platform` as an argument, so their property tests run on every runner. Commands use `uv` and work in any shell (for example `uv run pytest -q`).

## Tasks

- [ ] 1. Set up project scaffolding and shared models
  - [ ] 1.1 Create project structure, dependencies, and test configuration
    - Create `pyproject.toml` with `requires-python = ">=3.14"`; runtime deps from Appendix B with the floors `pydantic>=2.12`, `pyyaml>=6.0.3`, `uvicorn[standard]>=0.36`, `uvloop>=0.22; sys_platform != 'win32'`, and `litellm` pinned to a release whose Windows wheel imports offline; dev deps `pytest`, `pytest-asyncio`, `hypothesis`, `respx`, `ruff`, `mypy`; pytest markers `integration`, `llm`, and `manual`; pytest `filterwarnings` turning `DeprecationWarning` from `asyncio` and `local_agent` into errors; ruff config; `[tool.mypy] python_version = "3.14"` with `strict = true`
    - Add `[tool.uv] required-environments` listing the five Supported_Platform markers (Windows x86_64, macOS arm64, macOS x86_64, Linux x86_64 glibc, Linux aarch64 glibc), then generate `uv.lock` with pinned versions so locking fails if any package lacks a wheel
    - Create `.gitattributes` with `* text=auto eol=lf` and `tests/fixtures/** -text`
    - Create the package skeleton under `src/local_agent/` (`config/`, `a2a_server/`, `orchestrator/`, `llm/`, `mcp_client/`, `session/`, `oscompat/` with `__init__.py` files) and `tests/{unit,property,integration,fixtures}/`
    - Create `tests/conftest.py` registering a Hypothesis profile `ci` (`max_examples=200`, `deadline=None`, `suppress_health_check=[HealthCheck.too_slow]`), an injectable fake `Clock`/`sleep`, and the platform markers `windows_only`, `posix_only`, `macos_only`, `linux_only` from the design
    - Create `config.example.yaml` matching the design's example config (including `cwd: ~/mcp` and the `npx` → `npx.cmd` note), with a note that wildcard binds need a reachable hostname for remote clients
    - _Requirements: 1.1, 13.1, 13.4, 14.1, 17.1, 17.25, 17.28, 17.29, 17.33_

  - [ ] 1.2 Define shared runtime data models
    - Create `src/local_agent/models.py` with `ToolCall`, `AssembledResponse`, `SystemMessage`/`UserMessage`/`AssistantMessage`/`ToolMessage` (with `to_openai()`), `ChatMessage`, `ContextRecord`, `TaskRecord`, `McpToolInfo`, `RegisteredTool`, `ToolResult` (with `as_message()`), `ServerStatus`, and content-item types, as specified in the design's Runtime models; use frozen dataclasses and no `from __future__ import annotations`
    - _Requirements: 8.4, 11.1, 11.2, 11.5_

- [ ] 2. Implement oscompat foundations
  - [ ] 2.1 Implement platform detection
    - Write `oscompat/__init__.py` with `PlatformFacts`, `PlatformVerdict`, pure `classify_platform(facts)` (CPython 3.14 GIL build; Windows build >= 19045 on native x86_64, macOS >= 14 on arm64/x86_64, Linux glibc >= 2.28 on x86_64/arm64; `aarch64` normalized to `arm64`; musl and `Py_GIL_DISABLED == 1` unsupported), and `gather_facts()` (native machine via `IsWow64Process2` on Windows)
    - _Requirements: 17.1, 17.2_

  - [ ]* 2.2 Write property test for platform classification
    - File: `tests/property/test_platform.py`
    - **Property 33: Platform classification**
    - **Validates: Requirements 17.1, 17.2**

  - [ ] 2.3 Implement Path_Field resolution
    - Write `oscompat/paths.py` with `PathFieldError`, pure `has_separator(value, platform)` and `resolve_path_field(value, *, base_dir, home, platform)` using `ntpath`/`posixpath` by `platform` (never the host module): `/` everywhere and `\` on Windows only, `~`/`~/`/`~\` expansion (no `~user`), missing home raises, relative joined to `base_dir`, lexical normalization, UNC and drive-absolute accepted, root-relative takes the base drive, drive-relative rejected
    - _Requirements: 13.1, 17.16, 17.17, 17.18_

  - [ ]* 2.4 Write property test for Path_Field resolution
    - File: `tests/property/test_paths.py`
    - **Property 34: Path_Field resolution**
    - **Validates: Requirements 17.16, 17.17, 17.18, 13.1**

  - [ ] 2.5 Implement environment-name rules
    - Write `oscompat/env.py` with pure `env_key`, `lookup_env` (ordinal upper-case match on Windows, exact match elsewhere), `merge_env` (configured names replace every inherited entry with the same `env_key`, keeping configured spelling), and `inherited_env` (SDK `DEFAULT_INHERITED_ENV_VARS` plus `COMSPEC`, `WINDIR`, `PROGRAMDATA`, `PROGRAMFILES`, `PROGRAMFILES(X86)` on Windows, as a module constant)
    - _Requirements: 17.21, 17.22, 13.2_

  - [ ]* 2.6 Write property test for environment-variable name matching
    - File: `tests/property/test_env_lookup.py`
    - **Property 37: Environment-variable name matching**
    - **Validates: Requirements 17.21, 13.2**

  - [ ]* 2.7 Write property test for subprocess environment merge
    - File: `tests/property/test_env_merge.py`
    - **Property 38: Subprocess environment merge**
    - **Validates: Requirements 17.22**

  - [ ] 2.8 Implement event loop factory selection
    - Write `oscompat/event_loop.py` with pure `select_loop_factory(platform, uvloop_new_event_loop)` returning `uvloop.new_event_loop` on macOS/Linux when available and `asyncio.EventLoop` otherwise, plus a wrapper that imports `uvloop` only when `sys.platform != "win32"`; use no event-loop policy API
    - _Requirements: 17.28, 17.29, 14.1_

  - [ ]* 2.9 Write unit tests for loop selection and deprecated API usage
    - File: `tests/unit/test_event_loop.py`
    - Test `select_loop_factory` for all six platform × uvloop combinations; scan `src/` for the deprecated policy names (`set_event_loop_policy`, `get_event_loop_policy`, `WindowsSelectorEventLoopPolicy`, `WindowsProactorEventLoopPolicy`, `DefaultEventLoopPolicy`, `AbstractEventLoopPolicy`, `asyncio.iscoroutinefunction`) and assert none are present
    - _Requirements: 17.28, 17.29_

- [ ] 3. Implement configuration loading
  - [ ] 3.1 Implement Backend_Config pydantic models
    - Write `config/models.py` with `ServerConfig`, `LLMConfig`, `AgentConfig`, `ApprovalKeywords` (non-empty, disjoint after casefold), `OrchestrationConfig`, `StdioServerConfig` (`command`, `args` never path-processed, optional `cwd` stored as `None` when omitted, `env`), `SseServerConfig` (discriminated on `transport`), `LoggingConfig`, and `BackendConfig` (`extra="forbid"`, `frozen=True`), with all defaults and ranges from the design
    - Use `SecretStr` for `llm.api_key`, `server.bearer_token`, MCP `env` values, and MCP `headers` values
    - _Requirements: 1.7, 5.1, 6.1, 6.2, 6.3, 8.9, 10.5, 11.8, 11.11, 13.4, 15.7, 16.1, 16.6, 17.19, 17.20_

  - [ ] 3.2 Implement config loader
    - Write `config/loader.py` with `ConfigIssue`, `ConfigError`, `resolve_config_path` (`AGENT_CONFIG` resolved as a Path_Field against the process cwd, or `./config.yaml`, no fallback), `decode_config_bytes` (strip one UTF-8 BOM, strict UTF-8 decode independent of locale, invalid UTF-8 reported with the file path and byte offset counted from the file start including the BOM), CRLF→LF normalization before parsing, `substitute_env` (`${NAME}` replacement via `oscompat.env.lookup_env`, `$${` escape, missing-variable issues with dotted paths), `load_config` (reads bytes before the loop exists), `load_config_from_text`, and `dump_config`
    - Resolve Path_Fields (`mcp_servers[*].cwd`, and `mcp_servers[*].command` when it contains a separator) after substitution and before validation with `resolve_path_field(base_dir=<config dir>)`; report `PathFieldError` as a `ConfigIssue` at the field's dotted path; leave bare command names and `args` untouched
    - Parse with `yaml.safe_load`; report 1-based line/column for `MarkedYAMLError`; reject non-mapping top level
    - Run pydantic validation and the raw-tree cross-checks (duplicate server names, permission server references and pattern lengths, non-loopback bind needing a >=32-character bearer token) so all issues are reported in one `ConfigError`; never include secret or substituted values in messages
    - _Requirements: 1.7, 5.3, 6.4, 13.1, 13.2, 13.3, 13.5, 13.6, 13.7, 13.8, 16.5, 17.16, 17.17, 17.18, 17.20, 17.21, 17.23, 17.24, 17.25_

  - [ ]* 3.3 Write property test for environment substitution
    - File: `tests/property/test_config_substitution.py`
    - **Property 24: Environment substitution**
    - **Validates: Requirements 13.2**

  - [ ]* 3.4 Write property test for missing environment variable errors
    - File: `tests/property/test_config_missing_env.py`
    - **Property 25: Missing environment variable error**
    - **Validates: Requirements 13.8**

  - [ ]* 3.5 Write property test for configuration round-trip
    - File: `tests/property/test_config_roundtrip.py`
    - **Property 26: Configuration round-trip**
    - **Validates: Requirements 13.5**

  - [ ]* 3.6 Write property test for complete validation report
    - File: `tests/property/test_config_validation.py`
    - **Property 27: Complete validation report**
    - **Validates: Requirements 1.7, 5.3, 6.1, 6.4, 13.3, 16.5**

  - [ ]* 3.7 Write property test for line-ending and BOM invariance
    - File: `tests/property/test_config_line_endings.py`
    - Generate CRLF/LF/BOM variants in memory, not from checked-out fixtures
    - **Property 35: Line-ending and BOM invariance**
    - **Validates: Requirements 17.23, 17.25, 13.7**

  - [ ]* 3.8 Write property test for invalid UTF-8 reporting
    - File: `tests/property/test_config_invalid_utf8.py`
    - **Property 36: Invalid UTF-8 reporting**
    - **Validates: Requirements 17.24**

  - [ ]* 3.9 Write unit tests for config loading
    - File: `tests/unit/test_config.py`
    - Test path resolution (including `AGENT_CONFIG` relative to cwd), defaults (Max_Tool_Iterations 10, Tool_Timeout 60, host `127.0.0.1`, empty CORS list), missing file without fallback, YAML line/column, stdio and SSE fields including `ftp://` rejection, omitted `cwd` stays `None`, drive-relative `C:tools\x.exe` error message at the field path, `~` with no home variable
    - _Requirements: 6.2, 6.3, 13.1, 13.4, 13.6, 13.7, 16.1, 16.6, 17.17, 17.18, 17.19_

- [ ] 4. Implement structured logging and secret redaction
  - [ ] 4.1 Implement `logging_setup.py`
    - Write `Redactor` (4-character window masking, short secrets matched whole, collapsed runs into one placeholder), `collect_secrets(cfg)`, `render_line(event) -> bytes` (`json.dumps(..., ensure_ascii=False, separators=(",", ":"))` encoded UTF-8 with `backslashreplace` plus exactly one `b"\n"`), `configure_logging` (structlog chain: contextvars, level, UTC millisecond timestamp, recursive redact processor, `render_line`, `make_filtering_bound_logger(level)`, `BytesLoggerFactory(sys.stderr.buffer)`), and `get_logger(component)`
    - _Requirements: 15.1, 15.5, 15.7, 17.26_

  - [ ]* 4.2 Write property test for secret redaction
    - File: `tests/property/test_redaction.py`
    - **Property 28: Secret redaction**
    - **Validates: Requirements 15.5, 9.7, 2.6**

  - [ ]* 4.3 Write property test for log line encoding
    - File: `tests/property/test_log_encoding.py`
    - **Property 47: Log line encoding**
    - **Validates: Requirements 17.26, 15.1**

  - [ ]* 4.4 Write unit tests for log format and level filtering
    - File: `tests/unit/test_logging.py`
    - Test single-line JSON with timestamp, level, component, event, `context_id`, `task_id`; `null` IDs outside a Task; level filtering and `info` default; output is bytes with LF only
    - _Requirements: 15.1, 15.7, 17.26_

- [ ] 5. Implement tool naming, schema translation, and result formatting
  - [ ] 5.1 Implement schema translator
    - Write `mcp_client/schema_translator.py` `translate(server, tool, llm_name)` returning an OpenAI function definition or `SchemaRejected` (meta-schema check via `validator_for(...).check_schema`, non-object `type` rejected, missing `type` defaulted to `object` with `properties` `{}`)
    - _Requirements: 7.2, 7.5, 7.6, 7.8_

  - [ ] 5.2 Implement Tool_Registry with naming and permission matching
    - Write `mcp_client/registry.py` with `sanitize`, `assign_names` (collision suffixes `_2`, `_3`, ... within 64 characters), `compile_pattern` (`*`/`?` only, everything else escaped, `fullmatch`), `is_protected`, `ToolRegistry.replace_server`/`remove_server` (rebuild names in config order, atomic snapshot swap, schema-rejection warnings), and immutable `RegistrySnapshot` (`resolve`, `openai_tools` for available servers only, `is_protected`, `tools_by_server`)
    - _Requirements: 5.1, 5.2, 7.3, 7.4, 7.7, 7.8, 7.9, 7.12_

  - [ ]* 5.3 Write property test for schema translation
    - File: `tests/property/test_schema_translator.py`
    - **Property 7: Schema translation preserves the input schema**
    - **Validates: Requirements 7.2, 7.5, 7.6, 7.8**

  - [ ]* 5.4 Write property test for LLM-facing names
    - File: `tests/property/test_registry_names.py`
    - **Property 5: LLM-facing name format, uniqueness, and determinism**
    - **Validates: Requirements 7.3, 7.7, 7.12**

  - [ ]* 5.5 Write property test for permission pattern matching
    - File: `tests/property/test_permission_patterns.py`
    - **Property 9: Permission pattern matching**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 5.6 Write property test for available-only tool exposure
    - File: `tests/property/test_registry_availability.py`
    - **Property 8: Only available servers' tools are exposed**
    - **Validates: Requirements 7.9**

  - [ ] 5.7 Implement result formatter
    - Write `mcp_client/result_formatter.py` with `content_to_text` (newline join, non-text descriptions with type, MIME type, URI; empty list text), `truncate` (notice with original length and limit), `timeout_text`, and `exception_text` (redacted `"<Type>: <message>"`)
    - _Requirements: 8.6, 8.9, 8.11, 9.1, 9.2, 9.7_

  - [ ]* 5.8 Write property test for MCP content conversion
    - File: `tests/property/test_content_to_text.py`
    - **Property 16: MCP content conversion**
    - **Validates: Requirements 8.6, 8.11**

  - [ ]* 5.9 Write property test for tool output truncation
    - File: `tests/property/test_truncate.py`
    - **Property 17: Tool output truncation**
    - **Validates: Requirements 8.9, 9.1**

- [ ] 6. Implement orchestration pure cores
  - [ ] 6.1 Implement approval decision parsing
    - Create `orchestrator/approval_gate.py` with `Decision` enum and `parse_approval_decision(message, keywords)` (first valid structured `decision` data part wins, else trimmed casefolded text keyword match, else `None`)
    - _Requirements: 5.12_

  - [ ]* 6.2 Write property test for approval decision parsing
    - File: `tests/property/test_approval_parsing.py`
    - **Property 10: Approval decision parsing**
    - **Validates: Requirements 5.12**

  - [ ] 6.3 Implement stream assembler
    - Write `llm/stream_assembler.py` `StreamAssembler.feed` (returns non-empty text, groups tool-call fragments by index: first id, first name, concatenated arguments) and `finish(existing_ids)` (ascending index order, `call_<uuid4 hex>` for missing IDs, argument text unchanged)
    - _Requirements: 3.2, 10.3, 10.11_

  - [ ]* 6.4 Write property test for tool-call fragment assembly
    - File: `tests/property/test_stream_assembler.py`
    - **Property 18: Tool-call fragment assembly**
    - **Validates: Requirements 10.3, 10.11**

  - [ ] 6.5 Implement context window trimming
    - Write `orchestrator/context_window.py` with `split_turns`, `trim_to_budget` (drop oldest whole turns, keep system prompt and last turn, raise `TurnTooLarge`, never mutate input), and `estimate_tokens` (`litellm.token_counter` via `asyncio.to_thread`, `ceil(chars/4)` fallback when tokenization fails, including a missing tiktoken cache; tool definitions counted)
    - _Requirements: 11.8, 11.9, 14.1_

  - [ ]* 6.6 Write property test for context window trimming
    - File: `tests/property/test_context_window.py`
    - **Property 20: Context window trimming**
    - **Validates: Requirements 11.8, 11.9**

  - [ ] 6.7 Implement prompt builder
    - Write `orchestrator/prompt_builder.py` with `DEFAULT_BASE_INSTRUCTIONS` and `build_system_prompt(base, tools)`: base or default first, exactly one `## Tool usage` section with the three required instructions, tools grouped under `### <server>` headings with description or "(no description provided)", or "No tools are currently available."
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.8, 12.9_

  - [ ]* 6.8 Write property test for system prompt structure
    - File: `tests/property/test_prompt_builder.py`
    - **Property 22: System prompt structure**
    - **Validates: Requirements 12.1, 12.2, 12.4, 12.8, 12.9**

  - [ ] 6.9 Implement Tool_Call preparation and status messages
    - Create `orchestrator/tool_dispatch.py` with `Prepared`, `Rejected`, `prepare_call` (unknown name, JSON parse / non-object, `jsonschema` errors with JSON path and keyword sorted by path, unavailable server, then `Prepared(protected=...)`), `denied_result`, and `start_status_message` (first path/file/uri/url string argument, truncated to 120 characters)
    - _Requirements: 4.2, 5.5, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 6.10 Write property test for tool start status message
    - File: `tests/property/test_start_status_message.py`
    - **Property 11: Tool start status message**
    - **Validates: Requirements 4.2**

  - [ ]* 6.11 Write property test for rejected Tool_Calls
    - File: `tests/property/test_prepare_call.py`
    - **Property 15: Rejected Tool_Calls are never sent**
    - **Validates: Requirements 9.3, 9.4, 9.5, 5.5**

  - [ ] 6.12 Implement Task_State transition table
    - Create `a2a_server/events.py` with the internal `TaskState` mapping to the pinned `a2a-sdk` enum names (isolated here), the `ALLOWED` transition set, `TERMINAL_STATES`, `STREAM_CLOSING_STATES`, and pure `is_allowed(current, new)`
    - _Requirements: 2.9, 3.5_

- [ ] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement OS decision cores for processes and sockets
  - [ ] 8.1 Implement executable resolution
    - Write `oscompat/process.py` with `Resolved`, `NotFound`, and pure `resolve_command(command, *, platform, path_value, pathext_value, base_dir, home, is_file)`: separator commands resolved as Path_Fields without `PATH`; otherwise `PATH` entries in order (empty skipped, quotes stripped on Windows, relative joined to the spawn cwd, no implicit current directory); on Windows `PATHEXT` extensions in order within each directory when the suffix is not in `PATHEXT` (default `.COM;.EXE;.BAT;.CMD`); `.cmd`/`.bat` → `batch`, `.exe`/`.com` → `native`, other suffixes such as `.ps1` → `NotFound` "not a launchable executable"; `NotFound` names the command and searched directories
    - _Requirements: 17.4, 17.5, 17.6_

  - [ ]* 8.2 Write property test for executable resolution
    - File: `tests/property/test_resolve_command.py`
    - Use an in-memory file set as the `is_file` oracle and record every probe
    - **Property 39: Executable resolution**
    - **Validates: Requirements 17.4, 17.5, 17.6**

  - [ ] 8.3 Implement Windows command-line construction and batch quoting
    - Extend `oscompat/process.py` with `BatchArgRejected(position, char)`, pure `quote_batch_args(batch_path, args)` (reject `"`, `%`, CR, LF in the path or any argument with the 1-based position of the first offender; double-quote the path and every argument; double trailing backslashes before the closing quote; empty → `""`), and pure `build_command_line(resolved, args, system_root)` (`list2cmdline` for native; `%SystemRoot%\System32\cmd.exe /d /e:ON /v:OFF /c "<quoted>"` for batch)
    - _Requirements: 17.7, 17.8, 17.9_

  - [ ]* 8.4 Write property test for Windows argument fidelity
    - Files: `tests/property/winargv_ref.py` (reference Microsoft C runtime / `CommandLineToArgvW` parser and cmd.exe quote-state scanner) and `tests/property/test_win_argv.py`
    - **Property 40: Windows argument fidelity for native executables**
    - **Validates: Requirements 17.7**

  - [ ]* 8.5 Write property test for batch-file quoting and rejection
    - File: `tests/property/test_batch_quoting.py` (uses `winargv_ref.py`)
    - **Property 41: Batch-file quoting and rejection**
    - **Validates: Requirements 17.8, 17.9**

  - [ ] 8.6 Implement spawn specification and termination plan
    - Extend `oscompat/process.py` with `SpawnSpec`, `SpawnRejected`, async `build_spawn_spec(cfg, config_dir, platform)` (file probes via `asyncio.to_thread`; `cwd` = resolved configured directory or config directory; `env` = `merge_env(inherited_env(os.environ), cfg.env)` with `PATH`/`PATHEXT` for resolution taken from the merged environment; POSIX `argv` with `start_new_session=True`; Windows `application` + `command_line` with `CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_NO_WINDOW`; `NotFound` and `BatchArgRejected` raise `SpawnRejected` with the resolver's message; `args` untouched), `Step`, `TerminationReport`, and pure `termination_plan(platform, job_assigned)`
    - _Requirements: 17.4, 17.6, 17.9, 17.10, 17.11, 17.19, 17.20, 17.22, 14.1_

  - [ ]* 8.7 Write property test for spawn specification
    - File: `tests/property/test_spawn_spec.py`
    - **Property 42: Spawn specification**
    - **Validates: Requirements 17.10, 17.19, 17.20**

  - [ ] 8.8 Implement bind planning
    - Write `oscompat/net.py` with `BindTarget`, `SockOpts`, pure `plan_binds(host, port)` (`localhost` → required `127.0.0.1` + optional `::1` without name resolution; literals → one target; other hostnames → first `getaddrinfo` result), pure `socket_options(platform, family)` (`SO_EXCLUSIVEADDRUSE` on Windows, `SO_REUSEADDR` on Linux only, never `SO_REUSEPORT`, `IPV6_V6ONLY` on IPv6), pure `classify_bind_error(exc)`, and `bind_all(host, port, platform)` (one warning and skip for an unavailable optional target; on any other failure close bound sockets and raise `BindError` naming host, port, and OS error) with an injectable socket layer
    - _Requirements: 1.8, 17.30, 17.31, 17.32_

  - [ ]* 8.9 Write property test for bind plan and socket options
    - File: `tests/property/test_bind_plan.py`
    - **Property 45: Bind plan and socket options**
    - **Validates: Requirements 17.30, 17.31, 17.32, 1.8**

- [ ] 9. Implement session store
  - [ ] 9.1 Implement SessionStore ABC and in-memory implementation
    - Write `session/store.py` (async ABC: `ensure_context`, `get_history`, `append`, `set_system_prompt`, `get_system_prompt`, `create_task`, `get_task`, `set_task_state`, `active_task`, `evict_idle`) and `session/memory_store.py` (per-Context records and locks, single system-prompt slot, `last_activity` updates, idle eviction skipping `working`/`input-required` Contexts, eviction callbacks for TaskStore and runner cleanup, background eviction loop every 30 s)
    - _Requirements: 11.1, 11.2, 11.6, 11.10, 11.11, 12.7_

  - [ ]* 9.2 Write property test for idle eviction
    - File: `tests/property/test_idle_eviction.py`
    - **Property 21: Idle eviction**
    - **Validates: Requirements 11.11**

  - [ ]* 9.3 Write unit tests for session store
    - File: `tests/unit/test_session_store.py`
    - Test empty history on creation, task-to-context mapping, single system-prompt slot replacement, `active_task`
    - _Requirements: 11.1, 11.2, 12.7_

- [ ] 10. Implement process spawning, tree termination, and the stdio transport
  - [ ] 10.1 Implement Windows ctypes declarations and the CTRL_BREAK helper
    - Write `oscompat/_win32.py` (ctypes declarations for `CreateJobObjectW`, `SetInformationJobObject` with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, `AssignProcessToJobObject`, `QueryInformationJobObject` `ActiveProcesses`, `TerminateJobObject`, `CreateToolhelp32Snapshot`/`Thread32First`/`Thread32Next`, `OpenThread`, `ResumeThread`, `CloseHandle`, `IsWow64Process2`, `SetConsoleCtrlHandler`, `HANDLER_ROUTINE`; importable on every OS, functions available only on Windows) and `oscompat/ctrl_break.py` (`python -m local_agent.oscompat.ctrl_break <pid>`: `FreeConsole`, `AttachConsole(pid)`, `SetConsoleCtrlHandler(NULL, TRUE)`, `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, 0)`)
    - _Requirements: 17.10, 17.11_

  - [ ] 10.2 Implement process spawning and tree termination
    - Extend `oscompat/process.py` with `TreeHandle`, `spawn(spec)` (POSIX `anyio.open_process(..., start_new_session=True)`; Windows: create job, `Popen` with the string command line suspended, assign to the job, resume the primary thread, `job_object_unavailable` warning and resume on failure; all blocking calls via `to_thread`; pipe I/O via worker threads; child stderr logged line by line at `debug`) and `terminate_tree(h, grace_s, stdin_wait_s)` executing `termination_plan` under a shielded cancel scope with bounded waits: close stdin, `killpg(pid, SIGTERM)` or the `ctrl_break` helper with 1 s timeout (`ctrl_break_failed` warning on failure), poll until empty or grace elapses, `killpg(SIGKILL)` / `TerminateJobObject` / `taskkill.exe /PID <pid> /T /F`, reap ≤ 1.5 s, `CloseHandle(job)`, `mcp_process_tree_stopped` log with survivors
    - Inject the OS-call layer and clock so the schedule is testable with a fake tree
    - _Requirements: 6.11, 14.1, 17.10, 17.11, 17.12, 17.15_

  - [ ]* 10.3 Write property test for process-tree termination schedule
    - File: `tests/property/test_terminate_tree.py`
    - Use a `FakeTree` with scripted member exits, a fake clock, and a recording OS-call shim
    - **Property 43: Process-tree termination schedule**
    - **Validates: Requirements 17.11, 17.15, 6.11**

  - [ ]* 10.4 Write unit tests for process helpers
    - File: `tests/unit/test_process.py`
    - Test `termination_plan` templates per platform and job outcome, `PATHEXT` default when unset, `.ps1` rejected as not launchable, `SpawnRejected` message for an unresolved command
    - _Requirements: 17.4, 17.6, 17.11_

  - [ ] 10.5 Implement the managed stdio transport
    - Write `mcp_client/mcp_compat.py` isolating `mcp` SDK import locations (`mcp.types` vs `mcp_types`, the JSON-RPC message adapter, `SessionMessage`, timeout argument types)
    - Write `mcp_client/stdio_transport.py` with pure `encode_line` (`model_dump_json(by_alias=True, exclude_unset=True)` + `"\n"`, UTF-8) and `decode_line` (strict UTF-8, SDK JSON-RPC adapter, `Exception` item on failure), and `managed_stdio_client(spec, *, grace)` yielding `(read_stream, write_stream)` of `SessionMessage`: bytes-level split on `b"\n"`, one trailing `b"\r"` stripped, writer task, and `terminate_tree` with `grace()` read at exit on normal exit, error, or cancellation
    - _Requirements: 6.7, 6.11, 17.11, 17.12, 17.27_

  - [ ]* 10.6 Write property test for stdio framing round-trip
    - File: `tests/property/test_stdio_framing.py`
    - **Property 48: stdio framing round-trip**
    - **Validates: Requirements 17.27**

- [ ] 11. Implement MCP client manager
  - [ ] 11.1 Implement McpSessionAdapter
    - Create `mcp_client/manager.py` with `McpSessionAdapter` wrapping `ClientSession` through `mcp_compat`: `list_all_tools(timeout)` following `next_cursor` until none, and `call_tool(name, args, timeout)` returning a `CallOutcome` (content items, `is_error`), isolating `mcp` SDK naming and timeout-type differences
    - _Requirements: 7.1, 7.10, 9.2_

  - [ ]* 11.2 Write property test for paginated tool discovery
    - File: `tests/property/test_list_all_tools.py`
    - **Property 32: Paginated tool discovery**
    - **Validates: Requirements 7.1**

  - [ ] 11.3 Implement ServerSupervisor and McpManager
    - Add `backoff_delay(n) = min(2**n, 30)`, `ServerState`, `ServerSupervisor.run` (stdio: `managed_stdio_client(await build_spawn_spec(cfg, config_dir, sys.platform), grace=...)`; SSE: `sse_client(url, headers=...)`; `SpawnRejected` treated as a failed attempt with the resolver's message; 15 s handshake, tool listing, registry replace/remove, up to 5 reconnects, Process_Tree terminated after a failed handshake and before each reconnection by leaving the transport context; logging with redaction), and `McpManager` (`start` concurrently with 17 s cap, disabled servers skipped, `stop(grace=5.0)` stopping all supervisors concurrently under `asyncio.timeout(1 + grace + 1.5)`, `refresh_available` keeping previous tools on failure, `availability`, `statuses`, `call_tool` with `asyncio.timeout` cancelling late responses, concurrent calls on one session)
    - Log `mcp_server_connected`, `mcp_server_failed`, and `tool_call_finished`-related timing
    - _Requirements: 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 7.10, 7.11, 9.2, 9.9, 14.3, 17.6, 17.9, 17.11, 17.15_

  - [ ]* 11.4 Write unit tests for McpManager with a fake transport
    - File: `tests/unit/test_mcp_manager.py`
    - Test backoff sequence `[1, 2, 4, 8, 16]` and stop after 5, re-registration after reconnect, refresh failure keeps tools, tool-list failure registers none, unavailable server, timeout, late reply discarded, exception text, `SpawnRejected` marks the server unavailable with the resolver's reason, `stop(grace)` forwards the grace to every transport
    - _Requirements: 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 7.10, 7.11, 9.2, 9.6, 9.7, 9.9, 17.6, 17.15_

- [ ] 12. Implement LLM gateway
  - [ ] 12.1 Implement LlmGateway and LlmStream
    - Write `llm/gateway.py` with `LlmUnreachable`, `LlmRejected(reason)`, `LlmInterrupted`, `LlmGateway.stream(messages, tools)` (LiteLLM `acompletion` with model, api_base, api_key only when set, temperature, max_tokens, timeout, `stream=True`, `num_retries=0`, tools and `tool_choice="auto"` only when non-empty), retry before first chunk (1 s, 2 s; 3 attempts), no retry on 4xx or after first chunk, and `LlmStream` yielding text chunks immediately with `result()` from `StreamAssembler`
    - Inject `sleep`/clock for tests; write one `llm_request_finished` log record per request with `duration_ms`, outcome, and error fields
    - Add a startup helper that sets `litellm.add_function_to_prompt = True` when `native_function_calling` is false
    - _Requirements: 10.1, 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 15.1, 15.4, 15.6_

  - [ ]* 12.2 Write property test for LLM retry policy
    - File: `tests/property/test_llm_retry.py`
    - **Property 19: LLM retry policy**
    - **Validates: Requirements 10.6, 10.9, 10.10**

  - [ ]* 12.3 Write unit tests for gateway request kwargs
    - File: `tests/unit/test_llm_gateway.py`
    - Test api_key only when set, defaults, tools omitted when disabled or empty, immediate chunk yielding
    - _Requirements: 10.1, 10.2, 10.4, 10.5_

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Implement event emitter, approval gate, dispatcher, and task runner
  - [ ] 14.1 Implement TaskEventEmitter
    - Extend `a2a_server/events.py` with `TaskEventEmitter`: `transition` (reject disallowed with `transition_rejected` log, persist state to Session_Store before enqueue, `task_state_changed` log, final flag on terminal and `input-required`), `status`, `token`, `close_artifact` (empty text, `append=true`, `last_chunk=true`), `tool_started`, `tool_completed` (exactly once per call ID), `approval_required`, `reprompt_approval`, seal after terminal state
    - Implement sinks: attached `EventQueue` via `TaskUpdater`, and `DetachedSink` persisting to the A2A TaskStore when no request is attached
    - _Requirements: 2.9, 3.3, 3.5, 3.9, 4.4, 4.5, 4.6, 4.9, 11.2, 14.6, 15.2_

  - [ ]* 14.2 Write property test for transition table and emitter seal
    - File: `tests/property/test_emitter.py`
    - **Property 4: Transition table and emitter seal**
    - **Validates: Requirements 2.9, 3.9, 4.9, 14.6**

  - [ ] 14.3 Implement ApprovalGate
    - Extend `orchestrator/approval_gate.py` with `PendingApproval` and `ApprovalGate` (`request` stores the pending call, transitions to `input-required` with LLM-facing name, original name, server, call ID, arguments, and `tool_approval_required` metadata, awaits the future under the optional approval timeout, transitions back to `working`; `resolve`; `discard`; at most one pending per Task)
    - _Requirements: 5.4, 5.6, 5.7, 5.8, 5.13, 5.14, 5.15, 2.7_

  - [ ]* 14.4 Write unit tests for ApprovalGate
    - File: `tests/unit/test_approval_gate.py`
    - Test status contents, approve, deny, timeout denial, indefinite wait without timeout, discard on cancel
    - _Requirements: 5.6, 5.8, 5.10, 5.14, 5.15_

  - [ ] 14.5 Implement ToolDispatcher
    - Extend `orchestrator/tool_dispatch.py` with `ToolDispatcher.run_turn` (rejected calls get results and `tool_completed(failed)`, non-protected calls run concurrently via `asyncio.gather`, then protected calls one at a time through the ApprovalGate in call order, results returned in Tool_Call order) and `_execute` (`tool_started`, `McpManager.call_tool` with Tool_Timeout, `ResultFormatter` conversion and truncation, `tool_completed`, `tool_call_finished` log with outcome and `duration_ms`; non-cancellation exceptions become error results)
    - _Requirements: 4.2, 4.3, 4.5, 4.6, 4.7, 4.8, 5.9, 5.10, 5.13, 8.2, 8.3, 8.5, 9.1, 9.2, 9.7, 9.8, 9.10, 15.3, 15.4_

  - [ ]* 14.6 Write property test for name round-trip and faithful dispatch
    - File: `tests/property/test_name_roundtrip.py`
    - **Property 6: Name round-trip and faithful dispatch**
    - **Validates: Requirements 7.4, 8.1, 8.2**

  - [ ] 14.7 Implement TaskRunner, TaskRunnerRegistry, and reasoning loop
    - Write `orchestrator/loop.py`: `TaskRunner` (`attach`, `start`, `cancel` with 2 s cap and pending approval discard, `wait_stream_closing`), `_run` (single "Discovering local tools..." `working` transition, per-session discovery refresh, user message append, exception-to-`failed` mapping with redacted messages), `_reasoning_loop` (round-trip counting, tools-disabled final request at Max_Tool_Iterations, assistant message before Tool_Results), `_build_request` (rebuild System_Prompt from live snapshot into the single slot, trim to budget), `_stream_response` (per-response artifact IDs, Token_Events for non-empty text, close marker, empty artifact for empty final response), `_finalize` (synthetic "not completed" Tool_Results, terminal transition under `asyncio.timeout(1.5)`), and `TaskRunnerRegistry` (including `cancel_all()` for Graceful_Shutdown)
    - Bind `context_id`/`task_id` structlog contextvars inside each runner; use `asyncio.get_running_loop()` and `inspect.iscoroutinefunction()` only
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 3.2, 3.3, 3.4, 3.6, 3.8, 4.1, 6.12, 8.7, 8.8, 8.10, 8.12, 10.7, 10.10, 11.3, 11.4, 11.5, 11.8, 11.9, 12.5, 12.6, 12.7, 14.4, 14.5, 17.14, 17.28_

  - [ ]* 14.8 Build scenario fakes and strategies for orchestration property tests
    - Create `tests/fakes.py` with `FakeLlmGateway`, `FakeMcpManager`, `RecordingEventSink`, and `tests/property/scenarios.py` with the `scenario_scripts()` strategy (per-turn text chunks, tool calls of each kind, latency and outcomes, approval decisions, cancel/failure injection) and a `run_scenario` helper using a fresh event loop and fake clock, plus a transcript normalizer that replaces generated IDs, timestamps, and durations with placeholders
    - _Requirements: 3.4, 8.5, 11.5, 17.3_

  - [ ]* 14.9 Write property test for stream reassembly and token framing
    - File: `tests/property/test_stream_framing.py`
    - **Property 1: Stream reassembly and token framing**
    - **Validates: Requirements 3.2, 3.3, 3.4**

  - [ ]* 14.10 Write property test for interrupted streams
    - File: `tests/property/test_stream_interrupted.py`
    - **Property 2: Interrupted streams keep the emitted prefix**
    - **Validates: Requirements 3.8, 10.10, 10.7**

  - [ ]* 14.11 Write property test for event ordering
    - File: `tests/property/test_event_ordering.py`
    - **Property 3: Event ordering**
    - **Validates: Requirements 2.4, 4.1, 4.3, 4.4, 4.5, 4.6**

  - [ ]* 14.12 Write property test for history integrity and result ordering
    - File: `tests/property/test_history_integrity.py`
    - **Property 12: History integrity and result ordering**
    - **Validates: Requirements 8.4, 8.5, 8.10, 9.8, 9.10, 11.4, 11.5, 14.5, 5.13**

  - [ ]* 14.13 Write property test for context isolation
    - File: `tests/property/test_context_isolation.py`
    - **Property 13: Context isolation**
    - **Validates: Requirements 11.6**

  - [ ]* 14.14 Write property test for iteration limit
    - File: `tests/property/test_iteration_limit.py`
    - **Property 14: Iteration limit**
    - **Validates: Requirements 8.8, 8.12**

  - [ ]* 14.15 Write property test for system prompt placement and single storage
    - File: `tests/property/test_system_prompt_placement.py`
    - **Property 23: System prompt placement and single storage**
    - **Validates: Requirements 8.7, 12.5, 12.6, 12.7**

  - [ ]* 14.16 Write unit tests for reasoning loop edge cases
    - File: `tests/unit/test_reasoning_loop.py`
    - Test empty final response artifact, failure message without traceback, user message appended before first request, no new LLM call on approval resume, approve uses identical arguments, cancel between approval and send
    - _Requirements: 2.5, 2.6, 4.7, 4.8, 5.7, 5.9, 11.3_

- [ ] 15. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Implement Graceful_Shutdown and signal handling
  - [ ] 16.1 Implement ShutdownController and signal installation
    - Write `oscompat/signals.py` with `ShutdownReason` (`SIGNAL`, `CONSOLE_CLOSE`), `ShutdownController` (`request` idempotent, second request forces `grace_s = 0`; `grace_s` 5.0 / 2.0 for console close; `run()` executing in order: `server.should_exit = True`, `TaskRunnerRegistry.cancel_all()`, wait for lifespan shutdown to finish `McpManager.stop(grace=grace_s)`, return exit status 0; `shutdown_started`/`shutdown_finished` logs; a `threading.Event` set when done), and `install(loop, ctl, platform)` returning an uninstall callable: POSIX `loop.add_signal_handler` for `SIGINT`/`SIGTERM`; Windows `signal.signal` for `SIGINT`/`SIGBREAK` via `call_soon_threadsafe`, plus a referenced ctypes `SetConsoleCtrlHandler` callback that schedules `CONSOLE_CLOSE` and blocks ≤ 4.5 s on the done event for `CTRL_CLOSE_EVENT` and returns `FALSE` for other events
    - _Requirements: 2.7, 6.11, 17.13, 17.14, 17.15_

  - [ ]* 16.2 Write property test for Graceful_Shutdown sequencing
    - File: `tests/property/test_shutdown_sequence.py`
    - Use fake server, runner registry, and MCP manager recording step start/finish
    - **Property 44: Graceful_Shutdown sequencing**
    - **Validates: Requirements 17.13, 17.14, 2.7**

  - [ ]* 16.3 Write unit tests for the Windows console control handler
    - File: `tests/unit/test_signals.py`
    - Call the handler callback directly: returns `FALSE` for non-close events; for `CTRL_CLOSE_EVENT` requests `CONSOLE_CLOSE` and blocks until the done event or 4.5 s; grace becomes 2 s
    - _Requirements: 17.15_

- [ ] 17. Implement A2A server surface, app wiring, and entry point
  - [ ] 17.1 Implement Agent_Card builder
    - Write `a2a_server/agent_card.py` with `endpoint_url(host, port)` (IPv6 bracketed, configured `localhost` kept) and `build_agent_card(cfg)` (streaming true, text modes, `local-tools-chat` skill, JSON-RPC interface at the endpoint URL)
    - _Requirements: 1.3, 1.4, 1.5, 17.30_

  - [ ]* 17.2 Write property test for agent card content
    - File: `tests/property/test_agent_card.py`
    - **Property 31: Agent card content**
    - **Validates: Requirements 1.3, 1.4, 1.5**

  - [ ] 17.3 Implement loopback and bearer-token middleware
    - Write `a2a_server/auth.py` with pure `is_loopback(host)`, `LoopbackGuardMiddleware` (403 for non-loopback clients when enabled; both `127.0.0.1` and `::1` sockets pass), and `BearerAuthMiddleware` (`hmac.compare_digest`, 401 before routing, agent card path exempt, `OPTIONS` passed through)
    - _Requirements: 1.2, 16.1, 16.2, 16.3, 16.7, 17.30_

  - [ ]* 17.4 Write property test for loopback classification and bearer check
    - File: `tests/property/test_auth.py`
    - **Property 30: Loopback classification and bearer check**
    - **Validates: Requirements 16.1, 16.2, 16.3**

  - [ ] 17.5 Implement health endpoint
    - Write `a2a_server/health.py` with pure `health_payload(statuses)` (`ok` iff every enabled server is available) and a Starlette `health_endpoint` reading only in-memory `McpManager` state
    - _Requirements: 15.8, 15.9_

  - [ ]* 17.6 Write property test for health status
    - File: `tests/property/test_health.py`
    - **Property 29: Health status**
    - **Validates: Requirements 15.8, 15.9**

  - [ ] 17.7 Implement HostRequestHandler and HostAgentExecutor
    - Write `a2a_server/a2a_compat.py` isolating `a2a-sdk` import names, and `a2a_server/executor.py` with `HostRequestHandler` (`DefaultRequestHandler` subclass; per-Context lock; "Context is busy" error; "Task is not awaiting input" error for messages to `working` Tasks) and `HostAgentExecutor` (`execute` creates Context/Task/runner or parses approval and resolves or re-prompts without history append; `cancel` delegates to `runner.cancel`; asserts new `task_id` uniqueness)
    - _Requirements: 2.1, 2.2, 2.3, 2.7, 2.13, 3.5, 3.6, 3.7, 5.8, 5.11, 11.7_

  - [ ]* 17.8 Write unit tests for executor admission
    - File: `tests/unit/test_executor.py`
    - Test new Task and Context IDs, Context association, unknown `context_id` creation, busy Context rejection, re-prompt on unrecognized approval, rejection of messages to `working` Tasks
    - _Requirements: 2.1, 2.2, 2.3, 2.13, 5.11, 11.7_

  - [ ] 17.9 Implement app wiring and lifespan
    - Write `app.py` with `Deps` dataclass (including the `ShutdownController`), `build_deps(cfg)`, `build_app(cfg, deps)` (agent card, JSON-RPC, and REST route factories from `a2a-sdk`, `/health` route, `LoopbackGuardMiddleware`, `CORSMiddleware` with exact origins outermost relative to auth, `BearerAuthMiddleware`), and lifespan (start `McpManager`, start idle eviction when configured; on shutdown stop eviction, cancel any remaining runners idempotently, and call `McpManager.stop(grace=shutdown_controller.grace_s)`)
    - _Requirements: 1.1, 1.2, 6.5, 6.11, 11.11, 16.2, 16.4, 16.6, 17.14, 17.15_

  - [ ] 17.10 Implement BackendServer and process entry point
    - Write `src/local_agent/server.py` with `BackendServer(uvicorn.Server)` whose `capture_signals()` is a no-op context manager, configured with `loop="none"` and `timeout_graceful_shutdown=2`
    - Write `__main__.py`: load config (print report and exit 2 on `ConfigError`, including invalid UTF-8 and Path_Field errors), configure logging, `classify_platform(gather_facts())` with one `platform_unsupported` warning, `os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")` before importing litellm, apply the function-calling fallback setting, `bind_all(plan)` before the loop exists (log host:port and exit 1 on `BindError`), create `asyncio.Runner(loop_factory=select_loop_factory(...))`, `signals.install(runner.get_loop(), ctl, sys.platform)`, `runner.run(server.serve(sockets=...))` with the shutdown controller driving `should_exit`, then `sys.exit(0)`
    - _Requirements: 1.6, 1.8, 10.8, 13.3, 13.6, 13.7, 14.1, 17.2, 17.13, 17.14, 17.24, 17.28, 17.29, 17.30, 17.31, 17.32_

  - [ ]* 17.11 Write unit tests for auth and CORS wiring
    - File: `tests/unit/test_app_security.py`
    - Test allowed vs disallowed origin headers, empty origin list, loopback without token, agent card exempt from auth, health protected on non-loopback bind
    - _Requirements: 1.2, 16.2, 16.4, 16.6, 16.7_

  - [ ]* 17.12 Write unit tests for BackendServer and startup ordering
    - File: `tests/unit/test_entrypoint.py`
    - Test that `BackendServer.capture_signals()` installs no signal handlers, that `LITELLM_LOCAL_MODEL_COST_MAP` is set before litellm is imported, and that config errors exit 2 before `bind_all` is called and bind errors exit 1 before any MCP_Server is spawned (with injected fakes)
    - _Requirements: 1.8, 13.3, 17.14, 17.28_

- [ ] 18. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 19. Write integration and platform-specific tests
  - [ ]* 19.1 Create integration fixtures
    - Create `tests/fixtures/mcp_echo_server.py` (stdio MCP server built with the `mcp` SDK: paginated tools, echo/slow/error/hang tools, optional crash, non-ASCII echo) and `tests/integration/conftest.py` with a scriptable fake LLM (patched `litellm.acompletion` or a local OpenAI-compatible stub), an app factory over `httpx.ASGITransport`, a real-uvicorn fixture on a free port, and a session-scoped finalizer that kills any leftover recorded pid
    - _Requirements: 6.5, 7.1, 10.2, 17.12_

  - [ ]* 19.2 Create the process-spawning MCP fixture
    - Create `tests/fixtures/mcp_spawning_server.py`: stdio MCP server that spawns a grandchild which spawns a great-grandchild; `get_pids` tool; flags `--cooperative`, `--ignore-graceful`, `--orphan`, `--no-initialize`, and `--spawn-first` (spawns the grandchild as its first instruction)
    - _Requirements: 6.11, 17.10, 17.11, 17.12_

  - [ ]* 19.3 Write A2A surface integration tests
    - File: `tests/integration/test_a2a_surface.py`
    - Test JSON-RPC send, SSE streaming, REST, get-task, cancel; each JSON-RPC error category; terminal-state and unknown-Task errors; first SSE event within 1 s; stream closes after final status
    - _Requirements: 1.1, 1.9, 2.8, 2.10, 2.11, 2.12, 3.1, 3.5, 3.7_

  - [ ]* 19.4 Write startup process tests
    - File: `tests/integration/test_process.py`
    - Run `python -m local_agent` as a subprocess: bind failure exit code 1 naming host:port, accept within 10 s, config error exit 2 before bind, invalid UTF-8 config exit with byte offset, one `platform_unsupported` warning when `oscompat` facts are overridden through the test-only environment hook
    - _Requirements: 1.6, 1.8, 13.3, 17.2, 17.24_

  - [ ]* 19.5 Write MCP stdio integration tests
    - File: `tests/integration/test_mcp_stdio.py`
    - Run the real SDK `ClientSession` over `managed_stdio_client` against the echo fixture on every runner: handshake, pagination, `call_tool`, concurrent calls from two Tasks, child kill and reconnection, hanging servers settle within 17 s, `"naïve 日本語 🚀"` round-trips under a non-UTF-8 locale (`LC_ALL=C` on POSIX, code page 1252 on Windows)
    - _Requirements: 6.5, 6.6, 6.7, 6.8, 7.1, 14.3, 17.27_

  - [ ]* 19.6 Write concurrency and cancellation integration tests
    - File: `tests/integration/test_concurrency.py`
    - Test 10 concurrent Tasks with first status within 500 ms, cancel during slow tool and during `input-required` within 2 s, failure isolation across Contexts, concurrent tool turn under slowest + 500 ms; run with asyncio debug and `slow_callback_duration=0.1`
    - _Requirements: 2.7, 8.3, 14.1, 14.2, 14.4, 14.7_

  - [ ]* 19.7 Write SSE disconnect integration test
    - File: `tests/integration/test_disconnect.py`
    - Drop the SSE stream mid-response on a real uvicorn socket; get-task later shows `completed` with the full artifact
    - _Requirements: 3.6_

  - [ ]* 19.8 Write health integration test
    - File: `tests/integration/test_health.py`
    - Health responds within 1 s with all servers listed while the fake LLM is unreachable
    - _Requirements: 15.8, 15.9_

  - [ ]* 19.9 Write opt-in function-calling fallback test
    - File: `tests/integration/test_llm_fallback.py`, marked `@pytest.mark.llm`, against a local Ollama
    - Prompt-based tool calls come back in the assembled Tool_Call format
    - _Requirements: 10.8_

  - [ ]* 19.10 Write process-tree cleanup integration test
    - File: `tests/integration/test_process_tree.py` (no marker; OS-specific assertions inside)
    - Parametrize `--cooperative` / `--ignore-graceful` / `--orphan` × `McpManager.stop()` / reconnection after killing only the server pid / failed handshake (`--no-initialize`); assert every recorded pid is gone within `1 + grace + 1.5` s (POSIX `os.kill(pid, 0)`; Windows `OpenProcess` + `WaitForSingleObject`) and `TerminationReport.forced` matches the behaviour
    - _Requirements: 6.7, 6.8, 6.11, 17.11, 17.12_

  - [ ]* 19.11 Write Windows-only process tests
    - File: `tests/integration/test_windows_process.py`, marked `windows_only`, with a `tests/fixtures/echo_args.cmd` batch fixture that forwards `%*` to a CRT program
    - `test_ctrl_break_reaches_tree`, `test_job_assignment_before_resume` (`--spawn-first`), `test_taskkill_fallback` (job creation forced to fail), `test_batch_args_literal` (`a&b`, `x|y`, `<z>`, `c^d`, `sp ace`, `trail\` literal; `%` and `"` rejected), `test_npx_resolution` (via child `PATHEXT`), `test_env_case_merge` (configured `Path` replaces inherited `PATH`), and a check that `winargv_ref` matches `python.exe -c "import sys, json; print(json.dumps(sys.argv[1:]))"` for a fixed corpus
    - _Requirements: 17.4, 17.7, 17.8, 17.9, 17.10, 17.11, 17.12, 17.22_

  - [ ]* 19.12 Write POSIX-only process and loop tests
    - File: `tests/integration/test_posix_process.py`, marked `posix_only`
    - `test_new_session_isolation` (child `getsid()` differs; `SIGINT` to the Backend's group does not reach the child), `test_sigterm_ignoring_child` (`SIGKILL` after 5 s ± 0.5 s), `test_uvloop_selected` (skipped when uvloop is not installed)
    - _Requirements: 17.10, 17.11, 17.29_

  - [ ]* 19.13 Write socket bind integration tests
    - File: `tests/integration/test_bind.py` (per-OS expectations)
    - `test_exclusive_bind_conflict` (second listener on the same port exits 1 naming host:port), `test_localhost_dual_stack` (binds `127.0.0.1` and `::1`, or warns once; `::1` assertions skipped when `socket.has_ipv6` is false)
    - _Requirements: 1.8, 17.30, 17.31, 17.32_

  - [ ]* 19.14 Write shutdown and signal integration tests
    - File: `tests/integration/test_shutdown.py`
    - `test_shutdown_exit_code`: POSIX sends `SIGTERM` and `SIGINT`, Windows sends `CTRL_BREAK_EVENT` to a Backend started with `CREATE_NEW_PROCESS_GROUP`; assert shutdown starts ≤ 1 s, active Tasks end `canceled`, MCP trees are gone, exit status 0, and a second signal still exits 0
    - `test_console_close`, marked `windows_only` and `@pytest.mark.manual`: trees gone within the 2 s grace
    - _Requirements: 6.11, 17.12, 17.13, 17.14, 17.15_

  - [ ]* 19.15 Write property test for cross-platform transcript equivalence
    - File: `tests/property/test_transcript_equivalence.py` with `@settings(derandomize=True, database=None)`, using the scenario fakes and normalizer; write the digest to `artifacts/transcripts-<os>.sha256`
    - **Property 46: Cross-platform transcript equivalence**
    - **Validates: Requirements 17.3**

- [ ] 20. Add the OS-matrix CI workflow
  - [ ] 20.1 Create `.github/workflows/ci.yml`
    - `test` job on `windows-latest`, `macos-latest`, `macos-15-intel`, `ubuntu-latest`, `ubuntu-24.04-arm` with `fail-fast: false`: checkout, `astral-sh/setup-uv`, `uv python install 3.14` (standard build, never `3.14t`), `uv sync --locked --no-build`, `ruff check .`, `ruff format --check .`, `mypy src`, `pytest tests/unit tests/property tests/integration -m "not manual and not llm"` with `HYPOTHESIS_PROFILE=ci` and the `DeprecationWarning` error filters, upload `artifacts/transcripts-<os>.sha256`
    - Windows-only step before tests: `uv run python -c "import litellm"` with `HTTPS_PROXY`/`HTTP_PROXY` set to `http://127.0.0.1:9`
    - `equivalence` job (`needs: test`) downloads the five digests and fails if they differ; the workflow fails if any job fails
    - _Requirements: 17.3, 17.33, 17.34_

- [ ] 21. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; integration fixtures (19.1, 19.2) and scenario fakes (14.8) are only needed by the optional tests
- Each task references specific requirements for traceability; each of the design's 48 correctness properties has exactly one property-test sub-task (P33–P48 cover Requirement 17)
- `src/local_agent/models.py`, `a2a_server/a2a_compat.py`, `mcp_client/mcp_compat.py`, and `src/local_agent/server.py` are additions to Appendix A that the design implies (shared runtime types, SDK name isolation, and the `BackendServer` subclass); `oscompat/env.py`, `oscompat/net.py`, `oscompat/_win32.py`, `oscompat/ctrl_break.py`, and `mcp_client/stdio_transport.py` are the design's own additions
- At implementation time, verify the pinned `a2a-sdk` (1.2.x), `mcp` (2.3.x), and `uvicorn` (>= 0.36, `capture_signals` override) against `a2a_compat.py`, `mcp_compat.py`, `stdio_transport.py`, and `server.py`; confirm the `macos-15-intel` runner label
- Platform-specific tests use the `windows_only`, `posix_only`, `macos_only`, and `linux_only` markers from `tests/conftest.py`; pure OS-rule tests take `platform` as a parameter and run on every runner
- Run tests with `uv run pytest -q`, integration tests with `uv run pytest -m integration`, and lint/type checks with `uv run ruff check` and `uv run mypy src`; these work the same in PowerShell, bash, and zsh

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "2.3", "2.5", "2.8", "3.1", "5.1"] },
    { "id": 2, "tasks": ["2.2", "2.4", "2.6", "2.7", "2.9", "3.2", "4.1", "5.2", "6.1", "6.3", "6.5", "6.7", "6.12", "8.1", "8.8", "9.1", "10.1"] },
    { "id": 3, "tasks": ["3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4.2", "4.3", "4.4", "5.3", "5.4", "5.5", "5.6", "5.7", "6.2", "6.4", "6.6", "6.8", "6.9", "8.2", "8.3", "8.9", "9.2", "9.3", "12.1", "14.1"] },
    { "id": 4, "tasks": ["5.8", "5.9", "6.10", "6.11", "8.4", "8.6", "12.2", "12.3", "14.2", "14.3", "17.1", "17.3", "17.5"] },
    { "id": 5, "tasks": ["8.5", "8.7", "10.2", "14.4", "17.2", "17.4", "17.6"] },
    { "id": 6, "tasks": ["10.3", "10.4", "10.5"] },
    { "id": 7, "tasks": ["10.6", "11.1"] },
    { "id": 8, "tasks": ["11.2", "11.3"] },
    { "id": 9, "tasks": ["11.4", "14.5"] },
    { "id": 10, "tasks": ["14.6", "14.7"] },
    { "id": 11, "tasks": ["14.8", "16.1", "17.7"] },
    { "id": 12, "tasks": ["14.9", "14.10", "14.11", "14.12", "14.13", "14.14", "14.15", "14.16", "16.2", "16.3", "17.8", "17.9"] },
    { "id": 13, "tasks": ["17.10", "17.11"] },
    { "id": 14, "tasks": ["17.12", "19.1", "19.2", "20.1"] },
    { "id": 15, "tasks": ["19.3", "19.4", "19.5", "19.6", "19.7", "19.8", "19.9", "19.10", "19.11", "19.12", "19.13", "19.14", "19.15"] }
  ]
}
```
