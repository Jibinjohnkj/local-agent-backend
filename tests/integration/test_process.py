"""Startup of the real Backend process: `python -m local_agent` as a subprocess (task 22.4).

`tests/unit/test_entrypoint.py` covers the order of the startup steps with fakes. These tests run
the whole process and check what a user sees: the exit status, the report on stderr, the log
records, and the files left behind.

* Bind failure: status 1, the host and port named, no MCP_Server spawned (1.8).
* Accept requests within 10 s of the start (1.6).
* Config error: status 2 before storage, MCP, or the bind (13.3, 13.12); invalid UTF-8 names the
  file and the byte offset (17.24).
* An unsupported platform gives one `platform_unsupported` warning (17.2). The platform facts are
  overridden through `sitecustomize.py` in a temporary directory on `PYTHONPATH`, so the child
  process sees them without any hook in the Backend itself.
* Storage failures (18.22, 18.24): status 2, `storage.path` named, the database file untouched, and
  no MCP_Server spawned.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from integration_harness import (
    DATABASE_NAME,
    BackendLauncher,
    BackendProcess,
    FakeLlmServer,
    config_dict,
    echo_server_entry,
    free_port,
    isolated_env,
    jsonrpc_call,
    message_params,
    read_request_log,
    wait_for_task_state,
)
from local_agent.storage.migrations import CURRENT_SCHEMA_VERSION, SCHEMA_V1

EXIT_FAILURE = 1
EXIT_CONFIG = 2
COMPLETED = "TASK_STATE_COMPLETED"
STARTUP_BUDGET_S = 10.0  # 1.6
EXIT_TIMEOUT_S = 60.0

SUPPORTED_FACTS: dict[str, Any] = {
    "system": "Darwin",
    "release": "14.4",
    "native_machine": "arm64",
    "libc": ["", ""],
    "implementation": "cpython",
    "version": [3, 14],
    "gil_disabled": False,
}
UNSUPPORTED_FACTS: dict[str, Any] = {
    **SUPPORTED_FACTS,
    "system": "Linux",
    "release": "",
    "native_machine": "x86_64",
}

FACTS_ENV = "LOCAL_AGENT_IT_PLATFORM_FACTS"
SITECUSTOMIZE = f'''\
"""Test-only: replace `oscompat.gather_facts` in the child process (task 22.4)."""

import json
import os

_raw = os.environ.get("{FACTS_ENV}")
if _raw:
    from local_agent import oscompat

    _data = json.loads(_raw)
    _facts = oscompat.PlatformFacts(
        system=_data["system"],
        release=_data["release"],
        native_machine=_data["native_machine"],
        libc=(_data["libc"][0], _data["libc"][1]),
        implementation=_data["implementation"],
        version=(_data["version"][0], _data["version"][1]),
        gil_disabled=_data["gil_disabled"],
    )
    oscompat.gather_facts = lambda: _facts
'''


# --- Helpers ------------------------------------------------------------------------------------


@contextmanager
def _listening_socket() -> Iterator[int]:
    """A socket that listens on a free loopback port, so a Backend cannot bind it."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield int(sock.getsockname()[1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(process: BackendProcess, event: str) -> list[dict[str, Any]]:
    return [record for record in process.log_records() if record.get("event") == event]


def _run_to_exit(process: BackendProcess) -> int:
    """Wait for a Backend that must stop by itself. It must not serve."""
    try:
        return process.wait(EXIT_TIMEOUT_S)
    finally:
        process.close()


def _start_with_config_bytes(
    backend_launcher: BackendLauncher, isolated_home: Path, data: bytes
) -> BackendProcess:
    """Start `python -m local_agent` on a config file with exactly these bytes."""
    workdir = backend_launcher.root / f"raw-{len(backend_launcher.processes) + 1}"
    workdir.mkdir()
    config_path = workdir / "config.yaml"
    config_path.write_bytes(data)
    process = BackendProcess(
        config_path=config_path,
        workdir=workdir,
        env=isolated_env(isolated_home),
        port=free_port(),
    )
    backend_launcher.processes.append(process)
    return process.start()


def _config_text(storage_dir: Path, **server: Any) -> str:
    config = config_dict(storage_dir=storage_dir, port=free_port())
    config["server"].update(server)
    return yaml.safe_dump(config, allow_unicode=True)


def _echo_entry(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    log = tmp_path / "echo-requests.jsonl"
    return echo_server_entry(request_log=log), log


# --- 1.8: bind failure --------------------------------------------------------------------------


def test_bind_failure_exits_1_naming_host_and_port(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    entry, echo_log = _echo_entry(tmp_path)
    with _listening_socket() as port:
        process = backend_launcher.launch(port=port, mcp_servers=[entry], wait=False)
        assert _run_to_exit(process) == EXIT_FAILURE

    assert f"127.0.0.1:{port}" in process.stderr_text()
    (record,) = _events(process, "bind_failed")
    assert record["host"] == "127.0.0.1"
    assert record["port"] == port
    assert not echo_log.exists(), "an MCP_Server was started before the bind failed"
    assert read_request_log(echo_log) == []


# --- 1.6: accept requests within 10 s -----------------------------------------------------------


def test_backend_accepts_requests_within_ten_seconds(
    backend_launcher: BackendLauncher,
    llm_stub: FakeLlmServer,
    storage_backend: Literal["memory", "sqlite"],
) -> None:
    started = time.monotonic()
    backend = backend_launcher.launch(
        storage_backend=storage_backend, llm_base=llm_stub.base_url, wait=False
    )
    backend.wait_healthy(STARTUP_BUDGET_S)
    elapsed = time.monotonic() - started
    assert elapsed < STARTUP_BUDGET_S

    with backend.client() as client:
        result = jsonrpc_call(client, "SendMessage", message_params("hello"))["result"]
        task = wait_for_task_state(client, result["task"]["id"])
    assert task["status"]["state"] == COMPLETED
    assert backend.stop() == 0


# --- 13.3 / 13.12: config errors ----------------------------------------------------------------


def test_config_error_exits_2_before_storage_mcp_or_bind(
    backend_launcher: BackendLauncher, isolated_home: Path, tmp_path: Path
) -> None:
    storage_dir = tmp_path / "never-created"
    entry, echo_log = _echo_entry(tmp_path)
    config = config_dict(storage_dir=storage_dir, mcp_servers=[entry])
    config["server"]["port"] = 70000  # outside 1 to 65535 (1.7)
    process = _start_with_config_bytes(
        backend_launcher, isolated_home, yaml.safe_dump(config).encode()
    )

    assert _run_to_exit(process) == EXIT_CONFIG
    report = process.stderr_text()
    assert "invalid configuration" in report
    assert "server.port" in report
    assert "70000" in report
    assert process.log_records() == [], "the Backend logged startup records after a config error"
    assert not storage_dir.exists(), "storage was touched after a config error"
    assert not echo_log.exists(), "an MCP_Server was started after a config error"


def test_missing_config_file_exits_2(
    backend_launcher: BackendLauncher, isolated_home: Path
) -> None:
    process = _start_with_config_bytes(backend_launcher, isolated_home, b"")
    process.config_path.unlink()

    assert _run_to_exit(process) == EXIT_CONFIG
    assert str(process.config_path) in process.stderr_text()


@pytest.mark.parametrize("with_bom", [False, True], ids=["plain", "bom"])
def test_invalid_utf8_config_exits_2_naming_file_and_byte_offset(
    backend_launcher: BackendLauncher, isolated_home: Path, tmp_path: Path, with_bom: bool
) -> None:
    bom = b"\xef\xbb\xbf" if with_bom else b""
    head = bom + "# naïve config ".encode()
    body = _config_text(tmp_path / "storage").encode()
    data = head + b"\xff\xfe" + b"\n" + body
    offset = len(head)  # counted from the start of the file, BOM included (17.24)
    process = _start_with_config_bytes(backend_launcher, isolated_home, data)

    assert _run_to_exit(process) == EXIT_CONFIG
    report = process.stderr_text()
    assert str(process.config_path) in report
    assert f"invalid UTF-8 at byte offset {offset}" in report
    assert process.log_records() == []
    assert not (tmp_path / "storage").exists()


# --- 17.2: platform_unsupported -----------------------------------------------------------------


def _facts_env(tmp_path: Path, facts: dict[str, Any]) -> dict[str, str]:
    hook_dir = tmp_path / "facts-hook"
    hook_dir.mkdir(exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    inherited = os.environ.get("PYTHONPATH")
    python_path = os.pathsep.join([str(hook_dir), *([inherited] if inherited else [])])
    return {"PYTHONPATH": python_path, FACTS_ENV: json.dumps(facts)}


def test_unsupported_platform_logs_one_warning_and_keeps_serving(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    backend = backend_launcher.launch(
        storage_backend="memory", env=_facts_env(tmp_path, UNSUPPORTED_FACTS)
    )
    assert backend.stop() == 0

    (record,) = _events(backend, "platform_unsupported")
    assert record["level"] == "warning"
    assert record["os"] == "Linux (non-glibc)"
    assert record["arch"] == "x86_64"
    assert record["python_build"] == "CPython 3.14 (GIL)"


def test_supported_platform_logs_no_platform_warning(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    backend = backend_launcher.launch(
        storage_backend="memory", env=_facts_env(tmp_path, SUPPORTED_FACTS)
    )
    assert backend.stop() == 0

    assert backend.log_records(), "the Backend wrote no log records, so the check proves nothing"
    assert _events(backend, "platform_unsupported") == []


# --- 18.24 / 18.22: storage failures ------------------------------------------------------------


def _assert_storage_refused(
    process: BackendProcess, echo_log: Path, *, path: Path, expected: tuple[str, ...] = ()
) -> None:
    assert _run_to_exit(process) == EXIT_CONFIG
    report = process.stderr_text()
    assert "invalid configuration" in report
    assert "storage.path" in report
    assert str(path) in report
    for text in expected:
        assert text in report
    assert _events(process, "bind_failed") == []
    assert not echo_log.exists(), "an MCP_Server was started after a storage failure"


def test_storage_path_that_is_a_file_exits_2_naming_storage_path(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    not_a_directory = tmp_path / "storage-is-a-file"
    not_a_directory.write_bytes(b"keep me")
    entry, echo_log = _echo_entry(tmp_path)

    process = backend_launcher.launch(storage_dir=not_a_directory, mcp_servers=[entry], wait=False)

    _assert_storage_refused(process, echo_log, path=not_a_directory)
    assert not_a_directory.read_bytes() == b"keep me"


def test_garbage_database_exits_2_and_stays_unchanged(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    storage_dir = tmp_path / "garbage-storage"
    storage_dir.mkdir()
    database = storage_dir / DATABASE_NAME
    database.write_bytes(b"this is certainly not a SQLite database file\n" * 200)
    before = (database.stat().st_size, _sha256(database))
    entry, echo_log = _echo_entry(tmp_path)

    process = backend_launcher.launch(storage_dir=storage_dir, mcp_servers=[entry], wait=False)

    _assert_storage_refused(process, echo_log, path=database)
    assert (database.stat().st_size, _sha256(database)) == before


def test_newer_schema_database_exits_2_and_stays_unchanged(
    backend_launcher: BackendLauncher, tmp_path: Path
) -> None:
    storage_dir = tmp_path / "newer-storage"
    storage_dir.mkdir()
    database = storage_dir / DATABASE_NAME
    newer = CURRENT_SCHEMA_VERSION + 98
    with closing(sqlite3.connect(database)) as conn:
        for statement in SCHEMA_V1:  # the real tables and indexes, as a newer Backend left them
            conn.execute(statement)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (newer,))
        conn.executemany(
            "INSERT INTO contexts (context_id, created_at_ms, last_activity_ms) VALUES (?, ?, ?)",
            [(f"context-{n}", n, n) for n in range(2000)],
        )
        conn.commit()
    before = (database.stat().st_size, _sha256(database))
    entry, echo_log = _echo_entry(tmp_path)

    process = backend_launcher.launch(storage_dir=storage_dir, mcp_servers=[entry], wait=False)

    _assert_storage_refused(
        process,
        echo_log,
        path=database,
        expected=(
            f"schema version {newer}",
            f"highest supported schema version is {CURRENT_SCHEMA_VERSION}",
        ),
    )
    assert (database.stat().st_size, _sha256(database)) == before
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(newer,)]
