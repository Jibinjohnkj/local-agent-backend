"""`BackendServer` and the startup order of `python -m local_agent` (17.14, 17.28, 18.28).

`main` takes every step as a field of `StartupSteps`. The `Rig` below replaces the steps with
recording fakes, so the tests read the call order and the exit status, and nothing starts a real
server, process, or MCP_Server. A few cases pass a real step (`ensure_storage_dir`,
`Database.open`, `StorageLock.acquire`, the config loader) to prove that the real failure maps to
the right exit status. "MCP" in the call order is the `serve` step: uvicorn's startup runs the
lifespan, which starts the MCP_Client.
"""

import asyncio
import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, cast

import pytest
import uvicorn
from starlette.types import ASGIApp, Receive, Scope, Send
from structlog.testing import capture_logs

from conftest import posix_only
from local_agent.__main__ import EXIT_CONFIG, EXIT_FAILURE, EXIT_OK, StartupSteps, main
from local_agent.a2a_server.app import Deps
from local_agent.config import loader
from local_agent.config.loader import ConfigError, ConfigIssue
from local_agent.config.models import BackendConfig
from local_agent.logging_setup import Redactor
from local_agent.models import Clock
from local_agent.oscompat import paths
from local_agent.oscompat.net import BindError
from local_agent.oscompat.paths import PathFieldError
from local_agent.oscompat.permissions import PermissionSetupError
from local_agent.oscompat.signals import ShutdownController, ShutdownReason
from local_agent.server import GRACEFUL_SHUTDOWN_TIMEOUT_S, BackendServer, build_server
from local_agent.storage import StorageError
from local_agent.storage.database import Database
from local_agent.storage.lock import StorageInUse, StorageLock
from local_agent.storage.migrations import MigrationError, SchemaError

IDLE_TIMEOUT_S: Final = 3600
STORAGE_STEPS: Final = frozenset(
    {"dir", "restrict", "lock", "open", "restrict_file", "migrate", "recover", "retention"}
)
AFTER_STORAGE: Final = ("prepare_llm", "bind", "build_app", "build_server", "mcp")

SQLITE_ORDER: Final = [
    "load_config",
    "configure_logging",
    "dir",
    "restrict",
    "lock",
    "open",
    "restrict_file",
    "migrate",
    "recover",
    "retention",
    "prepare_llm",
    "bind",
    "build_app",
    "build_server",
    "install_signals",
    "mcp",
    "uninstall_signals",
    "db_close",
    "unlock",
]
MEMORY_ORDER: Final = [
    "load_config",
    "configure_logging",
    "prepare_llm",
    "bind",
    "build_app",
    "build_server",
    "install_signals",
    "mcp",
    "uninstall_signals",
]


async def _app(scope: Scope, receive: Receive, send: Send) -> None:
    return None


# --- BackendServer ------------------------------------------------------------------------------


class _SignalLog:
    """Replaces `signal.signal` and `signal.raise_signal`, recording each call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.installed: list[int] = []
        self.raised: list[int] = []
        real_signal = signal.signal

        def record_signal(signalnum: int, handler: Any) -> Any:
            self.installed.append(signalnum)
            return real_signal(signalnum, handler)

        monkeypatch.setattr(signal, "signal", record_signal)
        monkeypatch.setattr(signal, "raise_signal", self.raised.append)


def test_capture_signals_installs_no_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(_app, "127.0.0.1", 0)
    seen = _SignalLog(monkeypatch)
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    with server.capture_signals():
        during = {sig: signal.getsignal(sig) for sig in before}

    assert seen.installed == []
    assert during == before
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_plain_uvicorn_server_does_install_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control for the test above: the override is what removes the handlers."""
    plain = uvicorn.Server(uvicorn.Config(_app, loop="none", log_config=None))
    seen = _SignalLog(monkeypatch)

    with plain.capture_signals():
        pass

    assert signal.SIGINT in seen.installed
    assert signal.SIGTERM in seen.installed


def test_capture_signals_does_not_raise_the_signal_again(monkeypatch: pytest.MonkeyPatch) -> None:
    server = build_server(_app, "127.0.0.1", 0)
    seen = _SignalLog(monkeypatch)

    with server.capture_signals():
        server.handle_exit(signal.SIGTERM, None)

    assert server.should_exit
    assert seen.raised == []


def test_build_server_config() -> None:
    server = build_server(_app, "127.0.0.1", 12345)

    assert isinstance(server, BackendServer)
    config = server.config
    assert (config.host, config.port) == ("127.0.0.1", 12345)
    assert config.loop == "none"
    assert config.lifespan == "on"
    assert config.log_config is None
    assert config.access_log is False
    assert config.server_header is False
    assert config.timeout_graceful_shutdown == GRACEFUL_SHUTDOWN_TIMEOUT_S


# --- LiteLLM environment variable ---------------------------------------------------------------

_PROBE: Final = """
import importlib.abc, json, os, sys

seen = {}

class Probe(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "litellm" or name.startswith("litellm."):
            seen.setdefault("value", os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"))
            raise ImportError("the probe stops the litellm import")
        return None

sys.meta_path.insert(0, Probe())
import local_agent.__main__ as entry
from local_agent.config.models import BackendConfig, LLMConfig

imported_by_entry_point = "value" in seen
try:
    entry.StartupSteps().prepare_llm(BackendConfig(llm=LLMConfig(model="m")))
except ImportError:
    pass
print(json.dumps({"early": imported_by_entry_point, "value": seen.get("value")}))
"""


@pytest.mark.parametrize(
    ("preset", "expected"), [(None, "True"), ("False", "False")], ids=["unset", "preset"]
)
def test_cost_map_variable_is_set_before_litellm_is_imported(
    preset: str | None, expected: str
) -> None:
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    if preset is not None:
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = preset
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))

    done = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["early"] is False, "importing the entry point imported litellm"
    assert report["value"] == expected, "litellm was imported before the variable was set"


# --- The rig ------------------------------------------------------------------------------------


class _FakeLock:
    def __init__(self, rig: Rig) -> None:
        self._rig = rig

    def release(self) -> None:
        self._rig.note("unlock")


class _FakeDatabase:
    def __init__(self, rig: Rig, path: Path) -> None:
        self._rig = rig
        self.path = path

    def close(self) -> None:
        self._rig.note("db_close")


class _NoRunners:
    def cancel_all(self) -> None:
        return None


class _FakeUvicornServer:
    """Stands in for `BackendServer`: `serve` marks the server started, or not (lifespan failed)."""

    should_exit = False

    def __init__(self, *, starts: bool = True) -> None:
        self.started = False
        self.served_with: list[socket.socket] | None = None
        self._starts = starts
        self.ctl: ShutdownController | None = None
        self.request_shutdown = False

    async def serve(self, sockets: Sequence[socket.socket]) -> None:
        self.served_with = list(sockets)
        self.started = self._starts
        if self.request_shutdown and self.ctl is not None:
            self.ctl.request(ShutdownReason.SIGNAL)


class _FakeHandle:
    def __init__(self) -> None:
        self.attached: object = None

    def attach(self, server: object) -> None:
        self.attached = server


@dataclasses.dataclass(frozen=True)
class _FakeDeps:
    server_handle: _FakeHandle
    shutdown_controller: ShutdownController


class Rig:
    """Recording fakes for every startup step. `fail_at[name]` is raised when step `name` runs."""

    def __init__(self, tmp_path: Path, *, backend: str = "sqlite", idle: int | None = None) -> None:
        self.root = tmp_path
        self.storage_dir = tmp_path / "data" / "storage"
        self.calls: list[str] = []
        self.args: dict[str, tuple[object, ...]] = {}
        self.fail_at: dict[str, Exception] = {}
        self.sockets: list[socket.socket] = []
        self.database: _FakeDatabase | None = None
        self.lock = _FakeLock(self)
        self.server = _FakeUvicornServer()
        self.handle = _FakeHandle()
        self.ctl = ShutdownController(self.server, _NoRunners())
        self.server.ctl = self.ctl
        self.cfg = self.make_config(backend=backend, idle=idle)

    def make_config(
        self, *, backend: str, idle: int | None, host: str = "127.0.0.1", port: int = 10000
    ) -> BackendConfig:
        orchestration = {} if idle is None else {"session_idle_timeout_s": idle}
        return BackendConfig.model_validate(
            {
                "llm": {"model": "fake/model"},
                "server": {"host": host, "port": port},
                "orchestration": orchestration,
                "storage": {"backend": backend, "path": str(self.storage_dir)},
            }
        )

    def note(self, name: str, *args: object) -> None:
        self.calls.append(name)
        self.args[name] = args
        failure = self.fail_at.get(name)
        if failure is not None:
            raise failure

    # The steps. Each one records its name, then raises its `fail_at` entry if there is one.

    def load_config(self, path: Path, env: Any) -> BackendConfig:
        self.note("load_config", path)
        return self.cfg

    def configure_logging(self, cfg: object, redactor: Redactor) -> None:
        self.note("configure_logging")

    def ensure_storage_dir(self, path: str) -> None:
        self.note("dir", path)

    def restrict_storage(self, directory: str, platform: str) -> None:
        self.note("restrict", directory)

    def acquire_lock(self, directory: Path, platform: str) -> StorageLock:
        self.note("lock", directory)
        return cast(StorageLock, self.lock)

    def open_database(self, path: Path) -> Database:
        self.note("open", path)
        self.database = _FakeDatabase(self, path)
        return cast(Database, self.database)

    def restrict_file(self, path: str, platform: str) -> None:
        self.note("restrict_file", path)

    def apply_migrations(self, db: Database, path: str) -> int:
        self.note("migrate", path)
        return 1

    def run_recovery(self, db: Database, clock: Clock) -> object:
        self.note("recover")
        return None

    def sweep_retention(self, db: Database, clock: Clock, idle_timeout_s: int) -> object:
        self.note("retention", idle_timeout_s)
        return None

    def prepare_llm(self, cfg: BackendConfig) -> None:
        self.note("prepare_llm")

    def bind_all(self, host: str, port: int, platform: str) -> list[socket.socket]:
        self.note("bind", host, port)
        sock = socket.socket()
        self.sockets.append(sock)
        return [sock]

    def build_app(
        self, cfg: BackendConfig, database: Database | None, config_dir: str, clock: Clock, r: Any
    ) -> tuple[ASGIApp, Deps]:
        self.note("build_app", database, config_dir)
        return _app, cast(Deps, _FakeDeps(self.handle, self.ctl))

    def build_server(self, app: ASGIApp, host: str, port: int) -> BackendServer:
        self.note("build_server", host, port)
        return cast(BackendServer, self.server)

    def install_signals(
        self, loop: Any, ctl: ShutdownController, platform: str
    ) -> Callable[[], None]:
        self.note("install_signals")
        return lambda: self.note("uninstall_signals")

    async def serve(self, server: Any, ctl: ShutdownController, sockets: Any) -> int:
        self.note("mcp")
        return EXIT_OK

    def steps(self, **overrides: Any) -> StartupSteps:
        base = StartupSteps(
            load_config=self.load_config,
            configure_logging=self.configure_logging,
            ensure_storage_dir=self.ensure_storage_dir,
            restrict_storage=self.restrict_storage,
            acquire_lock=self.acquire_lock,
            open_database=self.open_database,
            restrict_file=self.restrict_file,
            apply_migrations=self.apply_migrations,
            run_recovery=self.run_recovery,
            sweep_retention=self.sweep_retention,
            prepare_llm=self.prepare_llm,
            bind_all=self.bind_all,
            build_app=self.build_app,
            build_server=self.build_server,
            loop_factory=lambda: asyncio.new_event_loop,
            install_signals=self.install_signals,
            serve=self.serve,
        )
        return dataclasses.replace(base, **overrides)

    def run(self, steps: StartupSteps | None = None) -> int:
        return main(
            env={},
            cwd=self.root,
            platform=sys.platform,
            steps=self.steps() if steps is None else steps,
        )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path, idle=IDLE_TIMEOUT_S)


# --- Startup order ------------------------------------------------------------------------------


def test_sqlite_startup_runs_the_steps_in_order(rig: Rig) -> None:
    assert rig.run() == EXIT_OK

    assert rig.calls == SQLITE_ORDER
    db_path = rig.storage_dir / "local-agent.db"
    assert rig.args["dir"] == (str(rig.storage_dir),)
    assert rig.args["lock"] == (rig.storage_dir,)
    assert rig.args["open"] == (db_path,)
    assert rig.args["restrict_file"] == (str(db_path),)
    assert rig.args["migrate"] == (str(db_path),)
    assert rig.args["retention"] == (IDLE_TIMEOUT_S,)
    assert rig.args["bind"] == ("127.0.0.1", 10000)
    assert rig.args["build_app"][0] is rig.database


def test_started_server_gets_the_sockets_and_the_shutdown_handle(rig: Rig) -> None:
    steps = rig.steps(serve=StartupSteps().serve)

    assert rig.run(steps) == EXIT_OK

    assert rig.handle.attached is rig.server
    assert rig.server.served_with == rig.sockets


def test_sockets_database_and_lock_are_released_in_reverse_order(rig: Rig) -> None:
    assert rig.run() == EXIT_OK

    assert all(sock.fileno() == -1 for sock in rig.sockets)
    assert rig.calls[-2:] == ["db_close", "unlock"]


def test_retention_sweep_is_skipped_without_an_idle_timeout(tmp_path: Path) -> None:
    rig = Rig(tmp_path, idle=None)

    assert rig.run() == EXIT_OK

    assert rig.calls == [name for name in SQLITE_ORDER if name != "retention"]


# --- Configuration errors -----------------------------------------------------------------------


def test_config_error_exits_2_before_any_other_step(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    issue = ConfigIssue("llm.model", "is required")
    rig.fail_at["load_config"] = ConfigError([issue])

    assert rig.run() == EXIT_CONFIG

    assert rig.calls == ["load_config"]
    assert "llm.model: is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text",
    [None, "llm: [unclosed", "llm:\n  model: m\nstorage:\n  backend: postgres\n"],
    ids=["missing file", "yaml syntax error", "invalid storage backend"],
)
def test_real_loader_config_errors_exit_2_before_any_step(
    rig: Rig, text: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    if text is not None:
        (rig.root / "config.yaml").write_text(text, encoding="utf-8")

    assert rig.run(rig.steps(load_config=loader.load_config)) == EXIT_CONFIG

    assert rig.calls == []
    assert capsys.readouterr().err.startswith("invalid configuration:")


# --- Storage failures ---------------------------------------------------------------------------

DB_PATH_TEXT: Final = "/data/storage/local-agent.db"


def _garbage_database(rig: Rig) -> StartupSteps:
    rig.storage_dir.mkdir(parents=True)
    (rig.storage_dir / "local-agent.db").write_bytes(b"this is not a database " * 200)
    return rig.steps(open_database=StartupSteps().open_database)


def _path_is_a_file(rig: Rig) -> StartupSteps:
    rig.storage_dir.parent.mkdir(parents=True)
    rig.storage_dir.write_text("a file where the directory should be")
    return rig.steps(ensure_storage_dir=paths.ensure_storage_dir)


def _cannot_create(rig: Rig) -> StartupSteps:
    rig.storage_dir.parent.write_text("a file where the parent directory should be")
    return rig.steps(ensure_storage_dir=paths.ensure_storage_dir)


def _injected(step: str, exc: Exception) -> Callable[[Rig], StartupSteps]:
    def setup(rig: Rig) -> StartupSteps:
        rig.fail_at[step] = exc
        return rig.steps()

    return setup


def _unresolvable_path(rig: Rig) -> StartupSteps:
    def storage_directory(cfg: BackendConfig, env: Any, platform: str) -> str:
        raise PathFieldError("storage.path cannot be resolved")

    return rig.steps(storage_directory=storage_directory)


STORAGE_FAILURES: Final = [
    pytest.param(_path_is_a_file, id="path is a file"),
    pytest.param(_cannot_create, id="cannot create"),
    pytest.param(_unresolvable_path, id="path unresolvable"),
    pytest.param(
        _injected("restrict", PermissionSetupError("/data/storage", "Operation not permitted")),
        id="restrict fails",
    ),
    pytest.param(
        _injected("lock", StorageError("/data/storage/local-agent.lock", "Permission denied")),
        id="lock file cannot be opened",
    ),
    pytest.param(_garbage_database, id="garbage database"),
    pytest.param(
        _injected("restrict_file", PermissionSetupError(DB_PATH_TEXT, "Operation not permitted")),
        id="restrict database fails",
    ),
    pytest.param(_injected("migrate", SchemaError(DB_PATH_TEXT, 99, 1)), id="newer schema"),
    pytest.param(
        _injected("migrate", MigrationError(DB_PATH_TEXT, 1, 2, "disk is full")),
        id="migration fails",
    ),
    pytest.param(
        _injected("recover", StorageError(DB_PATH_TEXT, "disk I/O error")), id="recovery fails"
    ),
]


@pytest.mark.parametrize("setup", STORAGE_FAILURES)
def test_storage_failure_exits_2_before_bind(
    rig: Rig, setup: Callable[[Rig], StartupSteps], capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = rig.run(setup(rig))

    assert exit_code == EXIT_CONFIG
    for name in AFTER_STORAGE:
        assert name not in rig.calls
    assert rig.sockets == []
    report = capsys.readouterr().err
    assert report.startswith("invalid configuration:")
    assert "storage.path" in report
    if "lock" in rig.calls and "lock" not in rig.fail_at:
        assert "unlock" in rig.calls
    if rig.database is not None:
        assert "db_close" in rig.calls


def test_storage_failure_stops_the_sequence_at_the_failed_step(rig: Rig) -> None:
    rig.fail_at["migrate"] = SchemaError(DB_PATH_TEXT, 99, 1)

    assert rig.run() == EXIT_CONFIG

    assert rig.calls == [
        "load_config",
        "configure_logging",
        "dir",
        "restrict",
        "lock",
        "open",
        "restrict_file",
        "migrate",
        "db_close",
        "unlock",
    ]


# --- Held lock ----------------------------------------------------------------------------------


def test_held_lock_exits_1_before_the_database_is_opened(rig: Rig) -> None:
    rig.fail_at["lock"] = StorageInUse(str(rig.storage_dir / "local-agent.lock"))

    with capture_logs() as logs:
        exit_code = rig.run()

    assert exit_code == EXIT_FAILURE
    assert rig.calls == ["load_config", "configure_logging", "dir", "restrict", "lock"]
    [record] = [r for r in logs if r["event"] == "storage_in_use"]
    assert record["log_level"] == "error"
    assert record["database_path"] == str(rig.storage_dir / "local-agent.db")


@posix_only
def test_real_storage_lock_held_by_another_holder_exits_1(rig: Rig) -> None:
    rig.storage_dir.mkdir(parents=True)
    holder = StorageLock.acquire(rig.storage_dir, sys.platform)
    try:
        exit_code = rig.run(rig.steps(acquire_lock=StorageLock.acquire))
    finally:
        holder.release()

    assert exit_code == EXIT_FAILURE
    assert "open" not in rig.calls
    for name in AFTER_STORAGE:
        assert name not in rig.calls


# --- Bind errors --------------------------------------------------------------------------------


def test_bind_error_exits_1_before_any_mcp_server_starts(rig: Rig) -> None:
    rig.fail_at["bind"] = BindError("127.0.0.1", 10000, "Address already in use")

    with capture_logs() as logs:
        exit_code = rig.run()

    assert exit_code == EXIT_FAILURE
    for name in ("build_app", "build_server", "install_signals", "mcp"):
        assert name not in rig.calls
    assert rig.calls[-3:] == ["bind", "db_close", "unlock"]
    [record] = [r for r in logs if r["event"] == "bind_failed"]
    assert record["log_level"] == "error"
    assert (record["host"], record["port"]) == ("127.0.0.1", 10000)
    assert record["reason"] == "Address already in use"


def test_bind_error_with_memory_backend_exits_1_before_any_mcp_server_starts(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, backend="memory")
    rig.fail_at["bind"] = BindError("127.0.0.1", 10000, "Address already in use")

    assert rig.run() == EXIT_FAILURE

    assert rig.calls == ["load_config", "configure_logging", "prepare_llm", "bind"]


# --- Lifespan failure and clean exit ------------------------------------------------------------


@pytest.mark.parametrize(
    ("starts", "request_shutdown", "expected"),
    [(True, False, EXIT_OK), (False, False, EXIT_FAILURE), (False, True, EXIT_OK)],
    ids=["served", "lifespan failed", "shutdown requested"],
)
def test_exit_status_after_serve(
    rig: Rig, starts: bool, request_shutdown: bool, expected: int
) -> None:
    rig.server = _FakeUvicornServer(starts=starts)
    rig.server.ctl = rig.ctl
    rig.server.request_shutdown = request_shutdown

    assert rig.run(rig.steps(serve=StartupSteps().serve)) == expected


# --- storage.backend: memory --------------------------------------------------------------------


@pytest.fixture
def memory_rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path, backend="memory", idle=IDLE_TIMEOUT_S)


def test_memory_backend_runs_no_storage_step(memory_rig: Rig) -> None:
    assert memory_rig.run() == EXIT_OK

    assert memory_rig.calls == MEMORY_ORDER
    assert STORAGE_STEPS.isdisjoint(memory_rig.calls)
    assert memory_rig.args["build_app"][0] is None


def test_memory_backend_creates_no_file_under_the_storage_directory(memory_rig: Rig) -> None:
    defaults = StartupSteps()
    real_storage = {
        name: getattr(defaults, name)
        for name in (
            "storage_directory",
            "ensure_storage_dir",
            "restrict_storage",
            "acquire_lock",
            "open_database",
            "restrict_file",
            "apply_migrations",
            "run_recovery",
            "sweep_retention",
        )
    }
    watched = memory_rig.storage_dir
    watched.mkdir(parents=True)

    exit_code = memory_rig.run(memory_rig.steps(**real_storage))

    assert exit_code == EXIT_OK
    assert memory_rig.calls == MEMORY_ORDER
    assert list(watched.iterdir()) == []
    assert [p for p in memory_rig.root.rglob("*") if p not in (watched, watched.parent)] == []


def test_memory_backend_does_not_create_the_storage_directory(memory_rig: Rig) -> None:
    defaults = StartupSteps()

    exit_code = memory_rig.run(
        memory_rig.steps(
            ensure_storage_dir=defaults.ensure_storage_dir,
            open_database=defaults.open_database,
            acquire_lock=defaults.acquire_lock,
        )
    )

    assert exit_code == EXIT_OK
    assert not memory_rig.storage_dir.exists()
    assert not memory_rig.storage_dir.parent.exists()
