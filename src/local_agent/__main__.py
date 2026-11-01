"""The Backend process entry point: `python -m local_agent` (design.md, "Process lifecycle").

`main` runs the startup steps in a fixed order and returns the exit status:

1. Load the Backend_Config (`AGENT_CONFIG` or `./config.yaml`). A `ConfigError` prints its report
   and stops with status 2 before any other step (13.3, 13.12).
2. Configure logging, then log one `platform_unsupported` warning when the platform is outside the
   Supported_Platform set (17.2).
3. For `storage.backend: sqlite` only (`memory` touches no file, takes no lock, runs no migration or
   recovery, 18.2): resolve and create the Storage_Directory, restrict it, take the Storage_Lock,
   open the Storage_Database, restrict the database and its sidecars, apply migrations, run
   Restart_Recovery, and sweep expired Contexts once. Every step blocks and runs before the event
   loop exists (14.1). A storage failure prints a config-error style report naming `storage.path`
   and stops with status 2 (16.9, 18.21, 18.22, 18.24). A held lock logs `storage_in_use` and stops
   with status 1 (18.26).
4. Set `LITELLM_LOCAL_MODEL_COST_MAP` before anything imports litellm, apply the function-calling
   mode (10.8), and bind the listening sockets. A `BindError` logs the host and port and stops with
   status 1 (1.8), so no MCP_Server has been spawned yet.
5. Build the app, create the loop with `asyncio.Runner`, install the OS signal handlers, and run
   `BackendServer.serve(sockets=...)`. uvicorn's startup runs the lifespan, which starts the
   MCP_Client. A completed Graceful_Shutdown returns status 0 (17.14).

Nothing above `prepare_llm` may import litellm, directly or through `llm.gateway`, `a2a_server.app`
or the orchestrator. `build_app` and `prepare_llm` import those lazily for that reason.

Every step is a field of `StartupSteps`, so a test replaces any of them with a fake and records the
call order.
"""

import asyncio
import contextlib
import os
import socket
import sys
from asyncio import AbstractEventLoop
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Final

from starlette.types import ASGIApp

from local_agent import oscompat, server
from local_agent.config import loader
from local_agent.config.loader import ConfigError, ConfigIssue
from local_agent.config.models import BackendConfig, LoggingConfig, storage_directory
from local_agent.logging_setup import Redactor, collect_secrets, configure_logging, get_logger
from local_agent.models import Clock, SystemClock
from local_agent.oscompat import event_loop, net, paths, permissions, signals
from local_agent.oscompat.net import BindError
from local_agent.oscompat.paths import PathFieldError
from local_agent.oscompat.permissions import PermissionSetupError
from local_agent.server import BackendServer
from local_agent.storage import StorageError, migrations, recovery, retention
from local_agent.storage.database import Database
from local_agent.storage.lock import StorageInUse, StorageLock
from local_agent.storage.migrations import MigrationError, SchemaError

if TYPE_CHECKING:
    from local_agent.a2a_server.app import Deps
    from local_agent.oscompat.signals import ShutdownController

__all__ = [
    "DATABASE_FILE_NAME",
    "EXIT_CONFIG",
    "EXIT_FAILURE",
    "EXIT_OK",
    "StartupSteps",
    "main",
    "run",
]

EXIT_OK: Final = 0
EXIT_FAILURE: Final = 1
"""A held Storage_Lock (18.26), a bind failure (1.8), or a server that never started."""
EXIT_CONFIG: Final = 2
"""Configuration, Storage_Directory, permission, database, schema, migration, recovery failure."""

DATABASE_FILE_NAME: Final = permissions.STORAGE_FILES[0]
_SIDECAR_FILE_NAMES: Final = permissions.STORAGE_FILES[1:3]  # the -wal and -shm files

_log = get_logger("main")


# --- Default steps ------------------------------------------------------------------------------


def _open_database(path: Path) -> Database:
    return Database.open(path, create=True)


def _prepare_llm(cfg: BackendConfig) -> None:
    """Set the LiteLLM cost-map variable, then import litellm and apply the calling mode (10.8).

    `LITELLM_LOCAL_MODEL_COST_MAP` is read when litellm is imported, so the import stays below it.
    """
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from local_agent.llm.gateway import apply_function_calling_mode

    apply_function_calling_mode(cfg.llm)


def _build_app(
    cfg: BackendConfig,
    database: Database | None,
    config_dir: str,
    clock: Clock,
    redactor: Redactor,
) -> tuple[ASGIApp, Deps]:
    from local_agent.a2a_server.app import build_app, build_deps

    deps = build_deps(cfg, database, config_dir=config_dir, clock=clock, redactor=redactor)
    return build_app(cfg, deps), deps


async def _serve(
    uvicorn_server: BackendServer, ctl: ShutdownController, sockets: Sequence[socket.socket]
) -> int:
    """Run uvicorn and the `ShutdownController` side by side; return the process exit status.

    uvicorn runs the lifespan, which starts the MCP_Client on startup and calls
    `ctl.lifespan_finished()` as its last shutdown step. A server that stopped without a shutdown
    request and without ever finishing its startup (the lifespan failed) is status 1.
    """
    ctl_task = asyncio.create_task(ctl.run(), name="shutdown-controller")
    try:
        await uvicorn_server.serve(sockets=list(sockets))
    finally:
        ctl.lifespan_finished()  # uvicorn is done; this only matters if no lifespan ever ran
        if not ctl.requested:
            ctl_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ctl_task
    if ctl.requested or uvicorn_server.started:
        return EXIT_OK
    return EXIT_FAILURE


@dataclass(frozen=True)
class StartupSteps:
    """Every step of `main`, replaceable one by one in tests."""

    load_config: Callable[[Path, Mapping[str, str]], BackendConfig] = loader.load_config
    configure_logging: Callable[[LoggingConfig, Redactor], None] = configure_logging
    gather_facts: Callable[[], oscompat.PlatformFacts] = oscompat.gather_facts
    classify_platform: Callable[[oscompat.PlatformFacts], oscompat.PlatformVerdict] = (
        oscompat.classify_platform
    )
    storage_directory: Callable[[BackendConfig, Mapping[str, str], str], str | None] = (
        storage_directory
    )
    ensure_storage_dir: Callable[[str], None] = paths.ensure_storage_dir
    restrict_storage: Callable[[str, str], None] = permissions.restrict_storage
    acquire_lock: Callable[[Path, str], StorageLock] = StorageLock.acquire
    open_database: Callable[[Path], Database] = _open_database
    restrict_file: Callable[[str, str], None] = permissions.restrict_file
    apply_migrations: Callable[[Database, str], int] = migrations.apply
    run_recovery: Callable[[Database, Clock], object] = recovery.run
    sweep_retention: Callable[[Database, Clock, int], object] = retention.sweep_at_startup
    prepare_llm: Callable[[BackendConfig], None] = _prepare_llm
    bind_all: Callable[[str, int, str], Sequence[socket.socket]] = net.bind_all
    build_app: Callable[
        [BackendConfig, Database | None, str, Clock, Redactor], tuple[ASGIApp, Deps]
    ] = _build_app
    build_server: Callable[[ASGIApp, str, int], BackendServer] = server.build_server
    loop_factory: Callable[[], Callable[[], AbstractEventLoop]] = event_loop.default_loop_factory
    install_signals: Callable[[AbstractEventLoop, ShutdownController, str], Callable[[], None]] = (
        signals.install
    )
    serve: Callable[
        [BackendServer, ShutdownController, Sequence[socket.socket]], Coroutine[Any, Any, int]
    ] = _serve


# --- Reporting ----------------------------------------------------------------------------------


def _write_stderr(text: str, stream: BinaryIO | None = None) -> None:
    """Write `text` and one LF as UTF-8 bytes, whatever the console code page is (17.26)."""
    target = stream
    if target is None:
        target = getattr(sys.stderr, "buffer", None)
    if target is None:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
        return
    target.write(text.encode("utf-8") + b"\n")
    target.flush()


def _storage_failure(exc: Exception, storage_dir: str | None) -> ConfigError:
    """The config-error report for a storage failure: `storage.path`, the path, and the reason."""
    path = storage_dir or ""
    reason = str(exc)
    if isinstance(exc, StorageError):
        path, reason = exc.path, exc.reason
    elif isinstance(exc, PermissionSetupError):
        path, reason = exc.path, exc.os_reason
    elif isinstance(exc, SchemaError):
        path, reason = exc.path, exc.reason
    elif isinstance(exc, MigrationError):
        path = exc.path
        reason = (
            f"migration from schema version {exc.from_version} to {exc.target_version} "
            "failed and was rolled back"
        )
        if exc.reason:
            reason = f"{reason}: {exc.reason}"
    message = f"{path}: {reason}" if path else reason
    return ConfigError([ConfigIssue("storage.path", message)])


def _close_sockets(sockets: Sequence[socket.socket]) -> None:
    for sock in sockets:
        with contextlib.suppress(OSError):
            sock.close()


# --- Startup ------------------------------------------------------------------------------------


def _open_storage(
    cfg: BackendConfig,
    stack: contextlib.ExitStack,
    steps: StartupSteps,
    env: Mapping[str, str],
    platform: str,
    clock: Clock,
) -> Database:
    """The storage startup sequence of 18.28. The lock and the database are closed by `stack`."""
    directory = steps.storage_directory(cfg, env, platform)
    if directory is None:  # unreachable for `sqlite`; keeps the type honest
        raise StorageError("", "no storage directory for the sqlite backend")
    steps.ensure_storage_dir(directory)
    steps.restrict_storage(directory, platform)

    db_path = Path(directory) / DATABASE_FILE_NAME
    try:
        lock = steps.acquire_lock(Path(directory), platform)
    except StorageInUse:
        _log.error("storage_in_use", reason="storage is in use", database_path=str(db_path))
        raise
    stack.callback(lock.release)

    database = steps.open_database(db_path)
    stack.callback(database.close)
    steps.restrict_file(str(db_path), platform)
    for name in _SIDECAR_FILE_NAMES:
        sidecar = Path(directory) / name
        if sidecar.exists():
            steps.restrict_file(str(sidecar), platform)

    steps.apply_migrations(database, str(db_path))
    steps.run_recovery(database, clock)
    idle_timeout_s = cfg.orchestration.session_idle_timeout_s
    if idle_timeout_s is not None:
        steps.sweep_retention(database, clock, idle_timeout_s)
    return database


def main(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    platform: str | None = None,
    steps: StartupSteps | None = None,
) -> int:
    """Start the Backend, serve until Graceful_Shutdown, and return the exit status."""
    environment = os.environ if env is None else env
    plat = sys.platform if platform is None else platform
    the_steps = StartupSteps() if steps is None else steps
    clock: Clock = SystemClock()

    try:
        config_path = loader.resolve_config_path(
            environment, Path.cwd() if cwd is None else cwd, platform=plat
        )
        cfg = the_steps.load_config(config_path, environment)
    except ConfigError as exc:
        _write_stderr(str(exc))
        return EXIT_CONFIG

    redactor = Redactor(collect_secrets(cfg))
    the_steps.configure_logging(cfg.logging, redactor)
    verdict = the_steps.classify_platform(the_steps.gather_facts())
    if not verdict.supported:
        _log.warning(
            "platform_unsupported",
            os=verdict.os_label,
            arch=verdict.arch,
            python_build=verdict.build,
        )

    with contextlib.ExitStack() as stack:
        database: Database | None = None
        if cfg.storage.backend == "sqlite":
            try:
                database = _open_storage(cfg, stack, the_steps, environment, plat, clock)
            except StorageInUse:
                return EXIT_FAILURE
            except (
                StorageError,
                PathFieldError,
                PermissionSetupError,
                SchemaError,
                MigrationError,
            ) as exc:
                _write_stderr(str(_storage_failure(exc, cfg.storage.path)))
                return EXIT_CONFIG

        the_steps.prepare_llm(cfg)

        try:
            sockets = the_steps.bind_all(cfg.server.host, cfg.server.port, plat)
        except BindError as exc:
            _log.error(
                "bind_failed",
                host=exc.host,
                port=exc.port,
                reason=exc.os_error,
                message=str(exc),
            )
            return EXIT_FAILURE
        stack.callback(_close_sockets, sockets)

        config_dir = os.path.dirname(os.path.abspath(config_path))
        app, deps = the_steps.build_app(cfg, database, config_dir, clock, redactor)
        uvicorn_server = the_steps.build_server(app, cfg.server.host, cfg.server.port)
        deps.server_handle.attach(uvicorn_server)

        with asyncio.Runner(loop_factory=the_steps.loop_factory()) as runner:
            uninstall = the_steps.install_signals(runner.get_loop(), deps.shutdown_controller, plat)
            try:
                return runner.run(
                    the_steps.serve(uvicorn_server, deps.shutdown_controller, sockets)
                )
            finally:
                uninstall()


def run() -> None:
    """Console entry: run `main` and exit with its status."""
    sys.exit(main())


if __name__ == "__main__":
    run()
