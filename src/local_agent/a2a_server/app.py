"""App wiring: the `Deps` singletons, `build_deps`, `build_app`, and the Starlette lifespan.

`build_deps` constructs every shared object once, so a test can substitute fakes with
`dataclasses.replace(deps, ...)` or by building `Deps` directly. `build_app` turns `Deps` into a
Starlette app:

- Routes: the Agent_Card, the JSON-RPC route (`HostJsonRpcDispatcher`, used instead of the SDK's
  `create_jsonrpc_routes`, which takes no dispatcher class), the REST binding, and `/health`.
- Middleware, outermost first: `LoopbackGuardMiddleware` -> `CORSMiddleware` ->
  `BearerAuthMiddleware`. CORS sits outside auth, so a 401 for an allowed origin still carries its
  CORS headers (16.2, 16.4).
- Lifespan: start `McpManager`, start the retention task, and on shutdown stop retention, cancel
  any remaining runners, stop `McpManager`, close the `Database`, and tell the
  `ShutdownController` the lifespan is over (17.14).

The REST routes come last on purpose. The SDK adds a `Mount("/{tenant}")` that matches by path
shape alone and answers 404 for anything its children do not know, so any route listed after it
and sharing a one-segment prefix (`/health`) would never be reached.
"""

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Final

from a2a.server.routes import create_agent_card_routes, create_rest_routes
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import BaseRoute, Route
from structlog.typing import FilteringBoundLogger

from local_agent.a2a_server.agent_card import build_agent_card
from local_agent.a2a_server.auth import (
    BearerAuthMiddleware,
    LoopbackGuardMiddleware,
    is_loopback,
)
from local_agent.a2a_server.dispatch import HostJsonRpcDispatcher
from local_agent.a2a_server.errors import install_error_mappings
from local_agent.a2a_server.executor import HostAgentExecutor, HostRequestHandler
from local_agent.a2a_server.health import health_endpoint
from local_agent.config.models import BackendConfig
from local_agent.llm.gateway import LlmGateway
from local_agent.logging_setup import Redactor, collect_secrets, get_logger
from local_agent.mcp_client.manager import McpManager
from local_agent.mcp_client.registry import ToolRegistry
from local_agent.models import Clock, SystemClock
from local_agent.orchestrator.loop import RunnerDeps, TaskRunnerRegistry
from local_agent.oscompat.signals import ShutdownController, ShutdownServer
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import SessionStore
from local_agent.storage import retention
from local_agent.storage.database import Database
from local_agent.storage.task_store import ProjectedTaskStore

__all__ = [
    "MEMORY_DB_PATH",
    "RETENTION_INTERVAL_S",
    "Deps",
    "ServerHandle",
    "build_app",
    "build_deps",
    "lifespan",
]

RETENTION_INTERVAL_S: Final = 30.0
"""Seconds between retention sweeps (11.11 allows 60 s from expiry to deletion)."""

MEMORY_DB_PATH: Final = ":memory:"
"""The `db_path` shown in storage error logs when the backend is `memory`."""

_log = get_logger("a2a_server.app")


# --- Shutdown hand-off --------------------------------------------------------------------------


class ServerHandle:
    """The `should_exit` flag of the uvicorn server, which does not exist yet when Deps is built.

    `ShutdownController` needs the server, uvicorn needs the app, and the app needs `Deps`. The
    handle breaks that cycle: `build_deps` gives it to the controller, and `__main__` calls
    `attach(server)` once the `BackendServer` exists. A request made before `attach` is applied to
    the server when it is attached.
    """

    def __init__(self, server: ShutdownServer | None = None) -> None:
        self._server = server
        self._should_exit = False

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._should_exit = value
        if self._server is not None:
            self._server.should_exit = value

    def attach(self, server: ShutdownServer) -> None:
        self._server = server
        if self._should_exit:
            server.should_exit = True


# --- Deps ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Deps:
    """The wired singletons of one Backend process.

    `database` is `None` for the memory backend. `shutdown_controller` is created here because it
    needs `runners`; `__main__` runs it and installs the signal handlers.
    """

    config: BackendConfig
    session_store: SessionStore
    task_store: ProjectedTaskStore
    database: Database | None
    clock: Clock
    mcp_manager: McpManager
    registry: ToolRegistry
    llm_gateway: LlmGateway
    runners: TaskRunnerRegistry
    logger: FilteringBoundLogger
    shutdown_controller: ShutdownController
    server_handle: ServerHandle


def build_deps(
    cfg: BackendConfig,
    database: Database | None,
    *,
    config_dir: str | None = None,
    clock: Clock | None = None,
    redactor: Redactor | None = None,
) -> Deps:
    """Build every shared object for `cfg`.

    `storage.backend: sqlite` needs the opened `database` and gets `SQLiteSessionStore`;
    `memory` gets `InMemorySessionStore` and ignores `database`. Both get `ProjectedTaskStore`
    (18.1, 18.2). `config_dir` anchors relative `stdio` paths of MCP servers (17.19). It defaults
    to the working directory.
    """
    the_clock: Clock = clock if clock is not None else SystemClock()
    the_redactor = redactor if redactor is not None else Redactor(collect_secrets(cfg))
    sessions: SessionStore
    db: Database | None
    if cfg.storage.backend == "sqlite":
        if database is None:
            raise ValueError("storage.backend 'sqlite' needs an open Database")
        db = database
        sessions = SQLiteSessionStore(db, the_redactor, the_clock)
        db_path = str(db.path)
    else:
        db = None
        sessions = InMemorySessionStore(the_clock)
        db_path = MEMORY_DB_PATH

    registry = ToolRegistry([server.name for server in cfg.mcp_servers], cfg.permissions)
    manager = McpManager(
        cfg.mcp_servers,
        registry,
        config_dir=config_dir if config_dir is not None else os.getcwd(),
        clock=the_clock,
        redactor=the_redactor,
    )
    gateway = LlmGateway(cfg.llm, clock=the_clock)
    runners = TaskRunnerRegistry(
        RunnerDeps(
            sessions=sessions,
            llm=gateway,
            registry=registry,
            mcp=manager,
            orchestration=cfg.orchestration,
            base_instructions=cfg.agent.base_instructions,
            model=cfg.llm.model,
            redactor=the_redactor,
            clock=the_clock,
            db_path=db_path,
        )
    )
    handle = ServerHandle()
    return Deps(
        config=cfg,
        session_store=sessions,
        task_store=ProjectedTaskStore(sessions, runners, the_clock),
        database=db,
        clock=the_clock,
        mcp_manager=manager,
        registry=registry,
        llm_gateway=gateway,
        runners=runners,
        logger=get_logger("app"),
        shutdown_controller=ShutdownController(handle, runners, clock=the_clock),
        server_handle=handle,
    )


# --- Lifespan -----------------------------------------------------------------------------------


async def _retention_loop(deps: Deps, idle_timeout_s: int, interval_s: float) -> None:
    """Sweep expired Contexts now and then every `interval_s` until cancelled (11.11)."""
    while True:
        try:
            await retention.sweep(deps.session_store, deps.runners, deps.clock, idle_timeout_s)
        except Exception as exc:  # a failed sweep is retried on the next round
            deps.logger.error(
                "retention_sweep_failed", error_type=type(exc).__name__, reason=str(exc)
            )
        await asyncio.sleep(interval_s)


async def _stop_retention(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _shutdown(deps: Deps, retention_task: asyncio.Task[None] | None) -> None:
    """Stop retention, cancel leftover runners, stop MCP, close the database.

    A failing step is logged and does not skip the next one.
    """
    log = deps.logger
    try:
        await _stop_retention(retention_task)
    except Exception as exc:
        log.error("lifespan_stop_retention_failed", error_type=type(exc).__name__)
    try:
        await deps.runners.cancel_all()  # idempotent: the signal path already did this
    except Exception as exc:
        log.error("lifespan_cancel_runners_failed", error_type=type(exc).__name__)
    try:
        ctl = deps.shutdown_controller
        await deps.mcp_manager.stop(grace=lambda: ctl.grace_s)
    except Exception as exc:
        log.error("lifespan_mcp_stop_failed", error_type=type(exc).__name__)
    if deps.database is not None:
        try:
            await asyncio.to_thread(deps.database.close)
        except Exception as exc:
            log.error("lifespan_database_close_failed", error_type=type(exc).__name__)


def lifespan(
    deps: Deps, *, retention_interval_s: float = RETENTION_INTERVAL_S
) -> Callable[[Starlette], AbstractAsyncContextManager[None]]:
    """The Starlette lifespan of `deps`.

    Startup: `McpManager.start()` (6.5), then the retention task when `session_idle_timeout_s` is
    set (11.13). Shutdown: stop retention, cancel any remaining runners, `McpManager.stop(grace=
    shutdown_controller.grace_s)`, `Database.close()`, and finally
    `ShutdownController.lifespan_finished()`, which runs even if an earlier step failed.
    """

    @contextlib.asynccontextmanager
    async def run(_app: Starlette) -> AsyncIterator[None]:
        retention_task: asyncio.Task[None] | None = None
        try:
            await deps.mcp_manager.start()
            idle_timeout_s = deps.config.orchestration.session_idle_timeout_s
            if idle_timeout_s is not None:
                retention_task = asyncio.create_task(
                    _retention_loop(deps, idle_timeout_s, retention_interval_s),
                    name="retention",
                )
            yield
        finally:
            try:
                await _shutdown(deps, retention_task)
            finally:
                deps.shutdown_controller.lifespan_finished()

    return run


# --- App ----------------------------------------------------------------------------------------


def build_app(cfg: BackendConfig, deps: Deps) -> Starlette:
    """Build the Starlette app for `cfg` and `deps` (1.1, 1.2, 16.2, 16.4, 16.6)."""
    install_error_mappings()  # REST: ContextBusyError -> 409, TaskNotAwaitingInputError -> 400
    card = build_agent_card(cfg)
    executor = HostAgentExecutor(
        sessions=deps.session_store,
        runners=deps.runners,
        task_store=deps.task_store,
        approval_keywords=cfg.orchestration.approval_keywords,
        clock=deps.clock,
    )
    handler = HostRequestHandler(executor, task_store=deps.task_store, agent_card=card)
    jsonrpc = HostJsonRpcDispatcher(handler)  # renders HostContractError exactly (1.14, 1.15)
    routes: list[BaseRoute] = [
        *create_agent_card_routes(card),
        Route("/", jsonrpc.handle_requests, methods=["POST"]),  # replaces create_jsonrpc_routes
        Route("/health", health_endpoint, methods=["GET"]),
        *create_rest_routes(handler),
    ]
    loopback = is_loopback(cfg.server.host)
    token = cfg.server.bearer_token
    middleware = [
        Middleware(LoopbackGuardMiddleware, enabled=loopback),
        Middleware(
            CORSMiddleware,
            allow_origins=list(cfg.server.cors_allowed_origins),
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        ),
        Middleware(
            BearerAuthMiddleware,
            token=None if token is None else token.get_secret_value(),
            enabled=not loopback,
            exempt_paths={AGENT_CARD_WELL_KNOWN_PATH},
        ),
    ]
    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan(deps))
    app.state.mcp_manager = deps.mcp_manager  # read by `health_endpoint`
    app.state.deps = deps
    return app
