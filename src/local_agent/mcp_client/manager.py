"""MCP client manager: the session adapter, the per-server supervisor, and the manager.

`McpSessionAdapter` is the only class here that touches the `mcp` SDK's session API. It turns the
SDK's pydantic results into the Backend's own dataclasses (`McpToolInfo`, `ContentItem`) and
reports every timeout as the builtin `TimeoutError`, so callers need to know nothing about the
SDK's field spellings (snake_case in 2.x, camelCase in 1.x), its timeout argument type, or its
`MCPError` timeout code.

`ServerSupervisor` owns one enabled server: it opens the transport, runs the handshake, registers
the tools, waits until the connection ends, and reconnects with exponential backoff (6.5 to 6.10,
7.10, 7.11). Leaving the transport context always runs the Process_Tree termination of a `stdio`
server, so a failed handshake and every reconnection start from a dead tree (17.11).

`McpManager` owns the supervisors and is the only object the rest of the Backend talks to:
`start`, `stop`, `refresh_available`, `availability`, `statuses`, and `call_tool`.
"""

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Final, Protocol, Self

import anyio

from local_agent.config.models import SseServerConfig, StdioServerConfig
from local_agent.logging_setup import Redactor, get_logger
from local_agent.mcp_client.mcp_compat import (
    ClientSession,
    ReadStream,
    SessionMessage,
    WriteStream,
    is_read_timeout,
    page_params,
    read_attr,
    sse_client,
    to_read_timeout,
)
from local_agent.mcp_client.stdio_transport import managed_stdio_client
from local_agent.models import (
    AudioItem,
    Clock,
    ContentItem,
    EmbeddedResourceItem,
    ImageItem,
    McpToolInfo,
    ResourceLinkItem,
    ServerStatus,
    SystemClock,
    TextItem,
)
from local_agent.oscompat.process import DEFAULT_GRACE_S, SpawnRejected, build_spawn_spec

__all__ = [
    "CallOutcome",
    "McpManager",
    "McpServerUnavailable",
    "McpSession",
    "McpSessionAdapter",
    "ServerState",
    "ServerSupervisor",
    "ToolSession",
    "TransportFactory",
    "backoff_delay",
    "open_transport",
]

_log = get_logger("mcp_client.manager")


@dataclass(frozen=True)
class CallOutcome:
    """The result of one ``tools/call``: its content items and the server's ``isError`` flag."""

    content: tuple[ContentItem, ...]
    is_error: bool


class ToolSession(Protocol):
    """The part of ``mcp.ClientSession`` the adapter uses; tests can supply a fake."""

    async def list_tools(self, *, params: Any = None) -> Any: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: Any = None,
    ) -> Any: ...


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _convert_block(block: Any) -> ContentItem:
    """Convert one SDK content block to a `ContentItem`.

    A block of a type this Backend does not know becomes a text item describing the type, which
    is the same text `content_to_text` writes for other non-text items.
    """
    kind = getattr(block, "type", None)
    if kind == "text":
        return TextItem(text=str(block.text))
    if kind == "image":
        return ImageItem(mime_type=_optional_str(read_attr(block, "mime_type")))
    if kind == "audio":
        return AudioItem(mime_type=_optional_str(read_attr(block, "mime_type")))
    if kind == "resource":
        resource = block.resource
        return EmbeddedResourceItem(
            uri=_optional_str(getattr(resource, "uri", None)),
            mime_type=_optional_str(read_attr(resource, "mime_type")),
        )
    if kind == "resource_link":
        return ResourceLinkItem(
            uri=_optional_str(getattr(block, "uri", None)),
            mime_type=_optional_str(read_attr(block, "mime_type")),
        )
    return TextItem(text=f"[{kind} content]")


def _convert_tool(tool: Any) -> McpToolInfo:
    return McpToolInfo(
        name=str(tool.name),
        description=_optional_str(getattr(tool, "description", None)),
        input_schema=read_attr(tool, "input_schema"),
    )


class McpSessionAdapter:
    """Wraps an initialized ``ClientSession`` for tool discovery and tool calls."""

    def __init__(self, session: ToolSession) -> None:
        self._session = session

    async def list_all_tools(self, timeout: float) -> list[McpToolInfo]:  # noqa: ASYNC109
        """Request the tool list and follow ``next_cursor`` until the server returns none (7.1).

        The pages are concatenated in the order received. A cursor of any value, including an
        empty string, means another page; only an absent cursor ends the walk. ``timeout`` bounds
        the whole walk, not each page (7.10).

        Raises:
            TimeoutError: The walk did not finish within ``timeout`` seconds.
            Exception: Any error the session raises for a ``tools/list`` request.
        """
        tools: list[McpToolInfo] = []
        try:
            async with asyncio.timeout(timeout):
                cursor: str | None = None
                while True:
                    page = await self._session.list_tools(params=page_params(cursor))
                    tools.extend(_convert_tool(tool) for tool in page.tools)
                    cursor = read_attr(page, "next_cursor")
                    if cursor is None:
                        return tools
        except Exception as exc:
            if is_read_timeout(exc):
                raise TimeoutError(f"tools/list timed out after {timeout} seconds") from exc
            raise

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> CallOutcome:
        """Call one tool and return its content items and error flag (8.2, 9.1).

        ``timeout`` becomes the SDK's per-request read timeout. When the SDK reports that timeout,
        it is re-raised as the builtin ``TimeoutError`` (9.2). The caller still wraps the call in
        its own ``asyncio.timeout`` so a cancelled call is never left waiting.

        Raises:
            TimeoutError: The server did not answer within ``timeout`` seconds.
            Exception: Any other transport or protocol error the session raises.
        """
        try:
            result = await self._session.call_tool(
                name, args, read_timeout_seconds=to_read_timeout(timeout)
            )
        except Exception as exc:
            if is_read_timeout(exc):
                raise TimeoutError(f"tool {name!r} timed out after {timeout} seconds") from exc
            raise
        blocks: Sequence[Any] = getattr(result, "content", None) or ()
        return CallOutcome(
            content=tuple(_convert_block(block) for block in blocks),
            is_error=bool(read_attr(result, "is_error")),
        )


# ---------------------------------------------------------------------------------------------
# Transport and session construction
# ---------------------------------------------------------------------------------------------

HANDSHAKE_TIMEOUT_S: Final = 15.0
"""Longest a connection attempt may take to open the transport and finish `initialize` (6.7)."""

LIST_TIMEOUT_S: Final = 15.0
"""Longest a `tools/list` walk may take (7.10)."""

START_TIMEOUT_S: Final = 17.0
"""Cap on the startup connection phase: the 15 s handshake plus 2 s (6.6)."""

MAX_RECONNECTS: Final = 5
"""Reconnection attempts after a failure, before the supervisor gives up (6.8, 6.10)."""

_BACKOFF_CAP_S: Final = 30
_STDIN_WAIT_S: Final = 1.0
_REAP_WAIT_S: Final = 1.5

Streams = tuple[ReadStream[SessionMessage | Exception], WriteStream[SessionMessage]]
"""The `(read_stream, write_stream)` pair a transport yields and a `ClientSession` consumes."""


class McpSession(ToolSession, Protocol):
    """A `ToolSession` that can also run the MCP initialization handshake."""

    async def initialize(self) -> Any: ...


class TransportFactory(Protocol):
    """Opens the transport for one server; the context exit stops it.

    `grace` returns the Process_Tree grace period in seconds and is read when the context exits,
    so the value `McpManager.stop` is given reaches servers that are already running (6.11).
    """

    def __call__(
        self,
        cfg: StdioServerConfig | SseServerConfig,
        *,
        config_dir: str,
        grace: Callable[[], float],
    ) -> AbstractAsyncContextManager[Streams]: ...


SessionFactory = Callable[
    [ReadStream[SessionMessage | Exception], WriteStream[SessionMessage]],
    AbstractAsyncContextManager[McpSession],
]


@asynccontextmanager
async def open_transport(
    cfg: StdioServerConfig | SseServerConfig,
    *,
    config_dir: str,
    grace: Callable[[], float],
    platform: str = sys.platform,
) -> AsyncIterator[Streams]:
    """Open the real transport for `cfg` (6.2, 6.3).

    `stdio` starts the server through the managed transport, which owns the Process_Tree. `sse`
    uses the SDK's client. `SpawnRejected` from the executable resolver is raised as is, so the
    supervisor can log the resolver's own message (17.6).
    """
    if isinstance(cfg, StdioServerConfig):
        spec = await build_spawn_spec(cfg, config_dir, platform)
        async with managed_stdio_client(spec, grace=grace, server=cfg.name) as streams:
            yield streams
        return
    headers = {name: value.get_secret_value() for name, value in cfg.headers.items()}
    async with sse_client(str(cfg.url), headers=headers or None) as sse_streams:
        yield sse_streams


def open_session(
    read: ReadStream[SessionMessage | Exception], write: WriteStream[SessionMessage]
) -> AbstractAsyncContextManager[McpSession]:
    """The real session: an SDK `ClientSession` over the transport streams."""
    return ClientSession(read, write)


class _EofWatcher:
    """A read stream that records when the transport's stream ends.

    `ClientSession` treats an ended stream as a closed connection and fails pending requests, but
    it gives the supervisor no signal to wait on. This wrapper forwards every call to the
    transport's stream and sets `closed` the first time the stream ends or breaks.
    """

    def __init__(
        self, inner: ReadStream[SessionMessage | Exception], closed: asyncio.Event
    ) -> None:
        self._inner = inner
        self._closed = closed

    async def receive(self) -> SessionMessage | Exception:
        try:
            return await self._inner.receive()
        except anyio.EndOfStream, anyio.ClosedResourceError, anyio.BrokenResourceError:
            self._closed.set()
            raise

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> SessionMessage | Exception:
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration, anyio.ClosedResourceError, anyio.BrokenResourceError:
            self._closed.set()
            raise

    async def __aenter__(self) -> Self:
        await self._inner.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        # `last_context` and any other optional attribute of the SDK's context-aware streams.
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------------------------


def backoff_delay(n: int) -> int:
    """Seconds to wait before reconnection attempt `n + 1`: `min(2**n, 30)` (6.8). Pure.

    `n` counts failed attempts so far, starting at 0, so the delays run 1, 2, 4, 8, 16, 30, ...
    """
    if n < 0:
        raise ValueError(f"backoff attempt must not be negative, got {n}")
    # 2**6 already exceeds the cap; clamping the shift keeps a huge `n` cheap.
    return min(1 << min(n, 6), _BACKOFF_CAP_S)


class ServerState(Enum):
    """Where one configured server stands (15.8)."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class McpServerUnavailable(Exception):
    """A Tool_Call named a server that is not connected (9.6)."""

    def __init__(self, server: str, reason: str = "is unavailable") -> None:
        self.server = server
        super().__init__(f"MCP server '{server}' {reason}")


class _Stopped(Exception):
    """The supervisor was asked to stop while it was connecting."""


class ServerRegistry(Protocol):
    """The part of `ToolRegistry` a supervisor writes to."""

    def replace_server(self, server: str, tools: Iterable[McpToolInfo]) -> None: ...

    def remove_server(self, server: str) -> None: ...


Sleep = Callable[[float], Awaitable[None]]


def _leaves(exc: BaseException) -> list[BaseException]:
    """The innermost exceptions of `exc`; an ordinary exception is its own only leaf."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for member in exc.exceptions for leaf in _leaves(member)]
    return [exc]


def describe_failure(exc: BaseException, redactor: Redactor) -> str:
    """A one-line reason for a failed connection, with secrets redacted (15.5).

    The SDK's task groups wrap errors in exception groups, so the leaves are described. A
    `SpawnRejected` contributes the resolver's own message (6.7, 17.6).
    """
    parts: list[str] = []
    for leaf in _leaves(exc):
        if isinstance(leaf, SpawnRejected):
            parts.append(leaf.reason)
            continue
        detail = str(leaf)
        parts.append(f"{type(leaf).__name__}: {detail}" if detail else type(leaf).__name__)
    return redactor.redact("; ".join(dict.fromkeys(parts)))


async def _first_of(*waiters: Awaitable[Any]) -> None:
    """Wait until any one awaitable finishes, then cancel the others."""
    tasks = [asyncio.ensure_future(waiter) for waiter in waiters]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class ServerSupervisor:
    """Keeps one enabled server connected: connect, register tools, wait, back off, reconnect.

    `run` is the body of one `asyncio.Task`. It never raises an `Exception`; every failure is
    logged, turns the server unavailable, and removes its tools from the registry (6.8). After a
    failure it waits `backoff_delay(n)` seconds and tries again, up to `max_reconnects` times in a
    row, then keeps the server unavailable until the Backend restarts (6.10). A connection that
    reaches the handshake and the tool listing resets the count (6.8, 6.9).

    A failed initial connection takes the same path as a dropped one, so an unreachable server
    is retried in the background while the Backend serves requests (6.7).
    """

    def __init__(
        self,
        cfg: StdioServerConfig | SseServerConfig,
        registry: ServerRegistry,
        *,
        config_dir: str,
        transport_factory: TransportFactory = open_transport,
        session_factory: SessionFactory = open_session,
        grace: Callable[[], float] = lambda: DEFAULT_GRACE_S,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
        redactor: Redactor | None = None,
        handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S,
        list_timeout_s: float = LIST_TIMEOUT_S,
        max_reconnects: int = MAX_RECONNECTS,
    ) -> None:
        self.name = cfg.name
        self._cfg = cfg
        self._registry = registry
        self._config_dir = config_dir
        self._transport_factory = transport_factory
        self._session_factory = session_factory
        self._grace = grace
        self._clock: Clock = SystemClock() if clock is None else clock
        self._sleep = sleep
        self._redactor = Redactor(()) if redactor is None else redactor
        self._handshake_timeout_s = handshake_timeout_s
        self._list_timeout_s = list_timeout_s
        self._max_reconnects = max_reconnects
        self._state = ServerState.UNAVAILABLE if cfg.enabled else ServerState.DISABLED
        self._adapter: McpSessionAdapter | None = None
        self._settled = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._closed = asyncio.Event()
        self._deadline: asyncio.Timeout | None = None

    @property
    def state(self) -> ServerState:
        return self._state

    @property
    def adapter(self) -> McpSessionAdapter | None:
        """The adapter of the live session, or `None` while the server is not available."""
        return self._adapter

    @property
    def settled(self) -> asyncio.Event:
        """Set once the first connection attempt has either succeeded or failed (6.6)."""
        return self._settled

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Ask `run` to leave the transport context and return; a pending connect is aborted."""
        self._stop_event.set()
        if self._deadline is not None:
            self._deadline.reschedule(asyncio.get_running_loop().time())

    async def run(self) -> None:
        """Connect, serve, and reconnect until stopped or out of reconnection attempts."""
        failed_attempts = 0
        try:
            while not self._stop_event.is_set():
                connected, reason = await self._connection()
                if self._stop_event.is_set():
                    break
                if connected:
                    failed_attempts = 0
                retry_in = (
                    backoff_delay(failed_attempts)
                    if failed_attempts < self._max_reconnects
                    else None
                )
                _log.error(
                    "mcp_server_failed",
                    server=self.name,
                    reason=reason,
                    reconnect_attempt=failed_attempts,
                    retry_in_s=retry_in,
                )
                self._settled.set()
                if retry_in is None:
                    _log.error(
                        "mcp_server_gave_up",
                        server=self.name,
                        reconnect_attempts=self._max_reconnects,
                    )
                    break
                await _first_of(self._sleep(retry_in), self._stop_event.wait())
                failed_attempts += 1
        finally:
            self._mark_unavailable()
            self._settled.set()

    # -- one connection ----------------------------------------------------------------------

    async def _connection(self) -> tuple[bool, str]:
        """Run one connection from transport open to transport close.

        Returns whether the server became available, and the reason the connection ended. The
        reason is meaningful only when the supervisor was not asked to stop.
        """
        self._closed = asyncio.Event()
        connected = False
        reason = "the server closed the connection"
        started = self._clock.monotonic()
        try:
            async with AsyncExitStack() as stack:
                adapter = McpSessionAdapter(await self._connect(stack))
                tools = await self._discover(adapter)
                if tools is not None:
                    self._registry.replace_server(self.name, tools)
                self._adapter = adapter
                self._state = ServerState.AVAILABLE
                connected = True
                _log.info(
                    "mcp_server_connected",
                    server=self.name,
                    transport=self._cfg.transport,
                    tools=0 if tools is None else len(tools),
                    duration_ms=round((self._clock.monotonic() - started) * 1000),
                )
                self._settled.set()
                try:
                    await _first_of(self._closed.wait(), self._stop_event.wait())
                finally:
                    # Before the transport is torn down, which can take seconds (6.8).
                    self._mark_unavailable()
        except _Stopped:
            pass
        except Exception as exc:
            reason = describe_failure(exc, self._redactor)
        finally:
            self._mark_unavailable()
        return connected, reason

    async def _connect(self, stack: AsyncExitStack) -> McpSession:
        """Open the transport and session on `stack` and complete the handshake (6.5, 6.7)."""
        async with self._within(self._handshake_timeout_s, "MCP initialization handshake"):
            read, write = await stack.enter_async_context(
                self._transport_factory(self._cfg, config_dir=self._config_dir, grace=self._grace)
            )
            session = await stack.enter_async_context(
                self._session_factory(_EofWatcher(read, self._closed), write)
            )
            await session.initialize()
        return session

    async def _discover(self, adapter: McpSessionAdapter) -> list[McpToolInfo] | None:
        """List the tools; on failure the server stays connected with no tools (7.10)."""
        try:
            async with self._within(self._list_timeout_s, "tools/list"):
                return await adapter.list_all_tools(self._list_timeout_s)
        except _Stopped:
            raise
        except Exception as exc:
            _log.warning(
                "mcp_tool_list_failed",
                server=self.name,
                reason=describe_failure(exc, self._redactor),
            )
            return None

    @asynccontextmanager
    async def _within(self, seconds: float, what: str) -> AsyncIterator[None]:
        """Bound the body to `seconds`; `request_stop` ends it at once with `_Stopped`."""
        try:
            async with asyncio.timeout(seconds) as scope:
                self._deadline = scope
                try:
                    yield
                finally:
                    self._deadline = None
        except TimeoutError as exc:
            if self.stopping:
                raise _Stopped from exc
            if scope.expired():
                raise TimeoutError(f"{what} timed out after {seconds:g} seconds") from exc
            raise

    def _mark_unavailable(self) -> None:
        """Turn the server unavailable and drop its tools; safe to call repeatedly (6.8)."""
        if self._state is ServerState.DISABLED:
            return
        self._state = ServerState.UNAVAILABLE
        self._adapter = None
        self._registry.remove_server(self.name)

    # -- refresh -----------------------------------------------------------------------------

    async def refresh(self) -> None:
        """Request the tool list again and replace the registered tools (6.12).

        On any failure the previous tools stay registered and a warning names the server (6.13).
        A result that arrives after the connection was replaced or lost is dropped.
        """
        adapter = self._adapter
        if adapter is None:
            return
        try:
            tools = await adapter.list_all_tools(self._list_timeout_s)
            if self._adapter is adapter:
                self._registry.replace_server(self.name, tools)
        except Exception as exc:
            _log.warning(
                "mcp_tool_refresh_failed",
                server=self.name,
                reason=describe_failure(exc, self._redactor),
            )


# ---------------------------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------------------------


class McpManager:
    """All MCP_Server connections of the Backend (6.5 to 6.13, 14.3).

    `servers` is the Backend_Config list, in order. Disabled servers get no supervisor and no
    connection attempt (6.5). `config_dir` anchors relative `stdio` paths (17.19).

    The transport and session factories, the clock, and the sleep function can be replaced so a
    test can drive the whole lifecycle with a fake transport and no real waiting.
    """

    def __init__(
        self,
        servers: Sequence[StdioServerConfig | SseServerConfig],
        registry: ServerRegistry,
        *,
        config_dir: str,
        transport_factory: TransportFactory = open_transport,
        session_factory: SessionFactory = open_session,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
        redactor: Redactor | None = None,
        handshake_timeout_s: float = HANDSHAKE_TIMEOUT_S,
        list_timeout_s: float = LIST_TIMEOUT_S,
        start_timeout_s: float = START_TIMEOUT_S,
        max_reconnects: int = MAX_RECONNECTS,
    ) -> None:
        self._servers = tuple(servers)
        self._clock: Clock = SystemClock() if clock is None else clock
        self._start_timeout_s = start_timeout_s
        self._grace: Callable[[], float] = lambda: DEFAULT_GRACE_S
        self._supervisors: dict[str, ServerSupervisor] = {
            cfg.name: ServerSupervisor(
                cfg,
                registry,
                config_dir=config_dir,
                transport_factory=transport_factory,
                session_factory=session_factory,
                grace=self._current_grace,
                clock=self._clock,
                sleep=sleep,
                redactor=redactor,
                handshake_timeout_s=handshake_timeout_s,
                list_timeout_s=list_timeout_s,
                max_reconnects=max_reconnects,
            )
            for cfg in self._servers
            if cfg.enabled
        }
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def _current_grace(self) -> float:
        return self._grace()

    async def start(self) -> None:
        """Connect to every enabled server concurrently and return once all have settled (6.5).

        A server settles when its first attempt succeeds or fails. The wait is capped at
        `start_timeout_s` (17 s, 6.6); a server still connecting at the cap keeps connecting in
        the background and is unavailable until it finishes. Startup never fails because a server
        does (6.7).
        """
        if self._tasks:
            raise RuntimeError("McpManager.start() was already called")
        for name, supervisor in self._supervisors.items():
            self._tasks[name] = asyncio.create_task(supervisor.run(), name=f"mcp-supervisor-{name}")
        try:
            async with asyncio.timeout(self._start_timeout_s):
                await asyncio.gather(*(s.settled.wait() for s in self._supervisors.values()))
        except TimeoutError:
            _log.warning(
                "mcp_start_timed_out",
                timeout_s=self._start_timeout_s,
                pending=[n for n, s in self._supervisors.items() if not s.settled.is_set()],
            )

    async def stop(self, grace: float | Callable[[], float] = DEFAULT_GRACE_S) -> None:
        """Close every session and terminate every Process_Tree, concurrently (6.11, 17.15).

        `grace` is the Process_Tree grace period: 5 s normally, 2 s on a console close. Pass a
        callable (such as the `ShutdownController.grace_s` getter) to let it change during the
        stop: it is read while each tree waits, so a second Shutdown_Signal that drops it to 0
        ends the wait at once (17.14). All supervisors leave their transport contexts at once, so
        the trees stop in parallel. The wait is bounded by the stdin wait, the grace period as it
        was when `stop` began, and the reap wait.
        """
        self._grace = grace if callable(grace) else (lambda: grace)
        grace_s = self._grace()
        for supervisor in self._supervisors.values():
            supervisor.request_stop()
        tasks = dict(self._tasks)
        self._tasks.clear()
        if not tasks:
            return
        bound = _STDIN_WAIT_S + grace_s + _REAP_WAIT_S
        try:
            async with asyncio.timeout(bound):
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        except TimeoutError:
            _log.error("mcp_stop_timed_out", timeout_s=bound, servers=sorted(tasks))
            return
        for name, result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                _log.error("mcp_supervisor_crashed", server=name, error=type(result).__name__)

    async def refresh_available(self) -> None:
        """Re-list the tools of every available server, concurrently (6.12).

        A failing server keeps its previous tools and is logged; nothing here retries, so the
        caller never waits on a broken server beyond the list timeout (6.13).
        """
        await asyncio.gather(*(s.refresh() for s in self._supervisors.values()))

    def availability(self) -> Mapping[str, bool]:
        """`{server: available}` for every configured server; disabled ones are `False`."""
        return {
            cfg.name: (
                supervisor.state is ServerState.AVAILABLE
                if (supervisor := self._supervisors.get(cfg.name)) is not None
                else False
            )
            for cfg in self._servers
        }

    def statuses(self) -> list[ServerStatus]:
        """One `ServerStatus` per configured server, in Backend_Config order (15.8)."""
        available = self.availability()
        return [
            ServerStatus(name=cfg.name, enabled=cfg.enabled, available=available[cfg.name])
            for cfg in self._servers
        ]

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> CallOutcome:
        """Call `tool` on `server` and return its content and error flag (8.2, 9.1).

        Calls from different Tasks run concurrently on one session (14.3). `timeout` starts when
        the request is sent; when it passes, the waiting coroutine is cancelled, which makes the
        SDK drop the request, so a late reply is never observed (9.2, 9.9).

        Raises:
            McpServerUnavailable: The server is unknown, disabled, or not connected (9.6).
            TimeoutError: No reply within `timeout` seconds.
            Exception: Any transport or protocol error from the session (9.7).
        """
        supervisor = self._supervisors.get(server)
        adapter = None if supervisor is None else supervisor.adapter
        if adapter is None:
            raise McpServerUnavailable(server)
        started = self._clock.monotonic()
        outcome = "exception"
        try:
            async with asyncio.timeout(timeout):
                result = await adapter.call_tool(tool, args, timeout)
            outcome = "error" if result.is_error else "success"
            return result
        except TimeoutError:
            outcome = "timeout"
            raise
        finally:
            _log.debug(
                "mcp_call_finished",
                server=server,
                tool=tool,
                outcome=outcome,
                duration_ms=round((self._clock.monotonic() - started) * 1000),
            )
