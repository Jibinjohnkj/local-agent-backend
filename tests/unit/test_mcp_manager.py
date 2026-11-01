"""Unit tests for `McpManager` and `ServerSupervisor` against a fake transport.

The tests replace the transport factory, the session factory, the clock, and the sleep function, so
the whole lifecycle (connect, discover, drop, back off, reconnect, give up, stop) runs without a
subprocess, a socket, or real waiting. A few tests use real timeouts of tens of milliseconds where
the code under test owns the timer (`asyncio.timeout` around a handshake, a tool listing, or a
tool call).

The fakes:

- `FakeWorld.transport` has the `TransportFactory` call shape. Each open follows a scripted
  `Attempt` and yields a `FakeRead` stream plus a memory write stream. Leaving the context records
  the grace period the supervisor would give the Process_Tree (the callable is read at exit).
- `FakeWorld.session` has the session factory call shape. Its session drains the read stream the
  way the SDK's `ClientSession` does, which is what lets the supervisor notice a dropped
  connection: `Connection.drop()` ends the stream, and the manager sees the server go away.
- `GatedSleep` records each backoff delay and blocks until the test opens the gate, so a test can
  look at the state of a server while it waits to reconnect.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace, TracebackType
from typing import Any, Protocol, Self, cast

import anyio
import pytest
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp_types import REQUEST_TIMEOUT
from structlog.testing import capture_logs

from conftest import FakeClock, FakeSleep
from local_agent.config.models import SseServerConfig, StdioServerConfig
from local_agent.logging_setup import Redactor
from local_agent.mcp_client.manager import (
    CallOutcome,
    McpManager,
    McpServerUnavailable,
    McpSession,
    ServerState,
    ServerSupervisor,
    Streams,
    backoff_delay,
    describe_failure,
)
from local_agent.models import McpToolInfo, ServerStatus, TextItem
from local_agent.oscompat.process import DEFAULT_GRACE_S, SpawnRejected

type ServerConfig = StdioServerConfig | SseServerConfig

ALPHA = StdioServerConfig(name="alpha", transport="stdio", command="alpha-server")
BETA = SseServerConfig(name="beta", transport="sse", url="http://localhost:9/sse")
GAMMA = StdioServerConfig(name="gamma", transport="stdio", command="gamma-server")
OFF = StdioServerConfig(name="off", transport="stdio", command="off-server", enabled=False)

CONFIG_DIR = "/cfg"


# ---------------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------------


@dataclass
class Attempt:
    """How one connection attempt behaves. A field left at its default means "works"."""

    open_error: Exception | None = None  # the transport cannot be opened
    init_error: Exception | None = None  # `initialize` raises
    hang_init: bool = False  # `initialize` never returns
    list_error: Exception | None = None  # `tools/list` raises
    hang_list: bool = False  # `tools/list` never returns
    tools: tuple[str, ...] | None = None  # defaults to `<server>_tool`


class FakeRead:
    """The read side of a fake transport; `end()` models the server closing the connection."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self._queue: asyncio.Queue[SessionMessage | Exception | None] = asyncio.Queue()

    def end(self) -> None:
        self._queue.put_nowait(None)

    async def receive(self) -> SessionMessage | Exception:
        item = await self._queue.get()
        if item is None:
            self._queue.put_nowait(None)  # every later read ends too
            raise anyio.EndOfStream
        return item

    async def aclose(self) -> None:
        return None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> SessionMessage | Exception:
        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return None


type CallHandler = Callable[[str, dict[str, Any] | None], Awaitable[Any]]


def text_result(text: str, *, is_error: bool = False) -> Any:
    """A `tools/call` result as the SDK returns it: one text block and the `isError` flag."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], is_error=is_error)


class Connection:
    """One open fake connection. Tests change its fields to change the server's behavior."""

    def __init__(
        self,
        cfg: ServerConfig,
        config_dir: str,
        grace: Callable[[], float],
        attempt: Attempt,
        barrier: asyncio.Barrier | None,
    ) -> None:
        self.cfg = cfg
        self.config_dir = config_dir
        self.grace = grace
        self.barrier = barrier
        self.read = FakeRead(self)
        self.init_error = attempt.init_error
        self.hang_init = attempt.hang_init
        self.list_error = attempt.list_error
        self.hang_list = attempt.hang_list
        self.tools = list(attempt.tools if attempt.tools is not None else (f"{cfg.name}_tool",))
        self.call_handler: CallHandler | None = None
        self.calls: list[tuple[str, dict[str, Any] | None, Any]] = []
        self.list_calls = 0
        self.list_gate: asyncio.Event | None = None  # when set, `tools/list` waits for it
        self.closed = False
        self.grace_at_exit: float | None = None

    def drop(self) -> None:
        """The server closes the connection."""
        self.read.end()


class FakeSession:
    def __init__(self, connection: Connection) -> None:
        self._conn = connection

    async def initialize(self) -> Any:
        conn = self._conn
        if conn.barrier is not None:
            await conn.barrier.wait()
        if conn.hang_init:
            await asyncio.Event().wait()
        if conn.init_error is not None:
            raise conn.init_error
        return None

    async def list_tools(self, *, params: Any = None) -> Any:
        conn = self._conn
        conn.list_calls += 1
        if conn.list_gate is not None:
            await conn.list_gate.wait()
        if conn.hang_list:
            await asyncio.Event().wait()
        if conn.list_error is not None:
            raise conn.list_error
        tools = [
            SimpleNamespace(name=name, description=None, input_schema={"type": "object"})
            for name in conn.tools
        ]
        return SimpleNamespace(tools=tools, next_cursor=None)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: Any = None,
    ) -> Any:
        conn = self._conn
        conn.calls.append((name, arguments, read_timeout_seconds))
        if conn.call_handler is None:
            return text_result(f"{name}: ok")
        return await conn.call_handler(name, arguments)


class FakeWorld:
    """Scripted servers: the transport factory, the session factory, and every connection made.

    `scripts` maps a server name to its attempts in order; attempts past the end of a script use
    `default`. `opens[name]` counts every attempt, including those whose transport failed to open.
    """

    def __init__(
        self,
        scripts: Mapping[str, Sequence[Attempt]] | None = None,
        *,
        default: Attempt | None = None,
        barrier: asyncio.Barrier | None = None,
    ) -> None:
        self._scripts = scripts or {}
        self._default = default or Attempt()
        self._barrier = barrier
        self.connections: list[Connection] = []
        self.opens: dict[str, int] = {}

    def of(self, server: str) -> list[Connection]:
        return [c for c in self.connections if c.cfg.name == server]

    @asynccontextmanager
    async def transport(
        self, cfg: ServerConfig, *, config_dir: str, grace: Callable[[], float]
    ) -> AsyncIterator[Streams]:
        index = self.opens.get(cfg.name, 0)
        self.opens[cfg.name] = index + 1
        script = self._scripts.get(cfg.name, ())
        attempt = script[index] if index < len(script) else self._default
        if attempt.open_error is not None:
            raise attempt.open_error
        conn = Connection(cfg, config_dir, grace, attempt, self._barrier)
        self.connections.append(conn)
        write, write_peer = anyio.create_memory_object_stream[SessionMessage](1)
        try:
            yield conn.read, write
        finally:
            # The real transport reads `grace` here, while it terminates the Process_Tree.
            conn.grace_at_exit = grace()
            conn.closed = True
            write.close()
            write_peer.close()

    @asynccontextmanager
    async def _session(self, read: Any) -> AsyncIterator[McpSession]:
        conn = cast(Connection, read.connection)

        async def drain() -> None:
            # Like `ClientSession`, the session reads the stream; without this the supervisor
            # would never see the stream end.
            async for _ in read:
                pass

        task = asyncio.create_task(drain())
        try:
            yield FakeSession(conn)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def session(self, read: Any, write: Any) -> AbstractAsyncContextManager[McpSession]:
        return self._session(read)


class FakeRegistry:
    """Records what the supervisors register: `tools` is the live view per server."""

    def __init__(self) -> None:
        self.tools: dict[str, list[str]] = {}
        self.events: list[tuple[str, str]] = []

    def replace_server(self, server: str, tools: Iterable[McpToolInfo]) -> None:
        self.tools[server] = [tool.name for tool in tools]
        self.events.append(("replace", server))

    def remove_server(self, server: str) -> None:
        self.tools.pop(server, None)
        self.events.append(("remove", server))


class GatedSleep:
    """A sleep that records the delay and waits until `open()`; later sleeps return at once."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._gate = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        await self._gate.wait()

    def open(self) -> None:
        self._gate.set()


async def until(predicate: Callable[[], bool], what: str = "the condition") -> None:
    """Let the event loop run until `predicate()` holds; fail after about five seconds."""
    for _ in range(5000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"timed out waiting for {what}")


async def drain_loop() -> None:
    """Give every runnable task a few turns, to show that something does not happen."""
    for _ in range(20):
        await asyncio.sleep(0)


def events(logs: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [entry for entry in logs if entry["event"] == name]


class Build(Protocol):
    def __call__(
        self, servers: Sequence[ServerConfig], world: FakeWorld, **options: Any
    ) -> McpManager: ...


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
async def build(clock: FakeClock, sleep: FakeSleep, registry: FakeRegistry) -> AsyncIterator[Build]:
    """Make managers wired to a fake world; each is stopped at teardown."""
    managers: list[McpManager] = []

    def _build(servers: Sequence[ServerConfig], world: FakeWorld, **options: Any) -> McpManager:
        options.setdefault("sleep", sleep)
        manager = McpManager(
            servers,
            registry,
            config_dir=CONFIG_DIR,
            transport_factory=world.transport,
            session_factory=world.session,
            clock=clock,
            **options,
        )
        managers.append(manager)
        return manager

    yield _build
    for manager in managers:
        await manager.stop()


# ---------------------------------------------------------------------------------------------
# backoff_delay (6.8)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attempt", "delay"),
    [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16), (5, 30), (6, 30), (100, 30), (10**6, 30)],
)
def test_backoff_delay_doubles_and_is_capped_at_30(attempt: int, delay: int) -> None:
    assert backoff_delay(attempt) == delay


def test_backoff_delay_rejects_a_negative_attempt() -> None:
    with pytest.raises(ValueError, match="negative"):
        backoff_delay(-1)


# ---------------------------------------------------------------------------------------------
# Startup (6.5, 6.6, 6.7)
# ---------------------------------------------------------------------------------------------


async def test_start_connects_enabled_servers_and_skips_disabled(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld()
    manager = build([ALPHA, OFF, BETA], world)

    await manager.start()

    assert manager.availability() == {"alpha": True, "off": False, "beta": True}
    assert registry.tools == {"alpha": ["alpha_tool"], "beta": ["beta_tool"]}
    assert world.opens == {"alpha": 1, "beta": 1}  # no attempt for the disabled server


async def test_transport_receives_the_config_and_config_dir(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)

    await manager.start()

    (conn,) = world.of("alpha")
    assert conn.cfg == ALPHA
    assert conn.config_dir == CONFIG_DIR


async def test_statuses_follow_config_order(build: Build) -> None:
    world = FakeWorld(default=Attempt(), scripts={"gamma": [Attempt(open_error=OSError("down"))]})
    manager = build([GAMMA, OFF, ALPHA], world, max_reconnects=0)

    await manager.start()

    assert manager.statuses() == [
        ServerStatus(name="gamma", enabled=True, available=False),
        ServerStatus(name="off", enabled=False, available=False),
        ServerStatus(name="alpha", enabled=True, available=True),
    ]


async def test_servers_connect_concurrently(build: Build) -> None:
    # Each `initialize` waits for the other, so a sequential start would never get both through.
    world = FakeWorld(barrier=asyncio.Barrier(2))
    manager = build([ALPHA, BETA], world, handshake_timeout_s=5.0, start_timeout_s=5.0)

    await manager.start()

    assert manager.availability() == {"alpha": True, "beta": True}


async def test_start_stops_waiting_at_the_cap_and_stop_ends_the_pending_connection(
    build: Build,
) -> None:
    world = FakeWorld(scripts={"alpha": [Attempt(hang_init=True)]})
    manager = build([ALPHA, BETA], world, start_timeout_s=0.05)

    with capture_logs() as logs:
        await manager.start()

    assert manager.availability() == {"alpha": False, "beta": True}
    (warning,) = events(logs, "mcp_start_timed_out")
    assert warning["pending"] == ["alpha"]

    await manager.stop()

    assert [c.closed for c in world.of("alpha")] == [True]  # the pending transport was closed


async def test_start_twice_is_rejected(build: Build) -> None:
    manager = build([ALPHA], FakeWorld())
    await manager.start()

    with pytest.raises(RuntimeError, match="already called"):
        await manager.start()


# ---------------------------------------------------------------------------------------------
# Failed connections (6.7, 17.6)
# ---------------------------------------------------------------------------------------------


async def test_spawn_rejected_marks_the_server_unavailable_with_the_resolver_reason(
    build: Build, registry: FakeRegistry
) -> None:
    reason = "command 'nope' was not found on PATH"
    world = FakeWorld(default=Attempt(open_error=SpawnRejected(reason)))
    manager = build([ALPHA, BETA], world, max_reconnects=0)

    with capture_logs() as logs:
        await manager.start()

    assert manager.availability() == {"alpha": False, "beta": False}
    assert registry.tools == {}
    failed = events(logs, "mcp_server_failed")
    assert sorted(f["server"] for f in failed) == ["alpha", "beta"]
    # The resolver's own words, with no exception-type prefix.
    assert [f["reason"] for f in failed] == [reason, reason]
    assert world.connections == []  # no transport, no session


async def test_spawn_rejected_is_retried_like_any_failed_attempt(
    build: Build, sleep: FakeSleep
) -> None:
    attempts = [Attempt(open_error=SpawnRejected("cannot launch")), Attempt()]
    world = FakeWorld(scripts={"alpha": attempts})
    manager = build([ALPHA], world)

    await manager.start()
    await until(lambda: manager.availability()["alpha"], "the retry to connect")

    assert sleep.calls == [1]
    assert world.opens == {"alpha": 2}


async def test_failure_log_names_the_server_and_the_exception_type_and_text(
    build: Build,
) -> None:
    world = FakeWorld(default=Attempt(init_error=ConnectionError("connection refused by peer")))
    manager = build([ALPHA], world, max_reconnects=0)

    with capture_logs() as logs:
        await manager.start()

    (failed,) = events(logs, "mcp_server_failed")
    assert failed["server"] == "alpha"
    assert failed["reason"] == "ConnectionError: connection refused by peer"
    assert failed["log_level"] == "error"


async def test_failure_reason_is_redacted(build: Build) -> None:
    secret = "hunter2-secret-value"
    error = RuntimeError(f"401 for token {secret}")
    world = FakeWorld(default=Attempt(init_error=error))
    manager = build([ALPHA], world, max_reconnects=0, redactor=Redactor([secret]))

    with capture_logs() as logs:
        await manager.start()

    (failed,) = events(logs, "mcp_server_failed")
    assert secret not in failed["reason"]
    assert "[REDACTED]" in failed["reason"]


async def test_failed_handshake_leaves_the_transport_context(build: Build) -> None:
    # Leaving the context is what terminates a stdio server's Process_Tree (17.11).
    world = FakeWorld(default=Attempt(init_error=RuntimeError("bad handshake")))
    manager = build([ALPHA], world, max_reconnects=0)

    await manager.start()

    (conn,) = world.of("alpha")
    assert conn.closed
    assert manager.availability() == {"alpha": False}


async def test_handshake_that_hangs_times_out_and_closes_the_transport(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld(default=Attempt(hang_init=True))
    manager = build([ALPHA], world, handshake_timeout_s=0.05, max_reconnects=0)

    with capture_logs() as logs:
        await manager.start()

    (failed,) = events(logs, "mcp_server_failed")
    assert "handshake timed out after 0.05 seconds" in failed["reason"]
    assert world.of("alpha")[0].closed
    assert registry.tools == {}


def test_describe_failure_lists_the_leaves_of_an_exception_group() -> None:
    group = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [ValueError("first"), ExceptionGroup("inner", [KeyError("second"), ValueError("first")])],
    )

    text = describe_failure(group, Redactor(()))

    assert text == "ValueError: first; KeyError: 'second'"


def test_describe_failure_uses_the_spawn_rejected_reason_as_is() -> None:
    assert describe_failure(SpawnRejected("no such file"), Redactor(())) == "no such file"
    assert describe_failure(TimeoutError(), Redactor(())) == "TimeoutError"


# ---------------------------------------------------------------------------------------------
# Reconnection (6.8, 6.9, 6.10, 7.11)
# ---------------------------------------------------------------------------------------------


async def test_failures_back_off_1_2_4_8_16_and_stop_after_five_reconnects(
    build: Build, sleep: FakeSleep, registry: FakeRegistry
) -> None:
    world = FakeWorld(default=Attempt(open_error=ConnectionError("refused")))
    manager = build([ALPHA], world)

    with capture_logs() as logs:
        await manager.start()
        await until(lambda: bool(events(logs, "mcp_server_gave_up")), "the supervisor to give up")
        await drain_loop()

    assert sleep.calls == [1, 2, 4, 8, 16]
    assert world.opens == {"alpha": 6}  # the first attempt and five reconnections, then no more
    failures = events(logs, "mcp_server_failed")
    assert [f["reconnect_attempt"] for f in failures] == [0, 1, 2, 3, 4, 5]
    assert [f["retry_in_s"] for f in failures] == [1, 2, 4, 8, 16, None]
    (gave_up,) = events(logs, "mcp_server_gave_up")
    assert gave_up["server"] == "alpha"
    assert manager.availability() == {"alpha": False}
    assert registry.tools == {}


async def test_dropped_connection_removes_tools_then_reconnect_registers_the_new_list(
    build: Build, registry: FakeRegistry
) -> None:
    gated = GatedSleep()
    world = FakeWorld(
        scripts={"alpha": [Attempt(tools=("old_a", "old_b")), Attempt(tools=("new_a", "new_b"))]}
    )
    manager = build([ALPHA], world, sleep=gated)
    await manager.start()
    assert registry.tools == {"alpha": ["old_a", "old_b"]}
    first = world.of("alpha")[0]

    first.drop()
    await until(lambda: gated.calls == [1], "the supervisor to wait before reconnecting")

    # While it waits: unavailable, tools gone, old transport already closed, calls refused.
    assert manager.availability() == {"alpha": False}
    assert registry.tools == {}
    assert first.closed
    with pytest.raises(McpServerUnavailable):
        await manager.call_tool("alpha", "old_a", {}, 1.0)

    gated.open()
    await until(lambda: manager.availability()["alpha"], "the reconnection")

    assert registry.tools == {"alpha": ["new_a", "new_b"]}  # replaced, not merged (7.11)
    assert len(world.of("alpha")) == 2


async def test_backoff_restarts_from_one_second_after_a_successful_reconnect(
    build: Build, sleep: FakeSleep
) -> None:
    attempts = [Attempt(), Attempt(open_error=ConnectionError("refused")), Attempt(), Attempt()]
    world = FakeWorld(scripts={"alpha": attempts})
    manager = build([ALPHA], world)
    await manager.start()

    world.of("alpha")[0].drop()  # delay 1, then a failed attempt, delay 2, then a good connection
    await until(lambda: len(world.of("alpha")) == 2, "the second connection")
    assert sleep.calls == [1, 2]

    world.of("alpha")[1].drop()
    await until(lambda: len(world.of("alpha")) == 3, "the third connection")

    assert sleep.calls == [1, 2, 1]  # the successful connection reset the count
    assert manager.availability() == {"alpha": True}


async def test_a_dropped_server_gets_five_reconnections_with_the_same_backoff(
    build: Build, sleep: FakeSleep
) -> None:
    world = FakeWorld(scripts={"alpha": [Attempt()]}, default=Attempt(open_error=OSError("down")))
    manager = build([ALPHA], world)
    await manager.start()

    with capture_logs() as logs:
        world.of("alpha")[0].drop()
        await until(lambda: bool(events(logs, "mcp_server_gave_up")), "the supervisor to give up")

    # The first connection plus five reconnections that all fail, then no more attempts.
    assert sleep.calls == [1, 2, 4, 8, 16]
    assert world.opens == {"alpha": 6}


async def test_other_servers_stay_available_while_one_reconnects(
    build: Build, registry: FakeRegistry
) -> None:
    gated = GatedSleep()
    world = FakeWorld()
    manager = build([ALPHA, BETA], world, sleep=gated)
    await manager.start()

    world.of("alpha")[0].drop()
    await until(lambda: gated.calls == [1])

    assert manager.availability() == {"alpha": False, "beta": True}
    assert registry.tools == {"beta": ["beta_tool"]}
    outcome = await manager.call_tool("beta", "beta_tool", {}, 1.0)
    assert outcome.content == (TextItem(text="beta_tool: ok"),)


# ---------------------------------------------------------------------------------------------
# Tool discovery failures (7.10)
# ---------------------------------------------------------------------------------------------


async def test_tool_list_error_registers_no_tools_but_keeps_the_server_connected(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld(scripts={"alpha": [Attempt(list_error=RuntimeError("list exploded"))]})
    manager = build([ALPHA, BETA], world)

    with capture_logs() as logs:
        await manager.start()

    assert registry.tools.get("alpha", []) == []
    assert registry.tools["beta"] == ["beta_tool"]  # discovery continued for the other server
    assert manager.availability() == {"alpha": True, "beta": True}
    (warning,) = events(logs, "mcp_tool_list_failed")
    assert warning["server"] == "alpha"
    assert warning["reason"] == "RuntimeError: list exploded"
    assert warning["log_level"] == "warning"


async def test_tool_list_that_does_not_answer_registers_no_tools(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld(scripts={"alpha": [Attempt(hang_list=True)]})
    manager = build([ALPHA, BETA], world, list_timeout_s=0.05)

    with capture_logs() as logs:
        await manager.start()

    assert registry.tools.get("alpha", []) == []
    assert registry.tools["beta"] == ["beta_tool"]
    assert manager.availability() == {"alpha": True, "beta": True}
    (warning,) = events(logs, "mcp_tool_list_failed")
    assert warning["server"] == "alpha"
    assert "timed out" in warning["reason"]


# ---------------------------------------------------------------------------------------------
# Refresh (6.12, 6.13)
# ---------------------------------------------------------------------------------------------


async def test_refresh_replaces_the_tools_of_every_available_server(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld()
    manager = build([ALPHA, BETA, OFF], world)
    await manager.start()
    world.of("alpha")[0].tools = ["a1", "a2"]
    world.of("beta")[0].tools = []

    await manager.refresh_available()

    assert registry.tools == {"alpha": ["a1", "a2"], "beta": []}


async def test_refresh_failure_keeps_the_previous_tools_and_logs_a_warning(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld(scripts={"alpha": [Attempt(tools=("kept_1", "kept_2"))]})
    manager = build([ALPHA, BETA], world)
    await manager.start()
    world.of("alpha")[0].list_error = RuntimeError("refresh exploded")
    world.of("beta")[0].tools = ["fresh"]

    with capture_logs() as logs:
        await manager.refresh_available()

    assert registry.tools == {"alpha": ["kept_1", "kept_2"], "beta": ["fresh"]}
    assert manager.availability() == {"alpha": True, "beta": True}
    (warning,) = events(logs, "mcp_tool_refresh_failed")
    assert warning["server"] == "alpha"
    assert warning["reason"] == "RuntimeError: refresh exploded"


async def test_refresh_gives_up_at_the_list_timeout_and_keeps_the_previous_tools(
    build: Build, registry: FakeRegistry
) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world, list_timeout_s=0.05)
    await manager.start()
    world.of("alpha")[0].hang_list = True

    with capture_logs() as logs:
        await manager.refresh_available()  # returns instead of waiting for the server

    assert registry.tools == {"alpha": ["alpha_tool"]}
    (warning,) = events(logs, "mcp_tool_refresh_failed")
    assert warning["server"] == "alpha"


async def test_refresh_does_nothing_for_an_unavailable_server(
    build: Build, registry: FakeRegistry
) -> None:
    gated = GatedSleep()
    world = FakeWorld()
    manager = build([ALPHA], world, sleep=gated)
    await manager.start()
    first = world.of("alpha")[0]
    first.drop()
    await until(lambda: gated.calls == [1])

    with capture_logs() as logs:
        await manager.refresh_available()

    assert first.list_calls == 1  # only the discovery at connect time
    assert events(logs, "mcp_tool_refresh_failed") == []
    assert registry.tools == {}


async def test_a_refresh_that_finishes_after_the_connection_dropped_is_not_registered(
    build: Build, registry: FakeRegistry
) -> None:
    gated = GatedSleep()
    world = FakeWorld()
    manager = build([ALPHA], world, sleep=gated, list_timeout_s=5.0)
    await manager.start()
    conn = world.of("alpha")[0]
    conn.list_gate = asyncio.Event()
    conn.tools = ["stale"]

    refresh = asyncio.create_task(manager.refresh_available())
    await until(lambda: conn.list_calls == 2, "the refresh request")  # discovery was call 1
    conn.drop()
    await until(lambda: gated.calls == [1])
    conn.list_gate.set()
    await refresh

    assert registry.tools == {}  # the stale result did not bring the tools back


# ---------------------------------------------------------------------------------------------
# call_tool (9.2, 9.6, 9.7, 9.9, 14.3)
# ---------------------------------------------------------------------------------------------


async def test_call_tool_returns_content_and_the_error_flag(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()
    conn = world.of("alpha")[0]

    ok = await manager.call_tool("alpha", "echo", {"x": 1}, 2.5)

    assert ok == CallOutcome(content=(TextItem(text="echo: ok"),), is_error=False)
    assert conn.calls == [("echo", {"x": 1}, 2.5)]  # the timeout also reaches the SDK request

    async def failing(name: str, arguments: dict[str, Any] | None) -> Any:
        return text_result("tool says no", is_error=True)

    conn.call_handler = failing
    err = await manager.call_tool("alpha", "echo", {}, 2.5)
    assert err == CallOutcome(content=(TextItem(text="tool says no"),), is_error=True)


async def test_call_to_an_unknown_server_is_unavailable(build: Build) -> None:
    manager = build([ALPHA], FakeWorld())
    await manager.start()

    with pytest.raises(McpServerUnavailable) as caught:
        await manager.call_tool("nobody", "echo", {}, 1.0)

    assert caught.value.server == "nobody"
    assert str(caught.value) == "MCP server 'nobody' is unavailable"


async def test_call_to_a_disabled_server_is_unavailable(build: Build) -> None:
    world = FakeWorld()
    manager = build([OFF], world)
    await manager.start()

    with pytest.raises(McpServerUnavailable, match="'off'"):
        await manager.call_tool("off", "echo", {}, 1.0)

    assert world.connections == []


async def test_call_to_a_server_that_failed_to_connect_is_unavailable(build: Build) -> None:
    world = FakeWorld(default=Attempt(open_error=OSError("down")))
    manager = build([ALPHA], world, max_reconnects=0)
    await manager.start()

    with pytest.raises(McpServerUnavailable, match="'alpha'"):
        await manager.call_tool("alpha", "echo", {}, 1.0)


async def test_call_that_does_not_answer_raises_timeout_error(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()

    async def never(name: str, arguments: dict[str, Any] | None) -> Any:
        await asyncio.Event().wait()

    world.of("alpha")[0].call_handler = never

    with pytest.raises(TimeoutError):
        await manager.call_tool("alpha", "slow", {}, 0.05)

    assert manager.availability() == {"alpha": True}  # a slow tool does not drop the server


async def test_timeout_reported_by_the_sdk_becomes_timeout_error(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()

    async def sdk_timeout(name: str, arguments: dict[str, Any] | None) -> Any:
        raise MCPError(REQUEST_TIMEOUT, "Timed out while waiting for response")

    world.of("alpha")[0].call_handler = sdk_timeout

    with pytest.raises(TimeoutError, match="'slow'"):
        await manager.call_tool("alpha", "slow", {}, 30.0)


async def test_late_reply_is_discarded(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()
    conn = world.of("alpha")[0]
    release = asyncio.Event()
    cancelled: list[str] = []
    answered: list[str] = []

    async def slow(name: str, arguments: dict[str, Any] | None) -> Any:
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(name)  # the waiting request was dropped
            raise
        answered.append(name)
        return text_result("too late")

    conn.call_handler = slow
    with pytest.raises(TimeoutError):
        await manager.call_tool("alpha", "slow", {}, 0.05)

    release.set()  # the server "answers" after the caller gave up
    await drain_loop()

    assert cancelled == ["slow"]
    assert answered == []  # nothing consumed the late reply

    conn.call_handler = None
    outcome = await manager.call_tool("alpha", "next", {}, 1.0)
    assert outcome.content == (TextItem(text="next: ok"),)  # the session still works


async def test_call_exception_text_reaches_the_caller_unchanged(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()

    async def broken(name: str, arguments: dict[str, Any] | None) -> Any:
        raise RuntimeError("disk exploded at /var/data/x.db")

    world.of("alpha")[0].call_handler = broken

    with pytest.raises(RuntimeError) as caught:
        await manager.call_tool("alpha", "write", {}, 1.0)

    assert str(caught.value) == "disk exploded at /var/data/x.db"
    assert manager.availability() == {"alpha": True}


async def test_protocol_error_from_the_server_keeps_its_text(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()

    async def invalid(name: str, arguments: dict[str, Any] | None) -> Any:
        raise MCPError(-32602, "Invalid params: 'path' is required")

    world.of("alpha")[0].call_handler = invalid

    with pytest.raises(MCPError, match=r"Invalid params: 'path' is required"):
        await manager.call_tool("alpha", "read", {}, 1.0)


async def test_calls_run_concurrently_on_one_session(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()
    started: list[str] = []
    both_started = asyncio.Event()

    async def rendezvous(name: str, arguments: dict[str, Any] | None) -> Any:
        started.append(name)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()  # finishes only if the other call is in flight too
        return text_result(f"{name} done")

    world.of("alpha")[0].call_handler = rendezvous

    async with asyncio.timeout(5):
        first, second = await asyncio.gather(
            manager.call_tool("alpha", "one", {}, 5.0), manager.call_tool("alpha", "two", {}, 5.0)
        )

    assert first.content == (TextItem(text="one done"),)
    assert second.content == (TextItem(text="two done"),)
    assert len(world.of("alpha")) == 1


# ---------------------------------------------------------------------------------------------
# stop (6.11, 17.15)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("grace", [2.0, 7.5])
async def test_stop_forwards_the_grace_to_every_transport(
    build: Build, registry: FakeRegistry, grace: float
) -> None:
    world = FakeWorld()
    manager = build([ALPHA, BETA, GAMMA], world)
    await manager.start()
    assert [c.grace() for c in world.connections] == [DEFAULT_GRACE_S] * 3  # before stop

    await manager.stop(grace)

    assert [c.closed for c in world.connections] == [True] * 3
    assert [c.grace_at_exit for c in world.connections] == [grace] * 3
    assert manager.availability() == {"alpha": False, "beta": False, "gamma": False}
    assert registry.tools == {}


async def test_stop_uses_the_default_grace(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()

    await manager.stop()

    assert world.of("alpha")[0].grace_at_exit == DEFAULT_GRACE_S


async def test_stop_does_not_reconnect_a_server(build: Build) -> None:
    world = FakeWorld()
    manager = build([ALPHA], world)
    await manager.start()

    await manager.stop()
    await drain_loop()

    assert world.opens == {"alpha": 1}


async def test_stop_ends_a_server_that_is_waiting_to_reconnect(build: Build) -> None:
    gated = GatedSleep()  # never opened: the supervisor is parked in its backoff delay
    world = FakeWorld(default=Attempt(open_error=OSError("down")))
    manager = build([ALPHA], world, sleep=gated)
    await manager.start()
    await until(lambda: gated.calls == [1])

    async with asyncio.timeout(5):
        await manager.stop()

    assert world.opens == {"alpha": 1}
    assert manager.availability() == {"alpha": False}


async def test_stop_before_start_is_harmless(build: Build) -> None:
    manager = build([ALPHA], FakeWorld())

    await manager.stop()

    assert manager.availability() == {"alpha": False}


# ---------------------------------------------------------------------------------------------
# ServerSupervisor on its own
# ---------------------------------------------------------------------------------------------


async def test_supervisor_state_follows_the_connection(
    clock: FakeClock, registry: FakeRegistry
) -> None:
    gated = GatedSleep()
    world = FakeWorld()
    supervisor = ServerSupervisor(
        ALPHA,
        registry,
        config_dir=CONFIG_DIR,
        transport_factory=world.transport,
        session_factory=world.session,
        clock=clock,
        sleep=gated,
    )
    states: list[ServerState] = []  # a snapshot per check keeps each comparison independent

    def snapshot() -> ServerState:
        states.append(supervisor.state)
        return states[-1]

    def has_adapter() -> bool:
        return supervisor.adapter is not None

    assert snapshot() is ServerState.UNAVAILABLE
    assert not has_adapter()
    task = asyncio.create_task(supervisor.run())

    await supervisor.settled.wait()
    assert snapshot() is ServerState.AVAILABLE
    assert has_adapter()

    world.of("alpha")[0].drop()
    await until(lambda: gated.calls == [1])
    assert snapshot() is ServerState.UNAVAILABLE
    assert not has_adapter()

    supervisor.request_stop()
    async with asyncio.timeout(5):
        await task
    assert supervisor.stopping
