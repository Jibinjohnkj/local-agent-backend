"""Reusable helpers behind the integration fixtures of `tests/integration/conftest.py` (task 22.1).

Import what a test needs from here (`from integration_harness import ...`); the fixtures that wire
these helpers into pytest are in `tests/integration/conftest.py`.

* Config: `config_dict` builds a Backend_Config as plain data (always with `storage.path` set),
  `echo_server_entry` is the `mcp_servers` entry that runs `tests/fixtures/mcp_echo_server.py`.
* LLM, in process: `ScriptedCompletion` replaces `litellm.acompletion` and plays `fakes.LlmTurn`s.
* LLM, over HTTP: `FakeLlmServer` is an OpenAI-compatible `/v1/chat/completions` stub on a free
  port. `StubTurn`s script the chunks, the delay before each chunk, a stall until a
  `threading.Event` is set, an abrupt end, or an error status.
* App, in process: `open_app` builds the production app over a real Session_Store of either kind and
  serves it through `httpx.ASGITransport` (lifespan included) or through a real uvicorn socket.
* App, as a process: `BackendLauncher` / `BackendProcess` run `python -m local_agent` with a
  temporary config, Storage_Directory and free port, wait for `/health`, send the shutdown
  signal of the OS, kill hard, and restart on the same Storage_Directory.
* Processes: `PIDS` records pids (and Popen handles) so the session finalizer can kill leftovers.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, Literal, Self

import httpx
import yaml
from starlette.applications import Starlette

from fakes import DEFAULT_TURN, FakeRejection, LlmFailure, LlmTurn, RecordedRequest
from local_agent.a2a_server.app import Deps, build_app, build_deps
from local_agent.config.models import BackendConfig
from local_agent.models import SystemClock
from local_agent.server import build_server
from local_agent.storage import migrations, recovery
from local_agent.storage.database import Database

__all__ = [
    "A2A_HEADERS",
    "DATABASE_NAME",
    "ECHO_SERVER",
    "PIDS",
    "AppHandle",
    "BackendExited",
    "BackendLauncher",
    "BackendProcess",
    "FakeLlmServer",
    "PidRegistry",
    "ScriptedCompletion",
    "StubToolCall",
    "StubTurn",
    "config_dict",
    "deep_merge",
    "echo_server_entry",
    "free_port",
    "isolated_env",
    "jsonrpc_call",
    "kill_pid",
    "llm_tool_name",
    "message_params",
    "open_app",
    "pid_alive",
    "read_request_log",
    "wait_for_task_state",
    "wait_until_gone",
]

FIXTURES_DIR: Final = Path(__file__).resolve().parent / "fixtures"
ECHO_SERVER: Final = FIXTURES_DIR / "mcp_echo_server.py"
DATABASE_NAME: Final = "local-agent.db"
A2A_HEADERS: Final = {"A2A-Version": "1.0"}
LOOPBACK: Final = "127.0.0.1"
MODEL: Final = "openai/fake-model"
TERMINAL_STATES: Final = frozenset(
    {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}
)


# --- Processes ----------------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """True while a process with this pid exists (POSIX: `os.kill(pid, 0)`; Windows: a handle)."""
    if sys.platform == "win32":
        import ctypes

        synchronize, wait_timeout = 0x00100000, 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False  # ERROR_INVALID_PARAMETER: no such process
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) == wait_timeout)
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def kill_pid(pid: int) -> None:
    """End one process at once (`SIGKILL`; `TerminateProcess` on Windows). A missing pid is fine."""
    if sys.platform == "win32":
        sig = signal.SIGTERM  # `os.kill` calls `TerminateProcess` for any non-console signal.
    else:
        sig = signal.SIGKILL
    with contextlib.suppress(OSError):
        os.kill(pid, sig)


def wait_until_gone(pids: Sequence[int], timeout_s: float) -> list[int]:
    """Wait until none of `pids` exists. Returns the pids still alive after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    remaining = [pid for pid in pids if pid_alive(pid)]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.02)
        remaining = [pid for pid in remaining if pid_alive(pid)]
    return remaining


class PidRegistry:
    """Pids and Popen handles that a test run started, so a failing test cannot leak them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pids: set[int] = set()
        self._procs: list[subprocess.Popen[bytes]] = []

    def record(self, *pids: int) -> None:
        with self._lock:
            self._pids.update(pids)

    def track_process(self, popen: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._procs.append(popen)
            self._pids.add(popen.pid)

    def record_from_log(self, path: Path) -> list[int]:
        """Record every pid in an echo server request log (`--request-log`). Returns them."""
        pids = sorted({int(entry["pid"]) for entry in read_request_log(path)})
        self.record(*pids)
        return pids

    def kill_leftovers(self) -> list[int]:
        """Kill every recorded process that still runs. Returns the pids that had to be killed."""
        with self._lock:
            procs, pids = list(self._procs), set(self._pids)
            self._procs.clear()
            self._pids.clear()
        killed: list[int] = []
        for popen in procs:
            pids.discard(popen.pid)
            if popen.poll() is None:
                killed.append(popen.pid)
                if sys.platform != "win32":
                    with contextlib.suppress(OSError):
                        os.killpg(popen.pid, signal.SIGKILL)
                popen.kill()
                popen.wait(timeout=10)
        for pid in sorted(pids):
            if pid_alive(pid):
                killed.append(pid)
                kill_pid(pid)
        return killed


PIDS: Final = PidRegistry()


def read_request_log(path: Path) -> list[dict[str, Any]]:
    """The JSON lines the echo server wrote (`start`, `list_tools`, `call_tool`), or none."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def free_port() -> int:
    """A TCP port on 127.0.0.1 that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


# --- Config -------------------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """`base` with `overrides` laid over it; nested mappings merge, everything else replaces."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def llm_tool_name(server: str, tool: str) -> str:
    """The name the LLM sees for an MCP tool whose names need no sanitizing."""
    return f"{server}__{tool}"


def echo_server_entry(
    name: str = "echo",
    *,
    page_size: int | None = None,
    crash_after: int | None = None,
    request_log: Path | None = None,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """The `mcp_servers` entry that runs `tests/fixtures/mcp_echo_server.py` under this Python."""
    args = [str(ECHO_SERVER)]
    if page_size is not None:
        args += ["--page-size", str(page_size)]
    if crash_after is not None:
        args += ["--crash-after", str(crash_after)]
    if request_log is not None:
        args += ["--request-log", str(request_log)]
    return {
        "name": name,
        "transport": "stdio",
        "command": sys.executable,
        "args": [*args, *extra_args],
    }


def config_dict(
    *,
    storage_dir: Path,
    storage_backend: Literal["sqlite", "memory"] = "sqlite",
    port: int = 10000,
    llm_base: str | None = None,
    model: str = MODEL,
    mcp_servers: Sequence[Mapping[str, Any]] = (),
    permissions: Mapping[str, Sequence[str]] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A Backend_Config as plain data. `storage.path` is set for both backends (13.11)."""
    llm: dict[str, Any] = {"model": model, "api_key": "test-key", "request_timeout_s": 30}
    if llm_base is not None:
        llm["api_base"] = llm_base
    config: dict[str, Any] = {
        "server": {"host": LOOPBACK, "port": port},
        "llm": llm,
        "storage": {"backend": storage_backend, "path": str(storage_dir)},
        "mcp_servers": [dict(entry) for entry in mcp_servers],
        "permissions": {server: list(patterns) for server, patterns in (permissions or {}).items()},
        "logging": {"level": "info"},
    }
    return deep_merge(config, overrides) if overrides else config


def isolated_env(home: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a child Backend: per-user data locations point into `home`.

    `LOCALAPPDATA`, `HOME`, and `XDG_DATA_HOME` are the inputs of the default Storage_Directory
    (17.35), so a child that forgets `storage.path` still cannot touch the developer's real
    directory. Local addresses bypass any proxy, so the LLM stub is reached directly.
    """
    env = {key: value for key, value in os.environ.items() if key != "AGENT_CONFIG"}
    env.update(
        {
            "HOME": str(home),
            "LOCALAPPDATA": str(home / "LocalAppData"),
            "XDG_DATA_HOME": str(home / "xdg-data"),
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if extra:
        env.update(extra)
    return env


# --- A2A over HTTP (sync, for subprocess tests) -------------------------------------------------

_request_ids = iter(range(1, 1_000_000))


def message_params(
    text: str, *, context_id: str | None = None, task_id: str | None = None
) -> dict[str, Any]:
    """The `params` of a `SendMessage` / `SendStreamingMessage` call carrying one user text."""
    message: dict[str, Any] = {
        "messageId": f"it-{next(_request_ids)}",
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    return {"message": message}


def jsonrpc_call(client: httpx.Client, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """One JSON-RPC call to the Backend. Returns the whole response document."""
    envelope = {"jsonrpc": "2.0", "id": next(_request_ids), "method": method, "params": params}
    response = client.post("/", json=envelope, headers=A2A_HEADERS)
    response.raise_for_status()
    document: dict[str, Any] = response.json()
    return document


def wait_for_task_state(
    client: httpx.Client,
    task_id: str,
    states: Sequence[str] = tuple(TERMINAL_STATES),
    *,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """Poll `GetTask` until the Task is in one of `states` (ProtoJSON names). Returns the Task."""
    deadline = time.monotonic() + timeout_s
    task: dict[str, Any] = {}
    while time.monotonic() < deadline:
        document = jsonrpc_call(client, "GetTask", {"id": task_id})
        task = document.get("result", {})
        if task.get("status", {}).get("state") in states:
            return task
        time.sleep(0.05)
    raise TimeoutError(f"task {task_id} did not reach {list(states)}; last seen: {task}")


# --- LLM, in process ----------------------------------------------------------------------------


class ScriptedCompletion:
    """A stand-in for `litellm.acompletion` that plays `fakes.LlmTurn`s.

    Call `n` is answered by `turns[n]`, then by `exhausted`. Every call counts, so a retry of an
    `UNREACHABLE` turn uses up the next one. `calls` holds what each call sent. The chunks are the
    dictionaries the gateway reads, so the real `LlmGateway` runs on top of it.
    """

    def __init__(self, turns: Sequence[LlmTurn] = (), *, exhausted: LlmTurn = DEFAULT_TURN) -> None:
        self.turns: list[LlmTurn] = list(turns)
        self.exhausted = exhausted
        self.calls: list[RecordedRequest] = []

    def script(self, *turns: LlmTurn) -> None:
        """Replace the script and forget the calls so far."""
        self.turns = list(turns)
        self.calls.clear()

    async def __call__(self, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        index = len(self.calls)
        turn = self.turns[index] if index < len(self.turns) else self.exhausted
        self.calls.append(
            RecordedRequest(
                index,
                copy.deepcopy(list(kwargs.get("messages") or [])),
                copy.deepcopy(kwargs.get("tools")),
                attempts=1,
            )
        )
        if turn.on_request is not None:
            turn.on_request()
        if turn.failure is LlmFailure.UNREACHABLE:
            raise ConnectionError("connection refused")
        if turn.failure is LlmFailure.REJECTED:
            raise FakeRejection(turn.reason)
        return self._chunks(turn)

    async def _chunks(self, turn: LlmTurn) -> AsyncIterator[dict[str, Any]]:
        deltas = turn.deltas
        acts = turn.failure in (LlmFailure.INTERRUPTED, LlmFailure.STALL)
        limit = min(turn.fail_after, len(deltas)) if acts else -1
        for position in range(len(deltas) + 1):
            if position == limit:
                if turn.failure is LlmFailure.STALL:
                    await asyncio.get_running_loop().create_future()
                raise ConnectionError("stream broke")
            if position == len(deltas):
                return
            for _ in range(turn.latency):
                await asyncio.sleep(0)
            yield {"choices": [{"delta": dict(deltas[position])}]}


# --- LLM, over HTTP -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StubToolCall:
    """One streamed Tool_Call of a `StubTurn`."""

    name: str
    arguments: str = "{}"
    call_id: str = "call-1"


@dataclass(frozen=True)
class StubTurn:
    """One scripted answer of `FakeLlmServer`; request `n` gets turn `n`.

    `chunks` are text deltas and `tool_calls` follow them, one delta each. `chunk_delay_s` is the
    wait before every delta. After `hold_after` deltas the stream waits until `gate` is set.
    After `abort_after` deltas the connection ends without the closing chunk (a broken stream).
    A `status` other than 200 answers with an error body carrying `error_message` instead.
    """

    chunks: tuple[str, ...] = ()
    tool_calls: tuple[StubToolCall, ...] = ()
    chunk_delay_s: float = 0.0
    hold_after: int | None = None
    gate: threading.Event | None = None
    abort_after: int | None = None
    status: int = 200
    error_message: str = "invalid request"


STUB_EXHAUSTED: Final = StubTurn(chunks=("ok",))
"""What the stub answers once its script has run out."""


class _StubHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], stub: FakeLlmServer) -> None:
        super().__init__(address, _StubHandler)
        self.stub = stub


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _StubHttpServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        self._reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._reply(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        stub = self.server.stub
        turn = stub._begin(body)
        try:
            if turn.status != 200:
                self._reply(turn.status, {"error": {"message": turn.error_message}})
            else:
                self._stream(stub, turn)
        except OSError:
            stub._disconnected()
        self.close_connection = True

    def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _stream(self, stub: FakeLlmServer, turn: StubTurn) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        deltas: list[dict[str, Any]] = [{"content": text} for text in turn.chunks]
        for index, call in enumerate(turn.tool_calls):
            fragment = {
                "index": index,
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            deltas.append({"tool_calls": [fragment]})
        finish = "tool_calls" if turn.tool_calls else "stop"
        for position, delta in enumerate(deltas):
            if turn.abort_after is not None and position >= turn.abort_after:
                return  # the connection ends here, without the closing chunk
            if turn.hold_after is not None and position >= turn.hold_after and turn.gate:
                while not turn.gate.wait(0.05):
                    if stub.stopping.is_set():
                        return
            if turn.chunk_delay_s and stub.stopping.wait(turn.chunk_delay_s):
                return
            self._write_event(self._event(delta, None, position == 0))
            stub._chunk_sent()
        self._write_event(self._event({}, finish, first=not deltas))
        self._write_chunk(b"data: [DONE]\n\n")
        self._write_chunk(b"")  # the terminating zero-length chunk

    def _write_event(self, event: Mapping[str, Any]) -> None:
        self._write_chunk(b"data: " + json.dumps(event).encode() + b"\n\n")

    @staticmethod
    def _event(delta: Mapping[str, Any], finish: str | None, first: bool) -> dict[str, Any]:
        full = {"role": "assistant", **delta} if first else dict(delta)
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake-model",
            "choices": [{"index": 0, "delta": full, "finish_reason": finish}],
        }


class FakeLlmServer:
    """An OpenAI-compatible chat-completions stub on a free port, for subprocess tests.

    Point `llm.api_base` at `base_url` with a model such as `openai/fake-model`. `requests` holds
    the JSON body of every request in arrival order. Use it as a context manager, or call
    `start` and `stop`.
    """

    def __init__(
        self, turns: Sequence[StubTurn] = (), *, exhausted: StubTurn = STUB_EXHAUSTED
    ) -> None:
        self.turns: list[StubTurn] = list(turns)
        self.exhausted = exhausted
        self.requests: list[dict[str, Any]] = []
        self.chunks_sent = 0
        self.disconnects = 0
        self.stopping = threading.Event()
        self._cond = threading.Condition()
        self._http: _StubHttpServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    def script(self, *turns: StubTurn) -> None:
        """Replace the script and forget the requests so far."""
        with self._cond:
            self.turns = list(turns)
            self.requests.clear()
            self._cond.notify_all()

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}/v1"

    def start(self) -> Self:
        self._http = _StubHttpServer((LOOPBACK, 0), self)
        self.port = int(self._http.server_address[1])
        self._thread = threading.Thread(
            target=self._http.serve_forever, name="llm-stub", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stopping.set()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def wait_for_requests(self, count: int, timeout_s: float = 10.0) -> list[dict[str, Any]]:
        """Wait until at least `count` requests arrived. Returns all requests so far."""
        with self._cond:
            if not self._cond.wait_for(lambda: len(self.requests) >= count, timeout_s):
                raise TimeoutError(f"expected {count} LLM requests, saw {len(self.requests)}")
            return list(self.requests)

    def wait_for_chunks(self, count: int, timeout_s: float = 10.0) -> None:
        """Wait until the stub has sent at least `count` deltas in total."""
        with self._cond:
            if not self._cond.wait_for(lambda: self.chunks_sent >= count, timeout_s):
                raise TimeoutError(f"expected {count} chunks sent, saw {self.chunks_sent}")

    def _begin(self, body: dict[str, Any]) -> StubTurn:
        with self._cond:
            index = len(self.requests)
            self.requests.append(body)
            self._cond.notify_all()
            return self.turns[index] if index < len(self.turns) else self.exhausted

    def _chunk_sent(self) -> None:
        with self._cond:
            self.chunks_sent += 1
            self._cond.notify_all()

    def _disconnected(self) -> None:
        with self._cond:
            self.disconnects += 1
            self._cond.notify_all()


# --- App, in process ----------------------------------------------------------------------------


@dataclass
class AppHandle:
    """The production app over a real Session_Store, and a client that talks to it."""

    client: httpx.AsyncClient
    app: Starlette
    deps: Deps
    config: BackendConfig
    storage_dir: Path
    base_url: str


def _open_database(storage_dir: Path) -> Database:
    """The storage startup of `__main__` that matters to a test: open, migrate, recover (18.28)."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / DATABASE_NAME
    database = Database.open(path, create=True)
    try:
        migrations.apply(database, str(path))
        recovery.run(database, SystemClock())
    except BaseException:
        database.close()
        raise
    return database


@contextlib.asynccontextmanager
async def _serve_uvicorn(app: Starlette, *, timeout_s: float = 10.0) -> AsyncIterator[str]:
    """Serve `app` with the Backend's own `BackendServer` on a free port of this event loop."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOOPBACK, 0))
    port = int(sock.getsockname()[1])
    server = build_server(app, LOOPBACK, port)
    task = asyncio.create_task(server.serve(sockets=[sock]), name="uvicorn")
    try:
        async with asyncio.timeout(timeout_s):
            while not server.started and not task.done():  # noqa: ASYNC110 (uvicorn has no event)
                await asyncio.sleep(0.01)
        if not server.started:
            await task  # re-raises the startup error, if there was one
            raise RuntimeError("uvicorn stopped before it started")
        yield f"http://{LOOPBACK}:{port}"
    finally:
        server.should_exit = True
        try:
            async with asyncio.timeout(timeout_s):
                await task
        finally:
            sock.close()


@contextlib.asynccontextmanager
async def open_app(
    config: BackendConfig,
    storage_dir: Path,
    *,
    transport: Literal["asgi", "uvicorn"] = "asgi",
    timeout_s: float = 30.0,
) -> AsyncIterator[AppHandle]:
    """Build the production app for `config` and serve it until the block ends.

    `sqlite` opens `storage_dir/local-agent.db` (migrations and Restart_Recovery run, as at
    startup); `memory` touches no file. `asgi` calls the app through `httpx.ASGITransport` and runs
    the lifespan here, so MCP_Servers start and stop; `uvicorn` serves it on a free port, and
    uvicorn runs the lifespan. Leaving the block runs the Backend's shutdown (stops MCP, closes the
    database). To restart on the same data, leave the block and call `open_app` again.
    """
    database = None
    if config.storage.backend == "sqlite":
        database = await asyncio.to_thread(_open_database, storage_dir)
    lifespan_started = False
    try:
        deps = build_deps(config, database, config_dir=str(storage_dir))
        app = build_app(config, deps)
        async with contextlib.AsyncExitStack() as stack:
            if transport == "asgi":
                await stack.enter_async_context(app.router.lifespan_context(app))
                lifespan_started = True
                http = httpx.ASGITransport(app=app, client=(LOOPBACK, 50000))
                base_url = "http://localhost"
                client = httpx.AsyncClient(transport=http, base_url=base_url, timeout=timeout_s)
            else:
                base_url = await stack.enter_async_context(_serve_uvicorn(app))
                lifespan_started = True
                client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
            async with client:
                yield AppHandle(client, app, deps, config, storage_dir, base_url)
    finally:
        if database is not None and not lifespan_started:
            await asyncio.to_thread(database.close)  # startup failed before the lifespan owned it


# --- App, as a process --------------------------------------------------------------------------


class BackendExited(RuntimeError):
    """The Backend process ended before it answered `/health`."""


class BackendProcess:
    """One `python -m local_agent` child, started from a config file in its own working directory.

    On POSIX it starts in a new session (so signals sent to the test run do not reach it, and
    `SIGINT` to its group is a test of its own); on Windows in a new process group, so
    `CTRL_BREAK_EVENT` can be sent to it. Its output goes to files, never to pipes, so a chatty
    Backend cannot block. `restart` starts it again on the same config, port, and Storage_Directory.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        workdir: Path,
        env: Mapping[str, str],
        port: int,
        new_session: bool = True,
    ) -> None:
        self.config_path = config_path
        self.workdir = workdir
        self.port = port
        self.starts = 0
        self._env = dict(env)
        self._new_session = new_session
        self._popen: subprocess.Popen[bytes] | None = None
        self._stderr_path: Path | None = None
        self._stdout_path: Path | None = None

    # -- lifecycle --

    def start(self) -> Self:
        if self.running:
            raise RuntimeError("the Backend is already running")
        self.starts += 1
        self._stdout_path = self.workdir / f"stdout-{self.starts}.log"
        self._stderr_path = self.workdir / f"stderr-{self.starts}.log"
        flags: dict[str, Any] = {}
        if sys.platform == "win32":
            flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        elif self._new_session:
            flags["start_new_session"] = True
        env = {**self._env, "AGENT_CONFIG": str(self.config_path)}
        with self._stdout_path.open("wb") as out, self._stderr_path.open("wb") as err:
            self._popen = subprocess.Popen(
                [sys.executable, "-m", "local_agent"],
                cwd=self.workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                **flags,
            )
        PIDS.track_process(self._popen)
        return self

    def restart(self) -> Self:
        """Start again after the process ended (stopped, killed, or crashed)."""
        return self.start()

    @property
    def popen(self) -> subprocess.Popen[bytes]:
        if self._popen is None:
            raise RuntimeError("the Backend was never started")
        return self._popen

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def running(self) -> bool:
        return self._popen is not None and self._popen.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.popen.poll()

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    def client(self, *, timeout_s: float = 20.0) -> httpx.Client:
        """A sync client for this Backend with the `A2A-Version` header set."""
        return httpx.Client(base_url=self.url, headers=A2A_HEADERS, timeout=timeout_s)

    def wait_healthy(self, timeout_s: float = 30.0) -> Self:
        """Wait until `GET /health` answers 200; `BackendExited` if the process ends first."""
        deadline = time.monotonic() + timeout_s
        while True:
            code = self.popen.poll()
            if code is not None:
                raise BackendExited(f"exit status {code}\n{self.stderr_text()}")
            try:
                if httpx.get(f"{self.url}/health", timeout=1.0).status_code == 200:
                    return self
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"no /health answer in {timeout_s}s\n{self.stderr_text()}")
            time.sleep(0.1)

    # -- ending it --

    def send_shutdown_signal(self, *, interrupt: bool = False) -> None:
        """Ask for Graceful_Shutdown: `SIGTERM` (`SIGINT`), or `CTRL_BREAK_EVENT` on Windows."""
        if sys.platform == "win32":
            self.popen.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.popen.send_signal(signal.SIGINT if interrupt else signal.SIGTERM)

    def wait(self, timeout_s: float = 15.0) -> int:
        return self.popen.wait(timeout=timeout_s)

    def stop(self, timeout_s: float = 15.0) -> int:
        """Graceful_Shutdown, then wait. Kills the process and raises `TimeoutError` if it hangs."""
        if self.running:
            self.send_shutdown_signal()
        try:
            return self.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.kill()
            raise TimeoutError(
                f"the Backend did not stop in {timeout_s}s\n{self.stderr_text()}"
            ) from None

    def kill(self) -> int:
        """End the process at once (`SIGKILL` / `TerminateProcess`) and wait for its handle."""
        if self.running:
            self.popen.kill()
        return self.popen.wait(timeout=15)

    def close(self) -> None:
        """Teardown: end the process whatever its state. Never raises."""
        with contextlib.suppress(Exception):
            if self.running:
                self.kill()

    # -- output --

    def stderr_text(self) -> str:
        """The stderr of the latest run: the structured log, or the config error report."""
        path = self._stderr_path
        return path.read_text(encoding="utf-8", errors="replace") if path and path.exists() else ""

    def stdout_text(self) -> str:
        path = self._stdout_path
        return path.read_text(encoding="utf-8", errors="replace") if path and path.exists() else ""

    def log_records(self) -> list[dict[str, Any]]:
        """The JSON log records in stderr (lines that are not JSON objects are skipped)."""
        records: list[dict[str, Any]] = []
        for line in self.stderr_text().splitlines():
            with contextlib.suppress(ValueError):
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
        return records


@dataclass
class BackendLauncher:
    """Starts `BackendProcess`es under `root`, each with its own working directory and port.

    Launches share `root/storage` unless `storage_dir` is given, so a second launch is a restart
    (or a Storage_Lock conflict) on the same data. `close` ends everything that still runs.
    """

    root: Path
    home: Path
    processes: list[BackendProcess] = field(default_factory=list)

    @property
    def storage_dir(self) -> Path:
        return self.root / "storage"

    def launch(
        self,
        *,
        storage_backend: Literal["sqlite", "memory"] = "sqlite",
        storage_dir: Path | None = None,
        llm_base: str | None = None,
        mcp_servers: Sequence[Mapping[str, Any]] = (),
        permissions: Mapping[str, Sequence[str]] | None = None,
        overrides: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
        port: int | None = None,
        new_session: bool = True,
        wait: bool = True,
    ) -> BackendProcess:
        """Write a config, start `python -m local_agent`, and (with `wait`) wait for `/health`."""
        workdir = self.root / f"backend-{len(self.processes) + 1}"
        workdir.mkdir(parents=True)
        the_port = free_port() if port is None else port
        config = config_dict(
            storage_dir=storage_dir or self.storage_dir,
            storage_backend=storage_backend,
            port=the_port,
            llm_base=llm_base,
            mcp_servers=mcp_servers,
            permissions=permissions,
            overrides=overrides,
        )
        config_path = workdir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        process = BackendProcess(
            config_path=config_path,
            workdir=workdir,
            env=isolated_env(self.home, env),
            port=the_port,
            new_session=new_session,
        )
        self.processes.append(process)
        process.start()
        if wait:
            try:
                process.wait_healthy()
            except BaseException:
                process.close()
                raise
        return process

    def close(self) -> None:
        for process in reversed(self.processes):
            process.close()
