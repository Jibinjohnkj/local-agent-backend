"""Scenario scripts, the harness that runs them, and the kill-injection helper.

A **scenario script** (`ScenarioScript`) says what the LLM and the MCP_Servers do during one Task:
per LLM turn the text chunks and Tool_Calls (valid, unknown name, bad JSON, schema-invalid,
unavailable server, protected), per-call latency and outcome (success, error flag, timeout,
exception, unavailable, hang), the approval decision for each protected call, registry changes
between turns, a failure of the LLM (unreachable, rejected, interrupted, stalled), and an optional
cancel after the n-th event. `scenario_scripts()` draws them.

The harness turns a script into the fakes of `tests/fakes.py` and runs the real `TaskRunner`,
`ToolDispatcher`, `ApprovalGate`, and `TaskEventEmitter` over a real Session_Store:

    with open_store(store_kind) as env:               # fresh TemporaryDirectory for SQLite
        run = run_scenario(script, env)               # asyncio.run: a fresh event loop
        assert run.state is TaskState.COMPLETED

* `store_kind` is a parametrized fixture (`memory`, `sqlite`). Use the kind inside the test body
  with `open_store`, so every Hypothesis example gets its own store and directory.
* `run_plans` runs several Contexts concurrently, and several Tasks per Context one after another.
* `kill_run` runs a script on SQLite with `SimulatedCrash` at statement k and returns the path of
  the database the "killed process" left behind:

      with tempfile.TemporaryDirectory() as tmp:
          clean = kill_run(script, Path(tmp) / "clean", None)
          for k in (1, clean.statements // 2, clean.statements):
              killed = kill_run(script, Path(tmp) / f"killed-{k}", k)
              db = Database.open(killed.db_path, create=False)   # reopen, as at startup

  The statement count is exact for one Context. With several Contexts the interleaving can differ
  slightly between runs, because token estimates run on a worker thread; every k still yields a
  valid killed database.
* `build_transcript` and `normalize_transcript` produce the platform-neutral transcript of P46.

Every approval, cancel, and block is injected from the recording sink and the fakes, never by
timing, so a script produces the same events on every run.

A call's `kind` describes what the LLM emitted. It predicts the outcome only while the registry
does not change under the call, so generate with `world_changes=False` for assertions on outcomes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from hypothesis import strategies as st
from strategies import stored_texts, text_chars

from fakes import (
    DEFAULT_BEHAVIOR,
    ArtifactEvent,
    ClockSleep,
    FakeLlmGateway,
    FakeMcpManager,
    LlmFailure,
    LlmTurn,
    ManualTimers,
    RecordingEventSink,
    SinkEvent,
    StatusEvent,
    TickingClock,
    ToolBehavior,
    ToolOutcome,
    text_delta,
    tool_call_deltas,
)
from local_agent.a2a_server.events import KIND_APPROVAL_REQUIRED, TERMINAL_STATES, TaskState
from local_agent.config.models import OrchestrationConfig
from local_agent.logging_setup import Redactor
from local_agent.mcp_client.registry import ToolRegistry
from local_agent.models import ChatMessage, McpToolInfo, SystemMessage, TaskRecord
from local_agent.orchestrator.approval_gate import Decision
from local_agent.orchestrator.loop import RunnerDeps, TaskRunner
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.session.store import SessionStore
from local_agent.storage import StorageError
from local_agent.storage.database import Database
from local_agent.storage.migrations import apply
from storage_faults import ErrorSource, FaultFactory, SimulatedCrash, crash

__all__ = [
    "CONTEXT",
    "DEFAULT_WORLD",
    "STORE_KINDS",
    "TASK",
    "ApprovalChoice",
    "CallKind",
    "CallScript",
    "ContextPlan",
    "KillResult",
    "ScenarioRun",
    "ScenarioScript",
    "ServerSpec",
    "StoreEnv",
    "StoreKind",
    "TurnScript",
    "WorldChange",
    "WorldChangeKind",
    "WorldSpec",
    "build_fakes",
    "build_registry",
    "build_transcript",
    "kill_run",
    "normalize_transcript",
    "open_store",
    "render_deltas",
    "run_plans",
    "run_scenario",
    "scenario_scripts",
    "store_kind",
    "tagged_arguments",
    "transcript_json",
]

CONTEXT = "ctx-1"
TASK = "task-1"
DB_NAME = "local-agent.db"
MODEL = "fake/model"
HARNESS_TIMEOUT_S = 10.0
"""Real seconds a scenario may take before the harness reports a hang instead of waiting."""

type StoreKind = Literal["memory", "sqlite"]
STORE_KINDS: tuple[StoreKind, ...] = ("memory", "sqlite")


@pytest.fixture(scope="module", params=STORE_KINDS)
def store_kind(request: pytest.FixtureRequest) -> StoreKind:
    """Parametrizes a test over `InMemorySessionStore` and `SQLiteSessionStore` (11.10, 18.29).

    Import it into the test module (`from scenarios import store_kind`) and open the store inside
    the test body with `open_store(store_kind)`. It is module-scoped so it can be combined with
    `@given` without a Hypothesis health-check suppression; it holds only the kind, not a store.
    """
    return cast("StoreKind", request.param)


# --- Stores -------------------------------------------------------------------------------------


@dataclass
class StoreEnv:
    """A Session_Store ready for a scenario, with the clock and redactor it was built with."""

    kind: StoreKind
    store: SessionStore
    clock: TickingClock
    redactor: Redactor
    db_path: str
    path: Path | None = None
    factory: FaultFactory | None = None
    _reader: SessionStore | None = None

    @property
    def reader(self) -> SessionStore:
        """A store for reading what is committed. For SQLite it is a separate connection that is
        never faulted and not counted, and it works after a simulated kill, when `store` is dead."""
        return self.store if self._reader is None else self._reader

    @property
    def crashed(self) -> bool:
        """True once a `SimulatedCrash` ended the SQLite connection."""
        return self.factory is not None and self.factory.connection.crashed

    @property
    def statements(self) -> int:
        """SQL statements the writer has counted (0 for the in-memory store)."""
        return 0 if self.factory is None else self.factory.connection.calls


@contextlib.contextmanager
def open_store(
    kind: StoreKind,
    *,
    clock: TickingClock | None = None,
    secrets: Sequence[str] = (),
    directory: Path | None = None,
    fail_at: int | Sequence[int] | None = None,
    error: ErrorSource = crash,
) -> Iterator[StoreEnv]:
    """Open a fresh store. Call it before the event loop exists (`Database.open` blocks).

    For `sqlite`, the schema is applied on a plain connection and the store then opens through a
    `FaultFactory`, so its statement counter sees only what the scenario writes. `fail_at` makes
    those statements raise `error` (`crash` for a kill, `disk_io_error` or `disk_full_error` for a
    failed commit). Without `directory`, a `TemporaryDirectory` lives as long as the `with` block.
    `secrets` go to the store's `Redactor` and to the runner's.
    """
    ticking = clock if clock is not None else TickingClock()
    redactor = Redactor(secrets)
    if kind == "memory":
        if fail_at is not None:
            raise ValueError("fault injection needs the SQLite store")
        yield StoreEnv(kind, InMemorySessionStore(ticking), ticking, redactor, ":memory:")
        return

    with contextlib.ExitStack() as stack:
        root = directory
        if root is None:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        root.mkdir(parents=True, exist_ok=True)
        path = root / DB_NAME
        migrated = Database.open(path, create=True)
        try:
            apply(migrated, str(path))
        finally:
            migrated.close()
        # A second, fault-free connection for the harness's own reads, so they are neither counted
        # as statements of the run nor hit by the injected fault. It closes after the writer, which
        # may still hold the write lock of a transaction that a simulated kill abandoned.
        reader_db = Database.open(path, create=False)
        stack.callback(reader_db.close)
        points = (fail_at,) if isinstance(fail_at, int) else fail_at
        factory = FaultFactory(fail_at=points, error=error)
        db = Database.open(path, create=False, connection_factory=factory)
        stack.callback(db.close)
        store = SQLiteSessionStore(db, redactor, ticking)
        reader = SQLiteSessionStore(reader_db, redactor, ticking)
        yield StoreEnv(kind, store, ticking, redactor, str(path), path, factory, reader)


# --- The world: servers, tools, and changes to them ---------------------------------------------

READ_FILE = "files__read_file"
LIST_DIR = "files__list_dir"
WRITE_FILE = "files__write_file"  # protected by the default policy
FETCH = "web__fetch"
PING = "flaky__ping"  # its server is registered but not available


@dataclass(frozen=True)
class ServerSpec:
    name: str
    available: bool
    tools: tuple[McpToolInfo, ...]


@dataclass(frozen=True)
class WorldSpec:
    """The MCP_Servers of a scenario, in Backend_Config order, and the Permission_Policy."""

    servers: tuple[ServerSpec, ...]
    policy: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def _tool(name: str, description: str, **properties: str) -> McpToolInfo:
    """A tool whose `properties` (name to JSON type) are required, plus an optional `tag`.

    The `tag` argument carries the scenario's per-call key to `FakeMcpManager`.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            **{key: {"type": kind} for key, kind in properties.items()},
            "tag": {"type": "string"},
        },
        "required": list(properties),
        "additionalProperties": False,
    }
    return McpToolInfo(name=name, description=description, input_schema=schema)


DEFAULT_WORLD = WorldSpec(
    servers=(
        ServerSpec(
            "files",
            True,
            (
                _tool("read_file", "Read a file.", path="string"),
                _tool("list_dir", "List the working directory."),
                _tool("write_file", "Write a file.", path="string", content="string"),
            ),
        ),
        ServerSpec("web", True, (_tool("fetch", "Fetch a URL.", url="string"),)),
        ServerSpec("flaky", False, (_tool("ping", "Ping the flaky service."),)),
    ),
    policy={"files": ("write_*",)},
)


def build_registry(world: WorldSpec) -> ToolRegistry:
    registry = ToolRegistry([server.name for server in world.servers], world.policy)
    for server in world.servers:
        registry.replace_server(server.name, server.tools)
    return registry


class WorldChangeKind(StrEnum):
    SET_AVAILABLE = "set_available"  # flip the availability flag; the tools stay registered
    REMOVE_SERVER = "remove_server"  # unregister the tools and mark the server unavailable
    RESTORE_SERVER = "restore_server"  # register the server's tools again and mark it available


@dataclass(frozen=True)
class WorldChange:
    kind: WorldChangeKind
    server: str
    available: bool = True  # for SET_AVAILABLE


def _apply_change(change: WorldChange, mcp: FakeMcpManager, world: WorldSpec) -> None:
    match change.kind:
        case WorldChangeKind.SET_AVAILABLE:
            mcp.set_available(change.server, change.available)
        case WorldChangeKind.REMOVE_SERVER:
            mcp.registry.remove_server(change.server)
            mcp.set_available(change.server, False)
        case WorldChangeKind.RESTORE_SERVER:
            for spec in world.servers:
                if spec.name == change.server:
                    mcp.registry.replace_server(spec.name, spec.tools)
                    mcp.set_available(spec.name, True)


# --- Scenario scripts ---------------------------------------------------------------------------


class CallKind(StrEnum):
    """What kind of Tool_Call the LLM emitted (P3)."""

    VALID = "valid"
    PROTECTED = "protected"
    UNKNOWN_TOOL = "unknown_tool"
    BAD_JSON = "bad_json"  # includes valid JSON that is not an object
    SCHEMA_INVALID = "schema_invalid"
    UNAVAILABLE_SERVER = "unavailable_server"


class ApprovalChoice(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    TIMEOUT = "timeout"


def tagged_arguments(args: Mapping[str, Any], tag: str) -> str:
    """The raw JSON text of a call's arguments, with the scenario `tag` added."""
    return json.dumps({**args, "tag": tag})


@dataclass(frozen=True)
class CallScript:
    """One Tool_Call: what the LLM streams (`name`, `arguments`, `call_id`, `pieces`) and, for a
    call that gets sent, what the MCP_Server does (`behavior`). A protected call also has the
    user's `decision`. `call_id=None` makes the stream carry no ID, so the assembler invents one.
    `tag` is unique within the script and appears in the arguments of a call that is sent."""

    tag: str
    kind: CallKind
    name: str
    arguments: str
    behavior: ToolBehavior = DEFAULT_BEHAVIOR
    decision: ApprovalChoice = ApprovalChoice.APPROVE
    call_id: str | None = None
    pieces: int = 1

    @property
    def is_sent(self) -> bool:
        """Whether the call reaches an MCP_Server, assuming the registry does not change."""
        if self.kind is CallKind.VALID:
            return True
        return self.kind is CallKind.PROTECTED and self.decision is ApprovalChoice.APPROVE


@dataclass(frozen=True)
class TurnScript:
    """One chat-completion response.

    The text `chunks` and the Tool_Calls are streamed as deltas (`interleave` mixes them).
    `failure` ends the response early; `fail_after` counts deltas for INTERRUPTED and STALL.
    `world_changes` apply when the request of this turn arrives, which is after its System_Prompt
    was built: the next request sees them, and the Tool_Calls of this turn resolve against the
    names the LLM saw (7.4).
    """

    chunks: tuple[str, ...] = ()
    calls: tuple[CallScript, ...] = ()
    interleave: bool = False
    failure: LlmFailure | None = None
    fail_after: int = 0
    reason: str = "invalid request"
    latency: int = 0
    world_changes: tuple[WorldChange, ...] = ()


@dataclass(frozen=True)
class ScenarioScript:
    """Everything one Task does. A turn after the last scripted one answers "ok" with no calls."""

    user_text: str
    turns: tuple[TurnScript, ...]
    cancel_after_events: int | None = None
    max_tool_iterations: int = 10
    approval_timeout_s: int | None = None
    world: WorldSpec = DEFAULT_WORLD
    base_instructions: str | None = None
    context_window_budget: int = 8192
    drop_calls_without_tools: bool = False

    def __post_init__(self) -> None:
        if self.approval_timeout_s is None and any(
            call.decision is ApprovalChoice.TIMEOUT for call in self.calls
        ):
            raise ValueError("a TIMEOUT decision needs approval_timeout_s")

    @property
    def calls(self) -> tuple[CallScript, ...]:
        return tuple(call for turn in self.turns for call in turn.calls)


def render_deltas(turn: TurnScript) -> tuple[dict[str, Any], ...]:
    """The OpenAI-style deltas of `turn`: text chunks and Tool_Call fragments.

    Without `interleave` the text comes first. With it, the text and each call's fragments are
    dealt out round-robin, each stream keeping its own order (the id and name come first).
    """
    streams: list[list[dict[str, Any]]] = [[text_delta(chunk) for chunk in turn.chunks]]
    for index, call in enumerate(turn.calls):
        streams.append(
            tool_call_deltas(
                index, call.name, call.arguments, call_id=call.call_id, pieces=call.pieces
            )
        )
    if not turn.interleave:
        return tuple(delta for stream in streams for delta in stream)
    merged: list[dict[str, Any]] = []
    position = 0
    while any(position < len(stream) for stream in streams):
        merged.extend(stream[position] for stream in streams if position < len(stream))
        position += 1
    return tuple(merged)


def build_fakes(
    script: ScenarioScript,
    clock: TickingClock,
    sleep: ClockSleep,
    *,
    on_block: Callable[[], None] | None = None,
) -> tuple[FakeLlmGateway, FakeMcpManager]:
    """The LLM and MCP fakes a script describes. `on_block` runs when a call or stream hangs."""
    world = script.world
    mcp = FakeMcpManager(
        build_registry(world),
        {server.name: server.available for server in world.servers},
        behaviors={call.tag: call.behavior for call in script.calls},
        on_block=on_block,
    )

    def changes_for(turn: TurnScript) -> Callable[[], None] | None:
        def apply_changes() -> None:
            for change in turn.world_changes:
                _apply_change(change, mcp, world)

        return apply_changes if turn.world_changes else None

    turns = [
        LlmTurn(
            deltas=render_deltas(turn),
            failure=turn.failure,
            fail_after=turn.fail_after,
            reason=turn.reason,
            latency=turn.latency,
            on_request=changes_for(turn),
        )
        for turn in script.turns
    ]
    llm = FakeLlmGateway(
        turns,
        clock=clock,
        sleep=sleep,
        drop_calls_without_tools=script.drop_calls_without_tools,
        on_block=on_block,
    )
    return llm, mcp


# --- Running a script ---------------------------------------------------------------------------


@dataclass
class ScenarioRun:
    """What one Task did, and what the store held afterwards.

    `task`, `system_prompt`, and `history` are read through `StoreEnv.reader`, so they show the
    committed rows even after a simulated kill (`task` is `None` when the Task row was never
    committed). `admitted` is false when the creation commit failed and no runner started.
    """

    script: ScenarioScript
    context_id: str
    task_id: str
    runner: TaskRunner
    sink: RecordingEventSink
    llm: FakeLlmGateway
    mcp: FakeMcpManager
    timers: ManualTimers
    crashed: bool = False
    admitted: bool = True
    task: TaskRecord | None = None
    system_prompt: SystemMessage | None = None
    history: list[ChatMessage] = field(default_factory=list)

    @property
    def state(self) -> TaskState:
        """The Task's logical state in the emitter (the last committed one)."""
        return self.runner.emitter.state

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def char_estimate(
    model: str, messages: Sequence[ChatMessage], tools: Sequence[Mapping[str, Any]] | None
) -> int:
    """A deterministic token estimate, ceil(characters / 4), without the tokenizer."""
    chars = len(json.dumps([message.to_openai() for message in messages], ensure_ascii=False))
    if tools:
        chars += len(json.dumps(list(tools), ensure_ascii=False))
    return -(-chars // 4)


async def _drive_task(
    env: StoreEnv,
    script: ScenarioScript,
    *,
    context_id: str,
    task_id: str,
    probe_history: bool,
) -> ScenarioRun:
    clock = env.clock
    sleep = ClockSleep(clock)
    timers = ManualTimers(clock)
    finished = asyncio.Event()
    cancel_tasks: list[asyncio.Future[None]] = []
    timer_tasks: list[asyncio.Future[None]] = []
    resolver_tasks: list[asyncio.Future[None]] = []
    runners: list[TaskRunner] = []

    def request_cancel() -> None:
        """Cancel the Task once, from outside the runner, as the executor would."""
        if runners and not cancel_tasks:
            cancel_tasks.append(asyncio.ensure_future(runners[0].cancel()))

    llm, mcp = build_fakes(script, clock, sleep, on_block=request_cancel)
    deps = RunnerDeps(
        sessions=env.store,
        llm=llm,
        registry=mcp.registry,
        mcp=mcp,
        orchestration=OrchestrationConfig(
            max_tool_iterations=script.max_tool_iterations,
            approval_timeout_s=script.approval_timeout_s,
            context_window_budget=script.context_window_budget,
            per_session_discovery=True,
        ),
        base_instructions=script.base_instructions,
        model=MODEL,
        redactor=env.redactor,
        clock=clock,
        db_path=env.db_path,
        sleep=timers,
        estimate_tokens=char_estimate,
    )

    async def committed_messages() -> int | None:
        return len((await env.reader.get_history(context_id))[1])

    sink = RecordingEventSink(probe=committed_messages if probe_history else None)
    runner = TaskRunner(
        task_id=task_id,
        context_id=context_id,
        user_text=script.user_text,
        deps=deps,
        on_finished=lambda _runner: finished.set(),
    )
    runners.append(runner)
    run = ScenarioRun(script, context_id, task_id, runner, sink, llm, mcp, timers)

    decisions = {call.tag: call.decision for call in script.calls}

    # The `input-required` event closes the runner's stream: later events are dropped until the
    # Chat_App attaches again. The harness plays a Chat_App that does, so the sink sees every event.

    async def resume(decision: Decision) -> None:
        """What the executor does for a continuation message: attach the stream, move the Task
        back to `working` (committed and emitted), then hand the decision to the gate (5.8)."""
        runner.attach_sink(sink)
        await runner.emitter.transition(TaskState.WORKING)
        runner.approval_gate.resolve(decision)

    def end_process_if_crashed(_task: asyncio.Future[None]) -> None:
        """A simulated kill ends the whole process. If it hit the resolver while the runner waits
        for the decision, nothing else would ever wake the runner, so stop it here."""
        if env.crashed:
            runner.abandon()

    async def expire() -> None:
        """Let the approval timeout pass, with the Chat_App attached when it does (5.14)."""
        await timers.wait_armed()
        runner.attach_sink(sink)
        timers.fire()

    def on_event(event: SinkEvent) -> None:
        if isinstance(event, StatusEvent) and event.kind == KIND_APPROVAL_REQUIRED:
            arguments = (event.metadata or {}).get("arguments") or {}
            choice = decisions.get(str(arguments.get("tag")), ApprovalChoice.APPROVE)
            if choice is ApprovalChoice.TIMEOUT:
                timer_tasks.append(asyncio.ensure_future(expire()))
            else:
                decision = Decision.APPROVE if choice is ApprovalChoice.APPROVE else Decision.DENY
                resolver = asyncio.ensure_future(resume(decision))
                resolver.add_done_callback(end_process_if_crashed)
                resolver_tasks.append(resolver)
        if event.index == script.cancel_after_events:
            request_cancel()

    sink.add_listener(on_event)
    runner.attach_sink(sink)

    try:
        try:
            await env.store.create_task(task_id, context_id, clock.now_ms(), None)
        except StorageError:
            run.admitted = False  # the creation commit failed: the executor starts no runner
        else:
            runner.start()
            async with asyncio.timeout(HARNESS_TIMEOUT_S):
                await finished.wait()
    except SimulatedCrash:
        run.crashed = True
    except TimeoutError:
        pending = runner.approval_gate.pending
        runner.abandon()
        shown = [
            (e.state.value, e.text) if isinstance(e, StatusEvent) else (e.artifact_id, e.text)
            for e in sink.events[-8:]
        ]
        raise AssertionError(
            f"the scenario did not finish within {HARNESS_TIMEOUT_S} s: state "
            f"{runner.emitter.state.value}, pending approval {pending}, "
            f"pending timers {timers.pending}, tool calls {mcp.calls}, last events {shown}"
        ) from None
    except BaseException:
        runner.abandon()  # a hang or an interrupt: do not leave the runner behind
        raise
    finally:
        for timer in timer_tasks:
            timer.cancel()
        await asyncio.gather(*cancel_tasks, *resolver_tasks, *timer_tasks, return_exceptions=True)

    run.crashed = run.crashed or env.crashed
    run.task = await env.reader.get_task(task_id)
    run.system_prompt, run.history = await env.reader.get_history(context_id)
    return run


async def _drive_plans(
    env: StoreEnv, plans: Sequence[ContextPlan], *, probe_history: bool
) -> list[ScenarioRun]:
    async def run_plan(plan: ContextPlan) -> list[ScenarioRun]:
        runs: list[ScenarioRun] = []
        for number, script in enumerate(plan.scripts, start=1):
            run = await _drive_task(
                env,
                script,
                context_id=plan.context_id,
                task_id=f"{plan.context_id}/task-{number}",
                probe_history=probe_history,
            )
            runs.append(run)
            if run.crashed:
                break
        return runs

    grouped = await asyncio.gather(*(run_plan(plan) for plan in plans))
    return [run for runs in grouped for run in runs]


@dataclass(frozen=True)
class ContextPlan:
    """The scripts of one Context: its Tasks run one after another, in this order."""

    context_id: str
    scripts: tuple[ScenarioScript, ...]


def run_scenario(
    script: ScenarioScript,
    env: StoreEnv,
    *,
    context_id: str = CONTEXT,
    task_id: str = TASK,
    probe_history: bool = False,
) -> ScenarioRun:
    """Run `script` as one Task on a fresh event loop and fake clock, and read the store back.

    `probe_history` records the stored message count at each event (`StatusEvent.history_len`),
    which is how P3 checks that `tool_call_completed` comes before the Tool_Result is appended.
    """
    return asyncio.run(
        _drive_task(
            env, script, context_id=context_id, task_id=task_id, probe_history=probe_history
        )
    )


def run_plans(
    plans: Sequence[ContextPlan], env: StoreEnv, *, probe_history: bool = False
) -> list[ScenarioRun]:
    """Run Contexts concurrently on one fresh event loop (P13, P51), their Tasks in sequence.

    The runs are returned grouped by plan, in script order. Task IDs are `<context_id>/task-<n>`.
    """
    return asyncio.run(_drive_plans(env, plans, probe_history=probe_history))


# --- Kill injection -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KillResult:
    """A run on SQLite that may have been killed. The connection is closed; reopen `db_path`."""

    db_path: Path
    statements: int  # SQL statements the writer counted (including the one that crashed)
    crashed: bool
    runs: tuple[ScenarioRun, ...]


def kill_run(
    plans: ScenarioScript | Sequence[ContextPlan],
    directory: Path,
    crash_at: int | None,
    *,
    secrets: Sequence[str] = (),
    probe_history: bool = False,
) -> KillResult:
    """Run `plans` on SQLite in `directory`; the `crash_at`-th statement raises `SimulatedCrash`.

    `crash_at=None` is a clean run whose `statements` count gives the range of k (1-based). The
    crash abandons the connection without commit or rollback, as a kill does; the runner dies
    with it and nothing finalizes the Task. Returns the database path for the reopen.
    """
    if isinstance(plans, ScenarioScript):
        plans = [ContextPlan(CONTEXT, (plans,))]
    with open_store(
        "sqlite", directory=directory, secrets=secrets, fail_at=crash_at, error=crash
    ) as env:
        runs = run_plans(plans, env, probe_history=probe_history)
        assert env.path is not None
        return KillResult(env.path, env.statements, env.crashed, tuple(runs))


# --- Transcripts --------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def build_transcript(run: ScenarioRun) -> dict[str, Any]:
    """The Status_Event and Token_Event sequence, final history, and Task of a run as JSON data."""
    events: list[dict[str, Any]] = []
    for event in run.sink.events:
        if isinstance(event, StatusEvent):
            events.append(
                {
                    "type": "status",
                    "state": event.state.value,
                    "text": event.text,
                    "metadata": _jsonable(event.metadata),
                    "final": event.final,
                }
            )
        elif isinstance(event, ArtifactEvent):
            events.append(
                {
                    "type": "artifact",
                    "artifact_id": event.artifact_id,
                    "text": event.text,
                    "append": event.append,
                    "last_chunk": event.last_chunk,
                }
            )
    task = run.task
    return {
        "events": events,
        "system_prompt": None if run.system_prompt is None else run.system_prompt.content,
        "history": [message.to_openai() for message in run.history],
        "task": None
        if task is None
        else {
            "state": run.state.value,
            "status_message": task.status_message,
            "metadata": _jsonable(task.metadata),
            "created_at_ms": task.created_at_ms,
            "updated_at_ms": task.updated_at_ms,
            "artifacts": [
                {"artifact_id": a.artifact_id, "text": a.text, "closed": a.closed}
                for a in task.artifacts
            ],
        },
    }


_GENERATED_CALL_ID = re.compile(r"call_[0-9a-f]{32}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_DURATION_KEYS = frozenset({"duration_ms", "elapsed_ms", "duration_s", "elapsed_s"})
_TIMESTAMP_KEYS = frozenset({"timestamp", "created_at", "updated_at", "time"})


class _Placeholders:
    """Replaces generated values with `<kind-N>`, numbering by first appearance, so two runs with
    different generated IDs and times normalize to the same text."""

    def __init__(self) -> None:
        self._seen: dict[str, dict[str, str]] = {}

    def _name(self, kind: str, value: str) -> str:
        known = self._seen.setdefault(kind, {})
        if value not in known:
            known[value] = f"<{kind}-{len(known) + 1}>"
        return known[value]

    def text(self, value: str) -> str:
        value = _UUID.sub(lambda m: self._name("uuid", m.group(0)), value)
        value = _GENERATED_CALL_ID.sub(lambda m: self._name("call-id", m.group(0)), value)
        return _ISO_TIME.sub("<timestamp>", value)

    def walk(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {k: self.walk(v, str(k)) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.walk(item) for item in value]
        if key is not None and isinstance(value, int | float) and not isinstance(value, bool):
            if key in _DURATION_KEYS:
                return "<duration>"
            if key in _TIMESTAMP_KEYS or key.endswith("_at_ms"):
                return "<timestamp>"
        if isinstance(value, str):
            return self.text(value)
        return value


def normalize_transcript(transcript: Any) -> Any:
    """Replace generated IDs (UUIDs, `call_<hex>` Tool_Call IDs), timestamps (`*_at_ms`, ISO
    times), and durations (`duration_ms`) by placeholders, numbered by first appearance."""
    return _Placeholders().walk(transcript)


def transcript_json(transcript: Any) -> str:
    """The normalized transcript as canonical JSON text, for comparing or hashing (P46)."""
    return json.dumps(
        normalize_transcript(transcript), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


# --- Strategies ---------------------------------------------------------------------------------

_chunk_text = st.text(alphabet=text_chars, max_size=6)
_path_values = st.text(alphabet=text_chars, max_size=10)


def _result_texts(secrets: Sequence[str]) -> st.SearchStrategy[str]:
    """Tool result, exception, and rejection texts; with `secrets`, often containing one."""
    plain = st.text(alphabet=text_chars, max_size=10)
    if not secrets:
        return plain
    return st.one_of(
        plain,
        st.builds(
            lambda before, secret, after: before + secret + after,
            plain,
            st.sampled_from(list(secrets)),
            plain,
        ),
    )


def _valid_arguments(name: str) -> st.SearchStrategy[dict[str, Any]]:
    match name:
        case "files__read_file":
            return st.builds(lambda path: {"path": path}, _path_values)
        case "files__write_file":
            return st.builds(
                lambda path, content: {"path": path, "content": content},
                _path_values,
                stored_texts(),
            )
        case "web__fetch":
            return st.builds(lambda url: {"url": url}, _path_values)
        case _:
            return st.just({})


def _invalid_arguments(name: str) -> st.SearchStrategy[dict[str, Any]]:
    """JSON objects that break the schema of `name`: wrong type, missing field, or extra field."""
    extra = st.just({"bogus": 1})
    match name:
        case "files__read_file":
            return st.sampled_from([{}, {"path": 5}, {"path": "a", "bogus": 1}])
        case "web__fetch":
            return st.sampled_from([{}, {"url": None}, {"url": "u", "bogus": True}])
        case _:
            return extra


_BAD_JSON_TEXTS = ('{"path": ', "not json", "", "[1, 2]", "42", "null", '"text"')
_UNKNOWN_NAMES = ("no_such_tool", "files__read_file_2", "", "Unknown tool \u00e9")


def _behaviors(secrets: Sequence[str], *, hang: bool) -> st.SearchStrategy[ToolBehavior]:
    outcomes = [ToolOutcome.SUCCESS] * 4 + [
        ToolOutcome.ERROR,
        ToolOutcome.TIMEOUT,
        ToolOutcome.EXCEPTION,
        ToolOutcome.UNAVAILABLE,
    ]
    if hang:
        outcomes.append(ToolOutcome.HANG)
    return st.builds(
        ToolBehavior,
        outcome=st.sampled_from(outcomes),
        text=_result_texts(secrets),
        latency=st.integers(min_value=0, max_value=3),
    )


@st.composite
def _call_scripts(
    draw: st.DrawFn, turn: int, index: int, *, secrets: Sequence[str], approvals: bool, hang: bool
) -> CallScript:
    kinds = [CallKind.VALID] * 4 + [
        CallKind.UNKNOWN_TOOL,
        CallKind.BAD_JSON,
        CallKind.SCHEMA_INVALID,
        CallKind.UNAVAILABLE_SERVER,
    ]
    if approvals:
        kinds += [CallKind.PROTECTED] * 3
    kind = draw(st.sampled_from(kinds))
    tag = f"t{turn}c{index}"
    call_id = None if draw(st.integers(min_value=0, max_value=4)) == 0 else f"call-{turn}-{index}"
    pieces = draw(st.integers(min_value=1, max_value=3))
    common: dict[str, Any] = {"tag": tag, "kind": kind, "call_id": call_id, "pieces": pieces}

    if kind in (CallKind.VALID, CallKind.PROTECTED):
        names = [READ_FILE, LIST_DIR, FETCH] if kind is CallKind.VALID else [WRITE_FILE]
        name = draw(st.sampled_from(names))
        args = draw(_valid_arguments(name))
        decision = (
            draw(st.sampled_from(list(ApprovalChoice) + [ApprovalChoice.APPROVE] * 2))
            if kind is CallKind.PROTECTED
            else ApprovalChoice.APPROVE
        )
        behavior = draw(_behaviors(secrets, hang=hang))
        return CallScript(
            name=name,
            arguments=tagged_arguments(args, tag),
            behavior=behavior,
            decision=decision,
            **common,
        )
    if kind is CallKind.UNKNOWN_TOOL:
        return CallScript(
            name=draw(st.sampled_from(_UNKNOWN_NAMES)),
            arguments=tagged_arguments({}, tag),
            **common,
        )
    if kind is CallKind.BAD_JSON:
        return CallScript(
            name=READ_FILE, arguments=draw(st.sampled_from(_BAD_JSON_TEXTS)), **common
        )
    if kind is CallKind.SCHEMA_INVALID:
        name = draw(st.sampled_from([READ_FILE, LIST_DIR, FETCH]))
        return CallScript(name=name, arguments=json.dumps(draw(_invalid_arguments(name))), **common)
    return CallScript(name=PING, arguments=tagged_arguments({}, tag), **common)


_SERVER_NAMES = ("files", "web", "flaky")


@st.composite
def _world_changes(draw: st.DrawFn) -> WorldChange:
    return WorldChange(
        kind=draw(st.sampled_from(list(WorldChangeKind))),
        server=draw(st.sampled_from(_SERVER_NAMES)),
        available=draw(st.booleans()),
    )


@st.composite
def _turn_scripts(
    draw: st.DrawFn,
    index: int,
    *,
    secrets: Sequence[str],
    approvals: bool,
    failures: bool,
    cancels: bool,
    world_changes: bool,
) -> TurnScript:
    chunks = tuple(draw(st.lists(_chunk_text, max_size=4)))
    count = draw(st.sampled_from([0, 0, 1, 1, 2, 3]))
    calls = tuple(
        draw(_call_scripts(index, i, secrets=secrets, approvals=approvals, hang=cancels))
        for i in range(count)
    )
    turn = TurnScript(
        chunks=chunks,
        calls=calls,
        interleave=draw(st.booleans()),
        latency=draw(st.integers(min_value=0, max_value=2)),
        world_changes=tuple(draw(st.lists(_world_changes(), max_size=2))) if world_changes else (),
    )
    if not failures or draw(st.integers(min_value=0, max_value=7)) != 0:
        return turn
    deltas = len(render_deltas(turn))
    options = [LlmFailure.UNREACHABLE, LlmFailure.REJECTED]
    if deltas:
        options.append(LlmFailure.INTERRUPTED)
        if cancels:
            options.append(LlmFailure.STALL)
    failure = draw(st.sampled_from(options))
    acts_midstream = failure in (LlmFailure.INTERRUPTED, LlmFailure.STALL)
    return TurnScript(
        chunks=turn.chunks,
        calls=turn.calls,
        interleave=turn.interleave,
        latency=turn.latency,
        world_changes=turn.world_changes,
        failure=failure,
        fail_after=draw(st.integers(min_value=1, max_value=deltas)) if acts_midstream else 0,
        reason=draw(_result_texts(secrets)),
    )


@st.composite
def scenario_scripts(
    draw: st.DrawFn,
    *,
    secrets: Sequence[str] = (),
    approvals: bool = True,
    failures: bool = True,
    cancels: bool = True,
    world_changes: bool = True,
    max_turns: int = 4,
) -> ScenarioScript:
    """Draw a scenario script.

    The flags narrow the draw: `approvals` allows protected calls, `failures` allows LLM failures,
    `cancels` allows an injected cancel (and the hanging tool and stalled stream that cause one),
    `world_changes` allows registry changes between turns. `secrets` are embedded in tool results,
    exceptions, and rejection reasons, for the redaction properties (P61).
    """
    turns = tuple(
        draw(
            _turn_scripts(
                index,
                secrets=secrets,
                approvals=approvals,
                failures=failures,
                cancels=cancels,
                world_changes=world_changes,
            )
        )
        for index in range(draw(st.integers(min_value=1, max_value=max_turns)))
    )
    has_timeout = any(
        call.decision is ApprovalChoice.TIMEOUT for turn in turns for call in turn.calls
    )
    return ScenarioScript(
        user_text=draw(stored_texts(max_size=10)),
        turns=turns,
        cancel_after_events=(
            draw(st.integers(min_value=1, max_value=30))
            if cancels and draw(st.integers(min_value=0, max_value=3)) == 0
            else None
        ),
        max_tool_iterations=draw(st.sampled_from([1, 2, 3, 10])),
        approval_timeout_s=1 if has_timeout else draw(st.sampled_from([None, None, 30])),
        drop_calls_without_tools=draw(st.booleans()),
    )
