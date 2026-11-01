"""Property 51: Storage write failures (task 17.18).

Two or three Contexts run concurrently on SQLite, one Task each. One or two consecutive commits of
one Task fail with an `OperationalError` (disk full or I/O error), and the run is checked against
the property: the database holds exactly the effects of the commits that succeeded, no event
depended on a failed commit, a failed creation leaves no Task or Context, otherwise the Task ends
`failed` with "could not be saved" (18.30) or, after two failed commits, emits a final `failed`
event and keeps its last committed state (18.31), the other Contexts match a fault-free run, and
every error log record names the database path and the `task_id` without a secret.

`open_store(fail_at=...)` counts statements of all Contexts together, so under interleaving it
cannot pick a commit *of one Task*, and a second index cannot follow the retry after the first
failure. `TrackingConnection` therefore subclasses the harness's `FaultInjectingConnection` and
counts the COMMITs of transactions that wrote to the target Context. It fails the planned ones and
never the ROLLBACK that follows. It also keeps a ledger of every write in a transaction that
committed, so the file can be compared with "exactly the effects of the commits that succeeded".

A fault-free run of the same plans first lists the target's commits. A failing commit is drawn
from that list, except `input-required` to `working`: the scenario harness plays the executor for
that commit, and the executor answers its failure with an A2A error while the Task keeps waiting.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hypothesis import event, example, given, settings
from hypothesis import strategies as st
from scenarios import (
    CallKind,
    CallScript,
    ContextPlan,
    ScenarioRun,
    ScenarioScript,
    StoreEnv,
    TurnScript,
    build_transcript,
    normalize_transcript,
    open_store,
    run_plans,
    scenario_scripts,
    tagged_arguments,
)
from structlog.testing import capture_logs

from fakes import LlmFailure, StatusEvent
from local_agent.a2a_server.events import TaskState
from local_agent.orchestrator.loop import STORAGE_FAILED_MESSAGE
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.storage.database import Database
from storage_faults import FaultInjectingConnection, disk_full_error, disk_io_error

CONTEXT_IDS = ("ctx-a", "ctx-b", "ctx-c")
TERMINAL_NAMES = frozenset({"completed", "failed", "canceled"})
STORAGE_EVENTS = frozenset(
    {
        "storage_write_failed",
        "final_status_unpersisted",
        "task_finalize_failed",
        "missing_results_history_unreadable",
    }
)

# Characters that occur in no fixed log text, status message, or temporary path.
SECRET_CHARS = "BFGHJKLMNPQSUVWXYZ!@#$%^&*~"

type Params = tuple[Any, ...]
type Op = tuple[str, Params]
type Error = Callable[[], BaseException]


# --- The ledger of committed writes ---------------------------------------------------------------

# SQL prefix of each write the store makes, the kind it is filed under, and the index of the
# parameter that names the Context or Task it belongs to.
_WRITES: tuple[tuple[str, str, int], ...] = (
    ("INSERT INTO contexts", "context", 0),
    ("INSERT INTO tasks", "task", 1),
    ("UPDATE tasks SET state", "state", 4),
    ("INSERT INTO messages", "message", 0),
    ("UPDATE contexts SET system_prompt", "prompt", 1),
    ("INSERT INTO artifacts", "artifact", 0),
    ("UPDATE contexts SET last_activity_ms", "activity", 1),
)


@dataclass
class Attempt:
    """One COMMIT of a transaction that wrote to the target Context."""

    ok: bool
    eligible: bool  # false for `input-required` to `working`, which the harness's executor makes


@dataclass
class Ledger:
    """What the committed transactions wrote, as the SQL parameters say."""

    contexts: dict[str, str | None] = field(default_factory=dict)  # context_id: system_prompt
    tasks: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # state, message
    statuses: list[tuple[str, str, str | None]] = field(default_factory=list)  # task, state, text
    messages: dict[tuple[str, int], Params] = field(default_factory=dict)
    artifacts: dict[tuple[str, str], Params] = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    unmodeled: list[str] = field(default_factory=list)

    def apply(self, ops: Sequence[Op]) -> None:
        for kind, p in ops:
            match kind:
                case "context":
                    self.contexts[p[0]] = None
                case "task":
                    self.tasks[p[0]] = (p[2], None)
                case "state":
                    self.tasks[p[4]] = (p[0], p[1])
                    self.statuses.append((p[4], p[0], p[1]))
                case "message":
                    self.messages[(p[0], p[1])] = p[2:7]
                case "prompt":
                    self.contexts[p[1]] = p[0]
                case "artifact":
                    self.artifacts[(p[0], p[1])] = p[2:5]


@dataclass(frozen=True)
class FaultPlan:
    """Fail these COMMITs (1-based among the target's) with `error`."""

    attempts: frozenset[int]
    error: Error


class TrackingConnection(FaultInjectingConnection):
    """Records the writes of each transaction and fails planned COMMITs of the target Context."""

    def __init__(
        self, conn: sqlite3.Connection, *, ledger: Ledger, target: str, plan: FaultPlan | None
    ) -> None:
        super().__init__(conn)
        self._ledger = ledger
        self._target = target
        self._plan = plan
        self._ops: list[Op] | None = None
        self._on_target = False
        self._resume_like = False

    def execute(
        self, sql: str, parameters: Sequence[object] | Mapping[str, object] = (), /
    ) -> sqlite3.Cursor:
        normalized = " ".join(sql.split())
        verb = normalized.split(" ", 1)[0].upper()
        if verb == "BEGIN":
            cursor = super().execute(sql, parameters)
            self._ops, self._on_target, self._resume_like = [], False, False
            return cursor
        if self._ops is None:
            return super().execute(sql, parameters)
        if verb == "COMMIT":
            self._before_commit()
            cursor = super().execute(sql, parameters)
            self._ledger.apply(self._ops)
            self._ops = None
            return cursor
        if verb == "ROLLBACK":
            self._ops = None
            return super().execute(sql, parameters)
        cursor = super().execute(sql, parameters)
        self._record(normalized, parameters)
        return cursor

    def _before_commit(self) -> None:
        if not self._on_target:
            return
        attempt = Attempt(ok=True, eligible=not self._resume_like)
        self._ledger.attempts.append(attempt)
        if self._plan is not None and len(self._ledger.attempts) in self._plan.attempts:
            attempt.ok = False
            raise self._plan.error()

    def _record(self, sql: str, parameters: Sequence[object] | Mapping[str, object]) -> None:
        assert self._ops is not None
        write = next((entry for entry in _WRITES if sql.startswith(entry[0])), None)
        if write is None:
            if sql.startswith(("INSERT", "UPDATE", "DELETE")):
                self._ledger.unmodeled.append(sql)
            return
        _, kind, owner_index = write
        assert not isinstance(parameters, Mapping)
        params = tuple(parameters)
        self._ops.append((kind, params))
        if self._belongs(params[owner_index]):
            self._on_target = True
            if kind == "state" and params[0] == "working" and params[5:] == ("input-required",):
                self._resume_like = True

    def _belongs(self, owner: object) -> bool:
        # Task IDs are `<context_id>/task-<n>`.
        return isinstance(owner, str) and (
            owner == self._target or owner.startswith(self._target + "/")
        )


@contextlib.contextmanager
def tracked_store(
    *, secrets: Sequence[str], target: str, plan: FaultPlan | None
) -> Iterator[tuple[StoreEnv, Ledger]]:
    """The harness's SQLite `StoreEnv`, with its writer replaced by a tracking one.

    The harness opens its own writer first (idle here); `env.store` then writes through the
    tracking connection, and `env.reader` still reads through a separate, fault-free one.
    """
    ledger = Ledger()

    def wrap(conn: sqlite3.Connection) -> sqlite3.Connection:
        tracking = TrackingConnection(conn, ledger=ledger, target=target, plan=plan)
        return tracking.as_connection()

    with open_store("sqlite", secrets=secrets) as env:
        assert env.path is not None
        db = Database.open(env.path, create=False, connection_factory=wrap)
        try:
            env.store = SQLiteSessionStore(db, env.redactor, env.clock)
            yield env, ledger
        finally:
            db.close()


# --- Reading the file back ------------------------------------------------------------------------


@dataclass(frozen=True)
class Rows:
    contexts: dict[str, str | None]
    tasks: dict[str, tuple[str, str | None]]
    messages: dict[tuple[str, int], Params]
    artifacts: dict[tuple[str, str], Params]


def read_rows(path: str) -> Rows:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return Rows(
            contexts=dict(conn.execute("SELECT context_id, system_prompt FROM contexts")),
            tasks={
                task: (state, message)
                for task, state, message in conn.execute(
                    "SELECT task_id, state, status_message FROM tasks"
                )
            },
            messages={
                (ctx, seq): (task, role, content, calls, call_id)
                for ctx, seq, task, role, content, calls, call_id in conn.execute(
                    "SELECT context_id, seq, task_id, role, content, tool_calls, tool_call_id"
                    " FROM messages"
                )
            },
            artifacts={
                (task, artifact): (ordinal, text, closed)
                for task, artifact, ordinal, text, closed in conn.execute(
                    "SELECT task_id, artifact_id, ordinal, text, closed FROM artifacts"
                )
            },
        )


# --- The cases ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    plans: tuple[ContextPlan, ...]
    secrets: tuple[str, ...]
    target: int  # index of the plan whose Task suffers the failure
    pick: int  # chooses the failing commit among the target's eligible ones
    double: bool  # the commit after it fails as well
    disk_full: bool  # "database or disk is full" instead of "disk I/O error"


@st.composite
def cases(draw: st.DrawFn) -> Case:
    secrets = tuple(
        draw(
            st.lists(
                st.text(SECRET_CHARS, min_size=4, max_size=10), min_size=1, max_size=2, unique=True
            )
        )
    )
    count = draw(st.integers(min_value=2, max_value=3))
    plans = tuple(
        ContextPlan(context_id, (draw(scenario_scripts(secrets=secrets, max_turns=3)),))
        for context_id in CONTEXT_IDS[:count]
    )
    return Case(
        plans=plans,
        secrets=secrets,
        target=draw(st.integers(min_value=0, max_value=count - 1)),
        pick=draw(st.integers(min_value=0, max_value=10_000)),
        double=draw(st.booleans()),
        disk_full=draw(st.booleans()),
    )


def cancel_race_case(*, double: bool) -> Case:
    """A cancel arrives while a commit of the target Task is in flight, and that commit fails.

    Found by the property: `TaskEventEmitter._commit` drops a failed commit when the awaiting
    coroutine was cancelled meanwhile. No `storage_write_failed` record is written and the Task
    ends `canceled` instead of `failed` (18.30).
    """
    target = ScenarioScript(
        user_text="",
        turns=(
            TurnScript(
                calls=(
                    CallScript(
                        tag="t0c0",
                        kind=CallKind.VALID,
                        name="files__read_file",
                        arguments=tagged_arguments({"path": ""}, "t0c0"),
                    ),
                )
            ),
        ),
        cancel_after_events=2,
        max_tool_iterations=1,
    )
    other = ScenarioScript(
        user_text="", turns=(TurnScript(failure=LlmFailure.UNREACHABLE, reason=""),)
    )
    return Case(
        plans=(ContextPlan("ctx-a", (target,)), ContextPlan("ctx-b", (other,))),
        secrets=("@@@@",),
        target=0,
        pick=6,
        double=double,
        disk_full=False,
    )


def injected_error(disk_full: bool, secret: str) -> Error:
    """An `OperationalError` whose message also carries a secret, so redaction has work to do."""
    base = disk_full_error if disk_full else disk_io_error

    def build() -> BaseException:
        return sqlite3.OperationalError(f"{base()} {secret}")

    return build


# --- Checks ---------------------------------------------------------------------------------------


def leaks(text: str, secrets: Sequence[str]) -> bool:
    """Does `text` contain a 4-character piece of a secret? (The Redactor removes all of them.)"""
    return any(secret[i : i + 4] in text for secret in secrets for i in range(len(secret) - 3))


def is_subsequence(emitted: Sequence[object], committed: Sequence[object]) -> bool:
    remaining = iter(committed)
    return all(any(item == candidate for candidate in remaining) for item in emitted)


def check_events(run: ScenarioRun, ledger: Ledger, *, unpersisted_final: bool) -> None:
    """No Status_Event or Token_Event that depended on a failed commit was emitted."""
    emitted = [(event.state.value, event.text) for event in run.sink.statuses]
    if unpersisted_final:
        # The one event that is allowed to precede its commit (18.31) is the last one.
        assert emitted[-1] == ("failed", STORAGE_FAILED_MESSAGE)
        emitted.pop()
    committed = [(state, text) for task, state, text in ledger.statuses if task == run.task_id]
    assert is_subsequence(emitted, committed), (emitted, committed)
    for token in run.sink.artifact_events:
        if token.last_chunk:  # the marker follows the commit that closes the artifact
            stored = ledger.artifacts.get((run.task_id, token.artifact_id))
            assert stored is not None
            assert stored[2] == 1, f"{token.artifact_id} was announced closed but is not"


def check_failed_task(
    run: ScenarioRun, rows: Rows, ledger: Ledger, attempts: list[Attempt], logs: list[Any]
) -> None:
    """The Task's end state after a failed commit that was not its creation (18.30, 18.31)."""
    trailing = 0
    for attempt in reversed(attempts):
        if attempt.ok:
            break
        trailing += 1
    unpersisted = trailing > 0
    state, message = rows.tasks[run.task_id]
    assert run.state is TaskState.FAILED

    events = run.sink.events
    stream_closed = (
        bool(events)
        and isinstance(events[-1], StatusEvent)
        and events[-1].final
        and events[-1].state is TaskState.INPUT_REQUIRED
    )
    if unpersisted:
        # 18.31: the row keeps its last committed state; a final `failed` event goes to the stream.
        assert trailing >= 2, f"commit failed and the Task made no further commit: {attempts}"
        assert state not in TERMINAL_NAMES, (state, message)
        assert (run.task_id, "failed", STORAGE_FAILED_MESSAGE) not in ledger.statuses
        assert any(r["event"] == "final_status_unpersisted" for r in logs)
    else:
        # 18.30: committed as `failed` with the "could not be saved" status message.
        assert (state, message) == ("failed", STORAGE_FAILED_MESSAGE), (state, message)
    if not stream_closed:
        last = events[-1]
        assert isinstance(last, StatusEvent), last
        assert (last.state, last.text, last.final) == (
            TaskState.FAILED,
            STORAGE_FAILED_MESSAGE,
            True,
        )
    check_events(run, ledger, unpersisted_final=unpersisted and not stream_closed)


def check_logs(
    logs: list[Any], run: ScenarioRun, env: StoreEnv, secrets: Sequence[str], failures: int
) -> None:
    """Every error record names the database path and the `task_id`; none holds a secret."""
    # Failures of the scripted LLM are logged at error level too, in any Context; they have
    # nothing to do with storage.
    errors = [
        record
        for record in logs
        if record["log_level"] == "error" and record.get("component") != "llm"
    ]
    for record in errors:
        assert record.get("db_path") == env.db_path, record
        assert record.get("task_id") == run.task_id, record
    assert [r["event"] for r in errors] == ["storage_write_failed"] * failures
    # Other components redact later, in the logging pipeline (P28), which `capture_logs` bypasses.
    # The records written for a failed commit carry their text already redacted.
    for record in logs:
        if record["event"] in STORAGE_EVENTS:
            for key, value in record.items():
                if key != "db_path":
                    assert not leaks(str(value), secrets), (key, value)


# Feature: local-agent-backend, Property 51: Storage write failures
@settings(max_examples=100, deadline=None)
@example(case=cancel_race_case(double=False))
@example(case=cancel_race_case(double=True))
@given(case=cases())
def test_storage_write_failures(case: Case) -> None:
    """**Validates: Requirements 18.30, 18.31, 14.7, 16.11**"""
    target = case.plans[case.target].context_id

    # A fault-free run lists the commits of the target Task.
    with tracked_store(secrets=case.secrets, target=target, plan=None) as (env, ledger):
        baseline = {run.context_id: run for run in run_plans(case.plans, env)}
    eligible = [i for i, attempt in enumerate(ledger.attempts, start=1) if attempt.eligible]
    first = eligible[case.pick % len(eligible)]
    plan = FaultPlan(
        frozenset({first, first + 1} if case.double else {first}),
        injected_error(case.disk_full, case.secrets[0]),
    )

    with (
        capture_logs() as logs,
        tracked_store(secrets=case.secrets, target=target, plan=plan) as (env, ledger),
    ):
        runs = {run.context_id: run for run in run_plans(case.plans, env)}
        assert env.path is not None
        rows = read_rows(str(env.path))
    run = runs[target]
    attempts = ledger.attempts
    failed = [number for number, attempt in enumerate(attempts, start=1) if not attempt.ok]

    assert failed, "the planned failure did not happen"
    assert not ledger.unmodeled, ledger.unmodeled

    # The database contains exactly the effects of the commits that succeeded.
    assert rows == Rows(ledger.contexts, ledger.tasks, ledger.messages, ledger.artifacts)

    # The Task_States and Message_Histories of the other Contexts equal those of a fault-free run.
    for context_id, other in runs.items():
        if context_id != target:
            assert normalize_transcript(build_transcript(other)) == normalize_transcript(
                build_transcript(baseline[context_id])
            ), context_id
            check_events(other, ledger, unpersisted_final=False)

    event(f"failed commits: {len(failed)}, ending in failure: {not attempts[-1].ok}")
    event(f"creation failed: {failed[0] == 1}")
    if failed[0] == 1:
        # The creation failed: the request got the storage error and no row exists.
        assert len(attempts) == 1
        assert not run.admitted
        assert run.task is None
        assert target not in rows.contexts
        assert not [task for task in rows.tasks if task.startswith(target + "/")]
        assert run.history == []
        assert run.system_prompt is None
        assert run.sink.events == []
        check_logs(logs, run, env, case.secrets, failures=0)
    else:
        assert run.admitted
        check_failed_task(run, rows, ledger, attempts, logs)
        assert not leaks(rows.tasks[run.task_id][1] or "", case.secrets)
        check_logs(logs, run, env, case.secrets, failures=len(failed))
