"""Property test for persisted artifact prefixes (task 17.3).

The emitter runs over a real `SQLiteSessionStore` whose connection is a `FaultInjectingConnection`.
Each example builds a script of assistant responses (chunk sequences), picks how the Task ends
(complete, LLM failure after chunk j, cancel after chunk j), and optionally a crash point k: the
k-th SQL statement after the Task row exists raises `SimulatedCrash`, which ends the run like a
kill. The database is then closed, reopened, and the persisted artifact texts are compared with the
text the `RecordingEventSink` saw.
"""

import asyncio
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from fakes import RecordingEventSink, TickingClock
from local_agent.a2a_server.events import TaskEventEmitter, TaskState, to_sdk
from local_agent.logging_setup import Redactor
from local_agent.models import AssistantMessage, TaskRecord
from local_agent.session.sqlite_store import SQLiteSessionStore
from local_agent.storage.database import Database
from local_agent.storage.migrations import apply
from storage_faults import FaultFactory, SimulatedCrash, crash

TASK = "task-1"
CONTEXT = "ctx-1"


class Interruption(StrEnum):
    NONE = "none"
    LLM_FAILURE = "llm_failure"
    CANCEL = "cancel"


@dataclass(frozen=True)
class Scenario:
    """The assistant responses of one Task and how the Task ends."""

    responses: tuple[tuple[str, ...], ...]  # non-empty chunks of each response, in order
    interruption: Interruption
    stop_after: int  # chunks emitted before the interruption, 0..total chunks

    def stops_after(self, emitted: int) -> bool:
        return self.interruption is not Interruption.NONE and emitted == self.stop_after


@dataclass(frozen=True)
class Driven:
    base: int  # statements counted before the emitter started
    total: int  # statements counted since
    crashed: bool
    completed_texts: tuple[str, ...]  # text of every response that was closed and non-empty


@dataclass(frozen=True)
class Observed:
    driven: Driven
    emitted: dict[str, str]
    task: TaskRecord | None
    assistant_texts: tuple[str | None, ...]


async def finish(emitter: TaskEventEmitter, interruption: Interruption) -> None:
    match interruption:
        case Interruption.LLM_FAILURE:
            await emitter.transition(TaskState.FAILED, "The response was interrupted.")
        case Interruption.CANCEL:
            await emitter.transition(TaskState.CANCELED, "Canceled.")
        case Interruption.NONE:
            await emitter.transition(TaskState.COMPLETED)


async def run_script(
    emitter: TaskEventEmitter,
    store: SQLiteSessionStore,
    scenario: Scenario,
    completed: list[str],
) -> None:
    """Do what the reasoning loop does for the streamed responses of one Task."""
    await emitter.transition(TaskState.WORKING, "Discovering local tools...")
    emitted = 0
    if scenario.stops_after(emitted):
        await finish(emitter, scenario.interruption)
        return
    for index, chunks in enumerate(scenario.responses):
        artifact_id = f"response-{index}"
        for position, chunk in enumerate(chunks):
            await emitter.token(artifact_id, chunk, first=position == 0)
            emitted += 1
            if scenario.stops_after(emitted):
                await finish(emitter, scenario.interruption)
                return
        await emitter.close_artifact(artifact_id)
        text = "".join(chunks)
        if text:
            await store.append(CONTEXT, [AssistantMessage(content=text)], task_id=TASK)
            completed.append(text)
    await finish(emitter, Interruption.NONE)


async def drive(
    store: SQLiteSessionStore,
    factory: FaultFactory,
    sink: RecordingEventSink,
    scenario: Scenario,
    db_path: str,
) -> Driven:
    clock = TickingClock()
    await store.create_task(TASK, CONTEXT, clock.now_ms(), None)
    base = factory.connection.calls
    emitter = TaskEventEmitter(
        store, sink, Redactor([]), clock, db_path, task_id=TASK, context_id=CONTEXT
    )
    completed: list[str] = []
    crashed = False
    try:
        await run_script(emitter, store, scenario, completed)
    except SimulatedCrash:
        crashed = True
    return Driven(base, factory.connection.calls - base, crashed, tuple(completed))


async def read_back(store: SQLiteSessionStore) -> tuple[TaskRecord | None, tuple[str | None, ...]]:
    task = await store.get_task(TASK)
    _, history = await store.get_history(CONTEXT)
    texts = tuple(m.content for m in history if isinstance(m, AssistantMessage))
    return task, texts


def execute(scenario: Scenario, *, crash_at: int | None = None, base: int = 0) -> Observed:
    """Run `scenario` on a fresh database. With `crash_at=k`, statement `base + k` is a crash."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "local-agent.db"
        factory = FaultFactory(fail_at=None if crash_at is None else base + crash_at, error=crash)
        db = Database.open(path, create=True, connection_factory=factory)
        sink = RecordingEventSink()
        try:
            apply(db, str(path))
            store = SQLiteSessionStore(db, Redactor([]), TickingClock())
            driven = asyncio.run(drive(store, factory, sink, scenario, str(path)))
        finally:
            db.close()
        reopened = Database.open(path, create=False)
        try:
            task, assistant_texts = asyncio.run(
                read_back(SQLiteSessionStore(reopened, Redactor([]), TickingClock()))
            )
        finally:
            reopened.close()
    return Observed(driven, sink.emitted, task, assistant_texts)


def assert_prefixes(observed: Observed) -> None:
    """Every persisted artifact text is a prefix of what was emitted for that artifact."""
    assert observed.task is not None
    for artifact in observed.task.artifacts:
        emitted = observed.emitted.get(artifact.artifact_id, "")
        assert emitted.startswith(artifact.text), (artifact.artifact_id, artifact.text, emitted)


def assert_exact(scenario: Scenario, observed: Observed) -> None:
    """Without a crash, the persisted texts equal the emitted texts and the history."""
    assert not observed.driven.crashed
    task = observed.task
    assert task is not None
    persisted = {a.artifact_id: a.text for a in task.artifacts}
    assert persisted == observed.emitted
    closed = [a for a in task.artifacts if a.closed]
    if scenario.interruption is Interruption.NONE:
        assert len(closed) == len(task.artifacts) == len(scenario.responses)
    else:
        assert all(a.closed for a in task.artifacts[:-1])
    # A response that completed has its text in the Message_History too.
    assert observed.assistant_texts == observed.driven.completed_texts
    for text in observed.driven.completed_texts:
        assert text in persisted.values()
    expected_state = {
        Interruption.NONE: TaskState.COMPLETED,
        Interruption.LLM_FAILURE: TaskState.FAILED,
        Interruption.CANCEL: TaskState.CANCELED,
    }[scenario.interruption]
    assert task.state == to_sdk(expected_state)


chunk_text = st.text(min_size=1, max_size=6)
responses_strategy = st.lists(
    st.lists(chunk_text, max_size=4).map(tuple), min_size=1, max_size=3
).map(tuple)


# Feature: local-agent-backend, Property 52: Persisted artifact text is a prefix of emitted text
@given(
    responses=responses_strategy,
    interruption=st.sampled_from(list(Interruption)),
    stop_pick=st.integers(min_value=0, max_value=12),
    crash_pick=st.none() | st.integers(min_value=0, max_value=10_000),
)
def test_persisted_artifact_text_is_a_prefix_of_emitted_text(
    responses: tuple[tuple[str, ...], ...],
    interruption: Interruption,
    stop_pick: int,
    crash_pick: int | None,
) -> None:
    """**Validates: Requirements 18.12, 3.4, 3.8**"""
    total_chunks = sum(len(chunks) for chunks in responses)
    scenario = Scenario(responses, interruption, stop_pick % (total_chunks + 1))

    clean = execute(scenario)
    assert_exact(scenario, clean)
    assert_prefixes(clean)

    if crash_pick is None or clean.driven.total == 0:
        return
    crash_at = 1 + crash_pick % clean.driven.total
    crashed = execute(scenario, crash_at=crash_at, base=clean.driven.base)
    assert crashed.driven.crashed
    assert_prefixes(crashed)
