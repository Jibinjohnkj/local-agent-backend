"""Unit tests for the in-memory Session_Store (Requirements 2.1, 11.1, 11.2, 11.7, 12.7).

Each test drives `InMemorySessionStore` through its public interface. The Last_Activity tests read
`last_activity_ms` from the store's private `_contexts` map, because the interface does not expose
it to readers.
"""

from __future__ import annotations

import pytest

from local_agent.a2a_server.events import LIVE_STATES, TaskState, to_sdk
from local_agent.models import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from local_agent.session.memory_store import InMemorySessionStore
from local_agent.session.store import (
    UNSET,
    ArtifactWrite,
    StaleState,
    StateChange,
    StorageError,
    StorageTxn,
    TaskQuery,
)

T0 = 1_000


def _store() -> InMemorySessionStore:
    return InMemorySessionStore()


def _activity(store: InMemorySessionStore, context_id: str) -> int:
    return store._contexts[context_id].last_activity_ms


def _change(
    task_id: str,
    new: TaskState,
    *,
    expected: frozenset[TaskState] | None = None,
    status_message: str | None = None,
) -> StateChange:
    return StateChange(
        task_id=task_id,
        expected_from=expected if expected is not None else frozenset(TaskState),
        new=new,
        status_message=status_message,
        metadata=None,
    )


# --- Creation and mapping -------------------------------------------------------------------


async def test_new_context_has_empty_history_and_no_system_prompt() -> None:
    store = _store()

    created = await store.create_task("t1", "c1", T0, None)

    assert created is True
    assert await store.get_history("c1") == (None, [])


async def test_unknown_context_has_empty_history() -> None:
    assert await _store().get_history("nope") == (None, [])


async def test_create_task_into_existing_context_reports_not_new() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    assert await store.create_task("t2", "c1", T0 + 1, None) is False


async def test_create_task_inserts_submitted_task_in_its_context() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, "snapshot")

    record = await store.get_task("t1")

    assert record is not None
    assert record.task_id == "t1"
    assert record.context_id == "c1"
    assert record.state == to_sdk(TaskState.SUBMITTED)
    assert record.created_at_ms == T0
    assert record.updated_at_ms == T0
    assert record.artifacts == ()


async def test_get_task_unknown_returns_none() -> None:
    assert await _store().get_task("nope") is None


async def test_tasks_map_to_their_own_contexts() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.create_task("t2", "c2", T0 + 1, None)
    await store.create_task("t3", "c1", T0 + 2, None)

    page = await store.list_tasks(TaskQuery(context_id="c1"))

    assert {task.task_id for task in page.tasks} == {"t1", "t3"}
    assert all(task.context_id == "c1" for task in page.tasks)
    record = await store.get_task("t2")
    assert record is not None
    assert record.context_id == "c2"


async def test_histories_of_different_contexts_are_isolated() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.create_task("t2", "c2", T0, None)

    await store.append("c1", [UserMessage("hello")], task_id="t1", now_ms=T0 + 1)

    assert (await store.get_history("c1"))[1] == [UserMessage("hello")]
    assert (await store.get_history("c2"))[1] == []


async def test_appends_keep_order_across_commits() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    messages: list[ChatMessage] = [
        UserMessage("q"),
        AssistantMessage("a"),
        ToolMessage(tool_call_id="x", content="r"),
    ]

    await store.append("c1", messages[:1], task_id="t1", now_ms=T0 + 1)
    await store.append("c1", messages[1:], task_id="t1", now_ms=T0 + 2)

    assert (await store.get_history("c1"))[1] == messages


async def test_returned_history_list_is_a_copy() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.append("c1", [UserMessage("q")], now_ms=T0 + 1)

    _, history = await store.get_history("c1")
    history.append(UserMessage("injected"))

    assert (await store.get_history("c1"))[1] == [UserMessage("q")]


# --- System-prompt slot ---------------------------------------------------------------------


async def test_system_prompt_slot_is_replaced_not_appended() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    await store.set_system_prompt("c1", SystemMessage("first"), now_ms=T0 + 1)
    await store.set_system_prompt("c1", SystemMessage("second"), now_ms=T0 + 2)

    system_prompt, history = await store.get_history("c1")
    assert system_prompt == SystemMessage("second")
    assert history == []


async def test_system_prompt_can_be_cleared_and_unset_leaves_it_alone() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.set_system_prompt("c1", SystemMessage("keep"), now_ms=T0 + 1)

    await store.commit(StorageTxn(context_id="c1", now_ms=T0 + 2, system_prompt=UNSET))
    assert (await store.get_history("c1"))[0] == SystemMessage("keep")

    await store.set_system_prompt("c1", None, now_ms=T0 + 3)
    assert (await store.get_history("c1"))[0] is None


async def test_system_message_in_appends_is_rejected_and_changes_nothing() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    with pytest.raises(StorageError):
        await store.append("c1", [UserMessage("q"), SystemMessage("sneaky")], now_ms=T0 + 1)

    assert await store.get_history("c1") == (None, [])


# --- active_task ----------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(TaskState))
async def test_active_task_for_each_state(state: TaskState) -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    if state is not TaskState.SUBMITTED:
        await store.set_state("c1", _change("t1", state), now_ms=T0 + 1)

    expected = "t1" if state in LIVE_STATES else None
    assert await store.active_task("c1") == expected


async def test_active_task_unknown_context_is_none() -> None:
    assert await _store().active_task("nope") is None


async def test_active_task_ignores_finished_tasks_and_other_contexts() -> None:
    store = _store()
    await store.create_task("done", "c1", T0, None)
    await store.set_state("c1", _change("done", TaskState.COMPLETED), now_ms=T0 + 1)
    await store.create_task("live", "c1", T0 + 2, None)
    await store.create_task("other", "c2", T0 + 3, None)

    assert await store.active_task("c1") == "live"
    assert await store.active_task("c2") == "other"


# --- CAS and atomicity ----------------------------------------------------------------------


async def test_cas_failure_raises_stale_state_and_leaves_store_unchanged() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.set_system_prompt("c1", SystemMessage("old"), now_ms=T0 + 1)
    before_task = await store.get_task("t1")
    before_activity = _activity(store, "c1")

    txn = StorageTxn(
        context_id="c1",
        now_ms=T0 + 50,
        appends=(("t1", UserMessage("q")),),
        system_prompt=SystemMessage("new"),
        state=_change("t1", TaskState.COMPLETED, expected=frozenset({TaskState.WORKING})),
        artifacts=(ArtifactWrite("t1", "response-1", 0, "text", closed=True),),
    )
    with pytest.raises(StaleState):
        await store.commit(txn)

    assert await store.get_history("c1") == (SystemMessage("old"), [])
    assert await store.get_task("t1") == before_task
    assert _activity(store, "c1") == before_activity


async def test_set_state_cas_accepts_any_state_in_expected_from() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    expected = frozenset({TaskState.SUBMITTED, TaskState.WORKING})

    await store.set_state("c1", _change("t1", TaskState.WORKING, expected=expected), now_ms=T0 + 1)
    await store.set_state(
        "c1", _change("t1", TaskState.COMPLETED, expected=expected), now_ms=T0 + 2
    )
    with pytest.raises(StaleState):
        await store.set_state(
            "c1", _change("t1", TaskState.FAILED, expected=expected), now_ms=T0 + 3
        )

    record = await store.get_task("t1")
    assert record is not None
    assert record.state == to_sdk(TaskState.COMPLETED)


async def test_failing_txn_applies_nothing_for_unknown_task_in_artifacts() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    before_task = await store.get_task("t1")
    before_activity = _activity(store, "c1")

    txn = StorageTxn(
        context_id="c1",
        now_ms=T0 + 5,
        appends=(("t1", UserMessage("q")),),
        system_prompt=SystemMessage("new"),
        state=_change("t1", TaskState.WORKING),
        artifacts=(
            ArtifactWrite("t1", "response-1", 0, "ok", closed=False),
            ArtifactWrite("ghost", "response-1", 0, "bad", closed=False),
        ),
    )
    with pytest.raises(StorageError):
        await store.commit(txn)

    assert await store.get_history("c1") == (None, [])
    assert await store.get_task("t1") == before_task
    assert _activity(store, "c1") == before_activity


async def test_failing_txn_applies_nothing_for_unknown_context() -> None:
    store = _store()

    with pytest.raises(StorageError):
        await store.append("missing", [UserMessage("q")], now_ms=T0)

    assert await store.get_history("missing") == (None, [])
    assert "missing" not in store._contexts


async def test_txn_naming_a_task_of_another_context_applies_nothing() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.create_task("t2", "c2", T0, None)

    txn = StorageTxn(
        context_id="c1",
        now_ms=T0 + 1,
        appends=(("t1", UserMessage("q")),),
        state=_change("t2", TaskState.WORKING),
    )
    with pytest.raises(StorageError):
        await store.commit(txn)

    assert (await store.get_history("c1"))[1] == []
    record = await store.get_task("t2")
    assert record is not None
    assert record.state == to_sdk(TaskState.SUBMITTED)


async def test_artifact_ordinal_collision_applies_nothing() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.commit(
        StorageTxn(
            context_id="c1",
            now_ms=T0 + 1,
            artifacts=(ArtifactWrite("t1", "response-1", 0, "a", closed=True),),
        )
    )

    txn = StorageTxn(
        context_id="c1",
        now_ms=T0 + 2,
        appends=(("t1", UserMessage("q")),),
        artifacts=(ArtifactWrite("t1", "response-2", 0, "b", closed=False),),
    )
    with pytest.raises(StorageError):
        await store.commit(txn)

    assert (await store.get_history("c1"))[1] == []
    record = await store.get_task("t1")
    assert record is not None
    assert [a.artifact_id for a in record.artifacts] == ["response-1"]


async def test_successful_txn_applies_every_part() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    await store.commit(
        StorageTxn(
            context_id="c1",
            now_ms=T0 + 5,
            appends=(("t1", UserMessage("q")), ("t1", AssistantMessage("a"))),
            system_prompt=SystemMessage("sys"),
            state=_change("t1", TaskState.WORKING, status_message="going"),
            artifacts=(
                ArtifactWrite("t1", "response-2", 1, "second", closed=True),
                ArtifactWrite("t1", "response-1", 0, "first", closed=True),
            ),
        )
    )

    assert await store.get_history("c1") == (
        SystemMessage("sys"),
        [UserMessage("q"), AssistantMessage("a")],
    )
    record = await store.get_task("t1")
    assert record is not None
    assert record.state == to_sdk(TaskState.WORKING)
    assert record.status_message == "going"
    assert record.updated_at_ms == T0 + 5
    assert [a.artifact_id for a in record.artifacts] == ["response-1", "response-2"]


# --- create_task collisions -----------------------------------------------------------------


async def test_task_id_collision_in_same_context_fails_and_keeps_original() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, "orig")
    await store.set_state("c1", _change("t1", TaskState.WORKING), now_ms=T0 + 1)
    before = await store.get_task("t1")

    with pytest.raises(StorageError):
        await store.create_task("t1", "c1", T0 + 9, None)

    assert await store.get_task("t1") == before
    assert _activity(store, "c1") == T0 + 1


async def test_task_id_collision_does_not_create_a_new_context() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    with pytest.raises(StorageError):
        await store.create_task("t1", "c2", T0 + 1, None)

    assert "c2" not in store._contexts
    assert await store.active_task("c2") is None
    record = await store.get_task("t1")
    assert record is not None
    assert record.context_id == "c1"


# --- Last_Activity --------------------------------------------------------------------------


async def test_create_task_sets_last_activity() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    assert _activity(store, "c1") == T0

    await store.create_task("t2", "c1", T0 + 10, None)
    assert _activity(store, "c1") == T0 + 10


async def test_same_state_status_message_does_not_change_last_activity() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)
    await store.set_state("c1", _change("t1", TaskState.WORKING), now_ms=T0 + 1)

    await store.set_state(
        "c1",
        _change("t1", TaskState.WORKING, status_message="calling tool"),
        now_ms=T0 + 100,
    )

    assert _activity(store, "c1") == T0 + 1
    record = await store.get_task("t1")
    assert record is not None
    assert record.status_message == "calling tool"
    assert record.state == to_sdk(TaskState.WORKING)


async def test_state_change_and_append_set_last_activity() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    await store.set_state("c1", _change("t1", TaskState.WORKING), now_ms=T0 + 10)
    assert _activity(store, "c1") == T0 + 10

    await store.append("c1", [UserMessage("q")], task_id="t1", now_ms=T0 + 20)
    assert _activity(store, "c1") == T0 + 20


async def test_system_prompt_and_artifact_writes_do_not_change_last_activity() -> None:
    store = _store()
    await store.create_task("t1", "c1", T0, None)

    await store.set_system_prompt("c1", SystemMessage("sys"), now_ms=T0 + 10)
    await store.commit(
        StorageTxn(
            context_id="c1",
            now_ms=T0 + 20,
            artifacts=(ArtifactWrite("t1", "response-1", 0, "text", closed=False),),
        )
    )

    assert _activity(store, "c1") == T0
