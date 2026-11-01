"""Property 13: Context isolation (task 17.14).

Several Contexts run concurrently on one Session_Store, each with one or two Tasks. Every
Context's Message_History must hold only messages that its own Tasks produced, and must equal the
history the same scripts produce when they run alone on a fresh store.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections import defaultdict
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    ContextPlan,
    ScenarioRun,
    StoreKind,
    build_transcript,
    normalize_transcript,
    open_store,
    run_plans,
    scenario_scripts,
    store_kind,
)

from local_agent.models import UserMessage

__all__ = ["store_kind"]  # the fixture is used by name below

CONTEXT_IDS = ("ctx-a", "ctx-b", "ctx-c")


@st.composite
def concurrent_plans(draw: st.DrawFn) -> tuple[ContextPlan, ...]:
    """Two or three Contexts, each running one or two Tasks in sequence.

    Each script's user text is prefixed with its Context ID and Task number, so a message that
    leaks into another Context's history is recognizable.
    """
    count = draw(st.integers(min_value=2, max_value=3))
    plans: list[ContextPlan] = []
    for context_id in CONTEXT_IDS[:count]:
        scripts = draw(
            st.lists(scenario_scripts(max_turns=3), min_size=1, max_size=2),
        )
        marked = tuple(
            dataclasses.replace(script, user_text=f"[{context_id}#{number}] {script.user_text}")
            for number, script in enumerate(scripts, start=1)
        )
        plans.append(ContextPlan(context_id, marked))
    return tuple(plans)


def histories(runs: list[ScenarioRun]) -> dict[str, list[list[dict[str, Any]]]]:
    """Per Context, the normalized final history (and System_Prompt) after each of its Tasks."""
    by_context: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for run in runs:
        transcript = build_transcript(run)
        by_context[run.context_id].append(
            normalize_transcript(
                {
                    "system_prompt": transcript["system_prompt"],
                    "history": transcript["history"],
                    "state": run.state.value,
                }
            )
        )
    return dict(by_context)


def assert_only_own_messages(runs: list[ScenarioRun], db_path: str | None) -> None:
    last_run = {run.context_id: run for run in runs}
    for context_id, run in last_run.items():
        users = [m.content for m in run.history if isinstance(m, UserMessage)]
        assert all(text.startswith(f"[{context_id}#") for text in users), (context_id, users)
    if db_path is None:
        return
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT context_id, task_id FROM messages").fetchall()
    for context_id, task_id in rows:
        assert task_id is not None
        assert task_id.startswith(f"{context_id}/task-"), (context_id, task_id)


# Feature: local-agent-backend, Property 13: Context isolation
@settings(max_examples=100, deadline=None)
@given(plans=concurrent_plans())
def test_context_isolation(store_kind: StoreKind, plans: tuple[ContextPlan, ...]) -> None:
    with open_store(store_kind) as env:
        together = run_plans(plans, env)
        assert_only_own_messages(together, env.db_path if env.kind == "sqlite" else None)
        concurrent = histories(together)

    for plan in plans:
        with open_store(store_kind) as alone_env:
            alone = histories(run_plans([plan], alone_env))
        assert concurrent[plan.context_id] == alone[plan.context_id], plan.context_id
