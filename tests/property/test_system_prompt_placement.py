"""Property 23: System prompt placement and single storage (task 17.16).

One or two Contexts run one or two Tasks each. Every script has reasoning steps, registry changes
between them, and its own base instructions. Every chat-completion request must carry the
System_Prompt that the live registry calls for as its first and only system message, and each
Context must hold at most one System_Prompt, the one of its latest request.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Sequence

from hypothesis import given, settings
from hypothesis import strategies as st
from scenarios import (
    ContextPlan,
    ScenarioRun,
    ScenarioScript,
    StoreKind,
    _apply_change,
    build_registry,
    open_store,
    run_plans,
    scenario_scripts,
    store_kind,
)
from strategies import stored_texts

from fakes import FakeMcpManager
from local_agent.models import SystemMessage
from local_agent.orchestrator.prompt_builder import build_system_prompt

__all__ = ["store_kind"]  # the fixture is used by name below

CONTEXT_IDS = ("ctx-a", "ctx-b")


@st.composite
def prompt_plans(draw: st.DrawFn) -> tuple[ContextPlan, ...]:
    """One or two Contexts of one or two scripts each, with varying base instructions."""
    count = draw(st.integers(min_value=1, max_value=2))
    base_texts = st.one_of(st.none(), st.just(""), stored_texts(max_size=20))
    plans: list[ContextPlan] = []
    for context_id in CONTEXT_IDS[:count]:
        scripts = draw(st.lists(scenario_scripts(max_turns=4), min_size=1, max_size=2))
        plans.append(
            ContextPlan(
                context_id,
                tuple(
                    dataclasses.replace(script, base_instructions=draw(base_texts))
                    for script in scripts
                ),
            )
        )
    return tuple(plans)


def expected_prompts(script: ScenarioScript, requests: int) -> list[str]:
    """The System_Prompt of each of the first `requests` requests, from a replay of the registry.

    The changes of turn i apply when request i arrives, after its prompt was built, so request i
    sees the changes of turns 0 to i-1.
    """
    world = script.world
    mcp = FakeMcpManager(
        build_registry(world), {server.name: server.available for server in world.servers}
    )
    prompts: list[str] = []
    for index in range(requests):
        snapshot = mcp.registry.snapshot()
        visible = [
            tool
            for tools in snapshot.tools_by_server(mcp.availability()).values()
            for tool in tools
        ]
        prompts.append(build_system_prompt(script.base_instructions, visible))
        if index < len(script.turns):
            for change in script.turns[index].world_changes:
                _apply_change(change, mcp, world)
    return prompts


def assert_requests_place_the_prompt(run: ScenarioRun) -> list[str]:
    expected = expected_prompts(run.script, len(run.llm.requests))
    for request, prompt in zip(run.llm.requests, expected, strict=True):
        roles = [message.get("role") for message in request.messages]
        assert roles[0] == "system", (request.index, roles)
        assert roles.count("system") == 1, (request.index, roles)
        assert request.system_messages == [{"role": "system", "content": prompt}], request.index
    return expected


def assert_single_stored_prompt(runs: Sequence[ScenarioRun]) -> None:
    """Per Context, the store holds the prompt of the latest request, and never a system message
    inside the history."""
    previous: dict[str, SystemMessage | None] = {}
    for run in runs:
        expected = assert_requests_place_the_prompt(run)
        # With no request, the slot keeps what the Context's earlier Task left.
        stored = SystemMessage(expected[-1]) if expected else previous.get(run.context_id)
        assert run.system_prompt == stored, (run.task_id, run.system_prompt, stored)
        assert not any(isinstance(message, SystemMessage) for message in run.history), run.task_id
        previous[run.context_id] = run.system_prompt


def assert_no_system_rows(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM messages WHERE role = 'system'").fetchone()
        contexts = conn.execute("SELECT COUNT(*), COUNT(system_prompt) FROM contexts").fetchone()
    assert rows == (0,)
    assert contexts[1] <= contexts[0]  # at most one System_Prompt per Context row


# Feature: local-agent-backend, Property 23: System prompt placement and single storage
@settings(max_examples=100, deadline=None)
@given(plans=prompt_plans())
def test_system_prompt_placement(store_kind: StoreKind, plans: tuple[ContextPlan, ...]) -> None:
    with open_store(store_kind) as env:
        runs = run_plans(plans, env)
        assert_single_stored_prompt(runs)
        if env.kind == "sqlite":
            assert_no_system_rows(env.db_path)
