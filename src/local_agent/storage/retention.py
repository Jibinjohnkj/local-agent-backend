"""Session_Idle_Timeout retention (11.11-11.14).

`select_expired` is the pure rule. The in-memory store applies it to its records, and the SQLite
store runs the same predicate in SQL, so the two can be checked against each other. `sweep` is one
pass: it asks the Session_Store to delete the expired Contexts in one transaction, then tells the
runner registry to drop whatever it still holds for them.

Scheduling lives in the lifespan task (once after Restart_Recovery, then every 30 s), and only when
`session_idle_timeout_s` is set (11.13). Nothing here reads the configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from local_agent.logging_setup import get_logger
from local_agent.storage.database import Database, run_in_transaction

if TYPE_CHECKING:
    from local_agent.models import Clock
    from local_agent.session.store import SessionStore

__all__ = ["ContextActivity", "ForgetContexts", "select_expired", "sweep", "sweep_at_startup"]

_log = get_logger("storage.retention")


@dataclass(frozen=True)
class ContextActivity:
    """What retention needs to know about one Context."""

    context_id: str
    last_activity_ms: int
    has_live_task: bool  # a Task in submitted, working, or input-required


class ForgetContexts(Protocol):
    """The part of `TaskRunnerRegistry` that retention uses."""

    def forget(self, context_ids: Sequence[str]) -> None:
        """Drop all in-memory state for the deleted Contexts."""
        ...


def select_expired(
    contexts: Sequence[ContextActivity], now_ms: int, idle_timeout_s: int
) -> list[str]:
    """Return the IDs of Contexts to delete, in input order.

    A Context is expired when its Last_Activity is older than the timeout and it has no live Task
    (11.11, 11.12). A `submitted` Task counts as live, so a Task about to start is never deleted
    under its runner.
    """
    timeout_ms = idle_timeout_s * 1000
    return [
        context.context_id
        for context in contexts
        if now_ms - context.last_activity_ms > timeout_ms and not context.has_live_task
    ]


async def sweep(
    store: SessionStore, runners: ForgetContexts, clock: Clock, idle_timeout_s: int
) -> list[str]:
    """Delete expired Contexts and forget their runners. Return the deleted `context_id`s."""
    cutoff_ms = clock.now_ms() - idle_timeout_s * 1000
    context_ids = await store.delete_expired(cutoff_ms)
    if context_ids:
        runners.forget(context_ids)
        _log.info("retention_sweep", deleted_contexts=len(context_ids))
    return context_ids


def sweep_at_startup(db: Database, clock: Clock, idle_timeout_s: int) -> list[str]:
    """One sweep for Contexts that expired while the Backend was stopped (11.11).

    Blocks on the writer thread through `Database.run_sync`, so it runs before the event loop
    exists, right after Restart_Recovery. No runner exists yet, so there is nothing to forget.
    """
    # Imported here: `session.memory_store` imports this module, so a top-level import of the
    # SQLite store could form a cycle.
    from local_agent.session.sqlite_store import _delete_expired

    cutoff_ms = clock.now_ms() - idle_timeout_s * 1000
    path = str(db.path)
    context_ids = db.run_sync(
        lambda conn: run_in_transaction(conn, lambda c: _delete_expired(c, cutoff_ms), path)
    )
    if context_ids:
        _log.info("retention_sweep", deleted_contexts=len(context_ids))
    return context_ids
