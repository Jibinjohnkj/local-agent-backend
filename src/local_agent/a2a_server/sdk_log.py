"""Log filter for one `a2a-sdk` message that is routine in this Backend (design.md, research table).

`a2a.server.agent_execution.active_task` logs "Task <id> already exists. Ignoring task
replacement." at ERROR when the executor enqueues a `Task` for a Task the store already holds. The
executor does that on purpose: the first event of every streaming request is a Task_Snapshot (3.10),
and for a continuation of an `input-required` Task the stored Task already exists. The SDK ignores
the replacement and forwards the stored Task, which is the behaviour the executor relies on. A new
Task never reaches this branch, because its snapshot is the creation commit.

`ContinuationSnapshotFilter` lowers that one record to DEBUG. Every other record of the logger,
including every other ERROR, passes unchanged. The filter is attached to the SDK logger by name,
so it sees only that module's records.
"""

import logging
from typing import Final

__all__ = [
    "ACTIVE_TASK_LOGGER",
    "ALREADY_EXISTS_PREFIX",
    "ALREADY_EXISTS_SUFFIX",
    "ContinuationSnapshotFilter",
    "install_active_task_log_filter",
]

ACTIVE_TASK_LOGGER: Final = "a2a.server.agent_execution.active_task"

# The SDK message is "Task %s already exists. Ignoring task replacement."; the task ID sits between.
ALREADY_EXISTS_PREFIX: Final = "Task "
ALREADY_EXISTS_SUFFIX: Final = " already exists. Ignoring task replacement."


class ContinuationSnapshotFilter(logging.Filter):
    """Downgrade the SDK's "already exists" ERROR record to DEBUG.

    The record is rewritten in place, so every handler sees it as a DEBUG record. It is then
    dropped unless the SDK logger itself would emit DEBUG, because the logger's own level check
    ran before this filter and used the original ERROR level.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= logging.DEBUG or not _is_replacement_notice(record):
            return True
        record.levelno = logging.DEBUG
        record.levelname = logging.getLevelName(logging.DEBUG)
        return logging.getLogger(record.name).isEnabledFor(logging.DEBUG)


def _is_replacement_notice(record: logging.LogRecord) -> bool:
    if not isinstance(record.msg, str):
        return False
    try:
        text = record.getMessage()
    except TypeError, ValueError:
        return False  # a malformed record is not ours; let the handlers deal with it
    return text.startswith(ALREADY_EXISTS_PREFIX) and text.endswith(ALREADY_EXISTS_SUFFIX)


def install_active_task_log_filter() -> None:
    """Attach `ContinuationSnapshotFilter` to the SDK's `active_task` logger. Idempotent."""
    sdk_logger = logging.getLogger(ACTIVE_TASK_LOGGER)
    if not any(isinstance(f, ContinuationSnapshotFilter) for f in sdk_logger.filters):
        sdk_logger.addFilter(ContinuationSnapshotFilter())
