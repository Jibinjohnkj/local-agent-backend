"""The SDK's "already exists" ERROR for continuation snapshots is logged at DEBUG (design.md)."""

import inspect
import logging
from collections.abc import Iterator

import pytest
from a2a.server.agent_execution import active_task

from local_agent.a2a_server.sdk_log import (
    ACTIVE_TASK_LOGGER,
    ALREADY_EXISTS_PREFIX,
    ALREADY_EXISTS_SUFFIX,
    ContinuationSnapshotFilter,
    install_active_task_log_filter,
)

SDK_TEMPLATE = "Task %s already exists. Ignoring task replacement."


@pytest.fixture
def sdk_logger() -> Iterator[logging.Logger]:
    """The SDK logger with exactly one filter installed, restored afterwards."""
    logger = logging.getLogger(ACTIVE_TASK_LOGGER)
    saved = list(logger.filters)
    logger.filters[:] = []
    install_active_task_log_filter()
    yield logger
    logger.filters[:] = saved


def test_pinned_sdk_still_logs_the_message_the_filter_matches() -> None:
    source = inspect.getsource(active_task)
    assert SDK_TEMPLATE in source
    assert f"{ALREADY_EXISTS_PREFIX}%s{ALREADY_EXISTS_SUFFIX}" == SDK_TEMPLATE


def test_install_is_idempotent(sdk_logger: logging.Logger) -> None:
    install_active_task_log_filter()
    install_active_task_log_filter()
    assert sum(isinstance(f, ContinuationSnapshotFilter) for f in sdk_logger.filters) == 1


def test_replacement_notice_is_recorded_at_debug(
    sdk_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    sdk_logger.error(SDK_TEMPLATE, "task-1")

    [record] = [r for r in caplog.records if r.name == ACTIVE_TASK_LOGGER]
    assert record.levelno == logging.DEBUG
    assert record.levelname == "DEBUG"
    assert record.getMessage() == "Task task-1 already exists. Ignoring task replacement."


def test_replacement_notice_is_dropped_when_debug_is_off(
    sdk_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)  # the root level; the SDK logger inherits it
    sdk_logger.setLevel(logging.NOTSET)

    sdk_logger.error(SDK_TEMPLATE, "task-1")

    assert [r for r in caplog.records if r.name == ACTIVE_TASK_LOGGER] == []


def test_other_errors_of_the_sdk_logger_keep_their_level(
    sdk_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    sdk_logger.error("Error during aclose")
    sdk_logger.error("Task %s failed to start", "task-1")
    sdk_logger.warning("Task %s already exists", "task-1")

    levels = [r.levelno for r in caplog.records if r.name == ACTIVE_TASK_LOGGER]
    assert levels == [logging.ERROR, logging.ERROR, logging.WARNING]


def test_a_malformed_record_is_left_alone(sdk_logger: logging.Logger) -> None:
    record = logging.LogRecord(
        ACTIVE_TASK_LOGGER, logging.ERROR, __file__, 1, SDK_TEMPLATE, (), None
    )
    record.args = ("one", "too many")  # formatting this record raises TypeError

    assert ContinuationSnapshotFilter().filter(record) is True
    assert record.levelno == logging.ERROR
