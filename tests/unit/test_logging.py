"""Unit tests for the log record format and level filtering (15.1, 15.7, 17.26)."""

import json
import re
from collections.abc import Iterator
from io import BytesIO
from typing import Any

import pytest
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from local_agent.config.models import LoggingConfig
from local_agent.logging_setup import Redactor, configure_logging, get_logger

_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00")
_LEVEL_ORDER = ["debug", "info", "warning", "error"]


@pytest.fixture(autouse=True)
def _isolated_logging() -> Iterator[None]:
    """Keep the global structlog configuration and bound context out of other tests."""
    clear_contextvars()
    yield
    clear_contextvars()
    structlog.reset_defaults()


def _configure(level: str | None = None) -> BytesIO:
    stream = BytesIO()
    cfg = LoggingConfig() if level is None else LoggingConfig.model_validate({"level": level})
    configure_logging(cfg, Redactor([]), stream=stream)
    return stream


def _records(stream: BytesIO) -> list[dict[str, Any]]:
    raw = stream.getvalue()
    if not raw:
        return []
    lines = raw.decode("utf-8").split("\n")
    assert lines.pop() == "", "output must end with a line feed"
    return [json.loads(line) for line in lines]


def _emit_all_levels() -> None:
    log = get_logger("levels")
    log.debug("d_event")
    log.info("i_event")
    log.warning("w_event")
    log.error("e_event")


# --- Record format (15.1) --------------------------------------------------------------------
def test_record_is_one_json_line_with_required_fields() -> None:
    stream = _configure("info")
    bind_contextvars(context_id="ctx-1", task_id="task-1")

    get_logger("orchestrator").info("task_state_changed", previous="submitted", new="working")

    raw = stream.getvalue()
    assert raw.count(b"\n") == 1
    (record,) = _records(stream)
    assert record["event"] == "task_state_changed"
    assert record["level"] == "info"
    assert record["component"] == "orchestrator"
    assert record["context_id"] == "ctx-1"
    assert record["task_id"] == "task-1"
    assert record["previous"] == "submitted"
    assert record["new"] == "working"
    assert isinstance(record["timestamp"], str)


def test_timestamp_is_utc_iso_8601_with_millisecond_precision() -> None:
    stream = _configure("info")

    get_logger("c").info("e")

    (record,) = _records(stream)
    assert _TIMESTAMP.fullmatch(record["timestamp"]), record["timestamp"]


def test_level_field_names_the_method_used() -> None:
    stream = _configure("debug")

    _emit_all_levels()

    assert [r["level"] for r in _records(stream)] == ["debug", "info", "warning", "error"]


def test_ids_are_null_outside_a_task() -> None:
    stream = _configure("info")

    get_logger("c").info("outside_task")

    (record,) = _records(stream)
    assert "context_id" in record
    assert "task_id" in record
    assert record["context_id"] is None
    assert record["task_id"] is None


def test_ids_come_from_bound_context_and_are_null_again_after_clearing() -> None:
    stream = _configure("info")
    log = get_logger("c")

    bind_contextvars(context_id="ctx-9", task_id="task-9")
    log.info("inside")
    clear_contextvars()
    log.info("outside")

    inside, outside = _records(stream)
    assert (inside["context_id"], inside["task_id"]) == ("ctx-9", "task-9")
    assert (outside["context_id"], outside["task_id"]) == (None, None)


def test_logger_created_before_configuration_uses_the_later_configuration() -> None:
    log = get_logger("early")
    stream = _configure("info")

    log.info("late_bound")

    (record,) = _records(stream)
    assert record["component"] == "early"
    assert record["event"] == "late_bound"


def test_exception_text_is_a_string_on_the_same_line() -> None:
    stream = _configure("info")

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("c").error("failed", exc_info=True)

    assert stream.getvalue().count(b"\n") == 1
    (record,) = _records(stream)
    assert isinstance(record["exception"], str)
    assert "ValueError: boom" in record["exception"]


def test_reconfiguring_replaces_the_previous_stream() -> None:
    first = _configure("info")
    second = _configure("info")

    get_logger("c").info("once")

    assert first.getvalue() == b""
    assert len(_records(second)) == 1


# --- Level filtering (15.7) ------------------------------------------------------------------
@pytest.mark.parametrize("level", _LEVEL_ORDER)
def test_records_below_the_minimum_level_are_omitted(level: str) -> None:
    stream = _configure(level)

    _emit_all_levels()

    expected = _LEVEL_ORDER[_LEVEL_ORDER.index(level) :]
    assert [r["level"] for r in _records(stream)] == expected


def test_info_is_the_default_minimum_level() -> None:
    assert LoggingConfig().level == "info"
    stream = _configure(None)

    _emit_all_levels()

    assert [r["event"] for r in _records(stream)] == ["i_event", "w_event", "e_event"]


def test_filtered_record_writes_nothing() -> None:
    stream = _configure("error")

    get_logger("c").warning("dropped")

    assert stream.getvalue() == b""


# --- Encoding (17.26) ------------------------------------------------------------------------
class _StrictBinaryStream:
    """A binary sink that rejects text and records each `write` call."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        assert isinstance(data, bytes), "log output must be written as bytes"
        self.chunks.append(data)
        return len(data)

    def flush(self) -> None:
        pass


def test_output_is_bytes_and_ends_with_exactly_one_lf() -> None:
    sink = _StrictBinaryStream()
    configure_logging(LoggingConfig(), Redactor([]), stream=sink)  # type: ignore[arg-type]

    get_logger("c").info("hello")

    data = b"".join(sink.chunks)
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert data.count(b"\n") == 1


def test_output_uses_lf_only_never_crlf() -> None:
    stream = _configure("info")
    log = get_logger("c")

    log.info("first")
    log.info("second")

    raw = stream.getvalue()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 2
    assert len(_records(stream)) == 2


def test_cr_and_lf_inside_values_are_escaped_so_the_record_stays_on_one_line() -> None:
    stream = _configure("info")

    get_logger("c").info("multi\r\nline", detail="a\nb\rc")

    raw = stream.getvalue()
    assert raw.count(b"\n") == 1
    assert b"\r" not in raw
    (record,) = _records(stream)
    assert record["event"] == "multi\r\nline"
    assert record["detail"] == "a\nb\rc"


def test_non_ascii_text_is_written_as_utf_8() -> None:
    stream = _configure("info")

    get_logger("c").info("caf\u00e9 \u65e5\u672c\u8a9e \U0001f600")

    raw = stream.getvalue()
    assert "caf\u00e9 \u65e5\u672c\u8a9e \U0001f600".encode() in raw
    assert b"\\u" not in raw
    (record,) = _records(stream)
    assert record["event"] == "caf\u00e9 \u65e5\u672c\u8a9e \U0001f600"


def test_lone_surrogate_does_not_break_the_record() -> None:
    stream = _configure("info")

    get_logger("c").info("bad", detail="x\ud800y")

    raw = stream.getvalue()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b"\\ud800" in raw
