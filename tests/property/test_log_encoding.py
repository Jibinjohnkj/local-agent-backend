"""Property test for log line encoding (Requirements 17.26, 15.1).

A record is written through the real logging pipeline (with a `Redactor`) into an in-memory byte
stream. Its text values are arbitrary Unicode, including CR, LF, NUL, non-BMP characters, and lone
surrogates. The same kind of event is also passed straight to `render_line`. In both cases the
bytes must be valid UTF-8, end with exactly one LF, hold no other CR or LF, and parse as JSON back
to the (redacted) event.
"""

import json
import re
from io import BytesIO
from typing import Any

import structlog
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from local_agent.config.models import LoggingConfig
from local_agent.logging_setup import Redactor, configure_logging, get_logger, render_line

# `exclude_categories=()` lets in the surrogate category, which `st.text` leaves out by default.
_ANY_CHAR = st.characters(exclude_categories=())
_SPECIAL_CHAR = st.sampled_from("\r\n\x00\u2028\u2029\U0001f680\ud83d\ude00\udc00\ud800")

# A high surrogate directly followed by a low surrogate is written as `\ud83d\ude00`, and every
# JSON parser reads that escape pair as one non-BMP character. The two are then no longer "lone"
# surrogates and cannot round-trip, whatever the encoder does. We leave that pair out.
_SURROGATE_PAIR = re.compile("[\ud800-\udbff][\udc00-\udfff]")
_TEXT = st.text(st.one_of(_ANY_CHAR, _SPECIAL_CHAR), max_size=30).filter(
    lambda s: _SURROGATE_PAIR.search(s) is None
)

_JSON_VALUE = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(-(2**53), 2**53), _TEXT),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(_TEXT, children, max_size=4),
    ),
    max_leaves=12,
)

_SECRETS = st.lists(st.text(_ANY_CHAR, min_size=1, max_size=8), max_size=3)


def _redacted(value: Any, redactor: Redactor) -> Any:
    """The expected event: every string, dictionary keys included, passed through `redact`."""
    if isinstance(value, str):
        return redactor.redact(value)
    if isinstance(value, list):
        return [_redacted(item, redactor) for item in value]
    if isinstance(value, dict):
        return {redactor.redact(k): _redacted(v, redactor) for k, v in value.items()}
    return value


def _check_line(line: bytes) -> Any:
    """Check the encoding rules on one rendered record and return the decoded JSON."""
    text = line.decode("utf-8")  # raises on invalid UTF-8
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert b"\r" not in line
    return json.loads(text)


# Feature: local-agent-backend, Property 47: Log line encoding
@settings(max_examples=200, deadline=None)
@given(
    message=_TEXT,
    fields=st.fixed_dictionaries({"detail": _JSON_VALUE, "other": _JSON_VALUE}),
    event=st.dictionaries(_TEXT, _JSON_VALUE, max_size=5),
    secrets=_SECRETS,
)
def test_log_line_encoding(
    message: str, fields: dict[str, Any], event: dict[str, Any], secrets: list[str]
) -> None:
    # `render_line` on its own: the line parses back to exactly the event it was given.
    assert _check_line(render_line(event)) == event

    # The whole pipeline: the line parses back to the redacted event.
    redactor = Redactor(secrets)
    stream = BytesIO()
    configure_logging(LoggingConfig(level="debug"), redactor, stream=stream)
    try:
        get_logger("test").info(message, **fields)
    finally:
        structlog.reset_defaults()

    record = _check_line(stream.getvalue())
    assert record["event"] == redactor.redact(message)
    # Field names are strings in the record, so they are redacted like every other string. A
    # secret can mask two names to the same text; the second would then replace the first.
    names = {name: redactor.redact(name) for name in fields}
    assume(len(set(names.values())) == len(names))
    for name, value in fields.items():
        assert record[names[name]] == _redacted(value, redactor)
