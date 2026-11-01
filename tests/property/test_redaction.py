"""Property test for secret redaction (Requirements 15.5, 9.7, 2.6).

The test configures the real logging pipeline with a `Redactor`, writes one record into an
in-memory byte stream, and checks the rendered line. A record carries secrets whole, as substrings,
and mixed with unrelated text, in the message, in a plain field, in a nested list, and in a nested
dict key.

Two things in a line are not secret data: the fixed JSON structure (field names, level,
timestamp) and the placeholder text. Secrets are drawn from an alphabet that cannot occur in
either of those, so a hit in the line can only come from a leaked secret. That alphabet leaves
out lowercase letters, digits, quotes, brackets, LF, and the letters of `[REDACTED]`.
"""

import json
from io import BytesIO
from typing import Any

import structlog
from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.config.models import LoggingConfig
from local_agent.logging_setup import Redactor, configure_logging, get_logger

_SECRET_CHARS = "BFGHIJKLMNOPQSUVWXYZ!@#$%^&*()~\t\u00e9\u00df\u65e5"
_UNRELATED_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 _-."

_SECRETS = st.lists(st.text(_SECRET_CHARS, min_size=1, max_size=12), min_size=1, max_size=4)
_UNRELATED = st.text(_UNRELATED_CHARS, max_size=6)


@st.composite
def _scenario(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """Draw some secrets and four text values that mix whole secrets, pieces, and other text."""
    secrets = draw(_SECRETS)

    def piece() -> st.SearchStrategy[str]:
        secret = st.sampled_from(secrets)
        part = secret.flatmap(
            lambda s: st.tuples(st.integers(0, len(s)), st.integers(0, len(s)), st.just(s)).map(
                lambda t: t[2][min(t[0], t[1]) : max(t[0], t[1])]
            )
        )
        return st.one_of(secret, part, _UNRELATED, st.text(_SECRET_CHARS, max_size=5))

    texts = [draw(st.lists(piece(), max_size=6).map("".join)) for _ in range(4)]
    return secrets, texts


def _windows(secrets: list[str]) -> set[str]:
    return {s[i : i + 4] for s in secrets for i in range(len(s) - 3)}


def _leaks(text: str, secrets: list[str]) -> bool:
    """Does `text` contain a 4-character secret substring, or a whole short secret?"""
    if any(w in text for w in _windows(secrets)):
        return True
    return any(len(s) < 4 and s in text for s in secrets)


def _walk(value: Any) -> list[str]:
    """Every string in a decoded JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in _walk(item)]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [k, *_walk(v)]]
    return []


# Feature: local-agent-backend, Property 28: Secret redaction
@settings(max_examples=200, deadline=None)
@given(_scenario())
def test_secret_redaction(scenario: tuple[list[str], list[str]]) -> None:
    secrets, (message, detail, item, key) = scenario
    redactor = Redactor(secrets)
    stream = BytesIO()
    configure_logging(LoggingConfig(level="debug"), redactor, stream=stream)
    try:
        get_logger("test").info(message, detail=detail, nested={key: [item]})
    finally:
        structlog.reset_defaults()

    line = stream.getvalue().decode("utf-8")
    assert line.endswith("\n")
    assert line.count("\n") == 1

    # No secret substring of 4 or more characters, and no whole short secret, in the raw line
    # or in any decoded string of the record.
    assert not _leaks(line, secrets)
    record = json.loads(line)
    for text in _walk(record):
        assert not _leaks(text, secrets)

    # Text that shares no 4-character window with a secret, and holds no short secret, is kept.
    if not _leaks(message, secrets):
        assert record["event"] == message
    if not _leaks(detail, secrets):
        assert record["detail"] == detail
    if not (_leaks(key, secrets) or _leaks(item, secrets)):
        assert record["nested"] == {key: [item]}
    for text in (message, detail, item, key):
        if not _leaks(text, secrets):
            assert redactor.redact(text) == text
