"""Property test for invalid UTF-8 reporting (Requirement 17.24).

A valid UTF-8 config text, optionally BOM-prefixed, gets an invalid byte sequence inserted at a
character boundary k. Loading it must fail with an error that names the file and reports k
counted from the first byte of the file, BOM included, and must never produce a Backend_Config.
The insertion point is a character boundary, so the original bytes hold no invalid sequence
before k.
"""

import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.config.loader import ConfigError, decode_config_bytes, load_config

_BOM = b"\xef\xbb\xbf"

# Sequences that are invalid from their first byte, whatever byte follows a character boundary:
# a stray continuation byte, bytes that never occur in UTF-8, a truncated multi-byte sequence
# (the next byte is always ASCII or a lead byte, never a continuation), an overlong encoding,
# and an encoded surrogate.
_INVALID_SEQUENCES = st.one_of(
    st.integers(0x80, 0xBF).map(lambda b: bytes([b])),
    st.sampled_from([b"\xc0", b"\xc1", b"\xf5", b"\xf8", b"\xfe", b"\xff"]),
    st.integers(0xC2, 0xF4).map(lambda b: bytes([b])),
    st.sampled_from([b"\xe2\x82", b"\xf0\x9f\x98", b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80"]),
)

# Any text that encodes to UTF-8 (Hypothesis excludes surrogates by default).
_TEXT = st.text(max_size=60)

_OFFSET_MESSAGE = re.compile(r"^invalid UTF-8 at byte offset (\d+)$")


@st.composite
def _corrupted(draw: st.DrawFn) -> tuple[bytes, int]:
    """Return `(file bytes, k)`: valid UTF-8 with an invalid sequence inserted at offset k."""
    text = draw(_TEXT)
    bom = draw(st.booleans())
    prefix = _BOM if bom else b""
    boundaries = [len(prefix)]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    k = draw(st.sampled_from(boundaries))
    original = prefix + text.encode("utf-8")
    invalid = draw(_INVALID_SEQUENCES)
    return original[:k] + invalid + original[k:], k


# Feature: local-agent-backend, Property 36: Invalid UTF-8 reporting
# **Validates: Requirements 17.24**
@settings(max_examples=100)
@given(case=_corrupted())
def test_invalid_utf8_is_reported_at_byte_offset(case: tuple[bytes, int]) -> None:
    data, k = case

    with pytest.raises(ConfigError) as decode_info:
        decode_config_bytes(data, "config.yaml")
    [issue] = decode_info.value.issues
    assert issue.path == "config.yaml"
    match = _OFFSET_MESSAGE.match(issue.message)
    assert match is not None, issue.message
    assert int(match.group(1)) == k

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_bytes(data)
        with pytest.raises(ConfigError) as load_info:
            load_config(path, {})
    [load_issue] = load_info.value.issues
    assert load_issue.path == str(path)
    load_match = _OFFSET_MESSAGE.match(load_issue.message)
    assert load_match is not None, load_issue.message
    assert int(load_match.group(1)) == k
