"""Property test for the stdio transport's framing (Requirement 17.27).

`encode_line` writes one JSON-RPC message as compact UTF-8 ended by a single LF, and `decode_line`
reads it back. The reader splits the byte stream on LF (dropping one CR before it) before any
decoding, so chunk boundaries inside a multi-byte sequence do not matter.
"""

from itertools import pairwise
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.mcp_client.mcp_compat import SessionMessage, jsonrpc_message_adapter
from local_agent.mcp_client.stdio_transport import _split_lines, decode_line, encode_line

# `st.text()` never draws surrogates, so every string here is encodable as UTF-8. It does draw
# LF, CR, U+2028 and other characters a naive line splitter could trip on.
_TEXT = st.text(max_size=30)
_JSON_VALUES: st.SearchStrategy[Any] = st.recursive(
    st.none() | st.booleans() | st.integers(-(2**31), 2**31) | _TEXT,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(_TEXT, children, max_size=4),
    max_leaves=10,
)
_OBJECTS = st.dictionaries(_TEXT, _JSON_VALUES, max_size=4)
_IDS = st.integers(0, 2**31) | _TEXT
_METHODS = st.text(min_size=1, max_size=20)


def _with_params(base: dict[str, Any], params: dict[str, Any] | None) -> dict[str, Any]:
    return base if params is None else {**base, "params": params}


_REQUESTS = st.builds(
    lambda i, m, p: _with_params({"jsonrpc": "2.0", "id": i, "method": m}, p),
    _IDS,
    _METHODS,
    st.none() | _OBJECTS,
)
_NOTIFICATIONS = st.builds(
    lambda m, p: _with_params({"jsonrpc": "2.0", "method": m}, p), _METHODS, st.none() | _OBJECTS
)
_RESPONSES = st.builds(lambda i, r: {"jsonrpc": "2.0", "id": i, "result": r}, _IDS, _OBJECTS)
_ERRORS = st.builds(
    lambda i, c, m, d: {
        "jsonrpc": "2.0",
        "id": i,
        "error": {"code": c, "message": m, **({} if d is None else {"data": d})},
    },
    _IDS,
    st.integers(-(2**31), 2**31),
    _TEXT,
    st.none() | _JSON_VALUES,
)
_MESSAGES = st.one_of(_REQUESTS, _NOTIFICATIONS, _RESPONSES, _ERRORS).map(
    lambda raw: SessionMessage(jsonrpc_message_adapter.validate_python(raw))
)


# Feature: local-agent-backend, Property 48: stdio framing round-trip
# **Validates: Requirements 17.27**
@settings(max_examples=200)
@given(messages=st.lists(_MESSAGES, min_size=1, max_size=5), crlf=st.data(), cuts=st.data())
def test_stdio_framing_round_trip(
    messages: list[SessionMessage], crlf: st.DataObject, cuts: st.DataObject
) -> None:
    encoded = [encode_line(m) for m in messages]

    # Each encoding is valid UTF-8 ending in exactly one LF with no other LF.
    for line in encoded:
        line.decode("utf-8")
        assert line.endswith(b"\n")
        assert line.count(b"\n") == 1

    # One message at a time: decoding the line without its terminator gives the message back.
    for message, line in zip(messages, encoded, strict=True):
        assert decode_line(line[:-1]) == message

    # A stream of the encodings, some ended by CRLF, cut at arbitrary byte offsets.
    stream = b"".join(
        line[:-1] + b"\r\n" if crlf.draw(st.booleans(), label="crlf") else line for line in encoded
    )
    points = sorted(
        cuts.draw(st.lists(st.integers(0, len(stream)), max_size=8), label="cut offsets")
    )
    bounds = [0, *points, len(stream)]

    buffer = bytearray()
    decoded: list[SessionMessage | Exception] = []
    for start, end in pairwise(bounds):
        buffer.extend(stream[start:end])
        decoded.extend(decode_line(line) for line in _split_lines(buffer))

    assert decoded == messages
    assert not buffer
