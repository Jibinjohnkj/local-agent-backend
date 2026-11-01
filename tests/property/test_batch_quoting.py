"""Property test for batch-file quoting and rejection (Requirements 17.8, 17.9).

`quote_batch_args(batch_path, args)` builds the inner line that `cmd.exe /c` runs for a `.cmd` or
`.bat` target. The test draws a batch path and an argument list from the characters that matter to
`cmd.exe` and the C runtime, with `"`, `%`, CR, and LF mixed in at a low rate so that both outcomes
occur.

- When any piece holds `"`, `%`, CR, or LF, the function must raise `BatchArgRejected`. The expected
  position and character are computed here by scanning the pieces in order. The batch path is
  position 0, and arguments count from 1.
- Otherwise the test checks the produced line against the two independent models in `winargv_ref`:
  a `cmd.exe` quote-state scan must read every `& | < > ^` and space that came from the path or an
  argument inside a quoted region (and no operator outside one), and the C runtime parser applied
  to the text after the batch path must return exactly the original arguments.

The state check locates each piece by counting characters (opening quote, text, doubled trailing
backslashes, closing quote, one joining space), so it does not rely on the implementation's own
helpers. NUL is excluded because a process argument cannot hold it.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from winargv_ref import cmd_quote_states, cmd_unquoted_metachars, parse_arguments

from local_agent.oscompat.process import BatchArgRejected, quote_batch_args

_FORBIDDEN = '"%\r\n'
_SENSITIVE = "&|<>^ "

_SAFE_CHARS = st.one_of(
    st.sampled_from(_SENSITIVE),
    st.sampled_from(" \t\\\\\\!()'\v\f\u00a0\u3000\u00e9\u20ac\U0001f600"),
    st.characters(
        exclude_characters="\x00" + _FORBIDDEN,
        exclude_categories=["Cs"],
    ),
)
_safe_text = st.text(alphabet=_SAFE_CHARS, max_size=14)
_any_text = st.text(
    alphabet=st.one_of(_SAFE_CHARS, st.sampled_from(_FORBIDDEN)),
    max_size=14,
)
# Mostly safe pieces, so the accepting branch is exercised as often as the rejecting one.
_piece = st.one_of(_safe_text, _safe_text, _safe_text, _any_text)
_path = st.one_of(_safe_text, _safe_text, _any_text)
_args = st.lists(_piece, max_size=6)


def _expected_rejection(pieces: list[str]) -> tuple[int, str] | None:
    """The first `(position, char)` in order, or `None` when every piece is allowed."""
    for position, piece in enumerate(pieces):
        for char in piece:
            if char in _FORBIDDEN:
                return position, char
    return None


def _check_accepted(batch_path: str, args: list[str], line: str) -> None:
    pieces = [batch_path, *args]

    # Lay the line out piece by piece: `"` text extra-backslashes `"`, joined by single spaces.
    offset = 0
    joiners: set[int] = set()
    sensitive: list[int] = []
    for index, piece in enumerate(pieces):
        trailing = len(piece) - len(piece.rstrip("\\"))
        assert line[offset] == '"'
        text_start = offset + 1
        assert line[text_start : text_start + len(piece)] == piece
        sensitive.extend(text_start + i for i, char in enumerate(piece) if char in _SENSITIVE)
        offset = text_start + len(piece) + trailing  # the closing quote
        assert line[offset - trailing : offset] == "\\" * trailing
        assert line[offset] == '"'
        offset += 1
        if index < len(pieces) - 1:
            assert line[offset] == " "
            joiners.add(offset)
            offset += 1
    assert offset == len(line)

    # cmd.exe sees every operator and space from the path or an argument inside quotes.
    states = cmd_quote_states(line)
    assert all(states[i] for i in sensitive)
    assert cmd_unquoted_metachars(line) == []
    assert {i for i, char in enumerate(line) if char == " " and not states[i]} == joiners

    # A C runtime program behind `%*` gets the original arguments back.
    path_end = len(line) if not args else min(joiners)
    assert parse_arguments(line[path_end:]) == args


# Feature: local-agent-backend, Property 41: Batch-file quoting and rejection
@given(batch_path=_path, args=_args)
def test_batch_file_quoting_and_rejection(batch_path: str, args: list[str]) -> None:
    """Validates: Requirements 17.8, 17.9."""
    expected = _expected_rejection([batch_path, *args])
    if expected is not None:
        with pytest.raises(BatchArgRejected) as rejected:
            quote_batch_args(batch_path, args)
        assert (rejected.value.position, rejected.value.char) == expected
        return
    _check_accepted(batch_path, args, quote_batch_args(batch_path, args))
