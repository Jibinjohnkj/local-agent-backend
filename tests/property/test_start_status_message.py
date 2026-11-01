"""Property test for the Tool start status message (Requirement 4.2).

`start_status_message(tool, args)` names the tool and, when an argument looks like a path, file,
URI or URL, shows that argument's value cut to 120 characters. The test generates arbitrary tool
names and argument objects, then checks the message against the property's wording.

"No other argument value" is hard to check with free text, because a short value such as "a" is
always a substring of "Calling". So every argument value that must stay out of the message starts
with its own marker, `V<index>§`. The generated text never contains `§`, which means a marker in the
message can only have come from that argument.
"""

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.orchestrator.tool_dispatch import start_status_message

_MARKERS = ("path", "file", "uri", "url")
_LIMIT = 120
_SEPARATOR = "§"
_WORDING_ALLOWANCE = 32  # room for "Calling ", " (", ")..." and a little slack

# Text that can never collide with a marker such as `V3§`.
_TEXT = st.text(max_size=200).filter(lambda text: _SEPARATOR not in text)
_TOOL_NAME = st.text(min_size=1, max_size=40).filter(lambda text: _SEPARATOR not in text)

# Keys that must match the rule: a marker word with arbitrary letter case and arbitrary padding.
_PATH_KEY_PADDING = st.text(alphabet="abc_-.", max_size=4)
_PATH_KEY = st.builds(
    lambda before, word, after, upper: (
        (before + word + after).upper() if upper else (before + word + after)
    ),
    _PATH_KEY_PADDING,
    st.sampled_from(_MARKERS),
    _PATH_KEY_PADDING,
    st.booleans(),
)
# Keys that must not match the rule.
_OTHER_KEY = st.text(alphabet="abcdxyz_-.0123456789", min_size=1, max_size=8).filter(
    lambda key: not any(marker in key.lower() for marker in _MARKERS)
)
_KEY = st.one_of(_PATH_KEY, _OTHER_KEY)

# Value shapes. Strings and containers hold the marker; numbers and None cannot leak text.
_KINDS = st.sampled_from(["str", "str", "str", "list", "dict", "int", "none"])


def _value(kind: str, index: int, text: str) -> Any:
    marker = f"V{index}{_SEPARATOR}"
    if kind == "str":
        return marker + text
    if kind == "list":
        return [marker + text]
    if kind == "dict":
        return {"nested": marker + text}
    if kind == "int":
        return index
    return None


@st.composite
def _arguments(draw: st.DrawFn) -> dict[str, Any]:
    """A JSON object whose keys are unique and whose order is the generated order."""
    keys = draw(st.lists(_KEY, max_size=8, unique=True))
    args: dict[str, Any] = {}
    for index, key in enumerate(keys):
        args[key] = _value(draw(_KINDS), index, draw(_TEXT))
    return args


def _contains_marker(message: str, value: Any, index: int) -> bool:
    """True when the message carries the marker that `_value` put into this argument."""
    if value is None or isinstance(value, int):
        return False
    return f"V{index}{_SEPARATOR}" in message


# Feature: local-agent-backend, Property 11: Tool start status message
@settings(max_examples=200)
@given(tool=_TOOL_NAME, args=_arguments())
def test_start_status_message(tool: str, args: dict[str, Any]) -> None:
    """**Validates: Requirements 4.2**"""
    message = start_status_message(tool, args)

    assert tool in message

    selected: int | None = None
    for index, (key, value) in enumerate(args.items()):
        if isinstance(value, str) and any(marker in key.lower() for marker in _MARKERS):
            selected = index
            break

    for index, value in enumerate(args.values()):
        if index == selected:
            continue
        assert not _contains_marker(message, value, index)

    if selected is None:
        return

    chosen = list(args.values())[selected]
    assert chosen[:_LIMIT] in message
    # Besides the tool name and the value, only a short fixed wording is allowed. A value longer
    # than the limit therefore cannot get past its 120-character cut.
    assert len(message) <= len(tool) + _LIMIT + _WORDING_ALLOWANCE
