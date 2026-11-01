"""Property test for approval decision parsing (Requirement 5.12).

`parse_approval_decision` reads a continuation message and decides approve, deny, or neither. The
test builds messages from random text parts, data parts and distractor parts (a `url` or `raw`
part that merely holds a keyword). It checks the result against a reference model written over
plain Python values, and checks that padding and re-casing the text parts changes nothing.
"""

from typing import Any

from a2a.types import Message, Part, Role
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from hypothesis import given
from hypothesis import strategies as st

from local_agent.config.models import ApprovalKeywords
from local_agent.orchestrator.approval_gate import Decision, parse_approval_decision

# Feature: local-agent-backend, Property 10: Approval decision parsing

# Letters whose upper/lower forms casefold back to the same text, so a re-cased keyword is still
# "the same keyword" under case-insensitive comparison.
_ALPHABET = "abcxyzABCXYZ019 -_\u00e9\u00c9\u00df"
_PADDING = st.text(alphabet=" \t\n\r\x0b\x0c\u00a0\u2003", max_size=4)

# Keywords are stored already trimmed, because the parser casefolds them without trimming.
_WORD = st.text(alphabet=_ALPHABET, min_size=1, max_size=8).filter(
    lambda w: w == w.strip() and w != ""
)


@st.composite
def _keywords(draw: st.DrawFn) -> ApprovalKeywords:
    """Approve and deny lists that stay disjoint under casefold, as the config model requires."""
    approve = draw(st.lists(_WORD, min_size=1, max_size=3))
    deny = draw(st.lists(_WORD, min_size=1, max_size=3))
    approve_folded = {w.casefold() for w in approve}
    deny = [w for w in deny if w.casefold() not in approve_folded]
    if not deny:
        deny = ["\u2603-deny"]  # cannot fold onto any word drawn from `_ALPHABET`
    return ApprovalKeywords(approve=approve, deny=deny)


def _recased(text: str, mask: list[bool]) -> str:
    """Upper-case the characters picked by `mask`, lower-case the rest."""
    return "".join(c.upper() if mask[i % len(mask)] else c.lower() for i, c in enumerate(text))


# A data part's payload, as plain Python. Most hold a `decision` that is exactly approve/deny,
# a near miss, or the wrong type, so the "exactly" rule and the type checks both get exercised.
_DECISIONS = st.one_of(
    st.sampled_from(["approve", "deny"]),
    st.sampled_from(["Approve", "DENY", " approve", "deny ", "yes", "no", ""]),
    st.sampled_from([5, True, None, ["approve"], {"decision": "deny"}]),
)
_PAYLOADS = st.one_of(
    st.fixed_dictionaries({"decision": _DECISIONS}),
    st.fixed_dictionaries({"other": st.text(max_size=5)}),
    st.fixed_dictionaries({"decision": _DECISIONS, "note": st.text(max_size=5)}),
    st.sampled_from(["approve", "deny", 3, None, ["approve"]]),  # not a JSON object
)


@st.composite
def _parts(draw: st.DrawFn, keywords: ApprovalKeywords) -> tuple[str, Any, Any]:
    """One part as `(kind, value, variant)`; text parts also carry a padded, re-cased variant."""
    kind = draw(st.sampled_from(["text", "text", "data", "url", "raw"]))
    if kind == "data":
        return kind, draw(_PAYLOADS), None
    pool = [*keywords.approve, *keywords.deny]
    text = draw(st.one_of(st.sampled_from(pool), st.text(alphabet=_ALPHABET, max_size=10)))
    if kind != "text":
        return kind, text, None
    mask = draw(st.lists(st.booleans(), min_size=1, max_size=4))
    variant = draw(_PADDING) + _recased(text, mask) + draw(_PADDING)
    return kind, text, variant


def _build(parts: list[tuple[str, Any, Any]], *, use_variant: bool) -> Message:
    built: list[Part] = []
    for kind, value, variant in parts:
        if kind == "text":
            built.append(Part(text=variant if use_variant else value))
        elif kind == "data":
            built.append(Part(data=json_format.ParseDict(value, Value())))
        elif kind == "url":
            built.append(Part(url=value))
        else:
            built.append(Part(raw=value.encode()))
    return Message(message_id="m1", role=Role.ROLE_USER, parts=built)


def _expected(parts: list[tuple[str, Any, Any]], keywords: ApprovalKeywords) -> Decision | None:
    """The three-rule definition from the design, over plain values."""
    for kind, value, _ in parts:
        if kind == "data" and isinstance(value, dict):
            decision = value.get("decision")
            if isinstance(decision, str) and decision in ("approve", "deny"):
                return Decision(decision)
    approve = {w.casefold() for w in keywords.approve}
    deny = {w.casefold() for w in keywords.deny}
    for kind, value, _ in parts:
        if kind == "text":
            folded = value.strip().casefold()
            if folded in approve:
                return Decision.APPROVE
            if folded in deny:
                return Decision.DENY
    return None


@st.composite
def _scenarios(draw: st.DrawFn) -> tuple[ApprovalKeywords, list[tuple[str, Any, Any]]]:
    keywords = draw(_keywords())
    parts = draw(st.lists(_parts(keywords), max_size=6))
    return keywords, parts


@given(_scenarios())
def test_approval_decision_parsing(
    scenario: tuple[ApprovalKeywords, list[tuple[str, Any, Any]]],
) -> None:
    """**Validates: Requirements 5.12**"""
    keywords, parts = scenario
    expected = _expected(parts, keywords)

    assert parse_approval_decision(_build(parts, use_variant=False), keywords) == expected
    # Whitespace padding and letter case on the text parts never change the result.
    assert parse_approval_decision(_build(parts, use_variant=True), keywords) == expected
