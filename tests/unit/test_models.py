"""Unit tests for the shared runtime models, SystemClock, and replace_lone_surrogates."""

import dataclasses
import time

import pytest
from a2a.types import TaskState
from hypothesis import given
from hypothesis import strategies as st

from local_agent.models import (
    ArtifactRecord,
    AssistantMessage,
    ContextRecord,
    SystemClock,
    SystemMessage,
    TaskRecord,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
    replace_lone_surrogates,
)


def test_message_to_openai_shapes() -> None:
    assert SystemMessage("be brief").to_openai() == {"role": "system", "content": "be brief"}
    assert UserMessage("hi").to_openai() == {"role": "user", "content": "hi"}
    assert ToolMessage("c1", "ok").to_openai() == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "ok",
    }


def test_assistant_message_to_openai_keeps_none_and_empty_distinct() -> None:
    assert AssistantMessage(None).to_openai() == {"role": "assistant", "content": None}
    assert AssistantMessage("").to_openai() == {"role": "assistant", "content": ""}


def test_assistant_message_to_openai_keeps_raw_arguments() -> None:
    call = ToolCall(id="c1", name="fs__read", arguments='{"path": not json')
    assert AssistantMessage(None, (call,)).to_openai() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "fs__read", "arguments": '{"path": not json'},
            }
        ],
    }


def test_tool_result_as_message() -> None:
    result = ToolResult("c1", "fs__read", "fs", "read", "text", "success")
    assert result.as_message() == ToolMessage(tool_call_id="c1", content="text")


def test_frozen_dataclasses_reject_assignment() -> None:
    msg = UserMessage("x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.content = "y"  # type: ignore[misc]


def test_context_record_is_mutable_and_uses_integer_milliseconds() -> None:
    ctx = ContextRecord("ctx", None, [], created_at_ms=1, last_activity_ms=2)
    ctx.history.append(UserMessage("hi"))
    ctx.last_activity_ms = 5
    assert ctx.history == [UserMessage("hi")]
    assert isinstance(ctx.created_at_ms, int)


def _task(artifacts: tuple[ArtifactRecord, ...]) -> TaskRecord:
    return TaskRecord(
        task_id="t",
        context_id="c",
        state=TaskState.TASK_STATE_WORKING,
        status_message=None,
        metadata={},
        created_at_ms=1,
        updated_at_ms=2,
        artifacts=artifacts,
    )


def test_task_record_accepts_ordered_artifacts() -> None:
    a = ArtifactRecord("response-1", 0, "a", True)
    b = ArtifactRecord("response-2", 1, "b", False)
    assert _task((a, b)).artifacts == (a, b)


def test_task_record_rejects_unordered_artifacts() -> None:
    a = ArtifactRecord("response-1", 1, "a", True)
    b = ArtifactRecord("response-2", 0, "b", True)
    with pytest.raises(ValueError, match="ordered"):
        _task((a, b))


def test_system_clock_reports_epoch_milliseconds_and_monotonic_seconds() -> None:
    clock = SystemClock()
    before = time.time_ns() // 1_000_000
    now = clock.now_ms()
    after = time.time_ns() // 1_000_000
    assert isinstance(now, int)
    assert before <= now <= after
    first = clock.monotonic()
    assert clock.monotonic() >= first


def test_replace_lone_surrogates_examples() -> None:
    assert replace_lone_surrogates("a\ud800b") == "a\ufffdb"
    assert replace_lone_surrogates("\udc00\ud83d") == "\ufffd\ufffd"
    assert replace_lone_surrogates("") == ""
    assert replace_lone_surrogates("caf\u00e9 \U0001f600\r\n") == "caf\u00e9 \U0001f600\r\n"


@given(st.text(alphabet=st.characters(codec="utf-8")))
def test_valid_text_is_unchanged(text: str) -> None:
    assert replace_lone_surrogates(text) == text


@given(st.text(alphabet=st.characters(categories=["Cs"]), min_size=1))
def test_every_surrogate_becomes_one_replacement_character(text: str) -> None:
    assert replace_lone_surrogates(text) == "\ufffd" * len(text)
    replace_lone_surrogates(text).encode("utf-8")
