"""Unit tests for reading prompt-based function calls out of streamed text (10.8)."""

import json
from typing import Any

import pytest

from local_agent.llm.stream_assembler import StreamAssembler, parse_prompted_tool_calls

KNOWN = frozenset({"echo", "add"})


def test_a_name_and_arguments_object_is_a_call() -> None:
    reply = '{"name": "echo", "arguments": {"text": "ping-42"}}'

    assert parse_prompted_tool_calls(reply, KNOWN) == [("echo", '{"text": "ping-42"}')]


def test_parameters_is_accepted_as_the_arguments_key() -> None:
    reply = '{"name": "add", "parameters": {"a": 1, "b": 2}}'

    assert parse_prompted_tool_calls(reply, KNOWN) == [("add", '{"a": 1, "b": 2}')]


def test_arguments_given_as_a_json_string_are_kept_as_is() -> None:
    reply = json.dumps({"name": "echo", "arguments": '{"text": "x"}'})

    assert parse_prompted_tool_calls(reply, KNOWN) == [("echo", '{"text": "x"}')]


def test_missing_arguments_become_an_empty_object() -> None:
    assert parse_prompted_tool_calls('{"name": "echo"}', KNOWN) == [("echo", "{}")]


def test_non_ascii_arguments_are_not_escaped() -> None:
    reply = '{"name": "echo", "arguments": {"text": "héllo ✓"}}'

    assert parse_prompted_tool_calls(reply, KNOWN) == [("echo", '{"text": "héllo ✓"}')]


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"name": "echo", "arguments": {"text": "a"}}\n```',
        '```\n{"name": "echo", "arguments": {"text": "a"}}\n```',
        '  \n{"name": "echo", "arguments": {"text": "a"}}\n  ',
        '{"function": {"name": "echo", "arguments": {"text": "a"}}}',
    ],
)
def test_fences_whitespace_and_function_wrapper_are_accepted(reply: str) -> None:
    assert parse_prompted_tool_calls(reply, KNOWN) == [("echo", '{"text": "a"}')]


def test_a_list_of_calls_gives_several_calls_in_order() -> None:
    reply = '[{"name": "echo", "arguments": {"text": "a"}}, {"name": "add", "arguments": {}}]'

    assert parse_prompted_tool_calls(reply, KNOWN) == [("echo", '{"text": "a"}'), ("add", "{}")]


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "The echo tool returned ping-42.",
        'Sure! {"name": "echo", "arguments": {"text": "a"}}',
        '{"name": "echo", "arguments": {"text": "a"}} and then more',
        '{"name": "delete_everything", "arguments": {}}',
        '{"name": "echo", "arguments": [1, 2]}',
        '{"name": 7, "arguments": {}}',
        '{"city": "Paris"}',
        '[{"name": "echo"}, {"name": "unknown"}]',
        "[]",
        '{"name": "echo", "arguments": {"text": ',
    ],
)
def test_anything_else_stays_text(reply: str) -> None:
    assert parse_prompted_tool_calls(reply, KNOWN) is None


def test_nothing_is_parsed_without_known_tools() -> None:
    assert parse_prompted_tool_calls('{"name": "echo", "arguments": {}}', frozenset()) is None


def _feed(assembler: StreamAssembler, *pieces: str) -> None:
    for piece in pieces:
        assembler.feed({"content": piece})


def test_finish_turns_a_prompted_reply_into_a_tool_call_with_a_generated_id() -> None:
    assembler = StreamAssembler(KNOWN)
    _feed(assembler, '{"name": "echo", ', '"arguments": {"text": "ping-42"}}')

    response = assembler.finish(frozenset())

    assert response.text == ""
    (call,) = response.tool_calls
    assert call.id.startswith("call_")
    assert call.name == "echo"
    assert json.loads(call.arguments) == {"text": "ping-42"}


def test_generated_ids_avoid_existing_ids_and_each_other() -> None:
    assembler = StreamAssembler(KNOWN)
    _feed(assembler, '[{"name": "echo"}, {"name": "echo"}]')

    response = assembler.finish({"call_a"})

    ids = [call.id for call in response.tool_calls]
    assert len(set(ids)) == 2
    assert "call_a" not in ids


def test_a_reply_that_is_not_a_call_stays_text() -> None:
    assembler = StreamAssembler(KNOWN)
    _feed(assembler, "All done.")

    response = assembler.finish(frozenset())

    assert response.text == "All done."
    assert response.tool_calls == ()


def test_the_text_is_never_parsed_in_native_mode() -> None:
    assembler = StreamAssembler()
    reply = '{"name": "echo", "arguments": {"text": "a"}}'
    _feed(assembler, reply)

    response = assembler.finish(frozenset())

    assert response.text == reply
    assert response.tool_calls == ()


def test_native_tool_call_fragments_win_over_the_text() -> None:
    assembler = StreamAssembler(KNOWN)
    fragment: dict[str, Any] = {
        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "add", "arguments": "{}"}}]
    }
    assembler.feed(fragment)
    _feed(assembler, '{"name": "echo", "arguments": {}}')

    response = assembler.finish(frozenset())

    assert [call.id for call in response.tool_calls] == ["call_1"]
    assert response.text == '{"name": "echo", "arguments": {}}'
