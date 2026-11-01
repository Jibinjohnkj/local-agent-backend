"""Unit tests for the LLM gateway request kwargs and chunk delivery (10.1, 10.2, 10.4, 10.5)."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import litellm
import pytest
from litellm.exceptions import APIConnectionError

from local_agent.config.models import LLMConfig
from local_agent.llm.gateway import (
    LlmGateway,
    LlmRejected,
    LlmUnreachable,
    apply_function_calling_mode,
)

_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "echo", "description": "Echo", "parameters": {"type": "object"}},
}


def _cfg(**overrides: Any) -> LLMConfig:
    return LLMConfig.model_validate({"model": "ollama_chat/llama3.1", **overrides})


def _gateway(cfg: LLMConfig, completion: Any = None) -> LlmGateway:
    # No test here retries, so the real sleep and clock are never exercised.
    return LlmGateway(cfg, completion=completion)


def _text_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


# --- request_kwargs ---------------------------------------------------------------------------


def test_defaults_are_passed_through() -> None:
    kwargs = _gateway(_cfg()).request_kwargs(_MESSAGES, None)

    assert kwargs == {
        "model": "ollama_chat/llama3.1",
        "messages": _MESSAGES,
        "api_base": None,
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 120,
        "stream": True,
        "num_retries": 0,
    }


def test_configured_values_are_passed_through() -> None:
    cfg = _cfg(
        api_base="http://localhost:11434",
        temperature=0.2,
        max_output_tokens=512,
        request_timeout_s=30,
    )

    kwargs = _gateway(cfg).request_kwargs(_MESSAGES, None)

    assert kwargs["api_base"] == str(cfg.api_base)
    assert kwargs["api_base"].startswith("http://localhost:11434")
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 512
    assert kwargs["timeout"] == 30
    assert kwargs["stream"] is True
    assert kwargs["num_retries"] == 0


def test_api_key_is_omitted_when_not_set() -> None:
    kwargs = _gateway(_cfg()).request_kwargs(_MESSAGES, None)

    assert "api_key" not in kwargs


def test_api_key_is_the_plain_secret_value_when_set() -> None:
    kwargs = _gateway(_cfg(api_key="sk-test-secret")).request_kwargs(_MESSAGES, None)

    assert kwargs["api_key"] == "sk-test-secret"
    assert isinstance(kwargs["api_key"], str)


@pytest.mark.parametrize("tools", [None, []])
def test_tools_and_tool_choice_are_omitted_when_there_are_no_tools(
    tools: list[dict[str, Any]] | None,
) -> None:
    kwargs = _gateway(_cfg()).request_kwargs(_MESSAGES, tools)

    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_tools_and_auto_tool_choice_are_sent_when_tools_exist() -> None:
    kwargs = _gateway(_cfg()).request_kwargs(_MESSAGES, [_TOOL])

    assert kwargs["tools"] == [_TOOL]
    assert kwargs["tool_choice"] == "auto"


def test_function_calling_mode_does_not_change_the_request_kwargs() -> None:
    with_native = _gateway(_cfg(native_function_calling=True)).request_kwargs(_MESSAGES, [_TOOL])
    without_native = _gateway(_cfg(native_function_calling=False)).request_kwargs(
        _MESSAGES, [_TOOL]
    )

    assert with_native == without_native


# --- apply_function_calling_mode --------------------------------------------------------------


def test_prompt_based_function_calling_is_enabled_when_native_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "add_function_to_prompt", False)

    apply_function_calling_mode(_cfg(native_function_calling=False))

    assert litellm.add_function_to_prompt is True


def test_function_calling_flag_is_left_alone_when_native_is_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "add_function_to_prompt", False)

    apply_function_calling_mode(_cfg(native_function_calling=True))

    assert litellm.add_function_to_prompt is False


# --- stream: kwargs reach LiteLLM and chunks are not buffered ---------------------------------


async def test_stream_calls_completion_with_the_request_kwargs() -> None:
    received: list[dict[str, Any]] = []

    async def completion(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        received.append(kwargs)

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            yield _text_chunk("hi")

        return chunks()

    gateway = _gateway(_cfg(api_key="sk-test-secret"), completion)
    stream = await gateway.stream(_MESSAGES, [_TOOL])
    assert [text async for text in stream] == ["hi"]

    assert received == [gateway.request_kwargs(_MESSAGES, [_TOOL])]


async def test_each_text_chunk_is_yielded_before_the_next_one_arrives() -> None:
    release_second = asyncio.Event()
    release_third = asyncio.Event()

    async def chunks() -> AsyncIterator[dict[str, Any]]:
        yield _text_chunk("one ")
        await release_second.wait()
        yield _text_chunk("two ")
        await release_third.wait()
        yield _text_chunk("three")

    async def completion(**_: Any) -> AsyncIterator[dict[str, Any]]:
        return chunks()

    stream = await _gateway(_cfg(), completion).stream(_MESSAGES, None)

    # The source is blocked on the second chunk, so these only succeed if nothing is buffered.
    assert await asyncio.wait_for(anext(stream), timeout=1) == "one "
    release_second.set()
    assert await asyncio.wait_for(anext(stream), timeout=1) == "two "
    release_third.set()
    assert await asyncio.wait_for(anext(stream), timeout=1) == "three"

    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert stream.result().text == "one two three"


async def test_chunks_without_text_are_skipped_and_result_holds_the_tool_calls() -> None:
    tool_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "echo", "arguments": '{"a":1}'},
                        }
                    ]
                }
            }
        ]
    }

    async def completion(**_: Any) -> AsyncIterator[dict[str, Any]]:
        async def chunks() -> AsyncIterator[dict[str, Any]]:
            yield {"choices": []}
            yield _text_chunk("calling")
            yield tool_chunk

        return chunks()

    stream = await _gateway(_cfg(), completion).stream(_MESSAGES, [_TOOL])
    texts = [text async for text in stream]

    assert texts == ["calling"]
    result = stream.result()
    assert result.text == "calling"
    assert [(call.id, call.name, call.arguments) for call in result.tool_calls] == [
        ("call_1", "echo", '{"a":1}')
    ]


# --- prompt-based function calling through the stream (10.8) ----------------------------------

_PROMPTED_REPLY = '{"name": "echo", "arguments": {"text": "ping-42"}}'


async def _stream_of(cfg: LLMConfig, tools: list[dict[str, Any]] | None, *texts: str) -> Any:
    async def completion(**_: Any) -> AsyncIterator[dict[str, Any]]:
        async def chunks() -> AsyncIterator[dict[str, Any]]:
            for text in texts:
                yield _text_chunk(text)

        return chunks()

    stream = await _gateway(cfg, completion).stream(_MESSAGES, tools)
    texts_out = [text async for text in stream]
    assert texts_out == list(texts)  # the text is still forwarded as it arrives (10.2)
    return stream.result()


async def test_a_prompted_reply_becomes_a_tool_call_when_native_calling_is_off() -> None:
    cfg = _cfg(native_function_calling=False)

    result = await _stream_of(cfg, [_TOOL], _PROMPTED_REPLY[:20], _PROMPTED_REPLY[20:])

    (call,) = result.tool_calls
    assert (call.name, call.arguments) == ("echo", '{"text": "ping-42"}')
    assert call.id


async def test_a_prompted_reply_for_an_unknown_tool_stays_text() -> None:
    cfg = _cfg(native_function_calling=False)
    reply = '{"name": "other", "arguments": {}}'

    result = await _stream_of(cfg, [_TOOL], reply)

    assert result.tool_calls == ()
    assert result.text == reply


async def test_a_prompted_reply_is_not_parsed_without_offered_tools() -> None:
    cfg = _cfg(native_function_calling=False)

    result = await _stream_of(cfg, None, _PROMPTED_REPLY)

    assert result.tool_calls == ()
    assert result.text == _PROMPTED_REPLY


async def test_a_prompted_reply_stays_text_when_native_calling_is_on() -> None:
    result = await _stream_of(_cfg(native_function_calling=True), [_TOOL], _PROMPTED_REPLY)

    assert result.tool_calls == ()
    assert result.text == _PROMPTED_REPLY


# --- error classification (10.6, 10.9) --------------------------------------------------------


class _ProviderError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _wrapped_connection_error(provider_status: int) -> Exception:
    """What LiteLLM raises for Ollama: APIConnectionError (status 500) over the provider error."""
    wrapper = APIConnectionError(
        message="Ollama_chatException - does not support tools",
        llm_provider="ollama_chat",
        model="gemma3n:latest",
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )
    wrapper.__context__ = _ProviderError("does not support tools", provider_status)
    return wrapper


async def test_a_provider_400_wrapped_in_a_connection_error_is_rejected_without_retry() -> None:
    attempts = 0

    async def completion(**_: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise _wrapped_connection_error(400)

    with pytest.raises(LlmRejected) as caught:
        await _gateway(_cfg(), completion).stream(_MESSAGES, [_TOOL])

    assert attempts == 1
    assert "does not support tools" in caught.value.reason


async def test_a_plain_connection_error_is_still_retried() -> None:
    attempts = 0
    sleeps: list[float] = []

    async def completion(**_: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise APIConnectionError(
            message="connection refused",
            llm_provider="ollama_chat",
            model="m",
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    gateway = LlmGateway(_cfg(), sleep=sleep, completion=completion)
    with pytest.raises(LlmUnreachable):
        await gateway.stream(_MESSAGES, None)

    assert attempts == 3
    assert sleeps == [1.0, 2.0]
