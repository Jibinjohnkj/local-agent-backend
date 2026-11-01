"""Property test for the LLM retry policy (Requirements 10.6, 10.9, 10.10)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from hypothesis import given, settings
from hypothesis import strategies as st
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    ServiceUnavailableError,
    Timeout,
)

from local_agent.config.models import LLMConfig
from local_agent.llm.gateway import LlmGateway, LlmInterrupted, LlmRejected, LlmUnreachable

Where = Literal["call", "first_chunk"]


class _FakeClock:
    """A monotonic clock that only moves when the fake sleep is awaited."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def now_ms(self) -> int:
        return int(self.now * 1000)

    def monotonic(self) -> float:
        return self.now


# --- Exceptions the endpoint can raise --------------------------------------------------------

_RETRYABLE: dict[str, Callable[[str], BaseException]] = {
    "connection": lambda m: APIConnectionError(m, "p", "model"),
    "timeout": lambda m: Timeout(m, "model", "p"),
    "internal": lambda m: InternalServerError(m, "p", "model"),
    "unavailable": lambda m: ServiceUnavailableError(m, "p", "model"),
    "bad_gateway": lambda m: BadGatewayError(m, "p", "model"),
    "builtin_timeout": lambda m: TimeoutError(m),
    "builtin_connection": lambda m: ConnectionError(m),
}

_REJECTIONS: dict[str, Callable[[str], BaseException]] = {
    "bad_request": lambda m: BadRequestError(m, "p", "model"),
    "authentication": lambda m: AuthenticationError(m, "p", "model"),
    "not_found": lambda m: NotFoundError(m, "p", "model"),
    "unrecognized": lambda m: ValueError(m),
}

# A reason starts with an alphanumeric so LiteLLM's prefix stripping cannot touch it.
_reason_text = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9 ]{0,24}", fullmatch=True)
_where = st.sampled_from(["call", "first_chunk"])


@dataclass(frozen=True)
class _Retryable:
    kind: str
    where: Where
    message: str


@dataclass(frozen=True)
class _Reject:
    kind: str
    where: Where
    reason: str


@dataclass(frozen=True)
class _Success:
    """Delivers a first chunk, then `extra_chunks` more, then ends or fails."""

    extra_chunks: int
    tail: Literal["complete", "fail"]
    tail_error_kind: str  # any exception, retryable types included, must not be retried


_attempt = st.one_of(
    st.builds(_Retryable, st.sampled_from(sorted(_RETRYABLE)), _where, _reason_text),
    st.builds(_Retryable, st.sampled_from(sorted(_RETRYABLE)), _where, _reason_text),
    st.builds(_Reject, st.sampled_from(sorted(_REJECTIONS)), _where, _reason_text),
    st.builds(
        _Success,
        st.integers(min_value=0, max_value=3),
        st.sampled_from(["complete", "fail"]),
        st.sampled_from(sorted({*_RETRYABLE, *_REJECTIONS})),
    ),
)
# Three entries cover every attempt the gateway may make. Later entries go unused after a
# success or a rejection.
_script = st.lists(_attempt, min_size=3, max_size=3)


def _make_error(kind: str, message: str) -> BaseException:
    factory = _RETRYABLE.get(kind) or _REJECTIONS[kind]
    return factory(message)


def _chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


class _FakeResponse:
    """What `acompletion(stream=True)` returns: an async iterator of chunks."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self) -> _FakeResponse:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        return None


class _ScriptedCompletion:
    """A stand-in for `litellm.acompletion` that plays one scripted outcome per call."""

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> _FakeResponse:
        outcome = self._script[self.calls]
        self.calls += 1
        match outcome:
            case _Retryable(kind, where, message):
                error = _make_error(kind, message)
                if where == "call":
                    raise error
                return _FakeResponse([error])
            case _Reject(kind, where, reason):
                error = _make_error(kind, reason)
                if where == "call":
                    raise error
                return _FakeResponse([error])
            case _Success(extra_chunks, tail, tail_error_kind):
                items: list[Any] = [_chunk("a") for _ in range(1 + extra_chunks)]
                if tail == "fail":
                    items.append(_make_error(tail_error_kind, "stream broke"))
                return _FakeResponse(items)
        raise AssertionError(outcome)


async def _drain(gateway: LlmGateway) -> tuple[str, Any]:
    """Run one request to its end and report how it ended."""
    try:
        stream = await gateway.stream([{"role": "user", "content": "hi"}], None)
    except LlmUnreachable:
        return "unreachable", None
    except LlmRejected as exc:
        return "rejected", exc.reason
    try:
        async for _ in stream:
            pass
    except LlmInterrupted:
        return "interrupted", None
    return "complete", None


# Feature: local-agent-backend, Property 19: LLM retry policy
@settings(max_examples=200)
@given(script=_script)
def test_llm_retry_policy(script: list[Any]) -> None:
    clock = _FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    completion = _ScriptedCompletion(script)
    gateway = LlmGateway(
        LLMConfig(model="ollama_chat/test"),
        sleep=fake_sleep,
        clock=clock,
        completion=completion,
    )

    # Reference model: walk the script the way the policy describes.
    attempts = 0
    expected: tuple[str, Any] | None = None
    for outcome in script:
        attempts += 1
        if isinstance(outcome, _Retryable):
            continue
        if isinstance(outcome, _Reject):
            expected = ("rejected", outcome.reason)
        elif outcome.tail == "fail":
            expected = ("interrupted", None)
        else:
            expected = ("complete", None)
        break
    else:
        expected = ("unreachable", None)

    result = asyncio.run(_drain(gateway))

    assert result == expected
    # At most 3 attempts, and none after a rejection, a success, or a failure after the first
    # chunk: the attempt count is exactly the position of the first non-retryable outcome.
    assert completion.calls == attempts
    assert completion.calls <= 3
    # 1 s before the second attempt, 2 s before the third, and no wait after the last attempt.
    assert sleeps == [1.0, 2.0][: attempts - 1]
    assert clock.now == 1_000.0 + sum(sleeps)
