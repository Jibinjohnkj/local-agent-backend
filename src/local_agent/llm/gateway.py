"""LiteLLM gateway: one streaming chat-completion request with retries and a finish log record.

`LlmGateway.stream` sends the request and returns an `LlmStream` once LiteLLM has delivered the
first chunk. Everything before that point is retried (10.6). After that point the gateway never
retries: any failure becomes `LlmInterrupted` (10.10). `LlmStream` yields text chunks as they
arrive (10.2) and hands the assembled response, including Tool_Calls, to the Orchestrator once
the stream is exhausted (10.3, 10.11).

Each request writes exactly one `llm_request_finished` record (15.1, 15.4, 15.6).
"""

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from collections.abc import Set as AbstractSet
from typing import Any, Final, Literal

import litellm
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    ServiceUnavailableError,
    Timeout,
)
from structlog.typing import FilteringBoundLogger

from local_agent.config.models import LLMConfig
from local_agent.llm.stream_assembler import StreamAssembler
from local_agent.logging_setup import get_logger
from local_agent.models import AssembledResponse, Clock, SystemClock

__all__ = [
    "LlmError",
    "LlmGateway",
    "LlmInterrupted",
    "LlmRejected",
    "LlmStream",
    "LlmUnreachable",
    "apply_function_calling_mode",
]

MAX_ATTEMPTS: Final = 3
"""One request plus two retries (10.6)."""

RETRY_DELAYS_S: Final = (1.0, 2.0)
"""Wait before the first and the second retry (10.6)."""

Sleep = Callable[[float], Awaitable[None]]
Completion = Callable[..., Awaitable[Any]]
Outcome = Literal["success", "error", "cancelled"]


class LlmError(Exception):
    """Base class for the failures the Orchestrator maps to a `failed` Task."""


class LlmUnreachable(LlmError):
    """All attempts failed before the first chunk with a retryable error (10.7)."""

    def __init__(self, message: str = "the local LLM is unreachable") -> None:
        super().__init__(message)


class LlmRejected(LlmError):
    """The endpoint rejected the request as invalid; `reason` is what it reported (10.9)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LlmInterrupted(LlmError):
    """The stream failed after the first chunk (10.10)."""

    def __init__(self, message: str = "the LLM response was interrupted") -> None:
        super().__init__(message)


def apply_function_calling_mode(cfg: LLMConfig) -> None:
    """Startup helper: switch LiteLLM to prompt-based function calling when needed (10.8).

    The flag is process-global. That is acceptable because the Backend talks to one model. When
    the model supports native function calling the flag is left alone.
    """
    if not cfg.native_function_calling:
        litellm.add_function_to_prompt = True


# --- Error classification ---------------------------------------------------------------------

_LITELLM_PREFIX = re.compile(r"^litellm\.\w+:\s*")

_RETRYABLE_TYPES = (
    APIConnectionError,
    Timeout,
    InternalServerError,
    ServiceUnavailableError,
    BadGatewayError,
    TimeoutError,
    ConnectionError,
)


def _client_error_status(exc: BaseException) -> int | None:
    """The first 4xx status (other than 408) on `exc` or an exception it was raised from.

    LiteLLM wraps some provider errors in `APIConnectionError` with status 500. Ollama's HTTP 400
    ("does not support tools") arrives that way; the real status stays on the chained error.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and 400 <= status < 500 and status != 408:
            return status
        current = current.__cause__ or current.__context__
    return None


def _is_retryable(exc: BaseException) -> bool:
    """True for a connection failure, a timeout, or a server-side (5xx) error (10.6).

    Any other status in the 4xx range, and anything we do not recognize, is not retried (10.9).
    `litellm.Timeout` carries status 408, so the timeout check has to come first. A 4xx status
    on the exception chain then wins over a wrapper type such as `APIConnectionError`.
    """
    if isinstance(exc, Timeout | TimeoutError):
        return True
    if _client_error_status(exc) is not None:
        return False
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _reason(exc: BaseException) -> str:
    """The endpoint's own message, without LiteLLM's `litellm.<Name>Error:` prefix."""
    message = getattr(exc, "message", None)
    text = message if isinstance(message, str) and message else str(exc)
    return _LITELLM_PREFIX.sub("", text) or type(exc).__name__


def _tool_names(tools: list[dict[str, Any]] | None) -> frozenset[str]:
    """Names of the offered OpenAI-style tool definitions."""
    names: set[str] = set()
    for tool in tools or ():
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


async def _close_quietly(response: object) -> None:
    """Release a stream we are abandoning. Failures here must not hide the real error."""
    aclose = getattr(response, "aclose", None)
    if aclose is None:
        return
    with contextlib.suppress(Exception):
        await aclose()


# --- Stream -----------------------------------------------------------------------------------


def _delta_of(chunk: Any) -> Any:
    """Return `chunk.choices[0].delta` for an object or a mapping, or None (e.g. usage chunks)."""
    choices = (
        chunk.get("choices") if isinstance(chunk, Mapping) else getattr(chunk, "choices", None)
    )
    if not choices:
        return None
    first = choices[0]
    return first.get("delta") if isinstance(first, Mapping) else getattr(first, "delta", None)


class _RequestLog:
    """Writes the single `llm_request_finished` record for one request (15.1, 15.4, 15.6)."""

    def __init__(self, logger: FilteringBoundLogger, clock: Clock, model: str) -> None:
        self._logger = logger
        self._clock = clock
        self._model = model
        self._started = clock.monotonic()
        self.attempts = 0
        self._written = False

    def finish(self, outcome: Outcome, error: BaseException | None = None) -> None:
        if self._written:
            return
        self._written = True
        elapsed_ms = max(0, int((self._clock.monotonic() - self._started) * 1000))
        fields: dict[str, Any] = {
            "duration_ms": elapsed_ms,
            "outcome": outcome,
            "model": self._model,
            "attempts": self.attempts,
        }
        if error is None:
            self._logger.info("llm_request_finished", **fields)
            return
        fields["error_type"] = type(error).__name__
        fields["error"] = str(error)
        self._logger.error("llm_request_finished", **fields)


class LlmStream(AsyncIterator[str]):
    """Text chunks of one response, in arrival order.

    Iterate to exhaustion, then call `result()`. Chunks that carry only Tool_Call fragments or
    metadata produce no text, so the iterator keeps reading until it has text to hand out.
    """

    def __init__(
        self,
        source: AsyncIterator[Any],
        first_chunk: Any,
        log: _RequestLog,
        response: object,
        prompted_tools: AbstractSet[str] = frozenset(),
    ) -> None:
        self._source = source
        self._pending: list[Any] = [first_chunk]
        self._log = log
        self._response = response
        self._assembler = StreamAssembler(prompted_tools)
        self._done = False
        self._error: LlmInterrupted | None = None
        self._result: AssembledResponse | None = None

    async def __anext__(self) -> str:
        if self._error is not None:
            raise self._error
        if self._done:
            raise StopAsyncIteration
        try:
            while True:
                chunk = self._pending.pop() if self._pending else await anext(self._source)
                text = self._assembler.feed(_delta_of(chunk))
                if text:
                    return text
        except StopAsyncIteration:
            self._done = True
            self._log.finish("success")
            raise
        except asyncio.CancelledError:
            await self._abort("cancelled")
            raise
        except Exception as exc:
            await self._abort("error", exc)
            interrupted = LlmInterrupted()
            interrupted.__cause__ = exc
            self._error = interrupted
            raise interrupted from exc

    async def aclose(self) -> None:
        """Stop reading early (for example after a cancel). Logs the request as cancelled."""
        if not self._done and self._error is None:
            await self._abort("cancelled")

    def result(self, existing_ids: AbstractSet[str] = frozenset()) -> AssembledResponse:
        """The assembled text and Tool_Calls. Valid only after the stream is exhausted.

        `existing_ids` are the Tool_Call IDs already in the Message_History; generated IDs avoid
        them (10.11). The first call fixes the result, so repeated calls return the same IDs.
        """
        if self._error is not None:
            raise self._error
        if not self._done:
            msg = "result() called before the stream was exhausted"
            raise RuntimeError(msg)
        if self._result is None:
            self._result = self._assembler.finish(existing_ids)
        return self._result

    async def _abort(self, outcome: Outcome, error: BaseException | None = None) -> None:
        self._done = True
        self._log.finish(outcome, error)
        await _close_quietly(self._response)


# --- Gateway ----------------------------------------------------------------------------------


class LlmGateway:
    """Sends chat-completion requests to the configured model through LiteLLM."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        sleep: Sleep = asyncio.sleep,
        clock: Clock | None = None,
        completion: Completion | None = None,
        logger: FilteringBoundLogger | None = None,
    ) -> None:
        """`sleep`, `clock`, and `completion` can be replaced in tests.

        When `completion` is not given, `litellm.acompletion` is looked up on each call.
        """
        self._cfg = cfg
        self._sleep = sleep
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._completion = completion
        self._logger = logger if logger is not None else get_logger("llm")

    def request_kwargs(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """The keyword arguments for `litellm.acompletion` (10.1, 10.4, 10.5)."""
        cfg = self._cfg
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "api_base": str(cfg.api_base) if cfg.api_base is not None else None,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_output_tokens,
            "timeout": cfg.request_timeout_s,
            "stream": True,
            "num_retries": 0,
        }
        if cfg.api_key is not None:
            kwargs["api_key"] = cfg.api_key.get_secret_value()
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LlmStream:
        """Start a request and return its stream once the first chunk has arrived.

        Raises `LlmUnreachable` after three failed attempts (10.7) and `LlmRejected` when the
        endpoint refuses the request (10.9). Either one is logged before it propagates.
        """
        kwargs = self.request_kwargs(messages, tools)
        log = _RequestLog(self._logger, self._clock, self._cfg.model)
        prompted = _tool_names(tools) if not self._cfg.native_function_calling else frozenset()
        try:
            return await self._open(kwargs, log, prompted)
        except LlmError as exc:
            log.finish("error", exc.__cause__ or exc)
            raise
        except asyncio.CancelledError:
            log.finish("cancelled")
            raise

    async def _open(
        self, kwargs: dict[str, Any], log: _RequestLog, prompted: AbstractSet[str]
    ) -> LlmStream:
        last_error: BaseException | None = None
        for attempt in range(MAX_ATTEMPTS):
            log.attempts = attempt + 1
            response: object = None
            try:
                completion = self._completion or litellm.acompletion
                response = await completion(**kwargs)
                source = response.__aiter__()  # type: ignore[attr-defined]
                first = await anext(source)
            except StopAsyncIteration:
                # The endpoint answered and closed the stream without a chunk. That is an empty
                # response, not a failure, so there is nothing to retry.
                return LlmStream(_EmptySource(), None, log, response, prompted)
            except Exception as exc:
                await _close_quietly(response)
                if not _is_retryable(exc):
                    raise LlmRejected(_reason(exc)) from exc
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    await self._sleep(RETRY_DELAYS_S[attempt])
                continue
            return LlmStream(source, first, log, response, prompted)
        unreachable = LlmUnreachable()
        raise unreachable from last_error


class _EmptySource(AsyncIterator[Any]):
    """An exhausted iterator, used when the endpoint sent no chunk at all."""

    async def __anext__(self) -> Any:
        raise StopAsyncIteration
