"""Structured logging and secret redaction.

Every log record is one JSON object on one line, encoded as UTF-8 and ended by exactly one LF
(15.1, 17.26). Before a record is rendered, every string in it passes through a `Redactor`, so
no configured secret reaches the log (15.5). The same `Redactor` also cleans the human-readable
error strings sent to the Chat_App and the LLM (2.6, 9.7).
"""

import json
import logging
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, BinaryIO

import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import add_log_level, format_exc_info
from structlog.typing import EventDict, FilteringBoundLogger, Processor, WrappedLogger

from local_agent.config.models import BackendConfig, LoggingConfig

__all__ = [
    "Redactor",
    "collect_secrets",
    "configure_logging",
    "get_logger",
    "render_line",
]

_WINDOW = 4
"""Secrets are matched by every substring of this length (see `Redactor`)."""

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


class Redactor:
    """Replace every part of a secret that appears in a text with a placeholder.

    We build the set of all 4-character substrings of every secret. To redact a text we slide a
    4-character window over it and mark every character covered by a window in that set. Each
    run of marked characters then becomes one placeholder. Secrets shorter than 4 characters
    have no window, so we look for them whole. The result contains no substring of 4 or more
    characters that comes from a secret. The price is occasional over-redaction of common
    4-grams, such as `Bear` from a `Bearer ...` header value. That is fine for local debug logs.
    """

    def __init__(self, secrets: Iterable[str], placeholder: str = "[REDACTED]") -> None:
        self._placeholder = placeholder
        windows: set[str] = set()
        short: set[str] = set()
        for secret in secrets:
            if not secret:
                continue
            if len(secret) < _WINDOW:
                short.add(secret)
                continue
            windows.update(secret[i : i + _WINDOW] for i in range(len(secret) - _WINDOW + 1))
        self._windows = frozenset(windows)
        self._short = frozenset(short)

    def redact(self, text: str) -> str:
        """Return `text` with every secret-derived run replaced by one placeholder. Pure."""
        if not self._windows and not self._short:
            return text
        masked = [False] * len(text)
        for i in range(len(text) - _WINDOW + 1):
            if text[i : i + _WINDOW] in self._windows:
                masked[i : i + _WINDOW] = [True] * _WINDOW
        for secret in self._short:
            start = text.find(secret)
            while start != -1:
                masked[start : start + len(secret)] = [True] * len(secret)
                start = text.find(secret, start + 1)
        if not any(masked):
            return text
        out: list[str] = []
        in_run = False
        for char, is_masked in zip(text, masked, strict=True):
            if is_masked:
                if not in_run:
                    out.append(self._placeholder)
                    in_run = True
            else:
                out.append(char)
                in_run = False
        return "".join(out)


def collect_secrets(cfg: BackendConfig) -> list[str]:
    """Gather the secret values to redact (15.5).

    That is `llm.api_key`, `server.bearer_token`, every `mcp_servers[*].headers` value, and every
    `mcp_servers[*].env` value. Empty values are skipped, and duplicates are dropped.
    """
    values: list[str] = []
    if cfg.llm.api_key is not None:
        values.append(cfg.llm.api_key.get_secret_value())
    if cfg.server.bearer_token is not None:
        values.append(cfg.server.bearer_token.get_secret_value())
    for server in cfg.mcp_servers:
        mapping = server.env if server.transport == "stdio" else server.headers
        values.extend(secret.get_secret_value() for secret in mapping.values())
    return list(dict.fromkeys(value for value in values if value))


def render_line(event: Mapping[str, Any]) -> bytes:
    """Encode one record as compact JSON, UTF-8, followed by exactly one LF (15.1, 17.26).

    `ensure_ascii=False` keeps non-ASCII text readable. JSON escaping turns CR and LF inside
    values into `\\r` and `\\n`, so the record cannot span lines. A lone surrogate cannot be
    encoded as UTF-8, so `backslashreplace` writes it as a `\\udXXX` escape instead of failing.
    """
    text = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
    return text.encode("utf-8", "backslashreplace") + b"\n"


def _redact_value(value: object, redactor: Redactor, active: frozenset[int]) -> object:
    """Redact every string inside `value`, rebuilding containers instead of mutating them."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redactor.redact(value)
    if isinstance(value, Mapping | list | tuple | set | frozenset):
        if id(value) in active:  # a container that contains itself
            return "[circular]"
        active = active | {id(value)}
        if isinstance(value, Mapping):
            return {
                (redactor.redact(k) if isinstance(k, str) else k): _redact_value(
                    v, redactor, active
                )
                for k, v in value.items()
            }
        return [_redact_value(item, redactor, active) for item in value]
    # Anything else (paths, exceptions, bytes) is rendered by `json.dumps(default=str)`. We turn
    # it into text here, so the redaction sees the same text the log would contain.
    return redactor.redact(str(value))


def _make_redact_processor(redactor: Redactor) -> Processor:
    def redact_processor(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
        redacted = _redact_value(event_dict, redactor, frozenset())
        assert isinstance(redacted, dict)  # a dict in, a dict out
        return redacted

    return redact_processor


def _timestamper(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Add a UTC ISO-8601 `timestamp` with millisecond precision."""
    event_dict["timestamp"] = datetime.now(UTC).isoformat(timespec="milliseconds")
    return event_dict


def _default_ids(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Make records outside a Task carry `null` for `context_id` and `task_id` (15.1)."""
    event_dict.setdefault("context_id", None)
    event_dict.setdefault("task_id", None)
    return event_dict


def _render_for_sink(
    _logger: WrappedLogger, _method: str, event_dict: MutableMapping[str, Any]
) -> bytes:
    """Render a record for `BytesLogger`, which appends the final LF itself.

    `render_line` already ends with one LF. If we passed that to `BytesLogger` the output would
    end with two, so we drop ours and let the sink add the single LF.
    """
    return render_line(event_dict).removesuffix(b"\n")


def configure_logging(
    cfg: LoggingConfig, redactor: Redactor, *, stream: BinaryIO | None = None
) -> None:
    """Install the structlog pipeline (15.1, 15.7, 17.26).

    Each record goes through: contextvars (`context_id`, `task_id`), level, timestamp, exception
    text, redaction of all strings, then JSON rendering. We write bytes to `stream`, which
    defaults to `sys.stderr.buffer`, so the console code page and Windows newline translation
    never touch the output. Calling this again replaces the previous setup.
    """
    processors: list[Processor] = [
        merge_contextvars,
        add_log_level,
        _timestamper,
        _default_ids,
        format_exc_info,  # turn exc_info into text so the redactor sees it
        _make_redact_processor(redactor),
        _render_for_sink,
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_LEVELS[cfg.level]),
        logger_factory=structlog.BytesLoggerFactory(
            stream if stream is not None else sys.stderr.buffer
        ),
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> FilteringBoundLogger:
    """Return a logger whose records carry `component`.

    The logger is lazy: it reads the configuration at the first call, so modules can create it at
    import time, before `configure_logging` runs. It satisfies the `warning(event, /, **fields)`
    protocol used by `mcp_client/registry.py` and `oscompat/net.py`.
    """
    logger: FilteringBoundLogger = structlog.get_logger(component=component)
    return logger
