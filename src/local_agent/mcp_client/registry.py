"""Tool_Registry: LLM-facing tool names, Permission_Policy matching, and immutable snapshots.

The module has two layers.

The pure layer (``sanitize``, ``assign_names``, ``compile_pattern``, ``is_protected``) turns
``(server, tool)`` pairs into unique LLM-facing names and decides which tools are protected.
It does no I/O, so the property tests can call it directly.

The stateful layer (``ToolRegistry``) remembers each server's accepted tools. Whenever a server
is added, replaced or removed, it rebuilds every name from scratch in Backend_Config order and
swaps in a brand-new ``RegistrySnapshot`` with one reference assignment. A request that holds an
older snapshot keeps resolving names against exactly what the LLM saw, so a reconnect in the
middle of a turn cannot misroute a Tool_Call (7.4, 7.12).
"""

import logging
import re
import threading
from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Protocol

from local_agent.mcp_client.schema_translator import SchemaRejected, translate
from local_agent.models import McpToolInfo, RegisteredTool

MAX_NAME_LENGTH = 64
"""Longest LLM-facing tool name accepted by OpenAI-compatible APIs (7.3)."""

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_PLACEHOLDER_NAME = "tool"
"""Name used to test-translate a tool before its real LLM-facing name exists."""


# --- Pure naming and matching -----------------------------------------------------------------


def sanitize(s: str) -> str:
    """Replace every character outside ``A-Z a-z 0-9 _ -`` with ``_`` (7.3).

    Each Unicode code point becomes exactly one underscore, so the result has the same length
    (in code points) as the input.
    """
    return _UNSAFE_NAME_CHARS.sub("_", s)


def assign_names(entries: Sequence[tuple[str, str]]) -> list[str]:
    """Give each ``(server, tool)`` pair a unique LLM-facing name, in input order (7.3, 7.7, 7.12).

    The base name is ``sanitize(f"{server}__{tool}")[:64]``. The first pair to claim a base keeps
    it unchanged. A later pair that collides tries ``_2``, ``_3``, ... and shortens the base so
    the suffixed name still fits in 64 characters. The function depends only on its input, so the
    same ordered entries always produce the same names.
    """
    used: set[str] = set()
    names: list[str] = []
    for server, tool in entries:
        base = sanitize(f"{server}__{tool}")[:MAX_NAME_LENGTH]
        name = base
        k = 2
        while name in used:
            suffix = f"_{k}"
            name = base[: MAX_NAME_LENGTH - len(suffix)] + suffix
            k += 1
        used.add(name)
        names.append(name)
    return names


@lru_cache(maxsize=1024)
def compile_pattern(p: str) -> re.Pattern[str]:
    """Translate a Permission_Policy pattern into a regex (5.1).

    Only ``*`` (any sequence, including empty) and ``?`` (any single character) are special.
    Every other character, including ``[``, ``]`` and regex metacharacters, is matched
    literally. ``DOTALL`` makes ``.`` match newlines too, so ``*`` and ``?`` really mean "any
    character". Callers use ``fullmatch``, and matching is case-sensitive.

    Compiled patterns are immutable, so results are cached.
    """
    parts: list[str] = []
    for ch in p:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts), re.DOTALL)


def is_protected(policy: Mapping[str, Sequence[str]], server: str, tool: str) -> bool:
    """True when ``tool`` matches at least one pattern listed for ``server`` (5.2).

    Patterns listed for other servers never apply, and ``tool`` is the original MCP_Tool name,
    not the LLM-facing one (5.1).
    """
    return any(compile_pattern(pattern).fullmatch(tool) for pattern in policy.get(server, ()))


# --- Warning logger ---------------------------------------------------------------------------


class WarningLogger(Protocol):
    """Anything with a structlog-style ``warning(event, **fields)`` method.

    ``logging_setup.get_logger(component)`` (task 4.1) returns a structlog bound logger, which
    satisfies this protocol. Until that exists, ``ToolRegistry`` falls back to a stdlib adapter.
    """

    def warning(self, event: str, /, **fields: object) -> None: ...


class _StdlibWarningLogger:
    """Adapts a stdlib ``logging.Logger`` to the ``WarningLogger`` protocol.

    Fields are appended as ``key=value`` pairs after the event name and also passed through
    ``extra`` so a handler can read them.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def warning(self, event: str, /, **fields: object) -> None:
        rendered = " ".join(f"{key}={value!r}" for key, value in fields.items())
        self._logger.warning(
            "%s %s", event, rendered, extra={"event": event, "fields": dict(fields)}
        )


# --- Immutable snapshot -----------------------------------------------------------------------


def _is_available(server: str, availability: Collection[str] | Mapping[str, bool] | None) -> bool:
    if availability is None:
        return True
    if isinstance(availability, Mapping):
        return bool(availability.get(server, False))
    return server in availability


@dataclass(frozen=True)
class RegistrySnapshot:
    """An immutable view of the registry, used by one request (one LLM turn).

    ``tools`` is ordered by server position in the Backend_Config, then by position in that
    server's tool list. Nothing in a snapshot changes after construction, so the names the LLM
    saw are the names later used to resolve its Tool_Calls.

    ``availability`` arguments accept a ``{server: bool}`` mapping (such as
    ``McpManager.availability()``), a collection of available server names, or ``None`` for "every
    server in the snapshot". The ``ToolRegistry`` only holds tools of servers whose supervisor
    registered them, so ``None`` is correct when no extra filtering is needed.
    """

    tools: tuple[RegisteredTool, ...] = ()
    _by_name: Mapping[str, RegisteredTool] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name = {tool.llm_name: tool for tool in self.tools}
        if len(by_name) != len(self.tools):
            raise ValueError("RegistrySnapshot tools must have unique llm_name values")
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    def get(self, llm_name: str) -> RegisteredTool | None:
        """The registered tool behind ``llm_name``, or ``None`` when the name is unknown."""
        return self._by_name.get(llm_name)

    def resolve(self, llm_name: str) -> tuple[str, str] | None:
        """``(server, original_tool)`` for an LLM-facing name, or ``None`` (7.4, 8.1)."""
        tool = self._by_name.get(llm_name)
        return None if tool is None else (tool.server, tool.tool)

    def is_protected(self, llm_name: str) -> bool:
        """True when the named tool is a Protected_Tool; unknown names are not protected."""
        tool = self._by_name.get(llm_name)
        return tool is not None and tool.protected

    def openai_tools(
        self, availability: Collection[str] | Mapping[str, bool] | None = None
    ) -> list[dict[str, Any]]:
        """OpenAI function definitions for tools of available servers only (7.9).

        Tools of unavailable servers are omitted. The definitions are deep copies, so a caller
        that edits them cannot change the snapshot.
        """
        return [
            deepcopy(tool.openai_def)
            for tool in self.tools
            if _is_available(tool.server, availability)
        ]

    def tools_by_server(
        self, availability: Collection[str] | Mapping[str, bool] | None = None
    ) -> dict[str, tuple[RegisteredTool, ...]]:
        """Registered tools grouped by server, in config order, for available servers only.

        The prompt builder uses this to write one ``### <server>`` section per server (12.2).
        Servers with no tools do not appear.
        """
        grouped: dict[str, list[RegisteredTool]] = {}
        for tool in self.tools:
            if _is_available(tool.server, availability):
                grouped.setdefault(tool.server, []).append(tool)
        return {server: tuple(tools) for server, tools in grouped.items()}


# --- Registry ---------------------------------------------------------------------------------


class ToolRegistry:
    """Holds each server's accepted tools and publishes ``RegistrySnapshot`` objects.

    ``server_order`` is the server names in Backend_Config order. It fixes which tool claims an
    unsuffixed name when two collide (7.7) and keeps names identical across restarts (7.12).
    A server missing from ``server_order`` sorts after the configured ones, in the order it was
    first registered. ``policy`` is the Permission_Policy (server name to patterns).

    ``replace_server`` and ``remove_server`` are synchronous and meant for the event loop thread.
    A lock still guards the rebuild, and readers call ``snapshot()``, which is one attribute read.
    """

    def __init__(
        self,
        server_order: Sequence[str],
        policy: Mapping[str, Sequence[str]] | None = None,
        *,
        logger: WarningLogger | None = None,
    ) -> None:
        self._order: dict[str, int] = {}
        for name in server_order:
            self._order.setdefault(name, len(self._order))
        self._policy: dict[str, tuple[str, ...]] = {
            server: tuple(patterns) for server, patterns in (policy or {}).items()
        }
        self._log: WarningLogger = logger or _StdlibWarningLogger(
            logging.getLogger("local_agent.mcp_client.registry")
        )
        self._accepted: dict[str, tuple[McpToolInfo, ...]] = {}
        self._snapshot = RegistrySnapshot()
        self._lock = threading.Lock()

    def snapshot(self) -> RegistrySnapshot:
        """The current immutable view. Hold it for the whole request."""
        return self._snapshot

    def replace_server(self, server: str, tools: Iterable[McpToolInfo]) -> None:
        """Register ``server``'s tools, replacing any it had before (7.1, 7.11).

        A tool whose input schema is rejected is left out and a warning naming the server and
        tool is logged. The remaining tools are still registered (7.8).
        """
        accepted: list[McpToolInfo] = []
        for tool in tools:
            outcome = translate(server, tool, _PLACEHOLDER_NAME)
            if isinstance(outcome, SchemaRejected):
                self._log.warning(
                    "mcp_tool_schema_rejected",
                    server=outcome.server,
                    tool=outcome.tool,
                    reason=outcome.reason,
                )
                continue
            accepted.append(tool)
        with self._lock:
            self._accepted[server] = tuple(accepted)
            self._snapshot = self._build_snapshot()

    def remove_server(self, server: str) -> None:
        """Drop all of ``server``'s tools, for example when it becomes unavailable (6.8)."""
        with self._lock:
            if self._accepted.pop(server, None) is None:
                return
            self._snapshot = self._build_snapshot()

    def _config_ordered_servers(self) -> list[str]:
        """Registered servers in config order, then any unconfigured ones in insertion order."""
        configured = sorted(
            (s for s in self._accepted if s in self._order), key=self._order.__getitem__
        )
        unknown = [s for s in self._accepted if s not in self._order]
        return configured + unknown

    def _build_snapshot(self) -> RegistrySnapshot:
        """Assign every name afresh from the ordered server list and build the new snapshot."""
        entries = [
            (server, tool)
            for server in self._config_ordered_servers()
            for tool in self._accepted[server]
        ]
        names = assign_names([(server, tool.name) for server, tool in entries])
        registered: list[RegisteredTool] = []
        for (server, tool), llm_name in zip(entries, names, strict=True):
            definition = translate(server, tool, llm_name)
            if isinstance(definition, SchemaRejected):
                # replace_server already screened this tool, and translate is deterministic.
                raise RuntimeError(
                    f"tool {tool.name!r} of server {server!r} passed screening but was rejected: "
                    f"{definition.reason}"
                )
            registered.append(
                RegisteredTool(
                    llm_name=llm_name,
                    server=server,
                    tool=tool.name,
                    description=tool.description,
                    openai_def=definition,
                    protected=is_protected(self._policy, server, tool.name),
                )
            )
        return RegistrySnapshot(tuple(registered))
