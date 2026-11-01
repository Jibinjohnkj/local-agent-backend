"""The Backend's A2A error contract (1.11, 1.12, 1.14, 1.15, 2.14, 11.7, 18.30).

Two Backend-defined errors, `ContextBusyError` and `TaskNotAwaitingInputError`, are
`HostContractError`s. The SDK's generic JSON-RPC mapping wraps `data` in `ErrorInfo` details, so
`HostJsonRpcDispatcher` (`dispatch.py`) renders these two itself from `jsonrpc_error()`. The REST
binding uses the SDK mapping table, which `install_error_mappings()` extends.

`accepts_version` is the pure check behind the `A2A-Version` gate.
"""

from __future__ import annotations

from typing import Any, ClassVar

from packaging.version import InvalidVersion, Version

from local_agent.a2a_server.a2a_compat import (
    A2A_ERROR_MAPPING,
    A2A_ERROR_REASONS,
    A2AError,
    ErrorMapping,
    InternalError,
    TaskState,
)

CONTEXT_BUSY_MESSAGE = "Context is busy"
NOT_AWAITING_INPUT_MESSAGE = "Task is not awaiting input"
STORAGE_WRITE_MESSAGE = "Storage could not be written"


class HostContractError(A2AError):
    """Base of the errors whose JSON-RPC `data` is a plain object, not SDK `ErrorInfo` details."""

    jsonrpc_code: ClassVar[int]

    def jsonrpc_error(self) -> dict[str, Any]:
        """The exact JSON-RPC `error` member."""
        return {"code": self.jsonrpc_code, "message": self.message, "data": self.data}


class ContextBusyError(HostContractError):
    """A new message arrived for a Context that has an Active_Task (1.14, 11.7)."""

    # Implementation-defined server error range, outside the A2A codes -32001...-32009.
    jsonrpc_code = -32050

    def __init__(self, context_id: str, active_task_id: str) -> None:
        super().__init__(
            CONTEXT_BUSY_MESSAGE,
            {"context_id": context_id, "active_task_id": active_task_id},
        )


class TaskNotAwaitingInputError(HostContractError):
    """A continuation arrived for a Task in `submitted` or `working` (1.15, 2.14)."""

    jsonrpc_code = -32602

    def __init__(self, task_id: str, state: TaskState | int) -> None:
        super().__init__(
            NOT_AWAITING_INPUT_MESSAGE,
            {"task_id": task_id, "state": TaskState.Name(state)},
        )


class StorageWriteError(InternalError):
    """The Task could not be committed at creation (18.30). JSON-RPC `-32603`, REST 500."""

    jsonrpc_code: ClassVar[int] = -32603

    def __init__(self, message: str | None = None, data: dict[str, Any] | None = None) -> None:
        super().__init__(message or STORAGE_WRITE_MESSAGE, data)


def install_error_mappings() -> None:
    """Register the Backend's error types in the SDK's REST tables. Idempotent.

    `ErrorInfo.metadata` holds only strings, and the `data` of both contract errors is built from
    strings. JSON-RPC needs no registration: `HostJsonRpcDispatcher` renders the contract errors,
    and the SDK already maps any unregistered type, `StorageWriteError` included, to `-32603`.
    """
    mappings = {
        ContextBusyError: ErrorMapping(409, "ABORTED", "CONTEXT_BUSY"),
        TaskNotAwaitingInputError: ErrorMapping(400, "INVALID_ARGUMENT", "INVALID_PARAMS"),
        StorageWriteError: ErrorMapping(500, "INTERNAL", "INTERNAL_ERROR"),
    }
    for error_type, mapping in mappings.items():
        A2A_ERROR_MAPPING[error_type] = mapping
        A2A_ERROR_REASONS[error_type] = mapping.reason


def accepts_version(header: str) -> bool:
    """True iff `header.strip()` is version 1.0, with optional trailing zero components (1.11).

    Pure. `1.0` and `1.0.0` pass. `1.1`, `1.0.1`, `2.0`, `0.3`, `1`, the empty string and
    unparsable text do not. Only the release segment is compared. A missing header never reaches
    this function (1.13).
    """
    try:
        release = Version(header.strip()).release
    except InvalidVersion:
        return False
    return len(release) >= 2 and release[:2] == (1, 0) and not any(release[2:])
