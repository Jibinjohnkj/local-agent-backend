"""The single place that names `a2a-sdk` classes used by the error contract and the handler.

The pinned release is the 1.2.x line. Other modules in `a2a_server` import these names from here
instead of from `a2a.*`, so a module move or rename in a later SDK release touches this file only
(design.md, "A2A interface contract"). Nothing here has behaviour of its own.
"""

from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.jsonrpc_models import JSONRPCError
from a2a.server.request_handlers import DefaultRequestHandler, DefaultRequestHandlerV2
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
from a2a.server.request_handlers.response_helpers import build_error_response
from a2a.server.tasks import TaskStore
from a2a.types import TaskState
from a2a.utils.errors import (
    A2A_ERROR_MAPPING,
    A2A_ERROR_REASONS,
    A2AError,
    ErrorMapping,
    InternalError,
    InvalidParamsError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
    VersionNotSupportedError,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0
from a2a.utils.proto_utils import to_stream_response, validate_proto_required_fields
from a2a.utils.version_validator import validate_version

__all__ = [
    "A2A_ERROR_MAPPING",
    "A2A_ERROR_REASONS",
    "A2AError",
    "AgentExecutor",
    "DefaultRequestHandler",
    "DefaultRequestHandlerV2",
    "ErrorMapping",
    "EventQueue",
    "PROTOCOL_VERSION_1_0",
    "InternalError",
    "InvalidParamsError",
    "JSONRPCError",
    "JsonRpcDispatcher",
    "RequestContext",
    "ServerCallContext",
    "TaskNotCancelableError",
    "TaskNotFoundError",
    "TaskState",
    "TaskStore",
    "UnsupportedOperationError",
    "VersionNotSupportedError",
    "build_error_response",
    "to_stream_response",
    "validate_version",
    "validate_proto_required_fields",
]
