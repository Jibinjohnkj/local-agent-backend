"""JSON-RPC dispatcher that renders the Backend's contract errors exactly (1.14-1.18, 2.14).

The SDK maps errors by exact type and wraps `data` in `ErrorInfo` details, so no handler-side
exception can produce `data: {"context_id": ...}`. `HostRequestHandler._admit` stores a
`HostContractError` in `context.state["host_contract_error"]` and raises it:

- Streaming (`SendStreamingMessage`, `SubscribeToTask`): `_process_streaming_request` keeps the SDK
  `validate_version` decorator, adds the strict 1.0 check, and returns the `_sse_events` async
  generator without awaiting the handler. Admission and start-up therefore run on the first
  iteration, after the `200 text/event-stream` headers are committed, and every rejection or later
  failure is one `event: error` SSE event (1.16, 1.17). Reading stops at the final event, so a
  `failed` Task is never followed by an error event.
- Non-streaming (`SendMessage`): the SDK builds its generic error dict and passes it to
  `_create_response(call_context, ...)`. The override substitutes the contract body when
  `call_context.state["host_contract_error"]` holds a `HostContractError`.
- Pre-dispatch errors keep the SDK's JSON form (1.18).

Both hooks are private SDK API of the pinned 1.2.x release, guarded by the contract tests.

Fallback if a later SDK release removes either hook: take over `handle_requests` for the two
streaming methods only, run the SDK's pre-dispatch steps, then call `_sse_events` directly.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Any, Final

from google.protobuf.json_format import MessageToDict
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from local_agent.a2a_server.a2a_compat import (
    PROTOCOL_VERSION_1_0,
    A2AError,
    InternalError,
    JsonRpcDispatcher,
    ServerCallContext,
    UnsupportedOperationError,
    VersionNotSupportedError,
    build_error_response,
    to_stream_response,
    validate_version,
)
from local_agent.a2a_server.errors import HostContractError, accepts_version
from local_agent.a2a_server.events import STREAM_CLOSING_STATES, from_sdk

logger = logging.getLogger(__name__)

HOST_CONTRACT_ERROR_KEY = "host_contract_error"

_METHOD_NOT_FOUND_CODE: Final = -32601
_SDK_METHOD_NOT_FOUND_MESSAGE: Final = "Method not found"
_MAX_REPORTED_METHOD_CHARS: Final = 100

_VERSION_HEADER: Final = "a2a-version"
_MAX_REPORTED_VERSION_CHARS: Final = 64
_INTERNAL_ERROR_MESSAGE: Final = "Internal error"


def jsonrpc_error_response(request_id: str | int | None, error: BaseException) -> dict[str, Any]:
    """The JSON-RPC error response for `error`, as the item of an SSE `error` event (1.14-1.17).

    A `HostContractError` keeps its exact message and plain `data`. Any other `A2AError` gets the
    SDK body. Anything else becomes `-32603` "Internal error" with no detail.
    """
    if isinstance(error, HostContractError):
        return contract_error_body(request_id, error)
    if isinstance(error, A2AError):
        return build_error_response(request_id, error)
    return build_error_response(request_id, InternalError(message=_INTERNAL_ERROR_MESSAGE))


def is_stream_closing(event: object) -> bool:
    """True for a Task or Status_Event whose state ends the stream (terminal or input-required)."""
    status = getattr(event, "status", None)
    if status is None or not hasattr(event, "HasField") or not event.HasField("status"):
        return False
    try:
        return from_sdk(status.state) in STREAM_CLOSING_STATES
    except ValueError:
        return False


def _check_version_strict(context: ServerCallContext) -> None:
    """Reject a present `A2A-Version` header that is not 1.0 (1.12). Absent is the SDK's call."""
    headers = context.state.get("headers")
    if not isinstance(headers, Mapping):
        return
    for name, value in headers.items():
        if isinstance(name, str) and name.casefold() == _VERSION_HEADER:
            header = str(value)
            if accepts_version(header):
                return
            shown = header[:_MAX_REPORTED_VERSION_CHARS]
            raise VersionNotSupportedError(
                message=f"A2A version {shown!r} is not supported. This server speaks A2A 1.0."
            )

def contract_error_body(request_id: str | int | None, error: HostContractError) -> dict[str, Any]:
    """The exact JSON-RPC error response for a contract error."""
    return {"jsonrpc": "2.0", "id": request_id, "error": error.jsonrpc_error()}


class HostJsonRpcDispatcher(JsonRpcDispatcher):
    """`JsonRpcDispatcher` whose JSON-RPC route renders `HostContractError`s without `ErrorInfo`."""

    async def handle_requests(self, request: Request) -> Response:
        """Name the unknown method in the SDK's `-32601` error (1.9).

        The SDK builds that error from `_generate_error_response`, which receives no method name,
        so the message is completed here from the already-parsed request body.
        """
        response = await super().handle_requests(request)
        if not isinstance(response, JSONResponse) or response.status_code != 200:
            return response
        try:
            body = json.loads(bytes(response.body))
        except ValueError:
            return response
        error = body.get("error") if isinstance(body, dict) else None
        if (
            not isinstance(error, dict)
            or error.get("code") != _METHOD_NOT_FOUND_CODE
            or error.get("message") != _SDK_METHOD_NOT_FOUND_MESSAGE
        ):
            return response
        try:
            method = (await request.json()).get("method")  # Starlette caches the parsed body
        except ValueError, AttributeError:
            return response
        if not isinstance(method, str):
            return response
        error["message"] = f"{_SDK_METHOD_NOT_FOUND_MESSAGE}: {method[:_MAX_REPORTED_METHOD_CHARS]}"
        return JSONResponse(body, status_code=200)

    @validate_version(PROTOCOL_VERSION_1_0)
    async def _process_streaming_request(
        self,
        request_id: str | int | None,
        request_obj: Any,
        context: ServerCallContext,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Streaming path: strict version check, then an SSE generator that does the work (1.16).

        Nothing is awaited here beyond the checks, so admission and handler start-up run on the
        first iteration, after the response headers are sent. Every outcome is an SSE event.
        """
        _check_version_strict(context)
        return self._sse_events(request_id, request_obj, context)

    async def _sse_events(
        self,
        request_id: str | int | None,
        request_obj: Any,
        context: ServerCallContext,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Admission, handler start-up, events, and at most one error item (1.16, 1.17)."""
        method = context.state.get("method")
        sent = 0
        try:
            if method == "SendStreamingMessage":
                stream = self.request_handler.on_message_send_stream(request_obj, context)
            elif method == "SubscribeToTask":
                stream = self.request_handler.on_subscribe_to_task(request_obj, context)
            else:
                raise UnsupportedOperationError(message="Stream not supported")
            async with aclosing(stream):
                async for event in stream:
                    result = MessageToDict(\n                        to_stream_response(event), preserving_proto_field_name=False\n                    )
                    yield {"jsonrpc": "2.0", "id": request_id, "result": result}
                    sent += 1
                    if is_stream_closing(event):
                        return
        except Exception as exc:
            if isinstance(exc, A2AError):
                logger.warning(
                    "stream_error method=%s id=%s events_sent=%d error=%s: %s",
                    method,\n                    request_id,\n                    sent,\n                    type(exc).__name__,\n                    exc,
                )
            else:
                logger.error(
                    "stream_error method=%s id=%s events_sent=%d",
                    method,\n                    request_id,\n                    sent,\n                    exc_info=True,
                )
            yield jsonrpc_error_response(request_id, exc)
    def _create_response(
        self,
        context: ServerCallContext,
        handler_result: AsyncGenerator[dict[str, Any]] | dict[str, Any],
    ) -> Response:
        """Non-streaming path: replace the SDK's generic error dict with the contract body."""
        stored = context.state.get(HOST_CONTRACT_ERROR_KEY)
        if (
            isinstance(stored, HostContractError)
            and isinstance(handler_result, dict)
            and "error" in handler_result
        ):
            request_id = handler_result.get("id", context.state.get("request_id"))
            return JSONResponse(contract_error_body(request_id, stored))
        return super()._create_response(context, handler_result)
