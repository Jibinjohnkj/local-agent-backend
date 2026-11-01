"""JSON-RPC dispatcher that renders the Backend's two contract errors exactly (1.14, 1.15, 2.14).

The SDK maps errors by exact type and wraps `data` in `ErrorInfo` details, so no handler-side
exception can produce `data: {"context_id": ...}`. `HostRequestHandler._admit` stores a
`HostContractError` in `context.state["host_contract_error"]` and raises it. It is an `A2AError`,
so the SDK aborts the request through its normal error path before a Task exists. This dispatcher
then replaces the SDK's generic error body with `{"jsonrpc": "2.0", "id": <id>, "error":
err.jsonrpc_error()}`:

- Streaming (`SendStreamingMessage`): the SDK fetches the first event eagerly, so the error
  surfaces before any SSE response starts. It reaches the SDK's outer `except A2AError`, which
  calls `_generate_error_response`. The override renders a `HostContractError` as a plain JSON
  response with HTTP 200.
- Non-streaming (`SendMessage`): the SDK builds its generic error dict and passes it to
  `_create_response(call_context, ...)`. The override substitutes the contract body when
  `call_context.state["host_contract_error"]` holds a `HostContractError`.

Both hooks are private SDK API of the pinned 1.2.x release. `test_error_contract_jsonrpc` guards
them (both methods, both errors, request ID echoed).

Fallback if a later SDK release removes either hook: move the same rendering into a small ASGI
wrapper around the JSON-RPC route. The wrapper replays the request body and rewrites error bodies
whose `ErrorInfo.reason` is `CONTEXT_BUSY` or `INVALID_PARAMS` with a Backend-owned marker. The
contract tests stay the same.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Final

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from local_agent.a2a_server.a2a_compat import (
    A2AError,
    JsonRpcDispatcher,
    JSONRPCError,
    ServerCallContext,
)
from local_agent.a2a_server.errors import HostContractError

logger = logging.getLogger(__name__)

HOST_CONTRACT_ERROR_KEY = "host_contract_error"

_METHOD_NOT_FOUND_CODE: Final = -32601
_SDK_METHOD_NOT_FOUND_MESSAGE: Final = "Method not found"
_MAX_REPORTED_METHOD_CHARS: Final = 100


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

    def _generate_error_response(
        self,
        request_id: str | int | None,
        error: Exception | JSONRPCError | A2AError,
    ) -> JSONResponse:
        """Streaming path (and any pre-dispatch error): render a contract error exactly."""
        if not isinstance(error, HostContractError):
            return super()._generate_error_response(request_id, error)
        logger.warning(
            "Request Error (ID: %s): Code=%s, Message='%s', Data=%s",
            request_id,
            error.jsonrpc_code,
            error.message,
            error.data,
        )
        return JSONResponse(contract_error_body(request_id, error), status_code=200)

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
