"""Raw response shapes of GetTask and CancelTask over JSON-RPC (task 25.3).

Both use the SDK's non-streaming path: HTTP 200, `application/json`, a bare Task as `result`
(no `{"task": ...}` wrapper), and JSON-RPC error bodies for failures.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from fakes import LlmTurn, text_delta
from integration_harness import A2A_HEADERS, AppHandle, ScriptedCompletion, message_params

pytestmark = pytest.mark.integration

COMPLETES = LlmTurn(deltas=(text_delta("all "), text_delta("done")))
TASK_NOT_FOUND = -32001


async def rpc(app: AppHandle, request_id: int, method: str, params: dict[str, Any]) -> Any:
    body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    return await app.client.post("/", json=body, headers=A2A_HEADERS)


async def test_get_and_cancel_response_shapes(app: AppHandle, fake_llm: ScriptedCompletion) -> None:
    # Validates: Requirements 1.19, 1.20
    fake_llm.script(COMPLETES)
    sent = await rpc(app, 1, "SendMessage", message_params("hi"))
    task_id = sent.json()["result"]["task"]["id"]

    got = await rpc(app, 2, "GetTask", {"id": task_id})
    assert got.status_code == 200
    assert got.headers["content-type"].split(";")[0] == "application/json"
    document = json.loads(got.content)
    assert document["jsonrpc"] == "2.0"
    assert document["id"] == 2
    result = document["result"]
    assert "task" not in result
    assert {"id", "contextId", "status", "artifacts"} <= result.keys()
    assert result["id"] == task_id

    for request_id, method in ((3, "GetTask"), (4, "CancelTask")):
        missing = await rpc(app, request_id, method, {"id": "no-such-task"})
        assert missing.status_code == 200
        assert missing.headers["content-type"].split(";")[0] == "application/json"
        error = json.loads(missing.content)
        assert error["jsonrpc"] == "2.0"
        assert error["id"] == request_id
        assert "result" not in error
        assert error["error"]["code"] == TASK_NOT_FOUND
