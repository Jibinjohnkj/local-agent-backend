"""Two real Backend processes on one Storage_Directory (task 22.16; 18.25, 18.26, 18.27).

* `test_storage_lock_conflict`: the second `python -m local_agent` on a Storage_Directory that a
  running Backend holds exits with status 1 within 2 s. It logs one `storage_in_use` record whose
  reason is `storage is in use` and which names the database path. It never opens the database
  (the file and its WAL are byte-identical afterwards) and never binds its port. The first Backend
  keeps serving: a Task that was `working` finishes and a new Task completes.
* `test_storage_lock_released_after_kill`: after the holder is ended hard (`SIGKILL` on POSIX,
  `TerminateProcess` on Windows) the OS drops the lock, so a new Backend starts on the same
  Storage_Directory with no cleanup and still sees the data the killed one stored (18.27).
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import closing
from pathlib import Path

from integration_harness import (
    DATABASE_NAME,
    BackendLauncher,
    BackendProcess,
    FakeLlmServer,
    StubTurn,
    jsonrpc_call,
    message_params,
    wait_for_task_state,
)
from local_agent.logging_setup import Redactor
from local_agent.storage.lock import LOCK_FILE_NAME

HARNESS_API_KEY = "test-key"  # `llm.api_key` in `integration_harness.config_dict`
EXIT_FAILURE = 1
COMPLETED = "TASK_STATE_COMPLETED"
WORKING = "TASK_STATE_WORKING"
CONFLICT_EXIT_BUDGET_S = 2.0  # 18.26
FIRST_REPLY = "first answer"


def _snapshot(storage_dir: Path) -> dict[str, bytes | None]:
    """The bytes of the database and its sidecars; `None` for a file that does not exist."""
    names = (DATABASE_NAME, f"{DATABASE_NAME}-wal", f"{DATABASE_NAME}-shm")
    return {
        name: (storage_dir / name).read_bytes() if (storage_dir / name).exists() else None
        for name in names
    }


def _port_refuses(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _send(backend: BackendProcess, text: str, *, wait: bool = True) -> str:
    """Send one message; return the Task id (after it completed, with `wait`)."""
    with backend.client() as client:
        params = message_params(text)
        params["configuration"] = {"returnImmediately": True}
        task_id = str(jsonrpc_call(client, "SendMessage", params)["result"]["task"]["id"])
        if wait:
            task = wait_for_task_state(client, task_id)
            assert task["status"]["state"] == COMPLETED
    return task_id


def test_storage_lock_conflict(backend_launcher: BackendLauncher, llm_stub: FakeLlmServer) -> None:
    gate = threading.Event()
    llm_stub.script(
        StubTurn(chunks=(FIRST_REPLY,)),
        StubTurn(chunks=("par", "tial"), hold_after=1, gate=gate),
    )
    storage_dir = backend_launcher.storage_dir
    database = storage_dir / DATABASE_NAME
    try:
        first = backend_launcher.launch(llm_base=llm_stub.base_url)
        _send(first, "complete me")

        # A second Task stays `working` while the second Backend tries to start.
        held_id = _send(first, "hold me", wait=False)
        llm_stub.wait_for_chunks(2)
        with first.client() as client:
            wait_for_task_state(client, held_id, [WORKING], timeout_s=10.0)
        before = _snapshot(storage_dir)
        assert before[DATABASE_NAME] is not None, "the first Backend has no database"
        lock_text = (storage_dir / LOCK_FILE_NAME).read_text(encoding="ascii").strip()
        assert lock_text == str(first.pid), "the lock file does not hold the holder's pid"

        started = time.monotonic()
        second = backend_launcher.launch(llm_base=llm_stub.base_url, wait=False)
        status = second.wait(30.0)
        elapsed = time.monotonic() - started

        assert status == EXIT_FAILURE, second.stderr_text()
        assert elapsed < CONFLICT_EXIT_BUDGET_S, f"the second Backend took {elapsed:.2f}s to exit"
        records = [r for r in second.log_records() if r.get("event") == "storage_in_use"]
        assert len(records) == 1, second.stderr_text()
        record = records[0]
        assert "storage is in use" in str(record["reason"])
        # The log redacts every 4-character run of the harness api_key (15.5), paths included.
        assert record["database_path"] == Redactor([HARNESS_API_KEY]).redact(str(database))
        assert str(record["database_path"]).endswith(DATABASE_NAME)
        assert _port_refuses(second.port), "the second Backend bound its port"
        assert _snapshot(storage_dir) == before, "the second Backend changed the database"
        assert (storage_dir / LOCK_FILE_NAME).read_text(encoding="ascii").strip() == str(first.pid)

        # The first Backend keeps serving and its Task continues.
        assert first.running
        gate.set()
        with first.client() as client:
            task = wait_for_task_state(client, held_id)
        assert task["status"]["state"] == COMPLETED
        _send(first, "one more")
        assert first.stop() == 0
    finally:
        gate.set()


def test_storage_lock_released_after_kill(
    backend_launcher: BackendLauncher, llm_stub: FakeLlmServer
) -> None:
    llm_stub.script(StubTurn(chunks=(FIRST_REPLY,)))
    first = backend_launcher.launch(llm_base=llm_stub.base_url)
    task_id = _send(first, "stored before the kill")

    first.kill()
    assert not first.running

    second = backend_launcher.launch(llm_base=llm_stub.base_url)
    assert second.running
    lock_text = (backend_launcher.storage_dir / LOCK_FILE_NAME).read_text(encoding="ascii")
    assert lock_text.strip() == str(second.pid)
    with second.client() as client:
        task = jsonrpc_call(client, "GetTask", {"id": task_id})["result"]
    assert task["status"]["state"] == COMPLETED
    assert second.stop() == 0
