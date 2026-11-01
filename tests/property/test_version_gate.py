"""Property test for the A2A-Version gate (Requirements 1.11, 1.12).

Two checks:

- `accepts_version` is true exactly when the stripped header parses as a version whose release is
  1.0 with optional trailing zero components. One test uses headers built from known release
  numbers, so the expected answer comes from the construction. The other uses arbitrary text and
  takes the parse from `packaging`, as the Property words it.
- A request that carries a rejected `A2A-Version` header gets `VersionNotSupportedError`
  (JSON-RPC `-32009`, REST 400 `VERSION_NOT_SUPPORTED`) on every A2A operation, and the
  Session_Store sees no write. The app is built by `build_app` and called through
  `httpx.ASGITransport`, over a recording wrapper of both Session_Store kinds. A request with an
  accepted header gets past the gate.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, Final

import httpx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from packaging.version import InvalidVersion, Version
from scenarios import (
    ScenarioScript,
    StoreEnv,
    StoreKind,
    build_fakes,
    char_estimate,
    open_store,
    store_kind,
)
from starlette.applications import Starlette

from fakes import ClockSleep, FakeLlmGateway, FakeMcpManager
from local_agent.a2a_server.app import build_app, build_deps
from local_agent.a2a_server.errors import accepts_version
from local_agent.a2a_server.events import TaskState
from local_agent.config.models import (
    BackendConfig,
    LLMConfig,
    OrchestrationConfig,
    StorageConfig,
)
from local_agent.models import TaskRecord, UserMessage
from local_agent.orchestrator.loop import RunnerDeps, TaskRunnerRegistry
from local_agent.session.store import (
    SessionStore,
    StateChange,
    StorageTxn,
    TaskPage,
    TaskQuery,
)
from local_agent.storage.task_store import ProjectedTaskStore

__all__ = ["store_kind"]  # the fixture is used by name below

VERSION_NOT_SUPPORTED_CODE: Final = -32009
SEED_CONTEXT: Final = "seed-context"
SEED_TASK: Final = "seed-task"
WRITE_METHODS: Final = frozenset(
    {"commit", "create_task", "set_a2a_snapshot", "delete_task", "delete_expired"}
)
"""The `SessionStore` methods that change what is stored (the rest only read)."""

VERSION_HEADER: Final = "A2A-Version"
PADDING_ASCII: Final = " \t"
PADDING_ANY: Final = " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0\u2003\u3000"


# --- Header strategies --------------------------------------------------------------------------

_SUFFIXES: Final = ("", "", "rc1", "a2", ".post1", ".dev0", "+local.1", "-1", "b3.post2.dev4")
_PREFIXES: Final = ("", "", "v", "V")


@st.composite
def structured_headers(draw: st.DrawFn, *, padding: str) -> tuple[str, bool]:
    """A well-formed version string and whether its release is 1.0 with only zeros after.

    The release is drawn as numbers, so the expected answer comes from the construction and not
    from any parser. The pre-release, post-release, dev, and local suffixes never change it.
    """
    near_miss = st.tuples(st.integers(0, 3), st.integers(0, 3))
    major, minor = draw(st.one_of(st.just((1, 0)), st.just((1, 0)), near_miss))
    tail = draw(st.lists(st.sampled_from([0, 0, 0, 1, 7]), max_size=3))
    release = (major, minor, *tail)
    parts = [str(number).zfill(draw(st.integers(1, 3))) for number in release]
    text = draw(st.sampled_from(_PREFIXES)) + ".".join(parts) + draw(st.sampled_from(_SUFFIXES))
    left = draw(st.text(alphabet=padding, max_size=2))
    right = draw(st.text(alphabet=padding, max_size=2))
    expected = (major, minor) == (1, 0) and not any(tail)
    return left + text + right, expected


_VERSION_LIKE: Final = "0123456789.vV!+-_ abcdeloprstw"
_PRINTABLE_ASCII: Final = "".join(chr(code) for code in range(0x20, 0x7F))


@st.composite
def version_headers(draw: st.DrawFn, *, wire: bool = False) -> str:
    """A header value: well-formed versions (accepted or not), near misses, and arbitrary text.

    With `wire` set, the value is printable ASCII plus space and tab, so an HTTP client can send it.
    """
    padding = PADDING_ASCII if wire else PADDING_ANY
    junk_alphabet = _PRINTABLE_ASCII if wire else st.characters(exclude_categories=["Cs"])
    kind = draw(st.integers(0, 4))
    if kind <= 1:
        return draw(structured_headers(padding=padding))[0]
    if kind == 2:
        return draw(st.sampled_from(["1.0", "1.0.0", "1.1", "1.0.1", "2.0", "0.3", "1", "", " "]))
    if kind == 3:
        return draw(st.text(alphabet=_VERSION_LIKE, max_size=10))
    return draw(st.text(alphabet=junk_alphabet, max_size=12))


def spec_accepts(header: str) -> bool:
    """The Property's wording: the stripped value parses as a version whose release is 1.0 with
    optional trailing zero components."""
    try:
        release = Version(header.strip()).release
    except InvalidVersion:
        return False
    return len(release) >= 2 and release[:2] == (1, 0) and all(n == 0 for n in release[2:])


# Feature: local-agent-backend, Property 58: A2A-Version gate
# **Validates: Requirements 1.11, 1.12**
@settings(max_examples=300, deadline=None)
@given(case=structured_headers(padding=PADDING_ANY))
def test_accepts_version_follows_the_release_numbers(case: tuple[str, bool]) -> None:
    header, expected = case

    assert accepts_version(header) is expected


# Feature: local-agent-backend, Property 58: A2A-Version gate
# **Validates: Requirements 1.11, 1.12**
@settings(max_examples=300, deadline=None)
@given(header=version_headers())
def test_accepts_version_matches_the_spec_wording(header: str) -> None:
    assert accepts_version(header) is spec_accepts(header)


def test_accepts_version_known_values() -> None:
    for accepted in ("1.0", "1.0.0", "1.00", " 1.0 ", "\t1.0.0.0\n", "v1.0"):
        assert accepts_version(accepted), accepted
    for rejected in ("1.1", "1.0.1", "2.0", "0.3", "1", "", " ", "abc", "1.0.x", "10.0"):
        assert not accepts_version(rejected), rejected


# --- The recording store ------------------------------------------------------------------------


class RecordingStore(SessionStore):
    """Forwards every call to `inner` and records the method names, so a test sees any write."""

    def __init__(self, inner: SessionStore) -> None:
        super().__init__()
        self._inner = inner
        self.calls: list[str] = []

    @property
    def writes(self) -> list[str]:
        return [name for name in self.calls if name in WRITE_METHODS]

    async def commit(self, txn: StorageTxn) -> None:
        self.calls.append("commit")
        await self._inner.commit(txn)

    async def create_task(
        self, task_id: str, context_id: str, now_ms: int, a2a_snapshot: str | None
    ) -> bool:
        self.calls.append("create_task")
        return await self._inner.create_task(task_id, context_id, now_ms, a2a_snapshot)

    async def get_history(self, context_id: str) -> Any:
        self.calls.append("get_history")
        return await self._inner.get_history(context_id)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        self.calls.append("get_task")
        return await self._inner.get_task(task_id)

    async def list_tasks(self, query: TaskQuery) -> TaskPage:
        self.calls.append("list_tasks")
        return await self._inner.list_tasks(query)

    async def active_task(self, context_id: str) -> str | None:
        self.calls.append("active_task")
        return await self._inner.active_task(context_id)

    async def set_a2a_snapshot(self, task_id: str, snapshot: str) -> None:
        self.calls.append("set_a2a_snapshot")
        await self._inner.set_a2a_snapshot(task_id, snapshot)

    async def get_a2a_snapshot(self, task_id: str) -> str | None:
        self.calls.append("get_a2a_snapshot")
        return await self._inner.get_a2a_snapshot(task_id)

    async def delete_task(self, task_id: str) -> None:
        self.calls.append("delete_task")
        await self._inner.delete_task(task_id)

    async def delete_expired(self, cutoff_ms: int) -> list[str]:
        self.calls.append("delete_expired")
        return await self._inner.delete_expired(cutoff_ms)


# --- The app under test -------------------------------------------------------------------------


@dataclasses.dataclass
class Rig:
    app: Starlette
    store: RecordingStore
    llm: FakeLlmGateway
    mcp: FakeMcpManager


def build_rig(env: StoreEnv) -> Rig:
    """The real app (`build_app`) over a recording wrapper of `env.store`, with scripted fakes."""
    cfg = BackendConfig(llm=LLMConfig(model="fake/model"))
    recording = RecordingStore(env.store)
    llm, mcp = build_fakes(ScenarioScript("unused", ()), env.clock, ClockSleep(env.clock))
    runners = TaskRunnerRegistry(
        RunnerDeps(
            sessions=recording,
            llm=llm,
            registry=mcp.registry,
            mcp=mcp,
            orchestration=OrchestrationConfig(per_session_discovery=True),
            base_instructions=None,
            model="fake/model",
            redactor=env.redactor,
            clock=env.clock,
            db_path=env.db_path,
            estimate_tokens=char_estimate,
        )
    )
    base = build_deps(cfg.model_copy(update={"storage": StorageConfig(backend="memory")}), None)
    deps = dataclasses.replace(
        base,
        session_store=recording,
        task_store=ProjectedTaskStore(recording, runners, env.clock),
        runners=runners,
        clock=env.clock,
    )
    return Rig(build_app(cfg, deps), recording, llm, mcp)


async def seed_completed_task(store: SessionStore, now_ms: int) -> None:
    """One finished Task, so cancel and get-task have something to act on."""
    await store.create_task(SEED_TASK, SEED_CONTEXT, now_ms, None)
    await store.commit(
        StorageTxn(
            context_id=SEED_CONTEXT,
            now_ms=now_ms + 1,
            appends=((SEED_TASK, UserMessage("hello")),),
            state=StateChange(
                SEED_TASK, frozenset({TaskState.SUBMITTED}), TaskState.WORKING, None, None
            ),
        )
    )
    await store.commit(
        StorageTxn(
            context_id=SEED_CONTEXT,
            now_ms=now_ms + 2,
            state=StateChange(
                SEED_TASK, frozenset({TaskState.WORKING}), TaskState.COMPLETED, None, None
            ),
        )
    )


async def snapshot(store: SessionStore) -> tuple[Any, ...]:
    """Everything stored: the seeded Task, the Task list, and the seeded Context's history."""
    page = await store.list_tasks(TaskQuery(page_size=100))
    return (
        await store.get_task(SEED_TASK),
        page.tasks,
        await store.get_history(SEED_CONTEXT),
        await store.get_a2a_snapshot(SEED_TASK),
    )


JSON_HEADERS: Final = {"Content-Type": "application/json"}
MESSAGE: Final = {"messageId": "m-1", "role": "ROLE_USER", "parts": [{"text": "hi"}]}

JSONRPC_CALLS: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    ("SendMessage", {"message": MESSAGE}),
    ("SendStreamingMessage", {"message": MESSAGE}),
    ("GetTask", {"id": SEED_TASK}),
    ("CancelTask", {"id": SEED_TASK}),
    ("ListTasks", {}),
    ("SubscribeToTask", {"id": SEED_TASK}),
)

REST_CALLS: Final[tuple[tuple[str, str, dict[str, Any] | None], ...]] = (
    ("POST", "/message:send", {"message": MESSAGE}),
    ("POST", "/message:stream", {"message": MESSAGE}),
    ("POST", f"/tasks/{SEED_TASK}:cancel", {}),
    ("GET", f"/tasks/{SEED_TASK}", None),
    ("GET", "/tasks", None),
    ("GET", f"/tasks/{SEED_TASK}:subscribe", None),
)


async def jsonrpc(
    client: httpx.AsyncClient, header: str, request_id: str | int, method: str, params: Any
) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    return await client.post(
        "/", content=json.dumps(body), headers={VERSION_HEADER: header, **JSON_HEADERS}
    )


def assert_jsonrpc_rejected(response: httpx.Response, request_id: str | int, method: str) -> None:
    where = f"{method} header-rejected: {response.status_code} {response.text[:300]}"
    payload = response.json()
    assert payload["jsonrpc"] == "2.0", where
    assert payload["id"] == request_id, where
    assert "result" not in payload, where
    assert payload["error"]["code"] == VERSION_NOT_SUPPORTED_CODE, where


def assert_rest_rejected(response: httpx.Response, where: str) -> None:
    detail = f"{where}: {response.status_code} {response.text[:300]}"
    assert response.status_code == 400, detail
    error = response.json()["error"]
    reasons = [d.get("reason") for d in error.get("details", []) if isinstance(d, dict)]
    assert reasons == ["VERSION_NOT_SUPPORTED"], detail


async def check_header(env: StoreEnv, header: str, request_id: str | int) -> None:
    rig = build_rig(env)
    await seed_completed_task(env.store, env.clock.now_ms())
    before = await snapshot(env.reader)
    accepted = spec_accepts(header)
    transport = httpx.ASGITransport(app=rig.app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        if accepted:
            # Past the gate: get-task answers with the seeded Task, over both bindings.
            response = await jsonrpc(client, header, request_id, "GetTask", {"id": SEED_TASK})
            payload = response.json()
            assert payload["result"]["id"] == SEED_TASK, response.text[:300]
            rest = await client.get(
                f"/tasks/{SEED_TASK}", headers={VERSION_HEADER: header, **JSON_HEADERS}
            )
            assert rest.status_code == 200, rest.text[:300]
            assert rest.json()["id"] == SEED_TASK
        else:
            for method, params in JSONRPC_CALLS:
                response = await jsonrpc(client, header, request_id, method, params)
                assert_jsonrpc_rejected(response, request_id, method)
            for verb, path, body in REST_CALLS:
                response = await client.request(
                    verb,
                    path,
                    content=None if body is None else json.dumps(body),
                    headers={VERSION_HEADER: header, **JSON_HEADERS},
                )
                assert_rest_rejected(response, f"{verb} {path}")

    assert rig.store.writes == [], rig.store.calls
    assert await snapshot(env.reader) == before
    assert rig.llm.requests == []
    assert rig.mcp.calls == []


request_ids: Final = st.one_of(
    st.integers(min_value=-(2**31), max_value=2**31),
    st.text(
        alphabet=st.characters(codec="ascii", exclude_categories=("Cc", "Cs")),
        min_size=1,
        max_size=8,
    ),
)


# Feature: local-agent-backend, Property 58: A2A-Version gate
# **Validates: Requirements 1.11, 1.12**
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(header=version_headers(wire=True), request_id=request_ids)
def test_rejected_header_yields_version_error_and_no_write(
    store_kind: StoreKind, header: str, request_id: str | int
) -> None:
    with open_store(store_kind) as env:
        asyncio.run(check_header(env, header, request_id))


def test_rejected_and_accepted_headers_both_reached() -> None:
    """Guard the strategy: the wire headers include both outcomes."""
    seen: list[bool] = []

    @settings(max_examples=100, deadline=None, database=None)
    @given(header=version_headers(wire=True))
    def collect(header: str) -> None:
        seen.append(spec_accepts(header))

    collect()
    assert True in seen
    assert False in seen
