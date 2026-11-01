"""Fixtures for the integration tests (task 22.1). The helpers behind them are in
`tests/integration_harness.py`.

* `isolated_home` (autouse): `HOME`, `LOCALAPPDATA`, and `XDG_DATA_HOME` point into a temporary
  directory, so no test can reach the developer's real per-user Storage_Directory.
* `storage_dir`: a temporary Storage_Directory for the test. Config built with it always sets
  `storage.path`.
* `storage_backend`: parametrizes a test over `memory` and `sqlite`.
* `fake_llm`: a `ScriptedCompletion` installed as `litellm.acompletion` (in-process tests).
* `llm_stub`: a started `FakeLlmServer` (OpenAI-compatible HTTP stub, for subprocess tests).
* `app_factory`: opens the production app in process, over `httpx.ASGITransport` or a real uvicorn
  socket (`transport="uvicorn"`). `app` is the same with defaults, for `storage_backend`.
* `backend_launcher`: starts `python -m local_agent` children and ends them at teardown.
* A session-scoped autouse finalizer kills any recorded pid still alive at the end of the run.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from integration_harness import (
    PIDS,
    AppHandle,
    BackendLauncher,
    FakeLlmServer,
    ScriptedCompletion,
    config_dict,
    open_app,
)
from local_agent.config.models import BackendConfig

type StorageBackend = Literal["memory", "sqlite"]
type AppOpener = Callable[..., contextlib.AbstractAsyncContextManager[AppHandle]]

STORAGE_BACKENDS: tuple[StorageBackend, ...] = ("memory", "sqlite")


@pytest.fixture(scope="session", autouse=True)
def _kill_leftover_pids() -> Iterator[None]:
    """At the end of the run, kill every recorded process that is still alive (17.12)."""
    yield
    leftovers = PIDS.kill_leftovers()
    if leftovers:
        sys.stderr.write(f"integration tests left processes running; killed pids {leftovers}\n")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "LocalAppData"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "xdg-data"))
    monkeypatch.delenv("AGENT_CONFIG", raising=False)
    return home


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    """A Storage_Directory that does not exist yet, inside the test's temporary directory."""
    return tmp_path / "storage"


@pytest.fixture(params=STORAGE_BACKENDS)
def storage_backend(request: pytest.FixtureRequest) -> StorageBackend:
    return cast("StorageBackend", request.param)


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> ScriptedCompletion:
    """Replace `litellm.acompletion` with a scriptable fake; `fake_llm.script(*turns)` sets it."""
    import litellm

    scripted = ScriptedCompletion()
    monkeypatch.setattr(litellm, "acompletion", scripted)
    return scripted


@pytest.fixture
def llm_stub() -> Iterator[FakeLlmServer]:
    with FakeLlmServer() as server:
        yield server


@pytest.fixture
def app_factory(storage_dir: Path) -> AppOpener:
    """`app_factory(...)` is an async context manager that yields an `AppHandle`.

    Keywords: `storage_backend` (default `memory`), `mcp_servers`, `permissions`, `overrides`
    (merged into the config), `transport` (`asgi` or `uvicorn`), `storage_dir` (to restart on the
    same data), and `config` (a ready `BackendConfig`, replacing the others).
    """

    def open_it(
        *,
        storage_backend: StorageBackend = "memory",
        mcp_servers: Sequence[Mapping[str, Any]] = (),
        permissions: Mapping[str, Sequence[str]] | None = None,
        overrides: Mapping[str, Any] | None = None,
        transport: Literal["asgi", "uvicorn"] = "asgi",
        storage_dir: Path = storage_dir,
        config: BackendConfig | None = None,
    ) -> contextlib.AbstractAsyncContextManager[AppHandle]:
        the_config = config or BackendConfig.model_validate(
            config_dict(
                storage_dir=storage_dir,
                storage_backend=storage_backend,
                mcp_servers=mcp_servers,
                permissions=permissions,
                overrides=overrides,
            )
        )
        return open_app(the_config, storage_dir, transport=transport)

    return open_it


@pytest.fixture
async def app(app_factory: AppOpener, storage_backend: StorageBackend) -> AsyncIterator[AppHandle]:
    """The app over each storage backend, with no MCP_Servers."""
    async with app_factory(storage_backend=storage_backend) as handle:
        yield handle


@pytest.fixture
def backend_launcher(tmp_path: Path, isolated_home: Path) -> Iterator[BackendLauncher]:
    launcher = BackendLauncher(root=tmp_path / "backends", home=isolated_home)
    launcher.root.mkdir()
    yield launcher
    launcher.close()
