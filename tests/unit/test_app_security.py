"""Auth and CORS wiring of `build_app` (task 20.18).

The real Starlette app from `build_app` is called through `httpx.ASGITransport`, with a real
(memory) Session_Store. The lifespan is not run: no request here reaches MCP, the LLM, or a runner.

Covered:

* CORS (1.2 context, 16.4, 16.6): an allowed origin gets `Access-Control-Allow-Origin`, a
  disallowed one (other host, scheme, or port) gets no CORS allow header, an empty or omitted
  origin list never gets one, and preflight follows the same rule.
* Loopback bind (16.7): no bearer token is needed, even when a token is configured.
* Non-loopback bind (16.2): `/health` and the A2A routes need the exact token, the Agent_Card path
  does not (1.2), and a rejected request creates no Task (16.3).
* The middleware order: a 401 for an allowed origin still carries CORS headers, and a preflight is
  answered before authentication.

Requirements: 1.2, 16.2, 16.4, 16.6, 16.7.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx
import pytest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette

from local_agent.a2a_server.app import build_app, build_deps
from local_agent.config.loader import load_config_from_text
from local_agent.config.models import (
    BackendConfig,
    LLMConfig,
    ServerConfig,
    StorageConfig,
)
from local_agent.session.store import TaskQuery

ALLOW_ORIGIN: Final = "access-control-allow-origin"
ALLOW_METHODS: Final = "access-control-allow-methods"
ALLOW_CREDENTIALS: Final = "access-control-allow-credentials"

TOKEN: Final = "k3y-" + "0123456789abcdef" * 2  # 36 characters, above the 32 minimum (16.5)
ALLOWED: Final = "http://localhost:3000"
LOOPBACK_PEER: Final = ("127.0.0.1", 50000)
REMOTE_PEER: Final = ("192.0.2.7", 50000)
JSON_HEADERS: Final = {"Content-Type": "application/json"}
VERSION: Final = {"A2A-Version": "1.0"}
MESSAGE: Final = {"messageId": "m-1", "role": "ROLE_USER", "parts": [{"text": "hi"}]}

NON_LOOPBACK_HOSTS: Final = ("0.0.0.0", "::", "192.0.2.7", "example.test")
LOOPBACK_HOSTS: Final = ("127.0.0.1", "127.9.9.9", "localhost", "LOCALHOST", "::1", "[::1]")


# --- Helpers ------------------------------------------------------------------------------------


def make_config(
    *,
    host: str = "127.0.0.1",
    token: str | None = None,
    origins: tuple[str, ...] = (),
) -> BackendConfig:
    return BackendConfig(
        llm=LLMConfig(model="fake/model"),
        storage=StorageConfig(backend="memory"),
        server=ServerConfig(
            host=host,
            bearer_token=token,  # pydantic coerces str -> SecretStr
            cors_allowed_origins=list(origins),
        ),
    )


def make_app(cfg: BackendConfig) -> Starlette:
    return build_app(cfg, build_deps(cfg, None))


def client_for(app: Starlette, peer: tuple[str, int] = LOOPBACK_PEER) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, client=peer)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def preflight(origin: str) -> dict[str, str]:
    return {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type,a2a-version",
    }


def jsonrpc_body() -> str:
    body: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {"message": MESSAGE},
    }
    return json.dumps(body)


async def task_count(app: Starlette) -> int:
    page = await app.state.deps.session_store.list_tasks(TaskQuery(page_size=100))
    return len(page.tasks)


# --- CORS: allowed versus disallowed origins (16.4) ---------------------------------------------


async def test_allowed_origin_gets_allow_origin_header() -> None:
    app = make_app(make_config(origins=(ALLOWED, "https://app.example.test")))

    async with client_for(app) as client:
        first = await client.get("/health", headers={"Origin": ALLOWED})
        second = await client.get("/health", headers={"Origin": "https://app.example.test"})

    assert first.status_code == 200
    assert first.headers[ALLOW_ORIGIN] == ALLOWED
    assert second.headers[ALLOW_ORIGIN] == "https://app.example.test"
    assert ALLOW_CREDENTIALS not in first.headers


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:3001",  # other port
        "https://localhost:3000",  # other scheme
        "http://localhost",  # port omitted
        "http://localhost:3000.evil.test",  # suffix attack
        "http://sub.localhost:3000",  # other host
        "http://evil.test",
        "null",
        "*",
    ],
)
async def test_disallowed_origin_gets_no_cors_headers(origin: str) -> None:
    app = make_app(make_config(origins=(ALLOWED,)))

    async with client_for(app) as client:
        response = await client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200  # CORS never blocks server-side; the browser does
    assert not [name for name in response.headers if name.startswith("access-control-")]


async def test_request_without_origin_gets_no_cors_headers() -> None:
    app = make_app(make_config(origins=(ALLOWED,)))

    async with client_for(app) as client:
        response = await client.get("/health")

    assert ALLOW_ORIGIN not in response.headers


async def test_preflight_from_allowed_origin_is_answered() -> None:
    app = make_app(make_config(origins=(ALLOWED,)))

    async with client_for(app) as client:
        response = await client.options("/", headers=preflight(ALLOWED))

    assert response.status_code == 200
    assert response.headers[ALLOW_ORIGIN] == ALLOWED
    methods = {method.strip() for method in response.headers[ALLOW_METHODS].split(",")}
    assert {"GET", "POST", "OPTIONS"} <= methods


async def test_preflight_from_disallowed_origin_is_refused_without_allow_headers() -> None:
    app = make_app(make_config(origins=(ALLOWED,)))

    async with client_for(app) as client:
        response = await client.options("/", headers=preflight("http://localhost:3001"))

    assert response.status_code == 400
    assert ALLOW_ORIGIN not in response.headers


# --- CORS: empty origin list (16.6) -------------------------------------------------------------


@pytest.mark.parametrize("origin", [ALLOWED, "http://127.0.0.1:3000", "https://example.test"])
async def test_empty_origin_list_returns_no_cors_headers(origin: str) -> None:
    app = make_app(make_config(origins=()))

    async with client_for(app) as client:
        simple = await client.get("/health", headers={"Origin": origin})
        pre = await client.options("/", headers=preflight(origin))

    assert simple.status_code == 200
    assert not [name for name in simple.headers if name.startswith("access-control-")]
    assert pre.status_code == 400
    assert ALLOW_ORIGIN not in pre.headers


async def test_omitted_origin_list_in_loaded_config_returns_no_cors_headers() -> None:
    cfg = load_config_from_text(
        "llm:\n  model: ollama_chat/llama3.1\nstorage:\n  backend: memory\n", {}, "test-config"
    )
    assert cfg.server.cors_allowed_origins == []
    app = make_app(cfg)

    async with client_for(app) as client:
        response = await client.get("/health", headers={"Origin": ALLOWED})

    assert response.status_code == 200
    assert ALLOW_ORIGIN not in response.headers


# --- Loopback bind: no token needed (16.7) ------------------------------------------------------


@pytest.mark.parametrize("host", LOOPBACK_HOSTS)
@pytest.mark.parametrize("path", ["/health", AGENT_CARD_WELL_KNOWN_PATH])
async def test_loopback_bind_needs_no_token(host: str, path: str) -> None:
    app = make_app(make_config(host=host))

    async with client_for(app) as client:
        response = await client.get(path)

    assert response.status_code == 200


async def test_loopback_bind_ignores_a_configured_token() -> None:
    app = make_app(make_config(host="127.0.0.1", token=TOKEN))

    async with client_for(app) as client:
        none = await client.get("/health")
        wrong = await client.get("/health", headers=bearer("wrong"))

    assert none.status_code == 200
    assert wrong.status_code == 200


async def test_loopback_bind_serves_a2a_routes_without_token() -> None:
    app = make_app(make_config())

    async with client_for(app) as client:
        rest = await client.get("/tasks", headers=VERSION)
        rpc = await client.post("/", content=json.dumps({"jsonrpc": "2.0"}), headers=JSON_HEADERS)

    assert rest.status_code == 200
    assert rpc.status_code != 401


async def test_loopback_bind_refuses_a_non_loopback_peer() -> None:
    app = make_app(make_config())

    async with client_for(app, REMOTE_PEER) as client:
        refused = await client.get("/health")
    async with client_for(app, ("::ffff:127.0.0.1", 50000)) as client:
        mapped = await client.get("/health")

    assert refused.status_code == 403
    assert mapped.status_code == 200  # an IPv4-mapped loopback peer is still loopback


# --- Non-loopback bind: the token protects everything but the Agent_Card (16.2) -----------------


@pytest.mark.parametrize("host", NON_LOOPBACK_HOSTS)
async def test_health_is_protected_on_a_non_loopback_bind(host: str) -> None:
    app = make_app(make_config(host=host, token=TOKEN))

    async with client_for(app, REMOTE_PEER) as client:
        missing = await client.get("/health")
        good = await client.get("/health", headers=bearer(TOKEN))

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert good.status_code == 200
    assert good.json() == {"status": "ok", "mcp_servers": []}


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "Bearer",
        "Bearer ",
        "Bearer wrong-token",
        f"Bearer {TOKEN[:-1]}",  # prefix of the token
        f"Bearer {TOKEN}x",  # token plus one character
        f"Bearer {TOKEN.upper()}",  # comparison is case-sensitive
        f"Bearer  {TOKEN}",  # extra space
        f"Basic {TOKEN}",
        TOKEN,  # no scheme
    ],
)
async def test_bad_credentials_are_rejected_with_401(authorization: str | None) -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN))
    headers = {} if authorization is None else {"Authorization": authorization}

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get("/health", headers=headers)

    assert response.status_code == 401


async def test_two_authorization_headers_are_rejected() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN))
    headers = httpx.Headers([("Authorization", f"Bearer {TOKEN}"), ("Authorization", "x")])

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get("/health", headers=headers)

    assert response.status_code == 401


async def test_agent_card_is_exempt_from_auth_on_a_non_loopback_bind() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN))

    async with client_for(app, REMOTE_PEER) as client:
        card = await client.get(AGENT_CARD_WELL_KNOWN_PATH)
        with_wrong_token = await client.get(AGENT_CARD_WELL_KNOWN_PATH, headers=bearer("wrong"))

    assert card.status_code == 200
    assert card.headers["content-type"].startswith("application/json")
    assert card.json()["name"]
    assert with_wrong_token.status_code == 200


@pytest.mark.parametrize(
    "path", ["/", "/tasks", "/message:send", "/message:stream", "/extendedAgentCard", "/nope"]
)
async def test_only_the_agent_card_is_exempt(path: str) -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN))

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get(path)

    assert response.status_code == 401


async def test_rejected_requests_create_no_task() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN))
    rest_body = json.dumps({"message": MESSAGE})

    async with client_for(app, REMOTE_PEER) as client:
        rpc = await client.post("/", content=jsonrpc_body(), headers=JSON_HEADERS)
        send = await client.post("/message:send", content=rest_body, headers=JSON_HEADERS)
        stream = await client.post(
            "/message:stream", content=rest_body, headers={**JSON_HEADERS, **bearer("wrong")}
        )

    assert (rpc.status_code, send.status_code, stream.status_code) == (401, 401, 401)
    assert await task_count(app) == 0


async def test_authorized_request_reaches_the_a2a_routes() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN))

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get("/tasks", headers={**VERSION, **bearer(TOKEN)})

    assert response.status_code == 200


# --- CORS and auth together: layer order --------------------------------------------------------


async def test_401_for_an_allowed_origin_still_carries_cors_headers() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN, origins=(ALLOWED,)))

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get("/health", headers={"Origin": ALLOWED})

    assert response.status_code == 401
    assert response.headers[ALLOW_ORIGIN] == ALLOWED


async def test_401_for_a_disallowed_origin_has_no_cors_headers() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN, origins=(ALLOWED,)))

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get("/health", headers={"Origin": "http://evil.test"})

    assert response.status_code == 401
    assert ALLOW_ORIGIN not in response.headers


async def test_preflight_needs_no_token_on_a_non_loopback_bind() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN, origins=(ALLOWED,)))

    async with client_for(app, REMOTE_PEER) as client:
        allowed = await client.options("/", headers=preflight(ALLOWED))
        refused = await client.options("/", headers=preflight("http://evil.test"))

    assert allowed.status_code == 200
    assert allowed.headers[ALLOW_ORIGIN] == ALLOWED
    assert refused.status_code == 400
    assert ALLOW_ORIGIN not in refused.headers


async def test_authorized_request_from_allowed_origin_gets_cors_headers() -> None:
    app = make_app(make_config(host="0.0.0.0", token=TOKEN, origins=(ALLOWED,)))

    async with client_for(app, REMOTE_PEER) as client:
        response = await client.get("/health", headers={"Origin": ALLOWED, **bearer(TOKEN)})

    assert response.status_code == 200
    assert response.headers[ALLOW_ORIGIN] == ALLOWED
