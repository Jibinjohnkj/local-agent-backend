"""Property test for the health payload (Requirements 15.8, 15.9).

`health_payload` lists every configured server once, as `available` only when it is both enabled
and available. The overall status is `ok` iff every enabled server is available, so a disabled
server never degrades it.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.a2a_server.health import health_payload
from local_agent.models import ServerStatus

_STATUSES = st.lists(
    st.builds(
        ServerStatus,
        name=st.text(min_size=1, max_size=12),
        enabled=st.booleans(),
        available=st.booleans(),
    ),
    max_size=10,
    unique_by=lambda server: server.name,
)


# Feature: local-agent-backend, Property 29: Health status
# **Validates: Requirements 15.8, 15.9**
@settings(max_examples=200)
@given(servers=_STATUSES)
def test_health_status(servers: list[ServerStatus]) -> None:
    payload = health_payload(servers)

    listed = payload["mcp_servers"]
    assert [entry["name"] for entry in listed] == [server.name for server in servers]
    for entry, server in zip(listed, servers, strict=True):
        expected = "available" if server.enabled and server.available else "unavailable"
        assert entry["status"] == expected
        assert set(entry) == {"name", "status"}

    all_enabled_available = all(server.available for server in servers if server.enabled)
    assert payload["status"] == ("ok" if all_enabled_available else "degraded")
