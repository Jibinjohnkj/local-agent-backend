"""Property test for the Agent_Card content (Requirements 1.3, 1.4, 1.5)."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from hypothesis import given, settings
from hypothesis import strategies as st

from local_agent.a2a_server.agent_card import build_agent_card
from local_agent.config.models import AgentConfig, BackendConfig, LLMConfig, ServerConfig

_TEXT = st.text(min_size=1, max_size=40)

_IPV4 = st.ip_addresses(v=4).map(str)
_IPV6 = st.ip_addresses(v=6).map(str)
# Lowercase DNS-style names, since URL parsing lowercases the host.
_HOSTNAME = st.one_of(
    st.just("localhost"),
    st.from_regex(r"[a-z][a-z0-9-]{0,10}[a-z0-9](\.[a-z][a-z0-9]{0,8})?", fullmatch=True),
)
_WELL_KNOWN = st.sampled_from(["0.0.0.0", "::", "::1", "127.0.0.1"])
_HOSTS = st.one_of(_IPV4, _IPV6, _HOSTNAME, _WELL_KNOWN)


@st.composite
def _configs(draw: st.DrawFn) -> BackendConfig:
    return BackendConfig(
        server=ServerConfig(host=draw(_HOSTS), port=draw(st.integers(1, 65535))),
        llm=LLMConfig(model="ollama_chat/llama3.1"),
        agent=AgentConfig(
            name=draw(_TEXT),
            description=draw(_TEXT),
            version=draw(_TEXT),
        ),
    )


# Feature: local-agent-backend, Property 31: Agent card content
# **Validates: Requirements 1.3, 1.4, 1.5**
@settings(max_examples=100)
@given(cfg=_configs())
def test_agent_card_content(cfg: BackendConfig) -> None:
    card = build_agent_card(cfg)

    assert card.capabilities.streaming is True
    assert card.name == cfg.agent.name
    assert card.description == cfg.agent.description
    assert card.version == cfg.agent.version
    assert card.name
    assert card.description
    assert card.version

    assert list(card.default_input_modes) == ["text/plain"]
    assert list(card.default_output_modes) == ["text/plain"]

    assert len(card.skills) >= 1
    for skill in card.skills:
        assert skill.id
        assert skill.name
        assert skill.description

    assert len(card.supported_interfaces) == 1
    url = card.supported_interfaces[0].url
    parts = urlsplit(url)
    host = cfg.server.host
    assert parts.scheme == "http"
    assert parts.port == cfg.server.port
    assert parts.hostname == host
    if ":" in host:
        assert isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address)
        assert parts.netloc == f"[{host}]:{cfg.server.port}"
    else:
        assert parts.netloc == f"{host}:{cfg.server.port}"
