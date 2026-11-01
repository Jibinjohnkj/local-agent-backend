"""Agent_Card builder: the A2A metadata document served at the well-known path.

We build the card once at startup from `BackendConfig`. The card advertises streaming, plain-text
input and output, one skill, and a single JSON-RPC interface whose URL points at the address the
A2A_Server binds to (1.3, 1.4, 1.5).
"""

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

from local_agent.config.models import BackendConfig

__all__ = ["SKILL_ID", "build_agent_card", "endpoint_url"]

SKILL_ID = "local-tools-chat"
_TEXT_MODES = ("text/plain",)


def endpoint_url(host: str, port: int) -> str:
    """Return the JSON-RPC endpoint URL for a bind address, such as `http://127.0.0.1:10000/`.

    The host is used exactly as configured, so `localhost` stays `localhost` and wildcard binds
    (`0.0.0.0`, `::`) appear literally (1.5). Any host containing a colon is an IPv6 literal, and
    URLs require those in square brackets so the port colon stays unambiguous.
    """
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}/"


def build_agent_card(cfg: BackendConfig) -> AgentCard:
    """Build the Agent_Card for `cfg`.

    Name, description, and version come from `cfg.agent`, which the config models already require
    to be non-empty. The single skill and the text-only modes are fixed because the Host Agent
    only ever talks plain text.
    """
    skill = AgentSkill(
        id=SKILL_ID,
        name="Local tools chat",
        description=(
            "Chat with a local LLM that can call tools exposed by configured MCP servers, "
            "asking for approval before it runs protected tools."
        ),
        tags=["chat", "tools", "mcp"],
        input_modes=list(_TEXT_MODES),
        output_modes=list(_TEXT_MODES),
    )
    interface = AgentInterface(
        url=endpoint_url(cfg.server.host, cfg.server.port),
        protocol_binding=TransportProtocol.JSONRPC.value,
        protocol_version=PROTOCOL_VERSION_CURRENT,
    )
    return AgentCard(
        name=cfg.agent.name,
        description=cfg.agent.description,
        version=cfg.agent.version,
        supported_interfaces=[interface],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=list(_TEXT_MODES),
        default_output_modes=list(_TEXT_MODES),
        skills=[skill],
    )
