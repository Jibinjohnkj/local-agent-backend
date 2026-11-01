"""System_Prompt construction: base instructions followed by exactly one tool usage section.

`build_system_prompt` is pure. The Orchestrator calls it on every chat-completion request with the
live registry snapshot, so a server that connects, disconnects, or refreshes shows up in the next
request (12.5).
"""

from collections.abc import Sequence

from local_agent.models import RegisteredTool

DEFAULT_BASE_INSTRUCTIONS = (
    "You are a helpful assistant running locally on the user's machine. "
    "Answer the user's questions clearly and concisely. "
    "Say so when you are unsure instead of guessing."
)

TOOL_USAGE_HEADING = "## Tool usage"
NO_TOOLS_SENTENCE = "No tools are currently available."
NO_DESCRIPTION_NOTE = "(no description provided)"

# The three fixed instructions of Requirement 12.3.
TOOL_USAGE_INSTRUCTIONS = (
    "- Call a tool when answering requires data that is reachable only through a listed tool.",
    "- When a tool result reports an error, tell the user which tool failed and why.",
    "- When a tool result reports that the user denied a tool call, tell the user that the "
    "requested action was denied and not performed.",
)


def _describe(tool: RegisteredTool) -> str:
    """One line of text for the tool: its description, or the "no description" note (12.9).

    Line breaks become spaces so a multi-line description cannot start a new heading or list item.
    """
    description = " ".join((tool.description or "").splitlines()).strip()
    return description or NO_DESCRIPTION_NOTE


def build_system_prompt(base: str | None, tools: Sequence[RegisteredTool]) -> str:
    """Build the System_Prompt from the base instructions and the available tools.

    `base` falls back to `DEFAULT_BASE_INSTRUCTIONS` when it is `None` or empty (12.8). `tools` must
    already be limited to tools of available servers. Each tool is listed once under a
    `### <server>` heading, with servers in order of first appearance (12.2). When `tools` is empty
    the section says that no tools are available and names none (12.4).
    """
    lines: list[str] = [base or DEFAULT_BASE_INSTRUCTIONS, "", TOOL_USAGE_HEADING, ""]
    lines.extend(TOOL_USAGE_INSTRUCTIONS)

    if not tools:
        lines.extend(["", NO_TOOLS_SENTENCE])
        return "\n".join(lines)

    by_server: dict[str, list[RegisteredTool]] = {}
    for tool in tools:
        by_server.setdefault(tool.server, []).append(tool)

    for server, server_tools in by_server.items():
        lines.extend(["", f"### {server}"])
        lines.extend(f"- {tool.llm_name}: {_describe(tool)}" for tool in server_tools)
    return "\n".join(lines)
