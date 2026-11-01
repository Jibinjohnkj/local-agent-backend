"""A stdio MCP server for integration tests, built on the low-level `mcp` SDK server API.

Run it as a script: `python mcp_echo_server.py [--page-size N] [--crash-after N] [--request-log P]`.

Tools (listed in this order, `--page-size` per page so `tools/list` is paginated):

* `echo(text)`: returns `text` unchanged, so non-ASCII text can be round-tripped (17.27).
* `slow(seconds, text)`: waits `seconds`, then returns `text`.
* `error(message)`: a result with `isError` set and `message` as text.
* `hang()`: never answers.
* `write_file(path, content)`: the Protected_Tool of the fixture. It writes nothing; it answers
  `wrote <path>`. Tests list it in `permissions` for the server.
* `crash()`: ends the process at once, without answering (`os._exit(3)`).
* `get_pid()`: the server's process ID as text.

Options:

* `--page-size N`: tools per `tools/list` page (default 2).
* `--crash-after N`: the N-th `tools/call` ends the process at once, before it answers.
* `--request-log PATH`: append one JSON line per event (`start`, `list_tools`, `call_tool`), so a
  test can show which requests reached the server, and the server's pid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

CRASH_EXIT_CODE = 3


def _schema(**properties: str) -> dict[str, Any]:
    """An input schema whose `properties` (name to JSON type) are all required."""
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in properties.items()},
        "required": list(properties),
        "additionalProperties": False,
    }


TOOLS: tuple[types.Tool, ...] = (
    types.Tool(
        name="echo", description="Return the text unchanged.", input_schema=_schema(text="string")
    ),
    types.Tool(
        name="slow",
        description="Wait, then return the text.",
        input_schema=_schema(seconds="number", text="string"),
    ),
    types.Tool(
        name="error", description="Fail with isError.", input_schema=_schema(message="string")
    ),
    types.Tool(name="hang", description="Never answer.", input_schema=_schema()),
    types.Tool(
        name="write_file",
        description="Protected tool; writes nothing.",
        input_schema=_schema(path="string", content="string"),
    ),
    types.Tool(name="crash", description="End the process.", input_schema=_schema()),
    types.Tool(name="get_pid", description="The server's process ID.", input_schema=_schema()),
)


def _text(text: str, *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], is_error=is_error
    )


def build_server(page_size: int, crash_after: int, request_log: str | None) -> Server:
    calls = 0

    def log(event: str, **fields: Any) -> None:
        if request_log is None:
            return
        line = json.dumps({"event": event, "pid": os.getpid(), **fields}, ensure_ascii=False)
        with open(request_log, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    async def on_list_tools(
        _ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        cursor = None if params is None else params.cursor
        log("list_tools", cursor=cursor)
        start = int(cursor) if cursor else 0
        end = start + page_size
        next_cursor = str(end) if end < len(TOOLS) else None
        return types.ListToolsResult(tools=list(TOOLS[start:end]), next_cursor=next_cursor)

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        nonlocal calls
        calls += 1
        arguments = params.arguments or {}
        log("call_tool", name=params.name, arguments=arguments)
        if crash_after and calls >= crash_after:
            os._exit(CRASH_EXIT_CODE)
        match params.name:
            case "echo":
                return _text(str(arguments["text"]))
            case "slow":
                await anyio.sleep(float(arguments["seconds"]))
                return _text(str(arguments["text"]))
            case "error":
                return _text(str(arguments["message"]), is_error=True)
            case "hang":
                await anyio.sleep_forever()
                return _text("unreachable")
            case "write_file":
                return _text(f"wrote {arguments['path']}")
            case "crash":
                os._exit(CRASH_EXIT_CODE)
            case "get_pid":
                return _text(str(os.getpid()))
            case _:
                return _text(f"unknown tool {params.name}", is_error=True)

    log("start")
    return Server("echo-fixture", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--page-size", type=int, default=2)
    parser.add_argument("--crash-after", type=int, default=0)
    parser.add_argument("--request-log", default=None)
    args = parser.parse_args(argv)
    anyio.run(serve, build_server(max(1, args.page_size), args.crash_after, args.request_log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
