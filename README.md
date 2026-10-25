# Local Agent Host

A local "Host Agent" for a chat app. It speaks the [A2A protocol](https://a2a-protocol.org) to your frontend, reasons with a local LLM through LiteLLM (Ollama, vLLM, or any OpenAI-compatible server), and runs tools through [MCP](https://modelcontextprotocol.io) servers on your machine.

```
Chat App  --A2A (JSON-RPC + SSE)-->  Host Agent  --LiteLLM-->  local LLM
                                         |
                                         +--MCP (stdio / SSE)-->  tool servers (files, git, postgres, ...)
```

> Status: the spec is in `.kiro/specs/local-agent-backend/` and implementation is in progress. This guide describes the interface defined there. Exact JSON field names follow the pinned `a2a-sdk` version (A2A 1.0 wire format).

## What you need

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- A local LLM that supports tool calling, for example [Ollama](https://ollama.com) with `llama3.1` or `qwen2.5`
- Node.js 20+ for the example filesystem MCP server (it runs through `npx`)

Windows, macOS, and Linux are all supported. The commands below work in bash, zsh, and PowerShell unless a section says otherwise.

## 1. Install

```sh
uv python install 3.14
uv sync
```

## 2. Configure

Copy `config.example.yaml` to `config.yaml` and adjust it. A minimal setup with one filesystem tool server looks like this:

```yaml
server:
  host: 127.0.0.1
  port: 10000

llm:
  model: ollama_chat/llama3.1
  api_base: http://localhost:11434

mcp_servers:
  - name: fs
    transport: stdio
    command: npx
    # Use an absolute path. Forward slashes work on Windows too: "C:/Users/you/notes"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/you/notes"]

# Tools that need your approval before they run
permissions:
  fs: ["write_*", "edit_*", "move_file", "create_directory"]
```

Notes:

- `${VAR}` in any string value is replaced with an environment variable, which keeps secrets out of the file.
- Relative paths (such as `cwd`) are resolved against the folder that contains `config.yaml`. `~` means your home folder.
- To use a different file, set `AGENT_CONFIG` to its path.

## 3. Run the Host Agent

Start the LLM first:

```sh
ollama pull llama3.1
ollama serve            # skip this if Ollama is already running
```

Then start the agent:

```sh
uv run python -m local_agent
```

To point at another config file:

```sh
# bash / zsh
AGENT_CONFIG=./my-config.yaml uv run python -m local_agent
```

```powershell
# PowerShell
$env:AGENT_CONFIG = "my-config.yaml"; uv run python -m local_agent
```

Logs are JSON lines on stderr. Look for `mcp_server_connected` for each tool server. Press Ctrl+C to stop; the agent shuts down its tool servers and exits with code 0.

Check that it is up:

```sh
curl http://127.0.0.1:10000/health
# {"status":"ok","mcp_servers":[{"name":"fs","status":"available"}]}

curl http://127.0.0.1:10000/.well-known/agent-card.json
```

On Windows PowerShell 5.1, type `curl.exe` instead of `curl`.

## 4. Send a chat message that needs a tool

Put a file named `todo.md` in your notes folder with a few lines in it. Then save this request as `request.json`:

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "SendStreamingMessage",
  "params": {
    "message": {
      "role": "ROLE_USER",
      "messageId": "msg-001",
      "parts": [{ "text": "What's on my todo list? Read todo.md in my notes folder." }]
    }
  }
}
```

Send it and keep the connection open for the stream (`-N` turns off buffering):

```sh
curl -N http://127.0.0.1:10000/ -H "Content-Type: application/json" -H "A2A-Version: 1.0" --data-binary "@request.json"
```

The answer can't come from the model alone, so the agent calls the filesystem server's `read_text_file` tool. You get a stream of Server-Sent Events like this (shortened):

```text
data: {"jsonrpc":"2.0","id":"1","result":{"task":{"id":"<task-id>","contextId":"<context-id>","status":{"state":"TASK_STATE_SUBMITTED"}}}}
data: {... "statusUpdate":{"status":{"state":"TASK_STATE_WORKING","message":{"parts":[{"text":"Discovering local tools..."}]}}}}
data: {... "statusUpdate":{... "text":"read_text_file: /home/you/notes/todo.md..." ... "metadata":{"kind":"tool_call_started","tool_name":"read_text_file","server":"fs","tool_call_id":"call_1"}}}
data: {... "statusUpdate":{... "metadata":{"kind":"tool_call_completed","tool_name":"read_text_file","server":"fs","tool_call_id":"call_1","outcome":"succeeded"}}}
data: {... "artifactUpdate":{"artifact":{"artifactId":"response-1","parts":[{"text":"You have"}]},"append":false}}
data: {... "artifactUpdate":{"artifact":{"artifactId":"response-1","parts":[{"text":" three items: ..."}]},"append":true}}
data: {... "artifactUpdate":{"artifact":{"artifactId":"response-1","parts":[{"text":""}]},"append":true,"lastChunk":true}}
data: {... "statusUpdate":{"status":{"state":"TASK_STATE_COMPLETED"},"final":true}}
```

How to read the stream:

- `statusUpdate` events show progress. Events with `metadata.kind` describe tool activity, so a UI can show "Reading todo.md..." while the tool runs.
- `artifactUpdate` events carry the answer, a few tokens at a time. Join the `text` parts in order. The last chunk is empty and has `lastChunk: true`.
- The stream closes after the final status: `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_CANCELED`, or `TASK_STATE_INPUT_REQUIRED`.

Other things you can do:

- Non-streaming call: use `"method": "SendMessage"`. You get one response with the finished task and its artifacts.
- Follow-up question in the same conversation: add `"contextId": "<context-id>"` to the `message`. The agent remembers earlier turns.
- Fetch a task later, for example after a dropped connection: `{"jsonrpc":"2.0","id":"3","method":"GetTask","params":{"id":"<task-id>"}}`
- Cancel a running task: `{"jsonrpc":"2.0","id":"4","method":"CancelTask","params":{"id":"<task-id>"}}`

## 5. Approve a protected tool

Tools listed under `permissions` pause for approval. Ask for something that writes:

```json
{ "text": "Add 'buy milk' to the end of todo.md." }
```

The stream ends with `TASK_STATE_INPUT_REQUIRED`. The status message names the tool and its arguments, and the metadata has `"kind": "tool_approval_required"`. To continue, send a message to the same task:

```json
{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "SendStreamingMessage",
  "params": {
    "message": {
      "role": "ROLE_USER",
      "messageId": "msg-002",
      "taskId": "<task-id>",
      "contextId": "<context-id>",
      "parts": [{ "data": { "decision": "approve" } }]
    }
  }
}
```

Use `"deny"` to refuse. A plain text part with `yes`, `approve`, `no`, or `deny` also works. If you deny, the agent tells the model and the model explains that the action wasn't performed.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `/health` shows a server as `unavailable` | Is Node.js installed and `npx` on your PATH? Search the logs for `mcp_server_failed`; it names the command and the folders searched. |
| Task fails with "local LLM is unreachable" | Is Ollama running? Does `llm.api_base` match its address? |
| The model answers without using tools | Use a model with tool calling (`llama3.1`, `qwen2.5`). For models without native support, set `llm.native_function_calling: false`. |
| "Context is busy" error | The conversation already has a running or paused task. Wait, approve or deny, or cancel it first. |
| Startup fails with "cannot bind" | Another process is using the port. Change `server.port`. |
| Startup fails with a config report | Each problem is listed with its field path, for example `mcp_servers.0.command`. |

## Security

By default the agent listens on `127.0.0.1` only and needs no token, so only programs on your machine can reach it. If you bind to another address, you must set `server.bearer_token` (at least 32 characters). Every request except the agent card then needs an `Authorization: Bearer <token>` header. Browser frontends must be listed in `server.cors_allowed_origins`.

## Development

```sh
uv run pytest -q                 # unit + property tests
uv run pytest -q -m integration  # integration tests (starts real MCP test servers)
uv run ruff check . ; uv run mypy src
```