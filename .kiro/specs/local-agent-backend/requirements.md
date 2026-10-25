# Requirements Document

## Introduction

The Local Agent Backend is the middleware layer of a local AI chat application. It runs as a single Python 3.14+ service on Windows, macOS, and Linux (Requirement 17) acting as the "Host Agent" and wears three hats at once:

1. **A2A Server (frontend facing)**: exposes Agent2Agent (A2A) protocol endpoints (JSON-RPC and REST) with Server-Sent Events (SSE) streaming, built on the official `a2a-sdk`.
2. **LLM Orchestrator (reasoning layer)**: talks to a local LLM through LiteLLM (Ollama, vLLM, or any OpenAI-compatible base URL), keeps conversation memory, and parses tool-calling output.
3. **MCP Client (tool facing)**: connects to locally configured Model Context Protocol (MCP) servers over stdio or SSE, discovers their tools, and executes tool calls on behalf of the LLM.

The design follows the host-agent patterns in `a2aproject/a2a-samples` (`samples/python/hosts/multiagent`), extended so the host bridges A2A (frontend communication) and MCP (tool execution). Appendices A–C contain the proposed directory structure, Python dependencies, and a request-flow sequence diagram.

## Glossary

- **Backend**: The complete Local Agent Backend Python service described in this document.
- **A2A_Server**: The Backend component that exposes A2A protocol endpoints and SSE streams to the Chat_App, implemented with the `a2a-sdk` package.
- **Chat_App**: The frontend UI that acts as the A2A client and sends user messages to the A2A_Server.
- **Agent_Card**: The A2A metadata document describing the Backend's identity, endpoint URL, capabilities (including streaming), and skills, served at the A2A well-known path.
- **Task**: An A2A unit of work identified by a `task_id`, created in response to a user message.
- **Context**: An A2A conversation grouping identified by a `context_id`; one Context contains one or more Tasks.
- **Task_State**: The A2A lifecycle state of a Task. This Backend uses `submitted`, `working`, `input-required`, `completed`, `failed`, and `canceled`.
- **Status_Event**: An A2A `TaskStatusUpdateEvent` streamed to the Chat_App that carries a Task_State and an optional human-readable status message.
- **Token_Event**: An A2A `TaskArtifactUpdateEvent` streamed to the Chat_App that carries an incremental chunk of LLM-generated text.
- **Orchestrator**: The Backend component that runs the reasoning loop: builds prompts, calls the LLM, detects Tool_Calls, dispatches them to the MCP_Client, and feeds Tool_Results back to the LLM.
- **LLM**: The locally hosted large language model reached through the LLM_Gateway.
- **LLM_Gateway**: The Backend component that wraps LiteLLM to issue streaming chat-completion requests with function-calling to the configured local LLM endpoint.
- **Session_Store**: The Backend component that maps a `context_id` to its LLM message history and maps each `task_id` to its owning `context_id` and Task_State.
- **Message_History**: The ordered list of system, user, assistant, and tool messages for one Context.
- **System_Prompt**: The system-role message the Prompt_Builder places at the start of every LLM request.
- **Prompt_Builder**: The Backend component that constructs the System_Prompt from configured instructions and the currently available tools.
- **MCP_Client**: The Backend component that manages connections to MCP_Servers using the official `mcp` Python SDK.
- **MCP_Server**: An external process or service exposing tools through the Model Context Protocol, reached over the stdio or SSE transport (for example File System, Git, or Postgres servers).
- **MCP_Tool**: A tool advertised by an MCP_Server, consisting of a name, a description, and a JSON Schema for its input.
- **Tool_Registry**: The Backend component that holds the merged catalog of MCP_Tools from all connected MCP_Servers and resolves each LLM-facing tool name to its owning MCP_Server.
- **Schema_Translator**: The Backend component that converts MCP_Tool definitions into the LLM function-calling tool format (OpenAI-compatible `{"type": "function", "function": {...}}`).
- **Tool_Call**: A function-call request emitted by the LLM, containing a call identifier, a tool name, and JSON-encoded arguments.
- **Tool_Result**: The content returned by an MCP_Server for a Tool_Call, or an error description when the Tool_Call fails.
- **Permission_Policy**: The configuration that marks specific MCP_Tools as requiring explicit user approval before execution.
- **Protected_Tool**: An MCP_Tool that the Permission_Policy marks as requiring user approval.
- **Approval_Gate**: The Backend component that pauses a Task before a Protected_Tool executes and resumes the Task based on the Chat_App's continuation message.
- **Config_Loader**: The Backend component that reads and validates the Backend configuration file and environment variables.
- **Backend_Config**: The validated configuration object covering server host/port, LLM settings, MCP_Server definitions, Permission_Policy, and orchestration limits.
- **Max_Tool_Iterations**: The configured maximum number of LLM-to-tool round trips the Orchestrator performs within a single Task (default 10).
- **Tool_Timeout**: The configured maximum duration in seconds for a single Tool_Call (default 60).
- **Approval_Keywords**: The configured text replies, compared case-insensitively after trimming, that the Approval_Gate treats as approval (default `approve`, `yes`) or denial (default `deny`, `no`).
- **Approval_Timeout**: The optional configured duration (1 to 86,400 seconds) after which a pending Protected_Tool Tool_Call is treated as denied; unset by default.
- **Max_Tool_Output_Size**: The configured maximum Tool_Result length in characters before truncation (default 32,000; range 1,000 to 1,000,000).
- **Context_Window_Budget**: The configured maximum token estimate for the messages of one chat-completion request (default 8,192; range 1,024 to 1,000,000).
- **Session_Idle_Timeout**: The optional configured duration (60 to 604,800 seconds) after which an idle Context is evicted from the Session_Store; unset by default.
- **Per_Session_Discovery**: The optional Backend_Config flag that makes the MCP_Client refresh tool lists from available MCP_Servers whenever a new Context is created.
- **LLM-facing name**: The sanitized, unique tool name of the form `<server_name>__<tool_name>` (at most 64 characters) that the Tool_Registry exposes to the LLM.
- **Supported_Platform**: One of the operating system and CPU architecture combinations listed in Requirement 17.1, running the CPython 3.14 standard (GIL-enabled) build.
- **Process_Tree**: A `stdio` MCP_Server process started by the MCP_Client together with every descendant process that remains in its process group (macOS and Linux) or job object (Windows), for example `npx.cmd` → `node` on Windows.
- **Shutdown_Signal**: An operating system request to stop the Backend: `SIGINT` or `SIGTERM` on macOS and Linux; a Ctrl+C (`CTRL_C_EVENT`), Ctrl+Break (`CTRL_BREAK_EVENT`), or console close (`CTRL_CLOSE_EVENT`) event on Windows.
- **Graceful_Shutdown**: The Backend shutdown sequence defined in Requirement 17.14, identical on every Supported_Platform.
- **Path_Field**: A Backend_Config value interpreted as a file system path: the `AGENT_CONFIG` environment variable value, a `stdio` MCP_Server command that contains a path separator, a `stdio` MCP_Server working directory, and any other Backend_Config field the Backend_Config schema declares as a path.
- **CI_Pipeline**: The automated continuous-integration workflow that installs the Backend and runs its test suites on every push and pull request.

## Requirements

### Requirement 1: A2A Server Exposure and Agent Card

**User Story:** As a Chat_App developer, I want the Backend to expose standard A2A endpoints, so that any A2A-conformant client can discover and talk to the Host Agent.

#### Acceptance Criteria

1. THE A2A_Server SHALL expose the A2A JSON-RPC endpoint, the A2A REST endpoints, and the SSE streaming endpoint using the `a2a-sdk` server components, all reachable at the host and port defined in the Backend_Config.
2. WHEN the A2A_Server receives a GET request at the A2A well-known agent card path, THE A2A_Server SHALL return the Agent_Card as a JSON document without requiring a bearer token, including when the bearer token requirement of Requirement 16 is active.
3. THE Agent_Card SHALL declare the `streaming` capability as true.
4. THE Agent_Card SHALL contain a non-empty name, a non-empty description, a non-empty version, an endpoint URL, default input modes that include text, default output modes that include text, and at least one skill, where each skill has a non-empty identifier, name, and description.
5. THE Agent_Card SHALL set the endpoint URL to the scheme, host, and port the A2A_Server is bound to according to the Backend_Config, so that a client following the endpoint URL reaches the A2A JSON-RPC endpoint.
6. WHEN the Backend starts, THE A2A_Server SHALL bind to the host and port defined in the Backend_Config and accept requests within 10 seconds of startup, not counting time spent on MCP_Server connections.
7. IF the configured port is outside the range 1 to 65535, THEN THE Config_Loader SHALL reject the Backend_Config at startup with an error naming the port field and its invalid value.
8. IF the A2A_Server cannot bind to the configured host and port (for example because the port is already in use, detected on every Supported_Platform as defined in Requirement 17.32), THEN THE Backend SHALL stop startup, exit with a non-zero exit status, and report an error naming the host and port.
9. IF the A2A_Server receives a request that is not parseable JSON, is not a valid JSON-RPC 2.0 request, names a method that A2A does not define, or carries parameters that violate the A2A schema for that method, THEN THE A2A_Server SHALL return the JSON-RPC error response defined for that category (parse error, invalid request, method not found, or invalid params), echo the request identifier when one can be read, include an error message naming the invalid element, and create or modify no Task.

### Requirement 2: Task Lifecycle Management

**User Story:** As a Chat_App user, I want each request to follow the standard A2A task states, so that the UI can show accurate progress and outcomes.

#### Acceptance Criteria

1. WHEN the A2A_Server receives a user message without a `task_id`, THE A2A_Server SHALL create a new Task whose `task_id` differs from every other `task_id` issued during the Backend's lifetime, record the Task in the Session_Store, and set the Task_State to `submitted`.
2. WHEN the A2A_Server receives a user message without a `context_id`, THE A2A_Server SHALL assign the Task a new `context_id` that differs from every existing `context_id` in the Session_Store, and include that `context_id` in the Task returned or streamed to the Chat_App.
3. WHEN the A2A_Server receives a user message with a `context_id` present in the Session_Store, THE A2A_Server SHALL associate the new Task with that Context so that the Session_Store maps the new `task_id` to that `context_id`.
4. WHEN the Orchestrator begins processing a Task in the `submitted` state, THE A2A_Server SHALL transition the Task_State to `working` and emit a Status_Event with Task_State `working` before emitting any Token_Event for that Task.
5. WHEN the Orchestrator produces a final LLM response containing no Tool_Calls, THE A2A_Server SHALL store the response text as a Task artifact, transition the Task_State to `completed`, and emit a Status_Event with Task_State `completed` marked as final.
6. IF an unrecoverable error occurs while processing a Task, where an unrecoverable error is the LLM_Gateway exhausting its retries (Requirement 10.7) or an unhandled exception raised by the Orchestrator outside a Tool_Call, THEN THE A2A_Server SHALL transition the Task_State to `failed` and emit a final Status_Event containing a human-readable message that states the error cause without stack traces or Backend_Config secret values, including when an unrelated Tool_Call failure occurs in the same Task at the same time; a Tool_Call failure (Requirement 9.8) SHALL NOT by itself trigger this transition.
7. WHEN the A2A_Server receives a cancel request for a Task in the `working` or `input-required` state, THE A2A_Server SHALL stop processing the Task within 2 seconds, transition the Task_State to `canceled`, discard any pending Tool_Call held by the Approval_Gate for that Task, and emit a final Status_Event with Task_State `canceled`.
8. IF the A2A_Server receives a message referencing a Task in the `completed`, `failed`, or `canceled` state, THEN THE A2A_Server SHALL reject the message with an A2A error indicating that the Task is in a terminal state, and leave the Task_State, artifacts, and Message_History unchanged.
9. THE A2A_Server SHALL permit only the Task_State transitions `submitted→working`, `working→input-required`, `input-required→working`, `working→completed`, `working→failed`, `input-required→failed`, `working→canceled`, and `input-required→canceled`; IF any other transition is attempted, THEN THE A2A_Server SHALL leave the Task_State unchanged and log the rejected transition with the `task_id`, current Task_State, and requested Task_State.
10. WHEN the A2A_Server receives a get-task request for a `task_id` present in the Session_Store, THE A2A_Server SHALL return the Task with its `task_id`, `context_id`, current Task_State, and all artifacts accumulated up to the time of the request.
11. IF the A2A_Server receives a message, get-task request, or cancel request referencing a `task_id` absent from the Session_Store, THEN THE A2A_Server SHALL respond with an A2A error indicating that the Task was not found and SHALL NOT create a Task or Context.
12. IF the A2A_Server receives a cancel request for a Task in the `completed`, `failed`, or `canceled` state, THEN THE A2A_Server SHALL respond with an A2A error indicating that the Task cannot be canceled and leave the Task_State unchanged.
13. IF the A2A_Server receives a user message without a `task_id` and with a `context_id` absent from the Session_Store, THEN THE A2A_Server SHALL create a new Context with the supplied `context_id` and an empty Message_History, and associate the new Task with that Context.

### Requirement 3: SSE Token Streaming

**User Story:** As a Chat_App user, I want to see the assistant's answer appear token by token, so that the chat feels responsive.

#### Acceptance Criteria

1. WHEN the Chat_App sends a streaming message request, THE A2A_Server SHALL respond with an SSE stream whose first event is emitted within 1 second of receiving the request and whose events are the A2A Status_Events and Token_Events of that Task only.
2. WHILE the LLM_Gateway receives non-empty text chunks from the LLM, THE A2A_Server SHALL emit each non-empty text chunk as one Token_Event within 100 ms of the LLM_Gateway receiving the chunk, in the order the LLM_Gateway received the chunks, and SHALL emit no Token_Event for empty text chunks or Tool_Call fragments.
3. THE A2A_Server SHALL emit all Token_Events of one assistant response into a single artifact with an artifact identifier unique within the Task, setting the append flag to false on the first chunk, to true on every subsequent chunk, and setting the last-chunk flag to true only on the final chunk of that assistant response.
4. WHEN an assistant response containing text completes, THE A2A_Server SHALL ensure that concatenating the text of that response's Token_Events in emission order equals the full assistant text stored for that response in the Message_History (stream reassembly property).
5. WHEN the Task reaches the `completed`, `failed`, `canceled`, or `input-required` state, THE A2A_Server SHALL emit one Status_Event carrying that Task_State marked as final and close the SSE stream within 1 second after emitting that Status_Event.
6. IF the Chat_App disconnects from the SSE stream before the Task reaches a final state, THEN THE Backend SHALL continue processing the Task to a final state and retain the Task, its Task_State, and all of its artifacts for retrieval through a get-task request until the owning Context is evicted from the Session_Store.
7. WHEN the Chat_App sends a non-streaming message request, THE A2A_Server SHALL return a single response containing the Task, its Task_State, and its accumulated artifacts only after the Task reaches the `completed`, `failed`, `canceled`, or `input-required` state.
8. IF the LLM stream fails after one or more Token_Events of an assistant response have been emitted, THEN THE A2A_Server SHALL emit no further Token_Events for that artifact, retain the already-emitted text in the Task artifact, and emit a final Status_Event with Task_State `failed` containing a human-readable error message indicating that the response was interrupted.
9. WHEN the A2A_Server emits a Status_Event with Task_State `canceled`, THE A2A_Server SHALL emit no further Token_Events or Status_Events for that Task on any SSE stream.

### Requirement 4: Intermediate Status Streaming

**User Story:** As a Chat_App user, I want to see what the agent is doing while tools run, so that the UI can render meaningful loading states.

#### Acceptance Criteria

1. WHEN the Orchestrator begins processing a Task and before the Orchestrator sends the first chat-completion request for that Task, THE A2A_Server SHALL emit exactly one Status_Event with Task_State `working` and the message "Discovering local tools...".
2. WHEN the Orchestrator dispatches a Tool_Call to the MCP_Client, THE A2A_Server SHALL emit, before the MCP_Client sends the Tool_Call to the MCP_Server, a Status_Event with Task_State `working` whose message contains the original MCP_Tool name and, if the Tool_Call arguments contain a top-level string argument whose name contains `path`, `file`, `uri`, or `url` (case-insensitive), the value of the first such argument truncated to at most 120 characters (for example "Reading config.json...").
3. WHEN a dispatched Tool_Call produces a Tool_Result, THE A2A_Server SHALL emit, before the Orchestrator appends that Tool_Result to the Message_History, a Status_Event with Task_State `working` whose message contains the original MCP_Tool name and the outcome `succeeded` or `failed`, where `failed` applies when the Tool_Result is flagged as an error by the MCP_Server, results from a Tool_Timeout, or results from a transport or protocol exception, and `succeeded` applies in all other cases.
4. THE A2A_Server SHALL attach machine-readable metadata to each tool-related Status_Event containing the event kind (one of `tool_call_started`, `tool_call_completed`, or `tool_approval_required`), the original MCP_Tool name, the owning MCP_Server name, the Tool_Call identifier, and, for `tool_call_completed` events only, the outcome `succeeded` or `failed`.
5. FOR ALL dispatched Tool_Calls in a Task, including Tool_Calls executed concurrently, THE A2A_Server SHALL emit exactly one `tool_call_started` Status_Event and exactly one `tool_call_completed` Status_Event per Tool_Call identifier, with the `tool_call_started` event emitted before the `tool_call_completed` event for the same Tool_Call identifier (event ordering property).
6. IF a Tool_Call is resolved without being sent to an MCP_Server because the tool name is absent from the Tool_Registry, the arguments are not valid JSON, the arguments fail input schema validation, the owning MCP_Server is unavailable, or the user denied the Tool_Call, THEN THE A2A_Server SHALL emit no `tool_call_started` Status_Event for that Tool_Call and SHALL emit exactly one `tool_call_completed` Status_Event with the outcome `failed`.
7. WHEN the Approval_Gate approves a suspended Protected_Tool Tool_Call, THE MCP_Client SHALL send that Tool_Call to the owning MCP_Server, unless the Task is canceled before the send (Requirements 2.7 and 14.4).
8. WHEN the Approval_Gate approves a suspended Protected_Tool Tool_Call, THE A2A_Server SHALL emit the `tool_call_started` Status_Event for that Tool_Call after the Task_State transitions back to `working` and before the MCP_Client sends the Tool_Call to the owning MCP_Server.
9. IF a Task has reached the `completed`, `failed`, or `canceled` state, THEN THE A2A_Server SHALL emit no further tool-related Status_Events for that Task, including for in-flight Tool_Calls that finish after the transition.

### Requirement 5: Human-in-the-Loop Permission Gate

**User Story:** As a Chat_App user, I want the agent to ask me before running risky tools such as file deletion, so that I stay in control of destructive actions.

#### Acceptance Criteria

1. THE Config_Loader SHALL load the Permission_Policy as a mapping from MCP_Server name to a list of 0 to 100 tool-name patterns, where each pattern is either an exact MCP_Tool name or a wildcard pattern using `*` (any sequence of characters) and `?` (any single character), matched case-sensitively against the original MCP_Tool name (not the LLM-facing name).
2. WHEN the Config_Loader has loaded the Permission_Policy, THE Tool_Registry SHALL mark every MCP_Tool of each MCP_Server listed in the Permission_Policy that matches at least one of that MCP_Server's patterns as a Protected_Tool.
3. IF the Permission_Policy references an MCP_Server name not defined in the Backend_Config, or contains an empty pattern, or a pattern longer than 128 characters, THEN THE Config_Loader SHALL reject the Backend_Config at startup with an error naming the MCP_Server and the invalid pattern.
4. WHEN the LLM emits a Tool_Call targeting a Protected_Tool whose arguments are valid JSON and pass validation against the MCP_Tool input schema, THE Approval_Gate SHALL suspend execution of that Tool_Call before the MCP_Client sends the Tool_Call to the MCP_Server.
5. IF a Tool_Call targeting a Protected_Tool has arguments that are not valid JSON or fail validation against the MCP_Tool input schema, THEN THE Orchestrator SHALL append the corresponding error Tool_Result defined in Requirement 9 without requesting user approval and without transitioning the Task_State to `input-required`.
6. WHEN the Approval_Gate suspends a Tool_Call, THE A2A_Server SHALL transition the Task_State to `input-required` and emit a Status_Event containing the LLM-facing tool name, the original MCP_Tool name, the MCP_Server name, the Tool_Call identifier, the parsed Tool_Call arguments, and a prompt asking the user to approve or deny, with machine-readable metadata containing the event kind `tool_approval_required`, the tool name, the MCP_Server name, and the Tool_Call identifier.
7. WHILE a Task is in the `input-required` state, THE Approval_Gate SHALL retain the pending Tool_Call (identifier, LLM-facing name, and arguments), all not-yet-executed Tool_Calls from the same LLM turn, and the Message_History in the Session_Store for the lifetime of the Backend process, so that processing resumes from the suspended Tool_Call without issuing a new LLM request.
8. WHEN the Chat_App sends a continuation message referencing the `task_id` of a Task in the `input-required` state and the message contains a recognizable approval decision, THE Approval_Gate SHALL transition the Task_State to `working`, emit a Status_Event with Task_State `working`, and resume processing.
9. WHEN the recognized approval decision is approve, THE MCP_Client SHALL execute the suspended Tool_Call with arguments identical to the original Tool_Call arguments, and THE Orchestrator SHALL handle the Tool_Result as defined in Requirements 8 and 9.
10. WHEN the recognized approval decision is deny, THE Orchestrator SHALL append a Tool_Result referencing the suspended Tool_Call identifier and stating that the user denied the Tool_Call, SHALL not send the Tool_Call to the MCP_Server, and SHALL continue the reasoning loop with the Task_State `working`.
11. IF a continuation message for a Task in the `input-required` state does not contain a recognizable approval decision, THEN THE Approval_Gate SHALL keep the Task in the `input-required` state, keep the pending Tool_Call unchanged, not append the message to the Message_History, and emit a Status_Event with Task_State `input-required` requesting an explicit approve or deny response.
12. THE Approval_Gate SHALL recognize an approval decision as either a structured data part whose `decision` value is exactly `approve` or `deny`, or a text part whose content, after trimming leading and trailing whitespace and comparing case-insensitively, exactly equals one of the configured approval keywords (default `approve`, `yes`) or denial keywords (default `deny`, `no`); if a message contains both a valid structured data part and a text part, the structured data part SHALL take precedence.
13. WHEN the LLM emits multiple Tool_Calls in one turn that include Protected_Tools, THE Approval_Gate SHALL request approval for one Protected_Tool Tool_Call at a time in Tool_Call order, with at most one pending approval per Task, and THE Orchestrator SHALL append all Tool_Results for that turn to the Message_History in Tool_Call order as defined in Requirement 8.
14. WHERE the Backend_Config sets an approval timeout between 1 and 86,400 seconds, IF no continuation message with a recognizable approval decision arrives within the approval timeout after the Task enters `input-required`, THEN THE Approval_Gate SHALL transition the Task_State to `working` and THE Orchestrator SHALL append a Tool_Result stating that the Tool_Call was denied because the approval timeout expired.
15. WHERE the Backend_Config does not set an approval timeout, THE Approval_Gate SHALL keep the Task in the `input-required` state until a recognizable approval decision or a cancel request arrives.

### Requirement 6: MCP Server Configuration and Connection

**User Story:** As a power user, I want to declare which local MCP servers the agent can use, so that I control the tools available to the LLM.

#### Acceptance Criteria

1. THE Config_Loader SHALL load from the Backend_Config a list of 0 to 50 MCP_Server definitions, each containing a name of 1 to 32 characters restricted to `a-z`, `A-Z`, `0-9`, `_`, and `-`, a transport type of `stdio` or `sse`, transport parameters, and an optional `enabled` flag that defaults to true when omitted.
2. THE Config_Loader SHALL accept, for `stdio` MCP_Servers, a required non-empty command (resolved to an executable as defined in Requirements 17.4 to 17.6), an optional argument list defaulting to empty, an optional working directory (resolved as defined in Requirements 17.16 to 17.19), and optional environment variables given as string key-value pairs (merged as defined in Requirement 17.22).
3. THE Config_Loader SHALL accept, for `sse` MCP_Servers, a required URL using the `http` or `https` scheme and optional HTTP headers given as string key-value pairs.
4. IF an MCP_Server definition is missing a required field, uses an unsupported transport type, has a name that violates the allowed length or character set, or has a name that duplicates another MCP_Server definition's name, THEN THE Config_Loader SHALL reject the Backend_Config at startup, stop startup, and report an error naming the MCP_Server and the invalid field.
5. WHEN the Backend starts, THE MCP_Client SHALL connect to every MCP_Server whose `enabled` flag is true, complete the MCP initialization handshake, mark each successfully initialized MCP_Server as available, and make no connection attempt to MCP_Servers whose `enabled` flag is false.
6. WHEN the Backend starts with more than one enabled MCP_Server, THE MCP_Client SHALL start all connection attempts concurrently, so that the total startup connection phase completes within 15 seconds plus 2 seconds regardless of the number of enabled MCP_Servers.
7. IF the MCP_Client does not complete the MCP initialization handshake with an MCP_Server within 15 seconds, or the connection attempt fails earlier (for example the `stdio` command cannot be started, the subprocess exits, the URL is unreachable, or the handshake returns an error), THEN THE MCP_Client SHALL write a log record naming the MCP_Server and the failure reason, mark that MCP_Server as unavailable, and continue startup with the remaining MCP_Servers, including when every MCP_Server is unavailable and the Tool_Registry is empty.
8. IF a connected MCP_Server terminates or drops its connection, THEN THE MCP_Client SHALL mark that MCP_Server as unavailable, remove the MCP_Server's tools from the Tool_Registry, and attempt reconnection up to 5 times with exponential backoff starting at a 1-second delay, doubling after each failed attempt, and capped at 30 seconds per delay.
9. WHEN a reconnection attempt to an unavailable MCP_Server completes the MCP initialization handshake, THE MCP_Client SHALL mark that MCP_Server as available and re-register its MCP_Tools in the Tool_Registry before any subsequent LLM request uses the tool list.
10. IF all 5 reconnection attempts to an MCP_Server fail, THEN THE MCP_Client SHALL stop reconnection attempts, write a log record naming the MCP_Server, and keep the MCP_Server marked as unavailable until the Backend restarts.
11. WHEN the Backend shuts down, THE MCP_Client SHALL close every MCP_Server session and terminate the Process_Tree of every `stdio` MCP_Server subprocess the Backend started, using the platform-specific termination sequence of Requirement 17.11 and forcibly killing any process of the Process_Tree still running 5 seconds after termination was requested (2 seconds in the Windows console-close case of Requirement 17.15), so that no process of any Backend-started Process_Tree remains running after shutdown completes.
12. WHERE the Backend_Config enables per-session discovery, WHEN a new Context is created, THE MCP_Client SHALL refresh the tool list from every MCP_Server currently marked as available before the first LLM request of that Context.
13. WHERE the Backend_Config enables per-session discovery, IF a tool-list refresh for an MCP_Server fails, THEN THE MCP_Client SHALL keep that MCP_Server's previously registered MCP_Tools in the Tool_Registry, write a log record naming the MCP_Server, and let the first LLM request of the Context proceed immediately with those previously registered MCP_Tools without waiting for a successful refresh.

### Requirement 7: Tool Discovery and Schema Translation

**User Story:** As a developer, I want MCP tool schemas translated automatically into the LLM's function-calling format, so that new tools work without code changes.

#### Acceptance Criteria

1. WHEN the MCP_Client completes the MCP initialization handshake with an MCP_Server, THE MCP_Client SHALL request the MCP_Server's tool list, follow every pagination cursor until the MCP_Server returns no further cursor, and register every returned MCP_Tool in the Tool_Registry under that MCP_Server's name.
2. THE Schema_Translator SHALL convert each MCP_Tool into an OpenAI-compatible function tool definition containing the LLM-facing name, the MCP_Tool description (or an empty string when the MCP_Tool has no description), and the translated input JSON Schema as the `parameters` object.
3. THE Tool_Registry SHALL assign each MCP_Tool an LLM-facing name of the form `<server_name>__<tool_name>`, built by replacing every character outside `a-z`, `A-Z`, `0-9`, `_`, and `-` with `_` and then truncating the result to a maximum of 64 characters.
4. FOR ALL registered MCP_Tools, resolving the LLM-facing name through the Tool_Registry SHALL return the original, unsanitized MCP_Server name and original, unsanitized MCP_Tool name (name round-trip property).
5. FOR ALL MCP_Tools whose input schema declares `type` as `object`, the `parameters` object produced by the Schema_Translator SHALL contain the same `type`, `properties`, and `required` values as the MCP_Tool input schema (schema preservation property).
6. IF an MCP_Tool input schema has no `type` field, THEN THE Schema_Translator SHALL produce a `parameters` object with `type` set to `object`, the `properties` and `required` values copied from the input schema when present, and an empty `properties` object when the input schema has no `properties` field.
7. IF two MCP_Tools produce the same LLM-facing name after sanitization, THEN THE Tool_Registry SHALL keep the name of the first-registered MCP_Tool (ordered by MCP_Server position in the Backend_Config, then by position in that MCP_Server's tool list) and append the suffix `_2`, `_3`, and so on to each later MCP_Tool, shortening the base name so that the suffixed name stays within 64 characters and every LLM-facing name is unique.
8. IF an MCP_Tool input schema fails validation against the JSON Schema meta-schema, or declares a `type` other than `object`, THEN THE Schema_Translator SHALL exclude that MCP_Tool from the Tool_Registry, log a warning naming the MCP_Server and MCP_Tool, and continue registering the remaining MCP_Tools of that MCP_Server.
9. WHEN the Orchestrator builds a chat-completion request, THE Tool_Registry SHALL provide tool definitions for MCP_Tools from MCP_Servers currently marked as available and SHALL omit all MCP_Tools from MCP_Servers marked as unavailable.
10. IF an MCP_Server returns an error for the tool list request or does not respond within 15 seconds, THEN THE MCP_Client SHALL log a warning naming the MCP_Server, register no MCP_Tools for that MCP_Server, and continue tool discovery for the remaining MCP_Servers.
11. WHEN the MCP_Client reconnects to an MCP_Server that was marked as unavailable, THE MCP_Client SHALL request that MCP_Server's tool list again and replace that MCP_Server's entries in the Tool_Registry with the newly returned MCP_Tools.
12. WHEN the Tool_Registry assigns LLM-facing names to the same set of MCP_Servers and MCP_Tools in the same Backend_Config order, THE Tool_Registry SHALL produce identical LLM-facing names on every Backend start.

### Requirement 8: Tool Execution and Result Injection

**User Story:** As a Chat_App user, I want the agent to run the tools the LLM selects and use their results, so that answers are grounded in my local data.

#### Acceptance Criteria

1. WHEN the LLM_Gateway returns a complete Tool_Call to the Orchestrator, THE Orchestrator SHALL resolve the Tool_Call's LLM-facing tool name through the Tool_Registry to exactly one owning MCP_Server name and one original MCP_Tool name before dispatching the Tool_Call.
2. WHEN the Orchestrator resolves a Tool_Call that does not target a Protected_Tool and whose arguments pass JSON parsing and input-schema validation, THE MCP_Client SHALL send the Tool_Call to the owning MCP_Server using the original MCP_Tool name and the parsed arguments, unmodified, and await the Tool_Result for at most the Tool_Timeout.
3. WHEN the LLM emits between 2 and 32 Tool_Calls in a single turn that do not target Protected_Tools, THE Orchestrator SHALL start every one of those Tool_Calls without waiting for any other Tool_Call in the same turn to finish, so that the turn's total tool execution time is no more than the duration of the slowest Tool_Call plus 500 ms.
4. WHEN a Tool_Result is received for a Tool_Call, THE Orchestrator SHALL append the Tool_Result to the Message_History as exactly one tool-role message whose Tool_Call identifier equals the identifier of the matching Tool_Call.
5. WHEN the LLM emits multiple Tool_Calls in a single turn, including turns that mix Protected_Tools and non-protected MCP_Tools, THE Orchestrator SHALL append the Tool_Results to the Message_History in the same order in which the LLM emitted the Tool_Calls, independent of completion order, so that each Tool_Call has exactly one Tool_Result.
6. WHEN the Orchestrator converts an MCP_Server response into a Tool_Result, THE Orchestrator SHALL join the text of all MCP text content items in their received order, separated by a single newline, and SHALL replace each non-text MCP content item (image, audio, or embedded resource) with a text description that contains the content type, the MIME type when provided, and the resource URI when provided.
7. WHEN all Tool_Results for a turn are appended to the Message_History, THE Orchestrator SHALL send the updated Message_History, including the System_Prompt as the first message and the translated tool definitions, to the LLM through the LLM_Gateway as the next reasoning step of the same Task.
8. WHEN the Orchestrator appends the Tool_Results of the turn that brings the count of LLM-to-tool round trips in a Task to Max_Tool_Iterations (one round trip being one LLM response that contains at least one Tool_Call, followed by appending that response's Tool_Results), THE Orchestrator SHALL send one more chat-completion request with tools disabled, stream the resulting text as Token_Events, and transition the Task_State to `completed`.
9. IF a Tool_Result's text exceeds the configured maximum tool output size (default 32,000 characters, configurable from 1,000 to 1,000,000 characters), THEN THE Orchestrator SHALL keep the first characters of the Tool_Result up to the maximum size and append a truncation notice stating the original length in characters and the maximum size applied.
10. WHEN the LLM emits one or more Tool_Calls in a turn, THE Orchestrator SHALL append the assistant message containing all of that turn's Tool_Calls (and any text streamed in that turn) to the Message_History before appending any of that turn's Tool_Results.
11. IF an MCP_Server returns a successful Tool_Result that contains no content items, THEN THE Orchestrator SHALL append a Tool_Result stating that the tool returned no content and continue the reasoning loop.
12. IF the LLM emits Tool_Calls in response to the tools-disabled final-answer request made after Max_Tool_Iterations is reached, THEN THE Orchestrator SHALL execute none of those Tool_Calls, keep only the response's text as the final assistant message in the Message_History, and transition the Task_State to `completed`.

### Requirement 9: Tool Error Handling and Self-Correction

**User Story:** As a Chat_App user, I want tool failures reported to the LLM instead of crashing the chat, so that the LLM can retry or explain what went wrong.

#### Acceptance Criteria

1. IF an MCP_Server returns a Tool_Result flagged as an error, THEN THE Orchestrator SHALL append the MCP_Server's error content, converted to text and subject to the configured maximum tool output size truncation rule of Requirement 8, to the Message_History as the tool-role message referencing that Tool_Call's identifier, and continue the reasoning loop.
2. IF a Tool_Call has not returned a Tool_Result within the Tool_Timeout (default 60 seconds) measured from the moment the MCP_Client sends the Tool_Call, THEN THE MCP_Client SHALL cancel the in-flight Tool_Call request to the MCP_Server, and THE Orchestrator SHALL append a Tool_Result stating the tool's LLM-facing name and that it timed out after the Tool_Timeout duration in seconds.
3. IF the LLM emits a Tool_Call with a tool name absent from the Tool_Registry, THEN THE Orchestrator SHALL append, without calling any MCP_Server, a Tool_Result stating the unknown tool name and listing every LLM-facing tool name currently in the Tool_Registry (or stating that no tools are available when the Tool_Registry is empty).
4. IF the LLM emits Tool_Call arguments that are not valid JSON or that parse to a JSON value other than an object, THEN THE Orchestrator SHALL append, without calling the MCP_Server, a Tool_Result containing the JSON parse error description (or a statement that the arguments must be a JSON object).
5. IF the parsed Tool_Call arguments fail validation against the MCP_Tool input schema, THEN THE Orchestrator SHALL append, without calling the MCP_Server, a Tool_Result listing every validation error, each with the JSON path of the offending argument and the violated schema constraint.
6. IF the owning MCP_Server is marked as unavailable when a Tool_Call is dispatched, THEN THE Orchestrator SHALL append, without sending the Tool_Call to any MCP_Server, a Tool_Result stating the MCP_Server name and that the MCP_Server is unavailable.
7. IF the MCP_Client raises a transport or protocol exception during a Tool_Call, THEN THE Orchestrator SHALL append a Tool_Result containing the exception type name and exception message, excluding stack traces and values of Backend_Config fields marked as secret, and continue the reasoning loop.
8. IF a Tool_Call fails for any reason covered by criteria 1 through 7, THEN THE Orchestrator SHALL keep the Task_State `working`, emit no `failed` Status_Event for that failure, and send the updated Message_History to the LLM for the next reasoning step, subject to the Max_Tool_Iterations limit.
9. IF an MCP_Server returns a response for a Tool_Call after the Tool_Timeout has elapsed for that Tool_Call, THEN THE Orchestrator SHALL discard the late response and leave the timeout Tool_Result unchanged in the Message_History.
10. WHEN the Orchestrator finishes processing a turn containing one or more failed Tool_Calls, THE Message_History SHALL contain exactly one tool-role message for each Tool_Call identifier emitted in that turn, in Tool_Call order, regardless of which Tool_Calls succeeded or failed.

### Requirement 10: LLM Gateway

**User Story:** As a power user, I want to point the agent at my local model server, so that all reasoning stays on my machine.

#### Acceptance Criteria

1. THE LLM_Gateway SHALL send every chat-completion request through LiteLLM using the model identifier and API base URL from the Backend_Config, and SHALL include the API key from the Backend_Config only when the Backend_Config sets an API key.
2. THE LLM_Gateway SHALL request streaming responses and SHALL yield each text chunk and each Tool_Call fragment to the Orchestrator as soon as LiteLLM delivers it, without waiting for the complete response.
3. WHEN the LLM streams a Tool_Call in multiple fragments, THE LLM_Gateway SHALL assemble the fragments by Tool_Call index into complete Tool_Calls, and SHALL return the complete Tool_Calls to the Orchestrator after the stream ends, in ascending Tool_Call index order.
4. THE LLM_Gateway SHALL pass the Tool_Registry's translated tool definitions in every chat-completion request unless the Orchestrator disables tools for that request; WHEN the Orchestrator disables tools or the Tool_Registry contains no tools, THE LLM_Gateway SHALL send the request with no tool definitions.
5. THE LLM_Gateway SHALL apply the temperature, maximum output tokens, and request timeout from the Backend_Config to every chat-completion request, using the defaults of temperature 0.7, maximum output tokens 2048, and request timeout 120 seconds for any of these fields that the Backend_Config omits.
6. IF a chat-completion request fails before the LLM_Gateway has yielded any chunk, because the LLM endpoint refuses the connection, does not respond within the request timeout, or returns a server-side error, THEN THE LLM_Gateway SHALL retry the request up to 2 times (3 attempts in total), waiting 1 second before the first retry and 2 seconds before the second retry.
7. IF all 3 attempts of a chat-completion request fail, THEN THE Orchestrator SHALL transition the Task_State to `failed` with an error message stating that the local LLM is unreachable, and the Message_History SHALL contain no assistant message from the failed request.
8. WHERE the Backend_Config marks the configured model as lacking native function-calling support, THE LLM_Gateway SHALL use LiteLLM's prompt-based function-calling fallback and SHALL return Tool_Calls to the Orchestrator in the same assembled Tool_Call format as for models with native support.
9. IF the LLM endpoint rejects a chat-completion request as invalid (for example an unknown model identifier or an authentication failure), THEN THE LLM_Gateway SHALL not retry the request, and THE Orchestrator SHALL transition the Task_State to `failed` with an error message stating the rejection reason reported by the LLM endpoint.
10. IF the LLM stream fails after the LLM_Gateway has yielded at least one chunk, THEN THE LLM_Gateway SHALL not retry the request, and THE Orchestrator SHALL transition the Task_State to `failed` with an error message stating that the LLM response was interrupted.
11. IF an assembled Tool_Call has no call identifier when the stream ends, THEN THE LLM_Gateway SHALL assign it a call identifier that is unique within the Message_History and SHALL pass the accumulated argument text to the Orchestrator unchanged, even when it is not valid JSON.

### Requirement 11: Stateful Session and Context Memory

**User Story:** As a Chat_App user, I want the agent to remember earlier turns in the same conversation, so that I can ask follow-up questions.

#### Acceptance Criteria

1. THE Session_Store SHALL map each `context_id` to exactly one Message_History, and SHALL create an empty Message_History when a new Context is created.
2. THE Session_Store SHALL map each `task_id` to exactly one owning `context_id` and to the Task's current Task_State, and SHALL update the stored Task_State on every Task_State transition before the corresponding Status_Event is emitted.
3. WHEN a Task is created in an existing Context, THE Orchestrator SHALL load that Context's Message_History and append the new user message to the end of the Message_History before sending the first chat-completion request for that Task.
4. WHEN a Task reaches the `completed`, `failed`, or `canceled` state, THE Session_Store SHALL contain the user message, every assistant message, every Tool_Call, and every Tool_Result produced during that Task up to that point, in production order.
5. FOR ALL Message_Histories, every tool-role message SHALL reference a Tool_Call identifier emitted by a preceding assistant message in the same Message_History, and every Tool_Call identifier in a Task that reached a terminal state SHALL have exactly one matching tool-role message; IF a Task is canceled or fails while a Tool_Call has no Tool_Result, THEN THE Orchestrator SHALL append a Tool_Result stating that the Tool_Call was not completed because the Task was canceled or failed (history integrity invariant).
6. WHILE two or more Tasks in different Contexts run concurrently, THE Session_Store SHALL keep each Context's Message_History isolated, such that no message produced by a Task appears in the Message_History of any Context other than the Task's owning Context.
7. IF a new user message without a `task_id` arrives for a Context that already has a Task in the `working` or `input-required` state, THEN THE A2A_Server SHALL reject the message with an A2A error stating that the Context is busy, and SHALL leave the existing Task's Task_State and the Context's Message_History unchanged.
8. WHEN the token estimate of the messages to be sent in a chat-completion request exceeds the configured context window budget (default 8,192 tokens, configurable from 1,024 to 1,000,000 tokens), THE Orchestrator SHALL remove the oldest complete user turns (a user message together with all assistant messages, Tool_Calls, and Tool_Results that followed it before the next user message) from that request, always keeping the System_Prompt and the most recent user turn, until the token estimate is less than or equal to the budget; the Message_History stored in the Session_Store SHALL remain unchanged by this trimming.
9. IF the token estimate of the System_Prompt plus the most recent user turn alone exceeds the configured context window budget, THEN THE Orchestrator SHALL transition the Task_State to `failed` with an error message stating that the conversation turn exceeds the context window budget, without sending the chat-completion request.
10. THE Session_Store SHALL expose an asynchronous storage interface so that the in-memory implementation can be replaced by a persistent implementation without changes to the Orchestrator.
11. WHERE the Backend_Config sets a session idle timeout (configurable from 60 to 604,800 seconds), THE Session_Store SHALL evict a Context, its Message_History, and its Task mappings within 60 seconds after the elapsed time since the Context's last Task creation or Task_State transition exceeds the session idle timeout, and SHALL NOT evict any Context that has a Task in the `working` or `input-required` state.

### Requirement 12: System Prompt Construction

**User Story:** As a developer, I want the agent to build a system prompt that describes available tools, so that the LLM uses MCP tools correctly.

#### Acceptance Criteria

1. WHEN the Orchestrator prepares a chat-completion request, THE Prompt_Builder SHALL build the System_Prompt by placing the base instructions text from the Backend_Config first, followed by exactly one tool usage section.
2. THE Prompt_Builder SHALL list in the tool usage section the LLM-facing name and description of every MCP_Tool in the Tool_Registry that belongs to an MCP_Server marked as available. Each MCP_Tool SHALL appear exactly once, grouped under the name of its owning MCP_Server.
3. THE Prompt_Builder SHALL include three instructions in the System_Prompt: (a) call a tool when answering requires data that is reachable only through a listed tool; (b) when a Tool_Result reports an error, tell the user which tool failed and why; (c) when a Tool_Result reports that the user denied a Tool_Call, tell the user that the requested action was denied and not performed.
4. IF the Tool_Registry contains zero MCP_Tools from available MCP_Servers when the Prompt_Builder builds the System_Prompt, THEN THE Prompt_Builder SHALL build a System_Prompt whose tool usage section states that no tools are currently available and lists no tool names.
5. WHEN the Tool_Registry contents change because an MCP_Server connects, disconnects, or refreshes its tool list, THE Prompt_Builder SHALL use the updated tool list in the System_Prompt of the next chat-completion request. This includes later requests within the same Task's reasoning loop.
6. THE Orchestrator SHALL place the System_Prompt as the first message of every chat-completion request.
7. THE Orchestrator SHALL store at most one System_Prompt in each Message_History. When the System_Prompt is rebuilt, the new System_Prompt SHALL replace the stored one rather than being appended as an additional message.
8. IF the Backend_Config omits the base instructions or sets them to an empty string, THEN THE Prompt_Builder SHALL use the documented default base instructions in place of the missing text.
9. IF an MCP_Tool in the Tool_Registry has no description or an empty description, THEN THE Prompt_Builder SHALL list that MCP_Tool's LLM-facing name with a note stating that no description is provided.

### Requirement 13: Configuration Loading

**User Story:** As a power user, I want one configuration file for the agent, so that I can set up the LLM, tools, and permissions in one place.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Config_Loader SHALL read the Backend_Config from the YAML file path given by the `AGENT_CONFIG` environment variable if that variable is set to a non-empty value (interpreted as a Path_Field per Requirements 17.16 to 17.18), or otherwise from `config.yaml` in the process working directory.
2. WHEN the Config_Loader reads the Backend_Config, THE Config_Loader SHALL replace every `${VAR_NAME}` reference (where `VAR_NAME` starts with a letter or underscore and contains only letters, digits, and underscores) inside string values with the value of the matching environment variable (name matching as defined in Requirement 17.21), including multiple references within one string value, before validating the Backend_Config.
3. IF the Backend_Config fails validation, THEN THE Config_Loader SHALL stop startup before the A2A_Server binds to its port and before the MCP_Client connects to any MCP_Server, report every validation error in a single report with the dotted field path of each invalid field, and exit the Backend process with a non-zero exit status.
4. WHEN a Backend_Config field marked optional is omitted from the YAML file, THE Config_Loader SHALL set that field to the default value defined for it in the Backend_Config schema, including Max_Tool_Iterations of 10, Tool_Timeout of 60 seconds, and an A2A_Server bind host of `127.0.0.1`.
5. FOR ALL valid Backend_Config objects, serializing to YAML and loading the result SHALL produce an equivalent Backend_Config with every field value equal to the original (configuration round-trip property).
6. IF the selected Backend_Config file does not exist or cannot be read, THEN THE Config_Loader SHALL stop startup with an error stating the resolved file path and the reason, exit the Backend process with a non-zero exit status, and SHALL NOT fall back to another file path.
7. IF the Backend_Config file content is not well-formed YAML or its top-level value is not a mapping, THEN THE Config_Loader SHALL stop startup with an error stating the file path and, for malformed YAML, the line and column of the parse failure, and exit the Backend process with a non-zero exit status.
8. IF a `${VAR_NAME}` reference names an environment variable that is not set, THEN THE Config_Loader SHALL stop startup with an error naming the missing variable and the field path containing the reference, without including the values of any other substituted environment variables in the error output, and exit the Backend process with a non-zero exit status.

### Requirement 14: Concurrency and Non-Blocking Execution

**User Story:** As a Chat_App user, I want several conversations to run at once without freezing, so that one slow tool does not block other chats.

#### Acceptance Criteria

1. THE Backend SHALL run the A2A_Server, Orchestrator, Session_Store, LLM_Gateway, and MCP_Client on a single asyncio event loop, selected as defined in Requirements 17.28 and 17.29, using asynchronous I/O, and SHALL run every synchronous blocking operation (file reads, subprocess management, CPU-bound token estimation) outside the event loop thread, so that no single operation blocks the event loop for longer than 100 ms.
2. WHILE a Task waits on an LLM response, a Tool_Call, or a user approval, THE Backend SHALL accept requests for other Contexts and emit the first Status_Event of a new Task in another Context within 500 ms of the A2A_Server receiving that Task's message, for up to 10 concurrently active Tasks.
3. WHILE a Tool_Call from one Task is in flight on an MCP_Server, THE MCP_Client SHALL send Tool_Calls from Tasks in other Contexts to the same MCP_Server without waiting for the in-flight Tool_Call to finish.
4. WHEN the A2A_Server receives a cancel request for a Task, THE Orchestrator SHALL cancel the Task's in-flight LLM request and in-flight Tool_Calls within 2 seconds of the A2A_Server receiving the cancel request.
5. WHEN a Task is canceled, THE Orchestrator SHALL discard any LLM output and Tool_Results that arrive after the cancellation without appending them to the Message_History, and SHALL append a Tool_Result stating that the Tool_Call was canceled for each Tool_Call in the Message_History that has no Tool_Result, so that the history integrity invariant holds.
6. WHEN a Task is canceled, THE A2A_Server SHALL emit no Token_Events or Status_Events for that Task after the final Status_Event with Task_State `canceled`.
7. IF an in-flight LLM request or Tool_Call in one Task fails, times out, or does not respond to cancellation within 2 seconds, THEN THE Backend SHALL keep the Task_State and Message_History of Tasks in other Contexts unchanged and continue processing those Tasks.

### Requirement 15: Observability

**User Story:** As a developer, I want structured logs of the agent's activity, so that I can debug tool and LLM behavior.

#### Acceptance Criteria

1. WHEN a Task_State transition occurs, an LLM request finishes, or a Tool_Call finishes, THE Backend SHALL write exactly one structured log record as a single machine-parseable line containing a UTC timestamp with millisecond precision, the log level, the emitting component name, the event kind, the `context_id`, and the `task_id`.
2. WHEN the Backend writes a log record for a Task_State transition, THE Backend SHALL include the previous Task_State and the new Task_State in that log record.
3. WHEN the Backend writes a log record for a Tool_Call, THE Backend SHALL include the LLM-facing tool name, the owning MCP_Server name, the Tool_Call identifier, and an outcome of `success`, `error`, `timeout`, or `denied` in that log record.
4. WHEN an LLM request or a Tool_Call finishes, whether by success, error, timeout, or cancellation, THE Backend SHALL record in the corresponding log record the elapsed duration as a non-negative integer number of milliseconds, measured from request dispatch to receipt of the final response, error, or cancellation.
5. THE Backend SHALL replace every value of a Backend_Config field marked as secret (API keys, MCP_Server HTTP headers, MCP_Server environment variables, and the A2A_Server bearer token) with a fixed redaction placeholder in every log record, such that no log record contains the secret value or any substring of it of 4 or more characters.
6. IF an LLM request or a Tool_Call fails, THEN THE Backend SHALL write the corresponding log record at error level including the error type and error message, with secret values redacted as defined in criterion 5.
7. WHERE the Backend_Config sets a minimum log level of `debug`, `info`, `warning`, or `error`, THE Backend SHALL omit log records below that level, and SHALL use `info` as the minimum log level when the field is omitted.
8. WHEN the A2A_Server receives a health request, THE A2A_Server SHALL respond within 1 second, without contacting the LLM or any MCP_Server, with the Backend status and, for each configured MCP_Server, its name and an availability status of `available` or `unavailable`.
9. WHEN the A2A_Server responds to a health request, THE A2A_Server SHALL report the Backend status as `ok` if every enabled MCP_Server is `available`, and as `degraded` if at least one enabled MCP_Server is `unavailable`.

### Requirement 16: Local Security Defaults

**User Story:** As a power user, I want the agent to stay local by default, so that my files and tools are not exposed to the network.

#### Acceptance Criteria

1. IF the Backend_Config omits the A2A_Server bind host, THEN THE Config_Loader SHALL set the A2A_Server bind host to `127.0.0.1`, and THE A2A_Server SHALL refuse connections that do not arrive on the loopback interface.
2. WHERE the Backend_Config binds the A2A_Server to a non-loopback address (any address outside `127.0.0.0/8` and other than `::1` or `localhost`, including `0.0.0.0` and `::`), THE A2A_Server SHALL require the bearer token configured in the Backend_Config on every request, including the health endpoint, except the Agent_Card request at the A2A well-known agent card path.
3. IF a request to an endpoint protected under criterion 2 carries no bearer token, or carries a bearer token that does not exactly match (case-sensitive, full-string comparison) the bearer token configured in the Backend_Config, THEN THE A2A_Server SHALL respond with HTTP status 401, SHALL NOT create, modify, or cancel any Task, and SHALL NOT append to any Message_History.
4. THE A2A_Server SHALL restrict CORS allowed origins to the list of exact origins (scheme, host, and port) configured in the Backend_Config, and SHALL omit CORS allow headers from responses to cross-origin requests whose origin is not in that list.
5. IF the Backend_Config binds the A2A_Server to a non-loopback address and does not configure a bearer token, or configures a bearer token shorter than 32 characters, THEN THE Config_Loader SHALL stop startup before the A2A_Server binds to any port and report an error naming the bearer token field path.
6. IF the Backend_Config omits the CORS allowed origins list, THEN THE Config_Loader SHALL set the CORS allowed origins list to empty, so that THE A2A_Server returns no CORS allow headers to any cross-origin request.
7. WHILE the A2A_Server is bound to a loopback address, THE A2A_Server SHALL process requests without requiring a bearer token.

### Requirement 17: Cross-Platform Support

**User Story:** As a power user on Windows, macOS, or Linux, I want the Backend to install and run the same way on my operating system, so that my configuration, MCP_Servers, and Chat_App behave identically wherever I run them.

#### Acceptance Criteria

1. THE Backend SHALL run on each Supported_Platform: Windows 10 (22H2) and Windows 11 on x86_64; macOS 14 Sonoma or later on arm64 and on x86_64; and Linux distributions with glibc 2.28 or later (manylinux_2_28) on x86_64 and on arm64 (aarch64); each using the CPython 3.14 standard (GIL-enabled) build.
2. WHEN the Backend starts on an operating system, CPU architecture, or Python build outside the Supported_Platform set (including Windows on arm64, Linux with musl libc, and the CPython 3.14 free-threaded build), THE Backend SHALL write one warning log record naming the detected operating system, architecture, and Python build, and continue startup.
3. FOR ALL requests and Backend_Configs covered by Requirements 1 to 16, THE Backend SHALL produce the same A2A responses, the same sequence of Status_Events and Token_Events, the same Message_History contents, and the same Backend_Config validation results on every Supported_Platform, ignoring generated identifiers, timestamps, and measured durations, and excepting only the platform-specific rules stated in criteria 8, 9, 15, 16, 17, 21, and 22 of this requirement (cross-platform equivalence property).
4. WHEN the MCP_Client starts a `stdio` MCP_Server whose command contains no path separator, THE MCP_Client SHALL resolve the command by searching, in order, the directories of the `PATH` value in the environment passed to that subprocess, and on Windows SHALL try each extension listed in that environment's `PATHEXT` value in order when the command has no extension (for example resolving `npx` to `npx.cmd`).
5. WHEN the MCP_Client starts a `stdio` MCP_Server whose command contains a path separator, THE MCP_Client SHALL use the command as a Path_Field resolved per criteria 16 to 18 without searching `PATH`, and on Windows SHALL try each `PATHEXT` extension in order when the resolved path has no extension and no file exists at that exact path.
6. IF command resolution under criterion 4 or 5 finds no executable file, THEN THE MCP_Client SHALL treat the MCP_Server as a failed connection attempt under Requirement 6.7 with a failure reason that names the command and states that no executable was found.
7. THE MCP_Client SHALL start each `stdio` MCP_Server process directly from the resolved executable and the configured argument list without invoking a command shell, so that each configured argument reaches the MCP_Server process as exactly one argument with unchanged characters (argument fidelity property).
8. WHERE the Backend runs on Windows and the resolved `stdio` executable is a batch file (`.cmd` or `.bat`), THE MCP_Client SHALL enclose each argument in double quotes so that `cmd.exe` passes the characters `&`, `|`, `<`, `>`, `^`, and spaces to the batch file literally.
9. WHERE the Backend runs on Windows and the resolved `stdio` executable is a batch file, IF a configured argument contains a double quote (`"`), a percent sign (`%`), a carriage return, or a line feed, THEN THE MCP_Client SHALL not start the subprocess and SHALL treat the MCP_Server as a failed connection attempt under Requirement 6.7 with a failure reason naming the argument position and the disallowed character.
10. WHEN the MCP_Client starts a `stdio` MCP_Server process, THE MCP_Client SHALL place that process in a new process group separate from the Backend (a new session on macOS and Linux; a new process group assigned to a job object on Windows), so that a Shutdown_Signal delivered to the Backend's console reaches the Backend and is not delivered directly to the Process_Tree.
11. WHEN the MCP_Client terminates a `stdio` MCP_Server (during Backend shutdown under Requirement 6.11, before a reconnection attempt under Requirement 6.8, or after a failed handshake under Requirement 6.7), THE MCP_Client SHALL close the subprocess's standard input, request graceful termination of the Process_Tree (`SIGTERM` to the process group on macOS and Linux; `CTRL_BREAK_EVENT` to the process group on Windows), and, for any process of the Process_Tree still running after the grace period, force termination of the whole Process_Tree (`SIGKILL` to the process group on macOS and Linux; termination of the job object on Windows, or `taskkill /PID <pid> /T /F` when the job object assignment failed).
12. WHEN the MCP_Client completes termination of a `stdio` MCP_Server under criterion 11, THE Process_Tree SHALL contain no running process, including descendant processes started by the MCP_Server process (process-tree cleanup property).
13. WHEN the Backend receives a Shutdown_Signal, THE Backend SHALL begin Graceful_Shutdown within 1 second.
14. THE Backend SHALL perform Graceful_Shutdown as the same ordered sequence on every Supported_Platform: stop accepting new A2A connections, transition every Task in the `working` or `input-required` state to `canceled` as defined in Requirement 2.7, terminate every `stdio` MCP_Server under criterion 11 and close every `sse` MCP_Server session, and exit the Backend process with exit status 0.
15. WHERE the Backend runs on Windows, WHEN the Backend receives a console close event (`CTRL_CLOSE_EVENT`), THE Backend SHALL use a grace period of 2 seconds instead of 5 seconds before forced Process_Tree termination, so that Graceful_Shutdown completes before Windows ends the Backend process.
16. THE Config_Loader SHALL accept `/` as a path separator in every Path_Field on every Supported_Platform, and WHERE the Backend runs on Windows THE Config_Loader SHALL also accept `\` as a path separator.
17. WHEN a Path_Field value is `~` or starts with `~/` (or, on Windows, `~\`), THE Config_Loader SHALL replace the leading `~` with the current user's home directory (`HOME` on macOS and Linux; `USERPROFILE` on Windows).
18. WHEN a Path_Field value is a relative path after `~` expansion, THE Config_Loader SHALL resolve the `AGENT_CONFIG` value against the process working directory and SHALL resolve every other Path_Field against the directory containing the loaded Backend_Config file.
19. WHEN a `stdio` MCP_Server definition omits the working directory, THE MCP_Client SHALL start the subprocess with the directory containing the loaded Backend_Config file as the working directory.
20. THE MCP_Client SHALL pass `stdio` MCP_Server argument values to the subprocess without path separator conversion, `~` expansion, or relative path resolution.
21. WHEN the Config_Loader resolves a `${VAR_NAME}` reference under Requirement 13.2, THE Config_Loader SHALL match `VAR_NAME` against environment variable names case-insensitively on Windows and case-sensitively on macOS and Linux, following each operating system's native environment semantics.
22. WHEN the MCP_Client builds the environment for a `stdio` MCP_Server, THE MCP_Client SHALL let each configured environment variable replace the inherited variable of the same name, comparing names case-insensitively on Windows and case-sensitively on macOS and Linux, so that the subprocess environment contains exactly one entry per name on Windows.
23. THE Config_Loader SHALL decode the Backend_Config file as UTF-8 independent of the operating system locale or console code page, and SHALL accept and ignore a leading UTF-8 byte order mark.
24. IF the Backend_Config file content is not valid UTF-8, THEN THE Config_Loader SHALL stop startup with an error stating the file path and the byte offset of the first invalid byte sequence, and exit the Backend process with a non-zero exit status.
25. FOR ALL Backend_Config files, replacing LF line endings with CRLF line endings, or mixing LF and CRLF, SHALL produce a Backend_Config equal to the one loaded from the LF version, and the line numbers reported under Requirement 13.7 SHALL count each CRLF or LF as one line break (line-ending invariance property).
26. THE Backend SHALL write every log record encoded as UTF-8 and terminated by a single LF character, independent of the operating system locale, console code page, or platform default line ending.
27. THE MCP_Client SHALL encode and decode `stdio` MCP messages as UTF-8 independent of the operating system locale or console code page.
28. THE Backend SHALL create its event loop only through the `loop_factory` argument of `asyncio.run` or `asyncio.Runner`, using an event loop that supports subprocesses (the default Proactor event loop on Windows), and SHALL use no event-loop policy API deprecated in Python 3.14 (such as `asyncio.set_event_loop_policy` or the `WindowsSelectorEventLoopPolicy` class).
29. WHERE the Backend runs on macOS or Linux and the `uvloop` package is installed, THE Backend SHALL use the `uvloop` event loop; WHERE the Backend runs on Windows or `uvloop` is not installed, THE Backend SHALL use the default asyncio event loop.
30. WHEN the Backend_Config sets the A2A_Server bind host to `localhost`, THE A2A_Server SHALL bind to `127.0.0.1` without name resolution and, on platforms where IPv6 is enabled, also to `::1`.
31. IF binding to `::1` fails under criterion 30 because IPv6 is disabled or unavailable, THEN THE A2A_Server SHALL write one warning log record and continue startup serving on `127.0.0.1` only.
32. WHEN the A2A_Server binds its listening socket, THE A2A_Server SHALL use exclusive address binding (`SO_EXCLUSIVEADDRUSE` on Windows; no `SO_REUSEPORT` on macOS and Linux), so that a bind to a host and port already in use by another listening process fails under Requirement 1.8 on every Supported_Platform.
33. WHEN the Backend is installed with CPython 3.14 on a Supported_Platform using binary-only installation (`pip install --only-binary=:all:` or `uv sync --no-build`), THE installation SHALL complete using pre-built wheels only, without building any distribution from source and without a compiler toolchain.
34. WHEN a push or pull request occurs, THE CI_Pipeline SHALL run the binary-only installation of criterion 33 and the unit, property, and integration test suites with CPython 3.14 on Windows x86_64, macOS arm64, macOS x86_64, Linux x86_64, and Linux arm64 runners, and SHALL report the run as failed if any job on any runner fails.


---

## Appendix A: Proposed Directory Structure

```text
Chatbot/
├── pyproject.toml                  # Project metadata (requires-python = ">=3.14"), dependencies, tool config (ruff, pytest, mypy)
├── uv.lock                         # Pinned dependency lock file
├── config.example.yaml             # Annotated example Backend_Config
├── README.md                       # Includes the Supported_Platform list (Requirement 17.1)
├── .gitattributes                  # Pins line endings so LF/CRLF test fixtures stay byte-exact on Windows checkouts
├── .github/
│   └── workflows/
│       └── ci.yml                  # OS matrix CI (Requirement 17.34), see sketch below
├── src/
│   └── local_agent/
│       ├── __init__.py
│       ├── __main__.py             # Entry point: `python -m local_agent`
│       ├── app.py                  # Builds Starlette/ASGI app, wires components, lifespan (startup/shutdown)
│       ├── config/
│       │   ├── __init__.py
│       │   ├── models.py           # Pydantic models: BackendConfig, LLMConfig, MCPServerConfig, PermissionPolicy
│       │   └── loader.py           # YAML loading, ${ENV} substitution, validation
│       ├── a2a_server/
│       │   ├── __init__.py
│       │   ├── agent_card.py       # AgentCard + skills definition
│       │   ├── executor.py         # HostAgentExecutor(AgentExecutor): bridges A2A requests to Orchestrator
│       │   ├── events.py           # Helpers emitting Status_Events / Token_Events via TaskUpdater
│       │   ├── auth.py             # Bearer-token middleware for non-loopback binds
│       │   └── health.py           # /health route
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── loop.py             # Reasoning loop: LLM → tool calls → results → LLM
│       │   ├── prompt_builder.py   # System_Prompt construction
│       │   ├── approval_gate.py    # Human-in-the-loop pause/resume for Protected_Tools
│       │   ├── tool_dispatch.py    # Argument parsing/validation, concurrent execution, result ordering
│       │   └── context_window.py   # Token estimation and history trimming
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── gateway.py          # LiteLLM streaming wrapper, retries
│       │   └── stream_assembler.py # Assembles streamed tool-call fragments
│       ├── mcp_client/
│       │   ├── __init__.py
│       │   ├── manager.py          # Connection lifecycle for stdio/SSE servers, reconnect/backoff
│       │   ├── registry.py         # Tool_Registry: naming, collision handling, resolution
│       │   ├── schema_translator.py# MCP Tool → OpenAI function tool format
│       │   └── result_formatter.py # MCP content → text Tool_Result, truncation
│       ├── session/
│       │   ├── __init__.py
│       │   ├── store.py            # Abstract async SessionStore interface
│       │   └── memory_store.py     # In-memory implementation with idle eviction
│       ├── oscompat/               # All OS-specific code lives here (named to avoid shadowing stdlib `platform`)
│       │   ├── __init__.py         # Supported_Platform detection and warning (17.1, 17.2)
│       │   ├── process.py          # PATH/PATHEXT resolution, batch-file quoting, process groups/job objects, tree termination (17.4–17.12)
│       │   ├── signals.py          # SIGINT/SIGTERM and Windows console control handlers → Graceful_Shutdown (17.13–17.15)
│       │   ├── paths.py            # Separators, `~` expansion, relative resolution (17.16–17.20)
│       │   └── event_loop.py       # loop_factory selection: uvloop on POSIX, default (Proactor) on Windows (17.28, 17.29)
│       └── logging_setup.py        # Structured logging + secret redaction (UTF-8, LF-terminated)
└── tests/
    ├── conftest.py
    ├── unit/                       # Pure-logic tests (translator, registry, assembler, history, config)
    ├── property/                   # Hypothesis property tests (round-trips, invariants, ordering)
    ├── integration/                # A2A client ↔ server, real stdio MCP test server, mocked LLM
    └── fixtures/
        ├── mcp_echo_server.py      # Minimal stdio MCP server used in tests
        └── mcp_spawning_server.py  # stdio MCP server that spawns a grandchild process (process-tree cleanup test, 17.12)
```

CI workflow sketch (`.github/workflows/ci.yml`). Runner labels are current GitHub-hosted labels at the time of writing; confirm availability (especially the Intel macOS label) when implementing:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest, macos-15-intel, ubuntu-latest, ubuntu-24.04-arm]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv python install 3.14
      - run: uv sync --locked --no-build          # binary wheels only (17.33)
      - run: uv run ruff check .
      - run: uv run mypy src
      - run: uv run pytest tests/unit tests/property tests/integration
```

## Appendix B: Python Dependencies

Runtime requirement: Python 3.14 or newer (`requires-python = ">=3.14"` in `pyproject.toml`). Exact versions are to be pinned in `uv.lock` (or an equivalent lock file) at implementation time, using the latest stable release of each package that satisfies the constraints below.

Python 3.14 compatibility: every package below must resolve to binary wheels (no source builds) for CPython 3.14 on every Supported_Platform: Windows x86_64 (`win_amd64`), macOS arm64 and x86_64 (`macosx_*_arm64`, `macosx_*_x86_64`, or `universal2`), and Linux x86_64 and arm64 (`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`). Coverage for macOS x86_64 and Linux arm64 is to be confirmed when `uv.lock` is generated and is enforced by the CI matrix (Requirement 17.34). Packages with compiled components need the minimum versions noted in the table; older releases lack 3.14 wheels. Pydantic V1 APIs (`pydantic.v1`) do not support Python 3.14 and are not to be used.

| Package | Purpose |
|---|---|
| `a2a-sdk` (with `http-server` extra) | Official A2A Python SDK: `A2AStarletteApplication`, `DefaultRequestHandler`, `AgentExecutor`, `TaskUpdater`, `InMemoryTaskStore`, A2A types. Current releases are 1.x; confirm these class names against the 1.x API at design time |
| `mcp` | Official Model Context Protocol Python SDK: `ClientSession`, `stdio_client`, `sse_client`. Current releases are 2.x; confirm these names and SSE transport support against the 2.x API at design time |
| `litellm` | Unified client for local LLMs (Ollama, vLLM, OpenAI-compatible endpoints) with streaming and function calling. 3.14 install depends on transitive wheels: requires `fastuuid>=0.14`, `tiktoken>=0.12`, and (for `uvicorn[standard]`) `uvloop>=0.22`; earlier litellm releases failed on 3.14 for these reasons |
| `starlette` | ASGI framework underlying the A2A server app (pulled in by `a2a-sdk`; used directly for health route and middleware) |
| `uvicorn[standard]` | ASGI server. The `standard` extra pulls `uvloop` (Linux/macOS only, `>=0.22` for 3.14), `httptools`, and `watchfiles`, all of which ship 3.14 wheels |
| `uvloop>=0.22; sys_platform != 'win32'` | Declared directly because the Backend selects it via `loop_factory` (Requirement 17.29). The environment marker keeps it off Windows, where uvloop does not exist |
| `sse-starlette` | SSE response support (dependency of `a2a-sdk` HTTP server) |
| `httpx` | Async HTTP client (A2A client in tests, MCP SSE transport) |
| `pydantic` (v2) | Backend_Config models and validation. Requires `pydantic>=2.12` (first release with Python 3.14 support and matching `pydantic-core` wheels) |
| `pydantic-settings` | Environment-variable overrides for configuration |
| `pyyaml` | YAML config parsing and serialization. Requires `pyyaml>=6.0.3` (first release with 3.14 wheels) |
| `jsonschema` | Validating Tool_Call arguments against MCP_Tool input schemas (its compiled dependency `rpds-py` ships 3.14 wheels in current releases) |
| `structlog` | Structured logging with context binding and redaction processors |
| `anyio` | Async primitives used by `mcp` and for task-group cancellation |

Platform-specific notes:

- Windows process-tree termination (Requirements 17.10–17.12) uses the standard library only: `subprocess.CREATE_NEW_PROCESS_GROUP`, `os.kill(pid, signal.CTRL_BREAK_EVENT)`, a job object created through `ctypes` (`kernel32.CreateJobObjectW`, `AssignProcessToJobObject`, `TerminateJobObject`, with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`), and `taskkill.exe /T /F` (part of Windows) as the fallback. macOS and Linux use `start_new_session=True` with `os.killpg`. `psutil` is not used. If it is added later, it ships stable-ABI (`abi3`) wheels that install on CPython 3.14; confirm per-platform wheels at lock time.
- `mcp` declares `pywin32` as a Windows-only dependency (`sys_platform == 'win32'`) for its own stdio process handling. Confirm a `cp314-win_amd64` pywin32 wheel at lock time, and check whether the SDK's built-in job-object handling already covers 17.10–17.12 so the Backend does not create a second job object for the same process.
- Signal handling: `loop.add_signal_handler` is unavailable on Windows. `oscompat/signals.py` uses `signal.signal` for `SIGINT`/`SIGBREAK` and a console control handler (`SetConsoleCtrlHandler` via `ctypes`) for `CTRL_CLOSE_EVENT`.
- `litellm` on Windows: a reported issue ([BerriAI/litellm#41855](https://github.com/BerriAI/litellm/issues/41855)) describes the `win_amd64` wheel bundling tiktoken cache files with CRLF line endings. tiktoken then rejects them by checksum and downloads replacements, so `import litellm` needs network access. Pin a release without this defect, or pre-populate `TIKTOKEN_CACHE_DIR`. The Windows CI job should import `litellm` with network access disabled.
- Windows on arm64 is not a Supported_Platform. Native CPython 3.14 `win_arm64` wheels are not reliably published for the full dependency tree; for example, tiktoken's `regex` dependency has reported source-build failures there ([openai/tiktoken#403](https://github.com/openai/tiktoken/issues/403)). The x86_64 build under Windows 11 emulation is untested.
- Linux musl (Alpine) and the CPython 3.14 free-threaded build (`3.14t`) are not Supported_Platforms. `musllinux` and `cp314t` wheel coverage across the compiled dependencies has not been verified.

Development / test dependencies:

| Package | Purpose |
|---|---|
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
| `hypothesis` | Property-based testing for round-trip, invariant, and ordering properties |
| `respx` | Mocking `httpx` calls (LLM endpoint) |
| `ruff` | Linting and formatting |
| `mypy` | Static type checking |

## Appendix C: Request Flow Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant UI as Chat App (A2A Client)
    participant A2A as A2A Server (HostAgentExecutor)
    participant ORC as Orchestrator
    participant SS as Session Store
    participant LLM as Local LLM (LiteLLM)
    participant MCP as MCP Client
    participant TS as Tool Server (MCP)

    Note over MCP,TS: Startup: connect to configured MCP servers (stdio/SSE), list_tools, translate schemas into Tool_Registry

    UI->>A2A: message/stream (text, context_id?)
    A2A->>SS: create Task, resolve/create Context
    A2A-->>UI: SSE Status: working
    A2A->>ORC: execute(task, context)
    ORC->>SS: load Message_History
    ORC-->>UI: SSE Status: "Discovering local tools..."
    ORC->>MCP: get tool definitions (Tool_Registry)
    ORC->>LLM: chat.completions(stream, system prompt + history + tools)

    loop Reasoning loop (≤ Max_Tool_Iterations)
        LLM-->>ORC: streamed chunks (text and/or tool_call fragments)
        ORC-->>UI: SSE Token_Event (append) per text chunk
        alt LLM emitted Tool_Calls
            ORC->>ORC: assemble fragments, resolve names, validate args
            alt Tool is Protected (permission required)
                ORC->>SS: persist pending Tool_Call + history
                ORC-->>UI: SSE Status: input-required (tool, server, args, approve/deny?)
                Note over UI,A2A: Stream ends with input-required; Task paused
                UI->>A2A: message/stream (task_id, decision)
                A2A->>ORC: resume(task, decision)
                ORC-->>UI: SSE Status: working
                alt Approved
                    ORC-->>UI: SSE Status: "Deleting notes.txt..."
                    ORC->>MCP: call_tool(server, tool, args)
                    MCP->>TS: tools/call
                    TS-->>MCP: result
                    MCP-->>ORC: Tool_Result
                else Denied
                    ORC->>ORC: Tool_Result = "User denied this action"
                end
            else Tool not protected
                par Concurrent Tool_Calls
                    ORC-->>UI: SSE Status: "Reading config.json..."
                    ORC->>MCP: call_tool(server, tool, args)
                    MCP->>TS: tools/call
                    alt Success
                        TS-->>MCP: result content
                        MCP-->>ORC: Tool_Result
                    else Error / timeout / unavailable
                        TS-->>MCP: isError / no response
                        MCP-->>ORC: Tool_Result (error text)
                    end
                    ORC-->>UI: SSE Status: tool finished (ok/failed)
                end
            end
            ORC->>SS: append Tool_Calls + Tool_Results (call order)
            ORC->>LLM: chat.completions(stream, updated history)
        else Final answer (no Tool_Calls)
            ORC->>SS: append assistant message
            ORC-->>UI: SSE Token_Event (last_chunk)
        end
    end

    alt Success
        A2A-->>UI: SSE Status: completed (final) and close stream
    else Unrecoverable error (e.g. LLM unreachable)
        A2A-->>UI: SSE Status: failed + error message (final) and close stream
    end
```
