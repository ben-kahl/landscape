# MCP Streamable HTTP Migration Design

## Goal

Replace Landscape's proof-of-concept stdio MCP server with a streamable HTTP MCP server as the only supported transport, so all clients and subagents can connect to one long-lived shared server instance instead of spawning one subprocess per client.

## Why This Slice Comes First

Automatic conversation ingestion is easier to implement on a shared long-lived MCP server than on the current stdio process-per-client model. The transport migration is therefore the correct first slice:

- it removes the per-subagent server spawning problem
- it creates a single process boundary for future background ingestion work
- it avoids layering new lifecycle behavior on top of a transport we already intend to replace

Auto conversation ingestion is explicitly deferred until after this migration is complete.

## Current State

Landscape currently exposes MCP through `src/landscape/mcp_server.py` using `FastMCP("landscape")` and a blocking `mcp.run(transport="stdio")` entrypoint. In practice this means:

- every client process launches its own `landscape-mcp` subprocess
- Codex subagents each get a separate Landscape MCP server process
- there is no shared in-memory state across clients
- any future background work would be tied to short-lived client-owned processes

This was acceptable for a proof of concept but is the wrong transport for a shared memory backend.

## Constraints

- The migration should use the transport support already provided by the installed MCP library instead of building a custom server layer.
- The existing tool surface and semantics should remain stable: `search`, `remember`, `add_entity`, `add_relation`, `graph_query`, `status`, and `conversation_history`.
- The cutover should be strict: streamable HTTP becomes the supported transport and stdio support is removed from Landscape-facing docs and entrypoints.
- The implementation should not require embedding MCP into the existing FastAPI app. That is a larger architecture choice than this migration needs.

## Transport Choice

The installed `FastMCP` version already supports:

- `stdio`
- `sse`
- `streamable-http`

Landscape should adopt `streamable-http` as the only supported MCP transport.

### Why not SSE

SSE would still require a network server, but it is operationally worse for this use case:

- separate handshake/message endpoints
- weaker fit for bidirectional session-style traffic
- no reason to choose it when the library already supports streamable HTTP directly

### Why not keep stdio as a fallback

Keeping stdio around would reduce immediate friction, but it would also preserve two deployment modes, two docs paths, and two behavior models right when the project is trying to harden its interfaces before phase 4. The user explicitly prefers taking the painful cut now to avoid accumulating transport debt.

## Recommended Architecture

### 1. Separate MCP app construction from server startup

Landscape should have one place that constructs and registers the `FastMCP` app and its tools. Startup logic should be a thin layer above that.

This means splitting today's single-file pattern into:

- an MCP app module that owns `FastMCP("landscape")` and all tool registration
- a small server entrypoint that runs that app using streamable HTTP

That keeps transport concerns out of tool definitions and makes the app easier to test.

### 2. Expose one long-lived HTTP MCP server

The canonical runtime becomes a long-lived process started explicitly by the developer or via Docker/Compose in a later follow-up. For this migration, the minimum viable shape is:

`uv run landscape-mcp --host 127.0.0.1 --port 8001`

Internally, this should call:

`mcp.run(transport="streamable-http")`

using the MCP library's built-in HTTP app and uvicorn handling.

### 3. Keep MCP separate from the FastAPI app

The Landscape API server and the MCP server should remain separate processes in this slice.

Reasons:

- lower migration risk
- clearer operational debugging
- fewer cross-lifecycle surprises
- avoids coupling MCP transport rollout to the existing FastAPI app

If we later want one unified application surface, that can be a separate design and migration.

## Configuration Model

The new MCP server entrypoint should support explicit host and port configuration so clients can point at a stable endpoint. The minimum configuration surface is:

- `--host`
- `--port`

It is acceptable to use sensible defaults for local development, but docs must show explicit configuration so client setup is unambiguous.

Landscape's existing environment variables for Neo4j, Qdrant, and Ollama remain unchanged.

## Client Configuration

Repository docs should switch from subprocess-launch MCP configuration to HTTP MCP configuration.

That means replacing instructions like:

- `command = "uv"`
- `args = ["run", "landscape-mcp"]`

with client configuration that targets the shared HTTP endpoint instead.

The exact client snippets should match what Codex and any documented MCP clients currently expect for remote/server URL configuration, but the project-level design requirement is simple: clients connect to a shared URL rather than spawning a local subprocess.

## Backward Compatibility

This migration intentionally does not preserve backward compatibility for stdio MCP clients. The compatibility strategy is documentation-based:

- update README and demo docs to the new HTTP connection model
- remove stdio-only framing from the MCP architecture description
- keep the tool API stable so only transport configuration changes for clients

That is a clean cutover instead of a prolonged mixed-mode period.

## Testing Strategy

The migration needs three categories of verification.

### Unit / app-construction coverage

- the MCP app can be imported without starting the server
- the streamable HTTP ASGI app can be constructed successfully
- the expected tools remain registered

### MCP behavior regression coverage

- existing MCP tool tests continue to pass against the shared app object
- no tool response shapes change as part of the transport migration

### Documentation / operator verification

- the repo documents how to start the HTTP MCP server
- demo and setup docs no longer instruct users to spawn stdio MCP subprocesses

An end-to-end remote MCP client smoke test would be useful later, but it is not required for the first migration if the app-construction and existing tool regression tests stay green.

## Implementation Boundaries

### In scope

- refactor MCP code so app construction is transport-agnostic
- switch server startup to streamable HTTP
- remove stdio as the supported transport in Landscape docs and entrypoints
- update demos and setup guidance to point clients at the shared MCP server URL
- add tests that cover app construction and preserve tool behavior

### Out of scope

- automatic conversation ingestion
- SSE transport support
- embedding the MCP server inside the existing FastAPI app
- Docker Compose automation for the MCP server
- authentication or multi-user access control for the MCP endpoint

## Risks and Mitigations

### Risk: client config churn

Users currently configured for subprocess MCP will need to update their config.

Mitigation:

- provide exact replacement config in docs
- keep the tool surface unchanged

### Risk: startup/config mistakes during cutover

Moving from subprocess launch to a separately started server introduces one more runtime step.

Mitigation:

- keep startup command simple
- document host and port explicitly
- add a lightweight status/verification path in docs

### Risk: future need for unified server hosting

It is possible Landscape eventually wants the MCP server mounted inside FastAPI or Compose-managed as part of the stack.

Mitigation:

- keep the MCP app creation isolated from startup logic now
- treat deployment unification as a separate later migration

## Success Criteria

This migration is complete when:

- Landscape exposes MCP only through a streamable HTTP server entrypoint
- repository docs point clients at a shared HTTP MCP endpoint instead of subprocess launch
- existing MCP tool tests still pass without response-shape regressions
- the code structure cleanly separates MCP app construction from transport startup
