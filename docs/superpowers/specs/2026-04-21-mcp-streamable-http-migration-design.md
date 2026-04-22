# Embedded MCP Streamable HTTP Migration Design

## Goal

Replace Landscape's proof-of-concept stdio MCP server with a streamable HTTP MCP server mounted inside the existing FastAPI application, making the FastAPI app the single long-lived shared runtime for both API and MCP access.

## Why This Direction

The project already has a long-lived FastAPI app in `src/landscape/main.py` with a real lifespan hook that initializes embeddings and Qdrant collections and closes storage clients on shutdown. Embedding MCP into that app is the better long-term architecture because it consolidates process lifecycle, avoids duplicate startup work, and creates one shared server surface for future background tasks such as automatic conversation ingestion.

The user explicitly prefers taking the larger cut now instead of carrying forward a separate MCP process model that would likely be revisited later.

## Current State

Landscape currently has two separate server concepts:

- the FastAPI application in `src/landscape/main.py`
- the MCP server in `src/landscape/mcp_server.py`

The MCP server currently uses:

- `FastMCP("landscape")`
- tool registration and startup in one file
- `mcp.run(transport="stdio")`

That creates a client-owned subprocess model:

- every MCP client process spawns its own Landscape MCP subprocess
- Codex subagents each get their own server instance
- in-memory state is not shared across clients
- future background work would be process-local and fragmented

That transport model is no longer aligned with the role of Landscape as a shared memory backend.

## Constraints

- Streamable HTTP becomes the only supported MCP transport.
- The existing MCP tool surface must remain stable: `search`, `remember`, `add_entity`, `add_relation`, `graph_query`, `status`, and `conversation_history`.
- The FastAPI app should remain the single hosted ASGI application after the migration.
- The migration should reuse the installed MCP library's built-in streamable HTTP ASGI app rather than reimplementing MCP protocol handling manually.
- Automatic conversation ingestion is out of scope for this slice and will be revisited after transport migration.

## Recommended Architecture

### 1. Split MCP app construction from runtime startup

Landscape should move away from a single file that both defines tools and starts the server. Instead it should have:

- an MCP app module responsible only for constructing `FastMCP("landscape")` and registering tools
- the FastAPI application module responsible for mounting that MCP ASGI app

This keeps tool definitions transport-agnostic and makes the resulting structure easier to test.

### 2. Mount streamable HTTP MCP inside FastAPI

The FastAPI app in `src/landscape/main.py` should mount the MCP streamable HTTP ASGI app under a dedicated path such as:

- `/mcp`

The mounted MCP app should come from:

- `mcp.streamable_http_app()`

This yields a single long-lived server process with one shared lifecycle while still preserving a clean protocol boundary between API routes and MCP routes.

### 3. Keep MCP and API logic separate even though they share one host app

Embedding MCP into FastAPI does not mean mixing the API handlers and MCP tool handlers together. The boundary should be:

- FastAPI routes stay in the API modules
- MCP tools stay in MCP-specific modules
- `main.py` is responsible only for wiring them into one ASGI application

That preserves conceptual separation while getting the operational benefits of one process.

## Why Embedding Is Better Here

### Pros

- One runtime owns initialization and shutdown.
  The existing FastAPI lifespan already loads the encoder and prepares Qdrant collections. MCP can reuse that instead of carrying its own lazy initialization gate.

- One shared process serves all clients.
  Codex subagents and other MCP consumers connect to the same server endpoint rather than spawning separate subprocesses.

- Better foundation for background work.
  A long-lived shared FastAPI runtime is a better place to host future non-blocking background ingestion tasks than short-lived per-client stdio servers.

- Cleaner operations.
  One server means one health story, one set of logs, one deployment surface, and fewer ways for API and MCP runtimes to drift.

- Easier future middleware and observability.
  If the project later adds auth, rate limiting, metrics, tracing, or request identifiers, the host app is already the central integration point.

### Cons

- Larger migration surface right now.
  This slice changes both transport and server topology at once.

- Tighter coupling between API and MCP availability.
  If the shared app fails to boot, both surfaces are affected.

- Care is required around mounting and lifespan assumptions.
  The mounted MCP ASGI app must not accidentally bypass or duplicate host-app initialization behavior.

These are acceptable tradeoffs given the user's stated preference for the long-term architecture.

## Initialization and Lifespan

The current FastAPI lifespan in `src/landscape/main.py` already does the important initialization:

- `encoder.load_model()`
- `qdrant_store.init_collection()`
- `qdrant_store.init_chunks_collection()`
- shutdown cleanup for Neo4j and Qdrant clients

After embedding MCP, that lifespan should become the single initialization path for both HTTP API requests and MCP tool calls.

This implies a corresponding cleanup in MCP code:

- remove the MCP-local lazy initialization gate once the mounted design is in place
- rely on FastAPI host startup instead of per-tool `await _ensure_init()`

That reduces duplicated lifecycle logic and removes the mismatch between a long-lived host app and a formerly standalone MCP process.

## Routing Model

The hosted application should expose:

- existing FastAPI API routes such as `/ingest`, `/query`, and `/healthz`
- mounted MCP streamable HTTP under `/mcp`

The intended result is a single server command for local development, for example:

`uv run uvicorn landscape.main:app --host 127.0.0.1 --port 8000`

with MCP clients pointing at:

`http://127.0.0.1:8000/mcp`

Using the existing app server command avoids inventing a separate MCP runtime command when the desired architecture is a single hosted app.

## Configuration Model

The transport-facing configuration becomes the host FastAPI server address plus mounted MCP path.

Canonical local assumptions:

- host app served by uvicorn
- MCP mounted at `/mcp`

Existing storage environment variables stay unchanged:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `QDRANT_URL`
- `OLLAMA_URL`

No new MCP-specific service discovery layer is needed in this slice beyond the mounted path.

## Client Configuration

Repository docs should switch from subprocess-launch MCP configuration to remote HTTP MCP configuration.

That means removing instructions that tell clients to spawn:

- `uv run landscape-mcp`

and replacing them with configuration that points at the shared mounted endpoint:

- `http://<host>:<port>/mcp`

The precise config snippets should match the clients documented in this repo, but the architecture requirement is simple: clients connect to the shared FastAPI-hosted MCP URL rather than launching a subprocess.

## Code Structure

The intended file responsibilities are:

- `src/landscape/mcp_app.py`
  Builds the `FastMCP` instance and registers all tools.

- `src/landscape/main.py`
  Owns FastAPI lifespan, API route inclusion, and MCP app mounting.

- `src/landscape/mcp_server.py`
  Removed entirely or reduced to a compatibility shim only if absolutely needed during the migration.

The preferred end state is to remove `mcp_server.py` as the public runtime entrypoint so the code no longer suggests that MCP is a separate server process.

## Backward Compatibility

This migration intentionally breaks stdio MCP startup compatibility.

The compatibility policy is:

- MCP tool behavior remains stable
- transport and runtime topology change
- documentation is updated to the new shared HTTP endpoint model

This is a hard cutover, not a mixed-mode transition.

## Testing Strategy

The migration needs four types of verification.

### 1. App-construction coverage

- the FastMCP app can be imported without starting a server
- the mounted streamable HTTP ASGI app can be created successfully
- the expected MCP tools remain registered

### 2. Host-app integration coverage

- `landscape.main:app` includes the existing API routes
- the MCP ASGI app is mounted at the intended path
- FastAPI lifespan remains the initialization path for shared resources

### 3. MCP behavior regression coverage

- existing MCP tool tests continue to pass against the embedded app object
- tool response shapes do not change

### 4. Documentation/operator coverage

- README and demo docs show the single-server startup path
- MCP setup docs no longer instruct users to spawn subprocesses

An end-to-end remote MCP smoke test would be useful later, but it is not required for the first migration if mounted-app construction and existing MCP behavior tests remain green.

## In Scope

- refactor MCP code so tool registration is separate from runtime startup
- embed streamable HTTP MCP into the FastAPI app
- remove stdio as a supported Landscape transport
- update docs and demos to point clients at the mounted MCP endpoint
- update tests to reflect the embedded architecture while preserving tool behavior

## Out of Scope

- automatic conversation ingestion
- SSE support
- auth and multi-user controls for the MCP endpoint
- Docker Compose automation for server startup
- broader FastAPI refactors unrelated to MCP embedding

## Risks and Mitigations

### Risk: mounting semantics are awkward with `FastMCP`

The MCP library's generated ASGI app may have assumptions about path handling or lifespan that need careful mounting.

Mitigation:

- keep the mounted path simple
- add integration coverage that verifies app construction and mounted routing
- avoid extra middleware complexity in this slice

### Risk: duplicated initialization paths

The current MCP server has its own lazy init gate while FastAPI already has a lifespan hook.

Mitigation:

- make FastAPI lifespan the only initialization path after embedding
- remove or bypass MCP-local initialization code

### Risk: client setup churn

Users configured for subprocess MCP will need to update their client configuration.

Mitigation:

- provide exact replacement setup in docs
- keep tool names and semantics unchanged

## Success Criteria

This migration is complete when:

- Landscape serves MCP through the FastAPI-hosted streamable HTTP endpoint only
- stdio MCP startup is removed from Landscape docs and runtime entrypoints
- `landscape.main:app` becomes the single hosted application surface
- existing MCP tool behavior remains unchanged from the client's perspective apart from transport configuration
