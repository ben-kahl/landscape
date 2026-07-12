# Landscape

Landscape is a local-first AI agent memory system for multi-hop reasoning across
documents and conversations. It stores memories in both Neo4j and Qdrant, then
combines vector similarity with graph traversal so an agent can answer questions
that require connecting facts spread across multiple sources.

The core difference from vector-only memory is that Landscape keeps extracted
entities and relationships in a real graph. If an answer depends on a path like
`Project Aurora -> Sarah -> Platform Team`, retrieval can traverse that path
directly instead of hoping one chunk is semantically close enough to contain the
whole answer. Extracted relationships also preserve temporal and numeric
qualifiers, so facts like "Eric watched 8 hours of Netflix today" retain the
`8 hours` and `today` metadata rather than collapsing to only
`ERIC -> WATCHED -> NETFLIX`.

Everything runs locally with Docker Compose, FastAPI, Neo4j, Qdrant, a
llama.cpp `llama-server` (OpenAI-compatible LLM inference), sentence-transformers
embeddings, LangChain, and Model Context Protocol.

## The killer demo

Seven questions across 1/2/3-hop bands, same killer-demo corpus (`tests/fixtures/killer_demo_corpus/`):

| Mode                | Questions answered | Notes                                              |
|---------------------|--------------------|----------------------------------------------------|
| Landscape (hybrid)  | 7 / 7              | Hits the 3-hop chain                               |
| Landscape (vector)  | 5 / 7              | Misses the 2-hop "who approved Aurora's database?" |
| ChromaDB (baseline) | 6 / 7              | Misses the 3-hop chain entirely                    |

The proof point is the 3-hop question: no single chunk contains the
"Aurora → Sarah → Platform Team" path, so chunk similarity cannot surface it;
graph traversal can. Reproduction commands are in
[Reproduce the benchmarks](#reproduce-the-benchmarks).

## Architecture

```mermaid
graph TD
    Client["MCP client\n(Claude Code / Cursor / custom)"]
    MCP["FastAPI /mcp\n(streamable HTTP)"]
    API["FastAPI\n/ingest  /query"]
    Pipeline["Ingestion pipeline\n→ LLM extraction\n→ embeddings (sentence-transformers)\n→ entity resolver"]
    Neo4j["Neo4j\ngraph traversal"]
    Qdrant["Qdrant\nvector search"]
    Llama["llama-server (llama.cpp)\nLLM extraction · OpenAI API"]

    Client -->|"search / remember / lookup_entity\nadd_entity / add_relation / graph_query / status"| MCP
    MCP --> API
    API --> Pipeline
    Pipeline --> Neo4j
    Pipeline --> Qdrant
    Pipeline -->|"extraction"| Llama
    MCP -->|"retrieve()"| Qdrant
    MCP -->|"Cypher"| Neo4j
```

## Current status

| Area | Status |
|---|---|
| Document ingestion | Markdown, text, PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, EPUB, RTF via markitdown; LLM extraction, chunking, entity resolution, Neo4j writes, Qdrant writes |
| Hybrid retrieval | Vector search, graph expansion, merge/rank, recency and distance scoring |
| Temporal memory | Bitemporal facts: separate system time (`ingested_at` / `system_until`) and valid time (`effective_from` / `effective_until`); supersession-aware retrieval for functional conflicts; `as_of` time-travel queries; negative-polarity facts stored and surfaced distinctly |
| Quantified facts | Relationship edges preserve counts, durations, prices, frequencies, and time scopes |
| Agent access | MCP server, conversation history, automatic conversation capture, LangChain retriever, FastAPI, local CLI |
| Benchmarks | Killer-demo retrieval benchmark, ChromaDB baseline, LongMemEval smoke harness |
| Phase 3.5 hardening | Complete: ranking tuning, LongMemEval beyond smoke, resolver improvements |
| Conversation capture | Automatic capture merged (PR #35); its trigger (a SessionEnd hook) was never wired up and is being redesigned as a Claude Code plugin with Stop-hook-driven incremental capture. Today, transcript ingestion works manually via `landscape ingest-transcript` |
| Current focus | W&B support demo direction (real-issue corpus merged in PR #40); planned next: `get_document` MCP tool, the conversation-capture plugin, and domain vocabulary packs |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design rationale, data
model details, benchmark notes, and known limitations.

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose
- A llama.cpp `llama-server` for LLM extraction — run in Docker via a Compose
  profile (`cpu`, `gpu-nvidia`, `gpu-amd`) or natively on the host (macOS/Metal).
  The model GGUF is pulled automatically on first start; no manual download.

```bash
git clone https://github.com/ben-kahl/landscape.git
cd landscape

./scripts/detect-stack.sh
docker compose up -d

uv sync --extra dev
uv run pytest -m "unit or smoke"               # CI-safe sanity check
uv run python scripts/demo_mcp_session.py      # supersession demo transcript
```

`scripts/detect-stack.sh` creates `.env`, sets `COMPOSE_PROFILES` and
`LLM_BASE_URL`, and chooses the right `llama-server` backend for the host:
NVIDIA GPU (`gpu-nvidia`), AMD GPU/ROCm (`gpu-amd`), CPU (`cpu`), or a host-run
`llama-server` on macOS (`host`). Each profile has a matching `llama-server`
service in `docker-compose.yml`, so `docker compose up -d` then starts the
detected backend, Neo4j, Qdrant, and the app together.

The default model is Qwen 3.5 9B (`Q4_K_M`) served by `llama-server` over an
OpenAI-compatible API. Switch models or point at a cloud provider with the
`LLM_PROFILE` env var (see `LLM_PROFILES` in `src/landscape/config.py`); the
`openai_gpt5` profile shows the cloud shape.

The checked-in `.env.example` keeps the higher-quality Nomic embedding model as
the default. That model requires Hugging Face remote model code, so the example
env file also sets `ALLOW_REMOTE_MODEL_CODE=true`. If you switch to a model
like `sentence-transformers/all-MiniLM-L6-v2`, you can remove that flag and
keep remote model code disabled.

### Test database isolation

Stack-backed tests use a separate Neo4j/Qdrant Compose stack by default so test
cleanup does not wipe your local development memory:

```bash
docker compose -f docker-compose.test.yml up -d
uv run pytest
```

Default test endpoints:

| Service | Test endpoint | Live dev endpoint |
|---|---|---|
| Neo4j Bolt | `bolt://localhost:17687` | `bolt://localhost:7687` |
| Neo4j Browser | `http://localhost:17474` | `http://localhost:7474` |
| Qdrant HTTP | `http://localhost:16333` | `http://localhost:6333` |
| Qdrant gRPC | `localhost:16334` | `localhost:6334` |

Tests refuse to wipe the live default Neo4j/Qdrant ports unless
`LANDSCAPE_ALLOW_LIVE_TEST_WIPE=1` is set deliberately. Override the test
endpoints with the `LANDSCAPE_TEST_NEO4J_*` and `LANDSCAPE_TEST_QDRANT_URL`
env vars. `llama-server` (default `http://localhost:8080/v1`) holds no state and
is not touched by tests.

The Docker `llama-server` backends download the model GGUF automatically via the
`-hf` flag on first start — no manual model pull needed:

```bash
docker compose --profile <cpu|gpu-nvidia|gpu-amd> up -d
```

On macOS, run `llama-server` natively (it can serve the same GGUF on Metal) and
let Docker reach it via `host.docker.internal`; `detect-stack.sh` selects the
`host` profile and sets `LLM_BASE_URL` accordingly.

Supported `COMPOSE_PROFILES` values: `cpu`, `gpu-nvidia`, `gpu-amd`, `host`.

## CLI

Use the CLI to inspect and operate the local Landscape stack:

```bash
uv run landscape --help
uv run landscape status --verbose
uv run landscape ingest /path/to/notes.md
uv run landscape ingest /path/to/paper.pdf --title "Aurora Design Doc"
uv run landscape ingest-dir ./docs            # walks every supported file
uv run landscape ingest-dir ./docs --glob "*.md"  # strict pattern mode
uv run landscape query "Who leads the project using PostgreSQL?"
uv run landscape graph counts
uv run landscape graph entity "Project Atlas"
uv run landscape graph neighbors "Project Atlas" --hops 2
uv run landscape seed killer-demo --confirm
uv run landscape wipe --confirm
```

`ingest` infers the source format from the file extension and converts it to
markdown via [markitdown](https://github.com/microsoft/markitdown) before
extraction. Supported formats today: markdown, text, PDF, DOCX, PPTX, XLSX,
XLS, HTML, CSV, JSON, XML, EPUB, RTF. Files with unrecognized extensions are
read as utf-8 text. `--source-type` is an override; by default the recorded
source type matches the input format (so a fact extracted from a PDF shows
`source_type: pdf` in provenance even though the chunker only saw markdown).

`ingest-dir` walks every file in the directory by default and dispatches by
extension; files without a known converter are skipped with a single log line.
Pass `--glob "<pattern>"` to opt into strict-pattern mode and ingest only
matching files.

The CLI defaults to host-reachable service URLs for local use: Neo4j on
`bolt://localhost:7687`, Qdrant on `http://localhost:6333`, and `llama-server`
on `http://localhost:8080/v1`. Explicit environment variables (`NEO4J_*`,
`QDRANT_URL`, `LLM_BASE_URL`) still override those defaults.

## Run the API and embedded MCP server

Start the shared FastAPI + MCP app:

```bash
uv run uvicorn landscape.main:app --host 127.0.0.1 --port 8000
```

The MCP endpoint is mounted at `http://127.0.0.1:8000/mcp`.

Retrieval logs under `logs/retrieval/` now hash `query_text` and `session_id`
by default. Raw values are only written when a caller explicitly enables
request-level debug logging with `debug=true`.

### Authentication

`/query`, `/ingest`, and `/mcp` all require a valid bearer token. `/healthz`
stays public.

Landscape uses **OAuth 2.1 Authorization Code + PKCE** (RFC 6749 + RFC 7636).
Clients self-register via dynamic registration — no manual token minting
required. MCP clients like Claude Code and Codex handle the full OAuth flow
automatically when pointed at the server URL.

**Connecting Claude Code or Codex.** Add the MCP server URL to your client
config:

```
http://localhost:8000/mcp
```

On first connection the client opens the browser to complete the OAuth flow,
obtains an access token, and stores it locally. Subsequent connections reuse
the token until it expires, then refresh automatically.

**Scopes.** Two scopes are available:

- `agent` — memory tools: `search`, `remember`, `lookup_entity`, `add_entity`,
  `add_relation`, `status`, `conversation_history`
- `graph_query` — raw read-only Cypher via the `/query` endpoint

Clients default to `agent` scope. Request `graph_query` explicitly if needed.

**Inspecting and revoking clients.** CLI commands read the local SQLite auth DB
directly — no network calls:

```bash
# Show all registered OAuth clients and their status
landscape auth list-clients

# Prevent a client from obtaining new tokens (access + refresh both invalidated)
landscape auth disable-client --client-id <client-id>

# Re-enable a previously disabled client
landscape auth enable-client --client-id <client-id>
```

**Where the auth DB lives.** Defaults to `~/.config/landscape/auth.db`.
Override with `AUTH_DB_PATH`. Under docker-compose the path is pinned to
`/var/lib/landscape/auth.db` on a named volume (`landscape_auth_data`)
so registered clients survive container rebuilds.

**Remote / cloud deployment.** Set `MCP_ISSUER_URL` to the public HTTPS URL
of your deployment:

```bash
MCP_ISSUER_URL=https://landscape.example.com
```

The issuer URL is embedded in OAuth discovery metadata (`/.well-known/oauth-authorization-server`),
so clients resolve the correct token and registration endpoints automatically.
HTTPS is required for remote deployments — the OAuth spec prohibits non-loopback
HTTP redirect URIs without TLS.

## MCP tools

Point any MCP client at `http://127.0.0.1:8000/mcp` (the FastAPI app must be
running; the `NEO4J_*`, `QDRANT_URL`, `LLM_BASE_URL`, and `LLM_PROFILE` env vars
apply to that server process). Multiple clients connected to the same URL share one Landscape
process. The legacy stdio launcher `landscape-mcp` is still available but is
process-per-client; prefer HTTP unless you need stdio.


| Tool | Description |
|---|---|
| `search` | Hybrid retrieve: vector similarity + graph traversal up to N hops; accepts `as_of` (ISO-8601) for time-travel queries against historical fact state |
| `remember` | Ingest free-text; extract entities and relations into the graph |
| `lookup_entity` | Entity-centered lookup: returns canonical entity + aliases + live facts for an exact name/alias match, or a miss + substring-match suggestions otherwise; supports `as_of` and `include_historical` |
| `add_entity` | Directly assert a named entity with type and provenance |
| `add_relation` | Assert a typed edge between two entities; supersedes functional conflicts |
| `graph_query` | Run a read-only Cypher query against the knowledge graph |
| `status` | Return a ~200-token summary: entity count, top entities, recent agent writes |
| `conversation_history` | Return chronological turns and entities mentioned in a session |

### Ingesting files from an agent conversation

There is intentionally no `remember_file` MCP tool. File ingestion happens
through the CLI, which the agent invokes via its existing shell access:

```bash
landscape ingest ~/Downloads/paper.pdf --session-id $SESSION --turn-id $TURN
```

The CLI reads the file directly from the agent's local filesystem (no
filesystem-topology problem), uses the markitdown converter dispatch
(PDF/DOCX/PPTX/XLSX/HTML/CSV/JSON/EPUB/RTF + markdown/text passthrough), and
records the same session/turn provenance the MCP write-back tools use. For
long-running ingestion of large PDFs, run the command in the background
(`&` / `nohup` / the agent harness's background-shell feature) so the agent
is not blocked on extraction.

### Automatic conversation capture (end of session)

Landscape captures conversation memory by reading a client's **completed
transcript at the end of a session** — not by streaming turns live. During a
session the agent already holds the conversation in its own context, so there is
nothing to gain from live capture; at `SessionEnd` a single hook hands the
transcript to the CLI, which parses it, runs one local `llama-server` salience
pass over the conversation, and ingests the selected turns:

```bash
landscape ingest-transcript            # reads the SessionEnd hook JSON (transcript_path) on stdin
landscape ingest-transcript PATH       # or pass a transcript file directly (manual / backfill)
```

This reuses the same direct-to-database runtime as `landscape ingest` and the
MCP server, so it needs **no HTTP endpoint and no API token** — only local
access to the Neo4j / Qdrant / llama-server stack on their configured ports. The
salience pass selects original turns (it does not rewrite them); selected turns
are ingested as one document and linked back to every contributing
`Conversation` / `Turn` for provenance.

`remember` remains the explicit synchronous document-ingest tool, and
`add_entity` / `add_relation` remain the precise structured write-back tools.

**Setup (Claude Code).** Copy or merge `hooks/claude-code/settings.example.json`
into `.claude/settings.json` or `~/.claude/settings.json`; it registers one
`SessionEnd` hook that runs `landscape ingest-transcript`. Set `LANDSCAPE_HOME`
to this checkout (e.g. in `~/.zshrc`) so the hook resolves regardless of which
project Claude Code is running in:

```bash
export LANDSCAPE_HOME="/absolute/path/to/landscape"   # this checkout
```

> **Note:** Automatic capture is currently supported for **Claude Code only**.
> The **opencode** (`hooks/opencode/`) and **codex** (`hooks/codex/`) example
> hooks are **outdated**: they targeted the removed push-based HTTP endpoints and
> do not yet have transcript-pull readers. They are unsupported until those
> readers land.

### Conversation capture provenance and privacy

Automatic capture stores only turns selected as durable, future-relevant memory:
identity/role, stable preferences, decisions, stable facts, relationships, and
state changes or corrections. Greetings, acknowledgements, tool chatter, task
mechanics, and transient clarifying questions are discarded before graph
ingestion.

Everything captured lands in your local Neo4j/Qdrant stack with
`Conversation` / `Turn` provenance. The salience and extraction calls use the
local `llama-server`, so capture does not require cloud APIs, but captured facts are still
persistent local memory. Treat enabling the capture hook as an explicit choice
to store useful conversation facts.

## Reproduce the benchmarks

```bash
uv sync --extra dev --extra bench
uv run python scripts/bench_retrieval.py    # Landscape hybrid + vector + graph
uv run python scripts/bench_chromadb.py     # ChromaDB baseline
```

The killer-demo corpus lives in `tests/fixtures/killer_demo_corpus/`. Results
are printed as a Markdown table; the numbers in [The killer demo](#the-killer-demo)
above were produced this way.

## Bitemporal facts

Every `MemoryFact` (and the live `MEMORY_REL` edge that mirrors it) carries two
independent time axes:

| Axis | Fields | Meaning |
|---|---|---|
| System time | `ingested_at`, `system_until` | When Landscape learned the fact and, if superseded, when a newer version took over. `system_until` is `NULL` for the current version. |
| Valid time | `effective_from`, `effective_until` | When the fact was true in the world. Extracted from the source text when an explicit calendar reference is present; otherwise `NULL`. |

Retrieval defaults to "current as of now": facts where `system_until IS NULL`.
The `as_of` parameter on `search` (API, CLI, and MCP) shifts the system-time
filter so callers can ask what Landscape *believed* at a past moment, even if
that belief has since been superseded. Pass `include_historical=true` to bypass
the system-time filter entirely.

When a functional fact is superseded, the prior version's `system_until` is
closed to the new fact's `effective_from` (not just "now"), so the system-time
chain reflects the actual succession point in the world rather than ingest
order.

## Design rationale and known limitations

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design rationale.
Three limitations worth calling out:

**Rel-type synonym drift.** Small local LLMs are non-deterministic about relationship type phrasing (`WORKS_FOR` vs `EMPLOYED_BY`). Landscape uses a closed vocabulary of 22 canonical types (with subtype annotations for richer semantics) and a `normalize_relation_type()` normalizer, but truly novel types pass through unchanged and will not trigger supersession. Demos that rely on temporal conflict resolution should use hand-constructed corpora.

**MCP tool-call reliability.** LLM agents invoking `add_relation` may invent relationship types outside the canonical vocabulary. These are stored as-is and do not trigger supersession rules. Monitor the `status` tool output for unexpected rel types in a live session.

**Entity resolver type-match strictness.** The resolver requires entity type agreement before merging; an agent that writes `("Sarah", "PERSON")` when the ingestion pipeline stored `("Sarah", "Employee")` will create a duplicate node rather than resolving to the existing one.

## License

Landscape is licensed under the Apache License 2.0. See
[LICENSE.txt](LICENSE.txt).
