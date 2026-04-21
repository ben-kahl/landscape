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

Everything runs locally with Docker Compose, FastAPI, Neo4j, Qdrant, Ollama,
LangChain, and Model Context Protocol.

## The killer demo

Seven questions across 1/2/3-hop bands, same killer-demo corpus (`tests/fixtures/killer_demo_corpus/`):

| Mode                 | P@k    | MRR    | Notes                                               |
|----------------------|--------|--------|-----------------------------------------------------|
| Landscape (hybrid)   | 100%   | 0.306  | Hits all 7 queries including the 3-hop chain        |
| Landscape (vector)   | 71.4%  | 0.149  | Misses the 2-hop "who approved Aurora's database?"  |
| ChromaDB*            | 86%    | 0.43*  | Misses the 3-hop chain entirely (P@1 = 0% at 3-hop) |

*ChromaDB is evaluated at chunk level; Landscape at entity level. Do not compare MRR numbers directly — the granularity differs. The apples-to-apples claim is per-question: ChromaDB answers 6/7 questions (all 1/2-hop), Landscape hybrid answers 7/7 including the one that requires "Aurora → Sarah → Platform Team" in a single traversal. No single chunk in the corpus contains that chain, so chunk similarity can never surface it.

The 3-hop question is the proof point. Reproduce the benchmark with:

```bash
uv sync --extra dev --extra bench
uv run python scripts/bench_retrieval.py    # Landscape hybrid + vector + graph
uv run python scripts/bench_chromadb.py     # ChromaDB baseline
```

## Architecture

```mermaid
graph TD
    Client["MCP client\n(Claude Code / Cursor / custom)"]
    MCP["landscape-mcp\n(stdio JSON-RPC)"]
    API["FastAPI\n/ingest  /query"]
    Pipeline["Ingestion pipeline\n→ LLM extraction\n→ entity resolver"]
    Neo4j["Neo4j\ngraph traversal"]
    Qdrant["Qdrant\nvector search"]
    Ollama["Ollama\nLLM + embeddings (local)"]

    Client -->|"search / remember / add_entity\nadd_relation / graph_query / status"| MCP
    MCP --> API
    API --> Pipeline
    Pipeline --> Neo4j
    Pipeline --> Qdrant
    Pipeline -->|"extraction + embeddings"| Ollama
    MCP -->|"retrieve()"| Qdrant
    MCP -->|"Cypher"| Neo4j
```

## Current status

| Area | Status |
|---|---|
| Text ingestion | LLM extraction, chunking, entity resolution, Neo4j writes, Qdrant writes |
| Hybrid retrieval | Vector search, graph expansion, merge/rank, recency and distance scoring |
| Temporal memory | Supersession-aware retrieval for functional relationship conflicts |
| Quantified facts | Relationship edges preserve counts, durations, prices, frequencies, and time scopes |
| Agent access | MCP server, conversation history, LangChain retriever, FastAPI, local CLI |
| Benchmarks | Killer-demo retrieval benchmark, ChromaDB baseline, LongMemEval smoke harness |
| Phase 3.5 hardening | In progress: ranking tuning, benchmark hardening, relation normalization, resolver improvements |
| Phase 4 | Next major feature area: expanded ingestion paths for documents, integrations, conversations, and multimodal memory |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design rationale, data
model details, benchmark notes, and known limitations.

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose
- Ollama, either in Docker via a Compose profile or running on the host

```bash
git clone https://github.com/ben-kahl/landscape.git
cd landscape

./scripts/detect-stack.sh
docker compose up -d

uv sync --extra dev
uv run pytest                                  # sanity check
uv run python scripts/demo_mcp_session.py      # supersession demo transcript
```

`scripts/detect-stack.sh` creates `.env`, sets `COMPOSE_PROFILES`, and chooses
the appropriate Ollama mode for the host: NVIDIA GPU, AMD GPU, CPU, or host
Ollama on macOS.

If the script selects Docker-managed Ollama, pull the default model once:

```bash
docker compose exec ollama-cpu ollama pull llama3.1:8b        # CPU profile
docker compose exec ollama-nvidia ollama pull llama3.1:8b     # NVIDIA profile
docker compose exec ollama-amd ollama pull llama3.1:8b        # AMD profile
```

On macOS, run Ollama on the host and let Docker reach it through
`host.docker.internal`:

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
docker compose up -d
```

You can still bypass detection and set `COMPOSE_PROFILES` manually in `.env`.
Supported profiles are `cpu`, `gpu-nvidia`, `gpu-amd`, and `host`.

## CLI

Use the CLI to inspect and operate the local Landscape stack:

```bash
uv run landscape --help
uv run landscape status --verbose
uv run landscape ingest /path/to/document.md
uv run landscape ingest /path/to/document.md --title "Architecture Notes" --source-type markdown
uv run landscape ingest-dir ./docs --glob "*.md"
uv run landscape query "Who leads the project using PostgreSQL?"
uv run landscape graph counts
uv run landscape graph entity "Project Atlas"
uv run landscape graph neighbors "Project Atlas" --hops 2
uv run landscape seed killer-demo --confirm
uv run landscape wipe --confirm
```

The CLI defaults to host-reachable service URLs for local use: Neo4j on
`bolt://localhost:7687`, Qdrant on `http://localhost:6333`, and Ollama on
`http://localhost:11434`. Explicit environment variables still override those
defaults.

## Use Landscape as MCP memory in Claude Code

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "landscape": {
      "command": "uv",
      "args": ["run", "--project", "/abs/path/to/landscape", "landscape-mcp"]
    }
  }
}
```

The seven MCP tools:

| Tool | Description |
|---|---|
| `search` | Hybrid retrieve: vector similarity + graph traversal up to N hops |
| `remember` | Ingest free-text; extract entities and relations into the graph |
| `add_entity` | Directly assert a named entity with type and provenance |
| `add_relation` | Assert a typed edge between two entities; supersedes functional conflicts |
| `graph_query` | Run a read-only Cypher query against the knowledge graph |
| `status` | Return a ~200-token summary: entity count, top entities, recent agent writes |
| `conversation_history` | Return chronological turns and entities mentioned in a session |

## Reproduce the benchmarks

```bash
uv sync --extra dev --extra bench
uv run python scripts/bench_retrieval.py    # Landscape hybrid + vector + graph
uv run python scripts/bench_chromadb.py     # ChromaDB baseline
```

Results are printed as a Markdown table. On the killer-demo corpus, hybrid retrieval stays at 7/7 P@k (100.0%) with 0.326 MRR and 64ms average latency; vector-only reaches 85.7% P@k, 0.213 MRR, and 42ms latency; graph-only remains at 0.0% P@k, 0.000 MRR, and 2ms latency. The killer-demo corpus lives in `tests/fixtures/killer_demo_corpus/`.

## Design rationale and known limitations

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design rationale.
Three limitations worth calling out here before phase 4:

**Rel-type synonym drift.** Small local LLMs are non-deterministic about relationship type phrasing (`WORKS_FOR` vs `EMPLOYED_BY`). Landscape uses a closed vocabulary of 18 canonical types (with subtype annotations for richer semantics) and a `normalize_relation_type()` normalizer, but truly novel types pass through unchanged and will not trigger supersession. Demos that rely on temporal conflict resolution should use hand-constructed corpora.

**MCP tool-call reliability.** LLM agents invoking `add_relation` may invent relationship types outside the canonical vocabulary. These are stored as-is and do not trigger supersession rules. Monitor the `status` tool output for unexpected rel types in a live session.

**Entity resolver type-match strictness.** The resolver requires entity type agreement before merging; an agent that writes `("Sarah", "PERSON")` when the ingestion pipeline stored `("Sarah", "Employee")` will create a duplicate node rather than resolving to the existing one.

## Pre-phase-4 checklist

- Align `AGENTS.md`, `README.md`, and `docs/ARCHITECTURE.md` to the implemented system.
- State explicit phase 3.5 exit criteria instead of relying on the original unchecked phase plan.
- Keep MCP/API/CLI documentation accurate as interfaces evolve.
- Add automatic agent-conversation ingestion so useful memory can be captured without requiring explicit fact write-back for every conversational detail.
- Harden the benchmark story so it is clear what is proven by killer-demo and what is still smoke-only in LongMemEval.
- Track reasoning-quality gaps explicitly: relation-direction normalization, semantic rel-type clustering, and stronger cross-type entity resolution.
- Add CI or document the canonical verification workflow if automated CI is still absent.
- Keep phase 4 scoped to new ingestion modes and integrations rather than mixing it with unrelated cleanup.

## License

Landscape is licensed under the Apache License 2.0. See
[LICENSE.txt](LICENSE.txt).
