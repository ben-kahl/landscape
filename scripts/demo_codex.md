# Landscape as a Memory Backend for OpenAI Codex

Evaluate Landscape — local-first graph + vector memory — as a persistent memory backend for a **non-local** model (gpt-5-codex via Codex CLI). The LLM runs in the cloud. The memory runs on your laptop. MCP bridges them.

## What this demonstrates

1. **Multi-hop reasoning** Codex can answer questions that require connecting facts from multiple documents, via Landscape's hybrid retrieval. Vector-only RAG typically fails on these.
2. **Continuous memory** Codex can write new facts back into the graph during a session. Supersession handles conflicts automatically.
3. **Cross-session recall** Facts persist across Codex sessions. A follow-up conversation can retrieve them.
4. **Cost-effective architecture** Heavy reasoning runs on the cloud model; storage, extraction, and retrieval stay local. No cloud bills for memory reads/writes.

## Prerequisites

- Docker Compose stack running locally (`docker compose up -d`). Neo4j + Qdrant + Ollama + FastAPI.
- `uv` installed.
- Codex CLI installed and logged in (`codex login`).

Verify the stack is up:
```
docker compose ps     # all services should be running
curl -s http://localhost:7474 | head -1   # Neo4j HTTP responding
```

## Setup

### 1. Seed Landscape with the killer-demo corpus

```
uv run python scripts/seed_killer_demo.py
```

Takes ~40 seconds. Ingests 7 Helios Robotics docs (org chart, team rosters, project details, a DB approval chain, an engineering sync). Each doc lands as a Turn in session `seed:killer-demo`.

Expected output:
```
Seeding Landscape from ...killer_demo_corpus
Session id: seed:killer-demo

Step 1/3  Wiping Neo4j + Qdrant state...
Step 2/3  Initializing encoder + Qdrant collections...
Step 3/3  Ingesting 7 docs under session 'seed:killer-demo'...
  [1/7] 01_org_chart.md  entities=8 relations=10 superseded=0
  ...
Done. entities=~40 live_relations=~60
Ready. Point an MCP client at `uv run landscape-mcp` to query.
```

### 2. Register Landscape as an MCP server in Codex

Add this to `~/.codex/config.toml` (create the file if it doesn't exist):

```toml
[mcp_servers.landscape]
command = "uv"
args = ["run", "landscape-mcp"]
cwd = "/ABSOLUTE/PATH/TO/landscape"

[mcp_servers.landscape.env]
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "landscape-dev"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
```

Replace `/ABSOLUTE/PATH/TO/landscape` with the repo root on your machine (e.g. `/home/you/Documents/landscape`).

**Note on Codex config format:** the schema has evolved. If Codex complains, check `codex mcp --help` for the current flag names. The inputs it needs are: `command=uv`, `args=["run", "landscape-mcp"]`, `cwd` set to the repo root, plus the env vars above.

### 3. Verify MCP connectivity

Launch Codex and ask it to list available tools:

```
codex
> List the MCP tools you have available.
```

You should see: `search`, `remember`, `add_entity`, `add_relation`, `graph_query`, `status`, `conversation_history`.

If the tools aren't visible, check `codex mcp` diagnostics and confirm the cwd path is correct.

## Demo script

Run these prompts in order in a single Codex session. After each one, watch for the MCP tool call in Codex's output.

### Turn 1 — wake-up context

> **You:** Before we start, check the Landscape memory and summarize what's there. Use the status tool.

**Expected:** Codex calls `landscape.status`. Response shows entity_count ~40, relation_count ~60, top_entities (by reinforcement — empty on a fresh seed), and recent_conversations listing `seed:killer-demo`. This is the ~200-token wake-up payload — cheap to load at session start.

### Turn 2 — single-hop retrieval (baseline)

> **You:** Who leads the Vision Team at Helios?

**Expected:** Codex calls `landscape.search` with a query like "Vision Team leader". Results include `Diego` (the intended answer) plus related entities. Codex should cite `Diego` in its reply.

### Turn 3 — two-hop reasoning

> **You:** Who approved the database choice for Project Aurora?

**Expected:** Codex calls `landscape.search` with hops=3 or similar. The answer requires traversing `Aurora —USES→ PostgreSQL ←APPROVED— Maya Chen` across two different source docs. Codex should surface `Maya Chen`. A vector-only RAG setup typically fails here because "Maya Chen" and "Project Aurora" never appear in the same chunk.

### Turn 4 — three-hop reasoning

> **You:** What team is the person who approved Aurora's database on?

**Expected:** Codex issues one or more `search` calls, then possibly a `graph_query` for precision. Path: `Aurora → USES → PostgreSQL ← APPROVED ← Maya Chen → MEMBER_OF → Platform Team`. Codex should answer `Platform Team`. This is the multi-hop capability — three edges, three source docs.

### Turn 5 — continuous memory write-back

> **You:** I just found out that Maya Chen was promoted to Director of Platform Engineering. Please record that.

**Expected:** Codex calls `landscape.add_relation` with something like `subject=Maya Chen, object=Director of Platform Engineering, rel_type=HAS_TITLE` (or a synonym). The current Landscape vocab will coerce this — see the known-limitations section of the repo `CLAUDE.md`. After the write, Codex should confirm the relation id and outcome (`created` or `reinforced`).

### Turn 6 — cross-turn recall in the same session

> **You:** What's Maya's title now?

**Expected:** Codex calls `search` with a session filter or just the general query. It should surface the new `Director of Platform Engineering` relation it just wrote — demonstrating that the write is immediately readable.

### Turn 7 — session replay

> **You:** Use the conversation_history tool to show me what we talked about in the `seed:killer-demo` session.

**Expected:** Codex calls `landscape.conversation_history(session_id="seed:killer-demo", limit=10)`. Returns the 7 ingested docs as turns with the entities each turn mentioned. Codex summarizes.

### Turn 8 — graph query for precision

> **You:** Run a Cypher query to show me every project and its tech stack.

**Expected:** Codex calls `landscape.graph_query` with a read-only Cypher pattern like `MATCH (p:Entity {type: "Project"})-[r:RELATES_TO {type: "USES"}]->(t:Entity) RETURN p.name, t.name`. Returns structured rows. If Codex tries a write query, Landscape's `cypher_guard` rejects it — Codex should fall back to read-only.

## Observations to record

Rough rubric — not a formal benchmark, but useful for judging whether Landscape is pulling its weight:

| Signal | What to watch for |
|---|---|
| Tool-call correctness | Does Codex reach for the right MCP tool? `search` for retrieval, `add_relation` for writes, `conversation_history` for replay. |
| Multi-hop success | Turns 3 and 4 should land the correct answer without you having to feed extra context. |
| Latency | Each MCP call typically 100–500ms locally. If Codex stalls for seconds, check the FastAPI app logs (`docker compose logs app`). |
| Noise in results | `search` returns up to `limit` results. Does Codex filter down to the relevant ones, or dump everything? |
| Write-read consistency | Turn 6 must surface the write from Turn 5. If it doesn't, the extraction pipeline's rel_type normalization is the likely culprit. |
| Supersession behavior | If you run Turn 5 twice with different titles, the second write should mark the first as superseded (check via Turn 7's `graph_query` or the `status` tool's `recent_agent_writes`). |

## What this corpus does NOT test

- Personal memory (preferences, family, life facts) — that needs the expanded vocab in `specs/vocab_expansion.md`. Until that ships, Codex will extract those facts into `RELATED_TO` fallbacks with weak supersession.
- Long documents — all corpus docs are < 200 tokens. Chunking / long-context behavior isn't exercised.
- Coreference across docs — handled by entity resolution on exact name matches plus fuzzy matching. Unusual phrasings ("he", "the lead", "that person") are not reliably resolved.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Codex can't see Landscape tools | `cwd` in config.toml is wrong; `uv` not on PATH; stack not running |
| `search` returns empty | Seed script didn't run or wiped mid-way — rerun `seed_killer_demo.py` |
| `add_relation` errors with "subject_type required" | Codex omitted `subject_type` / `object_type` — these are required in the current MCP server (Phase 3 change) |
| Multi-hop queries miss the answer | Try `hops=3` explicitly in the Codex prompt; the default is 2 |
| Cypher query rejected | Check for write keywords (`CREATE`, `SET`, `DELETE`) — `graph_query` is read-only by design |

## Resetting between demos

```
uv run python scripts/seed_killer_demo.py
```

The seed script wipes all graph and vector state before ingesting. Safe to run repeatedly.

## Related

- `scripts/demo_mcp_session.py` — automated 7-step MCP transcript (no Codex, pure Python MCP client). Good for CI.
- `scripts/demo_langchain_agent.py` — LangChain agent using Landscape as a custom retriever. Local-only.
- `tests/test_killer_demo.py` — pytest assertions proving the 1/2/3-hop paths are retrievable.
- `specs/vocab_expansion.md` — the planned vocab expansion. Once shipped, the `tests/fixtures/personal_memory_corpus/` seed will enable personal-memory / preference / family demos against Codex.
