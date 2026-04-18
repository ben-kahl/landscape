
# Landscape — Multi-Hop Reasoning Over Agent Memory

## What This Is

Landscape is a local-first agent memory system that enables multi-hop reasoning through hybrid Neo4j graph traversal and Qdrant vector similarity search. It gives AI agents persistent, evolving memory that supports structured reasoning — not just semantic recall.

The core pitch: ask a question that requires connecting 3 pieces of information across different documents. Landscape answers it by traversing the graph. Vector-only systems (MemPalace, basic RAG) fail because individual chunks aren't semantically similar enough to surface together.

## Problem Statement

Most agent memory systems store memories as vectors and retrieve by semantic similarity. This works for "find something similar" but fails when the answer requires connecting multiple pieces of information across sources. Landscape solves this with real graph traversal: memories are stored as both vector embeddings AND a queryable knowledge graph, and retrieval combines both.

# Agent Instructions
* Planning: Use Opus for architectural decisions, task decomposition, and creating a detailed plan.md
* Implementation: Delegate all command execution and file edits to !(CLAUDE ONLY)! sonnet subagents, ensuring they follow !(CLAUDE ONLY)! Opus generated plan exactly. For Codex: use gpt-5.4 for planning/orchestration and gpt-5.4-mini for subagent implementation
* Questions: If requirements are ambiguous, escalate to Opus immediately and get approval from user.
* Workflow is always: Plan -> Implement -> Test -> Review -> Push
* After completing a set of features, make sure to teach and walk the user through the code and logic.

## Landscape's Specific Differentiators

1. **Multi-hop graph traversal** — "Find everything within 3 hops of Entity X" is a single Cypher query. MemPalace can't do this (flat lookups). GraphRAG can but is batch-only.

2. **Hybrid retrieval with merge/rank** — Vector similarity candidates + graph neighborhood expansion, deduplicated and scored. Neither MemPalace (vector OR triple, never both) nor GraphRAG (separate query modes) combines them in a single ranked retrieval.

3. **LLM-powered entity extraction** — Structured output prompting catches implicit relationships, coreference, and semantic nuance that MemPalace's regex pipeline misses.

4. **Temporal supersession chains** — SUPERSEDES edges with timestamps and audit trail. MemPalace silently accumulates conflicting facts. GraphRAG has no temporal model.

5. **Continuous ingestion** — Agents write new memories during conversation. GraphRAG requires batch reindexing. MemPalace supports this via MCP add_drawer but without graph integration.

6. **Fully local** — Docker Compose: Neo4j + Qdrant + Ollama + FastAPI. No cloud accounts, no API costs.

## Tech Stack

### Core (all local, all Docker)
- **Python / FastAPI** — API layer and orchestration
- **Neo4j** — Knowledge graph with real Cypher traversal, multi-hop path finding, subgraph extraction. Already experienced with this from GovGraph.
- **Qdrant** — Purpose-built vector search engine. Built-in hybrid search (dense + sparse vectors), native payload filtering during HNSW traversal, built-in BM25 via sparse vectors, JSON payload metadata. Runs as a single Docker container. Advantages over pgvector: purpose-built for vector workloads, better filtering performance, native hybrid search without needing a separate FTS system, rich payload filtering, and a dedicated management UI.
- **Ollama** — Local LLM inference for entity extraction and embeddings. Runs Llama 3.1 8B or Mistral 7B on laptop GPU. Zero API costs.

### Frameworks & Libraries
- **LangChain** — Custom retriever interface (so Landscape plugs into any LangChain agent) and agent loop boilerplate. The hybrid retrieval logic is custom — LangChain provides the interface, not the implementation.
- **sentence-transformers** — Local embedding generation. all-MiniLM-L6-v2 for speed, nomic-embed-text for quality. Alternative: generate embeddings via Ollama directly.
- **MCP SDK (Python)** — Standard agent memory protocol. Claude Code, Cursor, and any MCP client can use Landscape as memory.

### Infrastructure
- **Docker Compose** — Single `docker-compose up` runs Neo4j + Qdrant + Ollama + FastAPI
- **pytest** — Integration tests + multi-hop retrieval quality benchmarks

### Future (Phase 4)
- **LLaVA (via Ollama)** — Local multimodal image understanding for CV extension
- **Tesseract** — Local OCR for text-heavy images

## Data Model

### Neo4j Graph Nodes
- `:Entity` — name, type, source_doc, timestamp, confidence, embedding_id, aliases[]
- `:Document` — title, source_type (text|image), ingested_at, content_hash
- `:Chunk` — text, position, embedding_id, doc_ref

### Neo4j Graph Edges
- `[:RELATES_TO]` — relationship_type, confidence, source_doc, extracted_at
- `[:EXTRACTED_FROM]` — extraction_method (llm|ocr|cv), model, timestamp
- `[:SUPERSEDES]` — reason, timestamp (temporal conflict resolution chain)
- `[:SAME_AS]` — confidence, method (entity resolution link)

### Qdrant Collections
- `entities` — Entity embeddings (384/768 dims depending on model), payload: {neo4j_node_id, name, type, source_doc, timestamp}
- `chunks` — Document chunk embeddings for broad context retrieval, payload: {doc_id, position, source_type}

## Architecture Phases

### Phase 1: Local Stack & Text Ingestion [START HERE]
Set up the fully local environment and build the core pipeline: text in → entities and relationships extracted → stored in graph + vector.

Tasks:
- [ ] Docker Compose: Neo4j + Qdrant + Ollama + FastAPI app
- [ ] Ollama setup with local model (Llama 3.1 8B or Mistral 7B)
- [ ] FastAPI ingestion endpoint — accepts text/markdown/plaintext
- [ ] LLM-powered entity + relationship extraction via structured output prompting
- [ ] Entity resolution: fuzzy name matching + coreference detection across documents
- [ ] Store entities as Neo4j nodes with properties
- [ ] Store relationships as typed Neo4j edges with metadata
- [ ] Generate embeddings locally (sentence-transformers or Ollama)
- [ ] Store embeddings in Qdrant with neo4j_node_id cross-references in payload
- [ ] Integration tests: ingest a doc, verify graph + vector state

**Deliverable:** Feed in markdown/text → populated graph + vector store, all local

### Phase 2: Hybrid Retrieval Engine [NEXT]
The core differentiator. Build the query layer that combines vector similarity with real graph traversal.

Tasks:
- [ ] Vector similarity search (Qdrant top-k nearest neighbors)
- [ ] Graph traversal from matched nodes — BFS/DFS, 1-3 hop configurable depth
- [ ] Hybrid merge: deduplicate and rank results from both retrieval paths
- [ ] Scoring function: vector similarity × graph distance × recency weighting
- [ ] Context assembly: format retrieved subgraph for LLM consumption
- [ ] LangChain custom retriever interface
- [ ] Temporal filtering: only retrieve currently-valid facts (supersession-aware)
- [ ] Benchmark script: hybrid vs vector-only vs graph-only on multi-hop test corpus
- [ ] Build the killer demo dataset (fictional company docs, 1/2/3-hop questions)

**Deliverable:** Hybrid retrieval API with benchmark proving multi-hop advantage

### Phase 3: Agent Integration & MCP [LATER]
Expose Landscape as an MCP server and build a demo agent.

Tasks:
- [ ] MCP server: expose search, ingest, graph query, and status as MCP tools
- [ ] LangChain agent with Landscape as custom memory backend
- [ ] Memory persistence across agent sessions
- [ ] Write-back: agent can add new memories during conversation
- [ ] Temporal conflict resolution with SUPERSEDES chains and audit trail
- [ ] Side-by-side demo: same questions with Landscape vs ChromaDB-only
- [ ] README with architecture diagram, benchmark results, and setup instructions

**Deliverable:** MCP-compatible memory server + compelling demo with comparison results

### Phase 4: Visual Ingestion / CV Extension [FUTURE]
Add computer vision as a second input modality.

Tasks:
- [ ] Multimodal LLM for image entity extraction (local LLaVA or Claude Vision API)
- [ ] Entity extraction from: architecture diagrams, screenshots, whiteboard photos
- [ ] OCR fallback for text-heavy images (Tesseract)
- [ ] Image-derived entities linked to text-derived entities in same graph
- [ ] Visual provenance: store source image reference on extracted nodes
- [ ] Demo: agent queries across both text and image-sourced memories

**Deliverable:** Agent builds memory from text + visual input, queries across both

## Key Design Decisions

**Why frame around multi-hop instead of "hybrid retrieval"?**
Everyone claims hybrid retrieval. Multi-hop is the specific, demonstrable capability that vector-only systems can't match. It's the thing MemPalace's knowledge graph claims to do but doesn't. Leading with the concrete capability makes the demo obvious and the differentiation undeniable.

**Why not extend MemPalace?**
MemPalace stores everything in a single ChromaDB collection. The palace hierarchy is computed on-demand by scanning metadata. Adding real graph traversal means rearchitecting the storage layer. Easier to build the right foundation.

**Why Neo4j over SQLite triples?**
"Find everything within 3 hops of Entity X" is one Cypher query. In SQLite it's recursive CTEs that scale poorly. The graph traversal IS the differentiator — it needs a real graph database.

**Why Qdrant over pgvector?**
Qdrant is purpose-built for vector search: native hybrid search (dense + sparse), payload filtering during HNSW traversal (not pre/post-filter), built-in quantization, and a management UI. pgvector is a Postgres extension — good for simple cases but Landscape needs a real vector engine. Qdrant also runs as a single Docker container with zero configuration, and its rich payload system lets us attach neo4j_node_id and metadata directly to vectors for cross-referencing.

**Why use LangChain at all?**
For plumbing, not for the interesting parts. LangChain provides the retriever interface (so Landscape plugs into any agent) and agent loop boilerplate. The hybrid retrieval logic — vector search, graph traversal, merge/rank — is all custom. That's what goes on the resume.

**Why Ollama over cloud APIs?**
Local-first is a hard requirement. Ollama runs Llama 3.1 8B on a laptop GPU. Extraction quality is sufficient for entity/relationship work. Zero API costs, fully offline. Cloud LLM is an optional upgrade path, not a dependency.

**What about MemPalace's low wake-up cost?**
Their 170-token L0+L1 loading is genuinely good design. Landscape should adopt a similar pattern: load a small identity + critical facts summary on wake, then search on demand. Don't stuff the graph into context — let the agent query it.

**How to benchmark fairly?**
Use LongMemEval (same as MemPalace) plus a custom multi-hop benchmark. On LongMemEval (single-hop semantic recall), Landscape should match MemPalace. On multi-hop questions, Landscape should clearly win. Showing both is more credible than cherry-picking.

## Known Limitations (must address later)

### Relation-type supersession is synonym-blind

Our supersession rule in `neo4j_store.upsert_relation` fires only when a new edge exactly matches an existing edge on `(subject, rel_type)`. Small local LLMs like Llama 3.1 8B are non-deterministic about rel_type phrasing — a single "Alice works at X" input can yield `WORKS_FOR`, `EMPLOYED_BY`, or `CURRENTLY_WORKS_AT` on different calls, and our rule treats those as independent additive relations. The result: when Alice moves from Acme to Zylos, only one of her Acme edges gets superseded, and the others remain live, quietly contradicting the current facts.

**Current mitigation (Phase 2, minimum-viable):**
- Closed vocabulary of 10 canonical rel_types declared in the extraction prompt (`WORKS_FOR, LEADS, MEMBER_OF, REPORTS_TO, APPROVED, USES, BELONGS_TO, LOCATED_IN, CREATED, RELATED_TO`).
- `normalize_relation_type()` in `extraction/schema.py` maps known synonyms → canonical (e.g. `EMPLOYED_BY → WORKS_FOR`) at the pipeline layer before `upsert_relation` is called.
- Unknown rel_types pass through unchanged to avoid destroying novel semantics.

**What this does NOT fix:**
- Reverse-direction synonyms (`APPROVED_BY` inverts subject/object) — deferred; would require swapping args on upsert.
- Brand-new rel_types the LLM invents that aren't in the synonym map — still create independent edges and won't supersede.
- Semantic near-synonyms that aren't literal string synonyms (`REPORTS_TO` vs `MANAGED_BY`) — different rel_types on opposite directions, same real-world relation.

**Path forward (Phase 2.5 / Phase 3 candidate):**
1. Embedding-based rel_type clustering: embed each novel rel_type string, collapse anything above a cosine threshold into the nearest canonical.
2. Cross-type supersession at upsert time: when a new edge from S lands with type T, look for live edges from S whose type is in the same semantic cluster as T, not just type-equal.
3. Once we have the retrieval benchmark in place, *measure* how much synonym drift actually hurts precision on multi-hop queries. The decision between 1 and 2 should be data-driven, not architectural.

Until that's done, any demo that relies on temporal conflict resolution should use hand-constructed corpora where supersession patterns are deterministic, or use the closed vocabulary aggressively. The integration test `test_temporal_filter_excludes_superseded` in `tests/test_retrieval_basic.py` deliberately builds its graph state via Cypher instead of LLM ingestion for exactly this reason.

### Supersession is now gated on functional rel types only

An earlier version of `upsert_relation` treated *every* `(subject, rel_type)` collision as supersession. That silently broke non-functional relations: when Diego led both Vision Team and Project Sentinel, the second ingest marked the first edge stale even though both facts are true. Same for a project that uses multiple technologies (`USES`), a person who approves multiple things (`APPROVED`), or a company with multiple offices (`LOCATED_IN`).

**Current behavior (fixed in Phase 2):**
- `FUNCTIONAL_RELATION_TYPES` in `extraction/schema.py` is a frozenset of rel types that are at-most-one-per-subject: `{WORKS_FOR, REPORTS_TO, BELONGS_TO}`.
- Case 2 supersession in `upsert_relation` only fires when the rel type is functional. For everything else, a second edge with a different object is additive — both coexist.
- This caught a real bug surfaced by the Helios Robotics killer-demo corpus; without this, Sentinel's `USES PyTorch` and Vision Team's `LEADS` edge from Diego were silently getting superseded.

**Edge cases not yet handled:**
- Functional-with-history semantics (e.g. `WORKS_FOR` — Alice genuinely moves companies). The current whitelist is correct for these: the old edge gets `valid_until` set and the new one is live.
- Pluggable-functionality: some orgs genuinely have multiple parents (`BELONGS_TO` across acquisitions). The fix is conservative — if you need multi-parent, omit `BELONGS_TO` from the functional set.
- The 7-doc killer-demo corpus validates the fix against an LLM extraction pipeline, not just unit tests. See `tests/test_killer_demo.py` and `tests/fixtures/killer_demo_corpus/`.
