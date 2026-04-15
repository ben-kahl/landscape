# Landscape — Multi-Hop Reasoning Over Agent Memory

## What This Is

Landscape is a local-first agent memory system that enables multi-hop reasoning through hybrid Neo4j graph traversal and Qdrant vector similarity search. It gives AI agents persistent, evolving memory that supports structured reasoning — not just semantic recall.

The core pitch: ask a question that requires connecting 3 pieces of information across different documents. Landscape answers it by traversing the graph. Vector-only systems (MemPalace, basic RAG) fail because individual chunks aren't semantically similar enough to surface together.

## Problem Statement

Most agent memory systems store memories as vectors and retrieve by semantic similarity. This works for "find something similar" but fails when the answer requires connecting multiple pieces of information across sources. Landscape solves this with real graph traversal: memories are stored as both vector embeddings AND a queryable knowledge graph, and retrieval combines both.

# Agent Instructions
* Planning: Use Opus for architectural decisions, task decomposition, and creating a detailed plan.md
* Implementation: Delegate all command execution and file edits to sonnet subagents, ensuring they follow Opus generated plan exactly.
* Workflow: If requirements are ambiguous, escalate to Opus immediately and get approval from user.

## Competitive Landscape

### MemPalace (Jovovich/Sigman, April 2026)
- 19K+ GitHub stars, 96.6% LongMemEval score
- Stores verbatim conversations in a single ChromaDB collection with spatial "palace" metaphor
- **Critical gaps (confirmed by independent code analysis):**
  - "Knowledge graph" is flat SQLite triple lookups — no traversal, no multi-hop
  - Entity resolution is slug-based exact matching only (alice_obrien)
  - Contradiction detection does not exist in codebase despite README claims
  - Retrieval is ChromaDB nearest-neighbor only — no hybrid search
  - Extraction is regex/keyword — no LLM, misses implicit relationships
  - 96.6% benchmark is just ChromaDB vector search, not the palace structure
- **Genuine strengths worth learning from:**
  - ~170 token wake-up cost (L0+L1 progressive loading)
  - Zero-LLM write path = zero API cost on writes
  - MCP integration pattern (PALACE_PROTOCOL in status output)
  - Verbatim storage philosophy avoids premature information loss

### Microsoft GraphRAG
- LLM-powered entity/relationship extraction into knowledge graph
- Community detection + hierarchical summarization for global queries
- Local Search (entity neighborhood), Global Search (community summaries), DRIFT Search
- **Key differences from Landscape:**
  - Document QA system, not agent memory — batch index then query, no continuous ingestion
  - No temporal awareness or fact evolution
  - No MCP integration for live agent read/write
  - Production path targets Azure cloud (Cosmos DB), not local-first
  - Community detection is genuinely novel — Landscape doesn't have this (yet)

### Zep Graphiti
- Real Neo4j graph with community detection, episodic memory, BFS retrieval, entity resolution
- Closest to Landscape's architecture
- **Gap:** Requires Neo4j cloud services and API costs. Not local-first. Commercial/proprietary.

## Landscape's Specific Differentiators

1. **Multi-hop graph traversal** — "Find everything within 3 hops of Entity X" is a single Cypher query. MemPalace can't do this (flat lookups). GraphRAG can but is batch-only.

2. **Hybrid retrieval with merge/rank** — Vector similarity candidates + graph neighborhood expansion, deduplicated and scored. Neither MemPalace (vector OR triple, never both) nor GraphRAG (separate query modes) combines them in a single ranked retrieval.

3. **LLM-powered entity extraction** — Structured output prompting catches implicit relationships, coreference, and semantic nuance that MemPalace's regex pipeline misses.

4. **Temporal supersession chains** — SUPERSEDES edges with timestamps and audit trail. MemPalace silently accumulates conflicting facts. GraphRAG has no temporal model.

5. **Continuous ingestion** — Agents write new memories during conversation. GraphRAG requires batch reindexing. MemPalace supports this via MCP add_drawer but without graph integration.

6. **Fully local** — Docker Compose: Neo4j + Qdrant + Ollama + FastAPI. No cloud accounts, no API costs.

## The Killer Demo

Feed the agent 5-10 documents about a fictional company (org charts, meeting notes, architecture decisions, project updates). Then ask questions at increasing complexity:

**1-hop (vector search handles this):**
> "What database does Project Atlas use?"
Vector similarity finds the chunk mentioning "Project Atlas uses PostgreSQL." Both approaches work.

**2-hop (vector search starts failing):**
> "Who approved the database choice for Project Atlas?"
Requires connecting: (1) Atlas uses PostgreSQL (arch doc) → (2) Sarah approved the PostgreSQL migration (meeting notes). Landscape traverses the graph. ChromaDB may miss this.

**3-hop (vector search fails reliably):**
> "What team does the person who approved Atlas's database work on?"
Requires: Atlas → PostgreSQL → Sarah → Platform Team. Three traversal steps. No single chunk contains this chain. Landscape follows the graph. Vector-only returns irrelevant results.

**The punchline:** Side-by-side comparison showing Landscape's retrieved context (correct subgraph) vs ChromaDB-only (semantically similar but wrong chunks). The graph path is the proof.

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

## Key Implementation Instructions

* Workflow is always: Plan -> Implement -> Test -> Review -> Push
