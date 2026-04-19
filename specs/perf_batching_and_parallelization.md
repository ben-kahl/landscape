# Retrieval + Ingest Performance: Batching & Parallelization

**Status:** Draft spec — not yet broken into an implementation plan.
**Origin:** Follow-up from the chunk-surfacing architecture review (2026-04-19). All findings reference source code on the `chunk-surfacing` branch at commit `dd77a75`.

## Goal

Cut wall-clock latency on the retrieval hot path and the ingest pipeline by exploiting batching and `asyncio.gather` opportunities that the existing code already has the primitives for — no architectural rework, no new dependencies.

## Why now

The chunk-surfacing feature just added a second parallel retrieval path (chunks alongside entities) and a second pair of filter helpers (`get_chunks_in_conversation`, `get_chunks_since`). Each was intentionally written to mirror the existing sequential style, which means the cost of the "do these in parallel" refactor compounds with every new retrieval surface we add. Fixing it now sets the pattern for future work and takes the pressure off before we scale up the benchmark corpus.

## Scope

**In:**
- Parallelize sequential Neo4j filter queries in `retrieve()`.
- Batch-encode chunks during ingest using the existing `encoder.embed_documents` path.
- Batch-encode entities during ingest and parallelize per-entity resolver calls.
- Parallelize the two Qdrant seed searches (`search_entities_any_type`, `search_chunks`) in `retrieve()`.
- Parallelize reinforcement writes (`touch_entities`, `touch_relations`).
- Derive or assert Qdrant `DIMS` from the configured embedding model instead of hardcoding.

**Out of scope:**
- Async wrapper for Ollama's sync `chat()` via `asyncio.to_thread`. Separate spec — it interacts with request cancellation semantics and deserves its own design pass.
- Missing Neo4j indexes. Belongs in ops/IaC, not application code.
- Relation-type synonym-blind supersession (tracked in CLAUDE.md limitations).
- Any change to embedding model defaults or chunk sizes.

## Current state — where the time goes

All line numbers are against commit `dd77a75` on the `chunk-surfacing` branch.

### Retrieval hot path (`src/landscape/retrieval/query.py`)

1. **Seed search** (lines ~63-79): `search_entities_any_type` awaited, then `search_chunks` awaited. Two independent Qdrant queries in series.
2. **Seed hydration + chunk→entity mapping** (lines ~101, ~123): serial awaits on `get_entity_by_element_id` and `get_entities_from_chunks`. Independent.
3. **Filters** (lines ~220-235): when both `session_id` and `since` are set, four independent Neo4j queries run back-to-back. Each is ~5-7 ms; combined ~20-28 ms sequential vs. ~7 ms parallel.
4. **Reinforcement writes** (lines ~265-266): `touch_entities` and `touch_relations` serial; fire-and-forget with respect to the response.

### Ingest pipeline (`src/landscape/pipeline.py`)

1. **Chunk encoding + upsert** (lines ~59-71): per-chunk `encoder.encode(chunk.text)` + per-chunk `qdrant_store.upsert_chunk`. `encoder.embed_documents(list[str])` exists and is unused in this path.
2. **Entity encoding + resolution** (lines ~81-127): per-entity `encoder.encode(...)` + per-entity `await resolver.resolve_entity(...)`. A 20-entity doc does 20 sequential encoder calls and 20 sequential Qdrant round-trips.

### Config safety (`src/landscape/storage/qdrant_store.py:18`)

`DIMS = 768` is hardcoded to match `nomic-ai/nomic-embed-text-v1.5`. Swapping `embedding_model` to a 384-dim variant like `all-MiniLM-L6-v2` silently corrupts Qdrant collections because the collection is created at the old dimension and subsequent upserts either fail obscurely or get truncated depending on the client path.

## Proposed changes

### 1. Parallelize filter queries in `retrieve()`

In the branch that handles simultaneous `session_id` + `since`, collect the four helper calls into one `asyncio.gather(...)`. Same treatment for the single-filter branches. Unpack into the same `allowlist` / `chunk_allowlist` sets as today. No behavior change — only scheduling.

### 2. Parallelize seed searches and hydration

Two gathers:
- `entity_hits, chunk_hits = await asyncio.gather(search_entities_any_type(...), search_chunks(...))`
- After seeds land: `hydrated_entities, chunk_to_ents = await asyncio.gather(hydrate(...), get_entities_from_chunks(...))`

This preserves the "chunk seeds feed entity expansion" ordering — only the *independent* work runs in parallel.

### 3. Parallelize reinforcement writes

`await asyncio.gather(touch_entities(...), touch_relations(...))` when `reinforce=True`. No correctness impact (both are writes against disjoint edge types).

### 4. Batch chunk encoding in ingest

Before the upsert loop, call `encoder.embed_documents([c.text for c in chunks])` once. Then `asyncio.gather(*[qdrant_store.upsert_chunk(c, v) for c, v in zip(chunks, vectors)])`. Be careful to preserve the `chunk_neo4j_id` cross-reference (already set earlier in the pipeline by the Neo4j chunk merge) — the batch change is strictly local to embedding + upsert.

### 5. Batch entity encoding + parallel resolution

Build one encoder call over all extracted entity names (formatted with their canonical types as today), then run resolver calls with `asyncio.gather`. Preserve the existing "new entity created within this ingest" dedupe — likely by resolving in two passes or by using a per-ingest dict that resolver calls can consult. Flag this in the implementation plan: the dedupe path is the reason the loop is serial today, and the refactor has to handle it explicitly rather than assume independence.

### 6. DIMS derived from model

In `config.py`, expose a small map `{model_name: dims}` covering the models we actually use (`nomic-embed-text-v1.5` → 768, `all-MiniLM-L6-v2` → 384). Alternatively, probe the encoder at load time and read `model.get_sentence_embedding_dimension()`. `qdrant_store.DIMS` becomes a function or a computed constant resolved from settings. Collection creation still uses a single source of truth — collections stay idempotent.

## Risks and tradeoffs

- **Entity resolution dedupe (change 5) is the only finding with real risk.** The serial loop exists partly because resolving entity A may create a canonical node that entity B should match against. Parallelizing naively produces duplicate canonical entities for near-synonyms written in the same document. The implementation plan must either (a) pre-group by normalized name before resolving, or (b) run resolution in dependency-ordered batches. This should be validated against the killer-demo corpus, which deliberately contains coreferent mentions.
- **Reinforcement parallelism (change 3) is cheap and safe.** Two separate edge types, no shared write conflicts.
- **Seed search parallelism (change 2) has no known risk** — the two Qdrant collections are independent.
- **Filter parallelism (change 1) is trivially safe** — four pure reads against different label/relationship patterns.
- **DIMS change (change 6) is pure config hygiene.** The failure mode it fixes is silent; the change itself can only surface the problem earlier, not hide it.

## Acceptance criteria

- `retrieve()` with `session_id` + `since` set makes at most one `await` boundary per independent-Cypher group (no more serial waits on independent reads). Verify by inspection and by adding a latency assertion around a mocked Neo4j.
- Ingest for a 10-chunk, 20-entity document calls `encoder.embed_documents` at most twice (once for chunks, once for entities) and issues Qdrant upserts via `asyncio.gather`. Verify with an `AsyncMock` on the encoder and on `qdrant_store.upsert_*`.
- Entity resolver still produces the same canonical-entity count on the Helios Robotics killer-demo corpus as the current implementation. Regression test: run the existing killer-demo ingest fixture and assert entity count equals the baseline.
- `qdrant_store.init_collection` requests the vector size that matches the currently configured embedding model. Unit test: parametrize over two supported models, assert the collection size matches.
- All existing tests (`uv run pytest -q`) still pass. The 6 pre-existing failures (`test_chromadb_baseline.*`, `test_llm_options.*`) are tracked separately and should not be touched by this work.

## Non-goals

- No scoring-function changes.
- No change to the chunk-surfacing feature itself.
- No change to what gets retrieved — only *how fast*.

## Follow-ups (explicitly deferred)

- Benchmarking harness that reports per-stage latency (seed / hydrate / filter / expand / rank). Currently we can eyeball but not measure. Worth a short spec after this lands.
- `asyncio.to_thread` wrapper for Ollama calls. Separate design — touches cancellation.
- Neo4j index creation as part of startup/migration. Ops concern.
