# Retrieval + Ingest Perf (Batching & Parallelization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut wall-clock latency on the retrieval hot path and the ingest pipeline by exploiting batching and `asyncio.gather` opportunities without architectural rework.

**Architecture:** Six localized refactors across `src/landscape/retrieval/query.py`, `src/landscape/pipeline.py`, `src/landscape/storage/qdrant_store.py`, and `src/landscape/config.py`. Each refactor preserves existing ordering semantics and adds parallel scheduling only where calls are proven independent. One ingest-side change (entity batching) carries real dedupe risk and is handled with intra-batch name+type grouping plus a killer-demo regression test.

**Tech Stack:** Python 3.11+, `asyncio`, `pytest`, `pytest-asyncio`, `unittest.mock.AsyncMock`, `qdrant-client`, `neo4j` async driver, `langchain-huggingface`.

**Source spec:** `specs/perf_batching_and_parallelization.md` (commit `f67f365`). All line numbers below are against `c60929c` on branch `phase-35`.

**Pre-existing test failures (do not touch):** `tests/test_chromadb_baseline.py`, `tests/test_llm_options.py`. Tracked separately.

---

## File Structure

### Created
- `tests/test_perf_batching.py` — latency-assertion unit tests for the parallelization refactors using `AsyncMock` + controlled delays. Single file so all perf regressions live together.

### Modified
- `src/landscape/config.py` — add `EMBEDDING_MODEL_DIMS` map + `embedding_dims` property on `Settings`.
- `src/landscape/storage/qdrant_store.py` — replace hardcoded `DIMS = 768` constant with lookup against `settings.embedding_dims`.
- `src/landscape/retrieval/query.py` — four `asyncio.gather` sites: filters, seed searches, seed-hydration-plus-chunk-mapping, reinforcement writes.
- `src/landscape/pipeline.py` — chunk batch encode + parallel upsert; entity batch encode + parallel resolve with intra-batch name-group dedupe.

### No new modules
All changes are localized to existing files. No helper modules, no new abstractions — changes are scheduling-only wherever possible.

---

## Task 1: Derive Qdrant DIMS from the configured embedding model

**Why first:** Smallest, purely-additive, and establishes the config pattern other tasks' tests depend on (tests construct settings overrides).

**Files:**
- Modify: `src/landscape/config.py:52-79`
- Modify: `src/landscape/storage/qdrant_store.py:18`, `44`, `55`
- Test: `tests/test_perf_batching.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_perf_batching.py` with:

```python
"""Tests for the batching + parallelization perf work (spec: specs/perf_batching_and_parallelization.md)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from landscape.config import EMBEDDING_MODEL_DIMS, Settings


def test_embedding_model_dims_map_contains_known_models():
    assert EMBEDDING_MODEL_DIMS["nomic-ai/nomic-embed-text-v1.5"] == 768
    assert EMBEDDING_MODEL_DIMS["sentence-transformers/all-MiniLM-L6-v2"] == 384


def test_settings_embedding_dims_resolves_from_model_name():
    s = Settings(embedding_model="nomic-ai/nomic-embed-text-v1.5")
    assert s.embedding_dims == 768
    s2 = Settings(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert s2.embedding_dims == 384


def test_settings_embedding_dims_raises_for_unknown_model():
    s = Settings(embedding_model="made-up/not-a-real-model")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        _ = s.embedding_dims
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_perf_batching.py -v`
Expected: `ImportError` on `EMBEDDING_MODEL_DIMS` (symbol not yet exported).

- [ ] **Step 3: Add the DIMS map and property to `config.py`**

Edit `src/landscape/config.py` — add after `LLM_PROFILES` block (around line 40):

```python
# Embedding model → output dimension. Extend this when switching/adding models.
# Source: published model card of each encoder.
EMBEDDING_MODEL_DIMS: dict[str, int] = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}
```

Add to the `Settings` class (after `model_post_init`):

```python
    @property
    def embedding_dims(self) -> int:
        try:
            return EMBEDDING_MODEL_DIMS[self.embedding_model]
        except KeyError as exc:
            known = ", ".join(sorted(EMBEDDING_MODEL_DIMS))
            raise ValueError(
                f"Unknown embedding model {self.embedding_model!r}. "
                f"Add it to EMBEDDING_MODEL_DIMS in src/landscape/config.py. "
                f"Known: {known}."
            ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_perf_batching.py -v`
Expected: all three tests PASS.

- [ ] **Step 5: Wire `qdrant_store` to read from settings**

Edit `src/landscape/storage/qdrant_store.py`:

Replace line 18:
```python
DIMS = 768
```
with:
```python
from landscape.config import settings

def _dims() -> int:
    return settings.embedding_dims
```

(Drop the duplicate `from landscape.config import settings` if it's already imported above — keep a single import.)

Replace `size=DIMS` on line ~44 and line ~55 with `size=_dims()`.

- [ ] **Step 6: Add an init_collection dimension test**

Append to `tests/test_perf_batching.py`:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,expected_dims",
    [
        ("nomic-ai/nomic-embed-text-v1.5", 768),
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ],
)
async def test_init_collection_uses_configured_dims(model_name, expected_dims):
    from landscape.storage import qdrant_store

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    fake_client.create_collection = AsyncMock()

    with patch.object(qdrant_store, "get_client", return_value=fake_client), \
         patch("landscape.storage.qdrant_store.settings") as mock_settings:
        mock_settings.embedding_dims = expected_dims

        await qdrant_store.init_collection()

    call = fake_client.create_collection.call_args
    assert call.kwargs["vectors_config"].size == expected_dims
```

- [ ] **Step 7: Run the new test**

Run: `uv run pytest tests/test_perf_batching.py -v`
Expected: all four tests PASS.

- [ ] **Step 8: Run full suite to confirm no regressions**

Run: `uv run pytest -q --ignore=tests/test_chromadb_baseline.py --ignore=tests/test_llm_options.py`
Expected: PASS (or existing-failure count unchanged).

- [ ] **Step 9: Commit**

```bash
git add src/landscape/config.py src/landscape/storage/qdrant_store.py tests/test_perf_batching.py
git commit -m "feat(config): derive Qdrant DIMS from configured embedding model

Replaces hardcoded DIMS=768 with a model→dims map on Settings. Surfaces
dimension mismatches at collection-init time instead of silently corrupting
writes when embedding_model is swapped."
```

---

## Task 2: Parallelize filter queries in `retrieve()`

**Files:**
- Modify: `src/landscape/retrieval/query.py:219-235`
- Test: `tests/test_perf_batching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_perf_batching.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_runs_both_filter_queries_in_parallel():
    """When session_id AND since are both set, the four Neo4j filter calls
    (entities-in-conv, entities-since, chunks-in-conv, chunks-since) must be
    gathered, not awaited serially."""
    from datetime import UTC, datetime
    from landscape.retrieval import query as query_mod

    call_log: list[tuple[str, float]] = []
    start_time = asyncio.get_event_loop().time()

    async def slow_return(label, value):
        # 50ms per call — if serial, total wait is 200ms; if parallel, 50ms.
        await asyncio.sleep(0.05)
        call_log.append((label, asyncio.get_event_loop().time() - start_time))
        return value

    with patch.object(
        query_mod.qdrant_store, "search_entities_any_type",
        AsyncMock(return_value=[]),
    ), patch.object(
        query_mod.qdrant_store, "search_chunks",
        AsyncMock(return_value=[]),
    ), patch.object(
        query_mod.encoder, "embed_query", return_value=[0.0] * 4,
    ), patch.object(
        query_mod.neo4j_store, "get_entities_from_chunks",
        AsyncMock(return_value=[]),
    ), patch.object(
        query_mod.neo4j_store, "get_entities_in_conversation",
        AsyncMock(side_effect=lambda *a, **kw: slow_return("ent_conv", ["e1"])),
    ), patch.object(
        query_mod.neo4j_store, "get_entities_since",
        AsyncMock(side_effect=lambda *a, **kw: slow_return("ent_since", ["e1"])),
    ), patch.object(
        query_mod.neo4j_store, "get_chunks_in_conversation",
        AsyncMock(side_effect=lambda *a, **kw: slow_return("chunk_conv", [])),
    ), patch.object(
        query_mod.neo4j_store, "get_chunks_since",
        AsyncMock(side_effect=lambda *a, **kw: slow_return("chunk_since", [])),
    ):
        await query_mod.retrieve(
            "q",
            session_id="s1",
            since=datetime(2026, 1, 1, tzinfo=UTC),
            reinforce=False,
        )

    # All four filter calls should start within ~20ms of each other (i.e.,
    # before any of them finishes). If serial, second finishes at ~100ms etc.
    # Parallel: all finish within a narrow band.
    timestamps = [t for _, t in call_log]
    assert max(timestamps) - min(timestamps) < 0.03, (
        f"filter calls not parallelized — spread: {max(timestamps) - min(timestamps):.3f}s"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_perf_batching.py::test_retrieve_runs_both_filter_queries_in_parallel -v`
Expected: FAIL with "filter calls not parallelized — spread: ~0.15s".

- [ ] **Step 3: Parallelize the filter branch**

Edit `src/landscape/retrieval/query.py`. Add `import asyncio` at top if not present (it is not — add it).

Replace lines 219-235:

```python
    if session_id is not None or since is not None:
        if session_id is not None and since is not None:
            conv_ids, since_ids, conv_cids, since_cids = await asyncio.gather(
                neo4j_store.get_entities_in_conversation(session_id),
                neo4j_store.get_entities_since(since),
                neo4j_store.get_chunks_in_conversation(session_id),
                neo4j_store.get_chunks_since(since),
            )
            allowlist = set(conv_ids) & set(since_ids)
            chunk_allowlist = set(conv_cids) & set(since_cids)
        elif session_id is not None:
            ents, chunks_in_conv = await asyncio.gather(
                neo4j_store.get_entities_in_conversation(session_id),
                neo4j_store.get_chunks_in_conversation(session_id),
            )
            allowlist = set(ents)
            chunk_allowlist = set(chunks_in_conv)
        else:
            assert since is not None
            ents, chunks_since = await asyncio.gather(
                neo4j_store.get_entities_since(since),
                neo4j_store.get_chunks_since(since),
            )
            allowlist = set(ents)
            chunk_allowlist = set(chunks_since)
```

- [ ] **Step 4: Run the new test**

Run: `uv run pytest tests/test_perf_batching.py::test_retrieve_runs_both_filter_queries_in_parallel -v`
Expected: PASS.

- [ ] **Step 5: Run the existing retrieval suite to confirm no semantic regression**

Run: `uv run pytest tests/test_retrieval_basic.py tests/test_conversation_history.py tests/test_cross_session_retrieval.py tests/test_chunk_surfacing.py -v`
Expected: PASS (modulo pre-existing failures in unrelated files).

- [ ] **Step 6: Commit**

```bash
git add src/landscape/retrieval/query.py tests/test_perf_batching.py
git commit -m "perf(retrieval): parallelize session/since filter queries

Four independent Neo4j reads in the session+since branch now run via
asyncio.gather instead of serial awaits. Verified with a latency-spread
assertion — all four calls must start before any finishes."
```

---

## Task 3: Parallelize seed searches + chunk→entity hydration

**Files:**
- Modify: `src/landscape/retrieval/query.py:67-123`
- Test: `tests/test_perf_batching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_perf_batching.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_runs_seed_searches_in_parallel():
    """search_entities_any_type and search_chunks query two different Qdrant
    collections and are independent — must be gathered."""
    from landscape.retrieval import query as query_mod

    start = asyncio.get_event_loop().time()
    times: dict[str, float] = {}

    async def slow_entities(*a, **kw):
        await asyncio.sleep(0.05)
        times["entities"] = asyncio.get_event_loop().time() - start
        return []

    async def slow_chunks(*a, **kw):
        await asyncio.sleep(0.05)
        times["chunks"] = asyncio.get_event_loop().time() - start
        return []

    with patch.object(query_mod.qdrant_store, "search_entities_any_type", side_effect=slow_entities), \
         patch.object(query_mod.qdrant_store, "search_chunks", side_effect=slow_chunks), \
         patch.object(query_mod.encoder, "embed_query", return_value=[0.0] * 4), \
         patch.object(query_mod.neo4j_store, "get_entities_from_chunks", AsyncMock(return_value=[])):
        await query_mod.retrieve("q", reinforce=False)

    assert abs(times["entities"] - times["chunks"]) < 0.03, (
        f"seed searches not parallel: entities={times['entities']:.3f}, "
        f"chunks={times['chunks']:.3f}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_perf_batching.py::test_retrieve_runs_seed_searches_in_parallel -v`
Expected: FAIL — gap between the two timestamps ≈ 0.05s.

- [ ] **Step 3: Parallelize the two seed searches**

Edit `src/landscape/retrieval/query.py`. Replace lines ~63-99 (the block starting with `query_vector = encoder.embed_query(query_text)` through the `retrieved_chunks.append(...)` loop):

```python
    query_vector = encoder.embed_query(query_text)

    # Seed searches: independent Qdrant queries against two collections.
    entity_hits, chunk_hits = await asyncio.gather(
        qdrant_store.search_entities_any_type(query_vector, limit=5),
        qdrant_store.search_chunks(query_vector, limit=5),
    )

    seed_sims: dict[str, float] = {}
    for hit in entity_hits:
        payload = hit.payload or {}
        neo4j_id = payload.get("neo4j_node_id")
        if not neo4j_id:
            continue
        seed_sims[neo4j_id] = max(seed_sims.get(neo4j_id, 0.0), float(hit.score))

    chunk_ids: list[str] = []
    chunk_score_by_id: dict[str, float] = {}
    retrieved_chunks: list[RetrievedChunk] = []
    for hit in chunk_hits:
        payload = hit.payload or {}
        cid = payload.get("chunk_neo4j_id")
        if not cid:
            continue
        chunk_ids.append(cid)
        chunk_score_by_id[cid] = float(hit.score)
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_neo4j_id=cid,
                text=payload.get("text", ""),
                doc_id=payload.get("doc_id", ""),
                source_doc=payload.get("source_doc", ""),
                position=int(payload.get("position", 0)),
                score=float(hit.score),
            )
        )
```

Leave lines 101 onward (`chunk_entities = await neo4j_store.get_entities_from_chunks(chunk_ids)`) unchanged — that call requires the chunk IDs that were just extracted, so it stays ordered.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_perf_batching.py::test_retrieve_runs_seed_searches_in_parallel -v`
Expected: PASS.

- [ ] **Step 5: Parallelize `_hydrate_entities` + `get_entities_from_chunks`**

These are independent once we have `chunk_ids` and `seed_sims.keys()`. But note: `get_entities_from_chunks` feeds new ids into `seed_sims`, and `_hydrate_entities` needs the final id list. So the order-preserving parallelization is:

1. Start `get_entities_from_chunks(chunk_ids)` as soon as `chunk_ids` is known.
2. Walk its result to update `seed_sims`.
3. Then hydrate.

Since step 2 depends on step 1's result, we can't gather (1) with anything. The only win here is that `get_entities_from_chunks` can start before the entity-hits loop finishes *populating* `seed_sims` — but that loop is CPU-only (microseconds), so the gain is nil.

**Decision:** skip hydration+chunk-mapping gathering. The spec's second bullet ("After seeds land: gather(hydrate, get_entities_from_chunks)") misreads the dependency — `_hydrate_entities` is called over `seed_sims.keys()` *after* `get_entities_from_chunks` augments it. Note this in the commit message.

- [ ] **Step 6: Run the retrieval suite**

Run: `uv run pytest tests/test_retrieval_basic.py tests/test_chunk_surfacing.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/landscape/retrieval/query.py tests/test_perf_batching.py
git commit -m "perf(retrieval): parallelize Qdrant seed searches

search_entities_any_type and search_chunks hit independent Qdrant
collections. Gather them. Hydrate-plus-chunk-map is NOT parallelized —
hydration depends on seed_sims after chunk mapping augments it, so the
spec's proposed second gather is order-incompatible."
```

---

## Task 4: Parallelize reinforcement writes

**Files:**
- Modify: `src/landscape/retrieval/query.py:263-266`
- Test: `tests/test_perf_batching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_perf_batching.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_runs_reinforcement_writes_in_parallel():
    from landscape.retrieval import query as query_mod

    start = asyncio.get_event_loop().time()
    times: dict[str, float] = {}

    async def slow_ents(*a, **kw):
        await asyncio.sleep(0.05)
        times["ents"] = asyncio.get_event_loop().time() - start

    async def slow_rels(*a, **kw):
        await asyncio.sleep(0.05)
        times["rels"] = asyncio.get_event_loop().time() - start

    # Seed the retrieval with a single entity hit so reinforce runs.
    fake_hit = MagicMock()
    fake_hit.payload = {"neo4j_node_id": "e1"}
    fake_hit.score = 0.9

    with patch.object(query_mod.qdrant_store, "search_entities_any_type", AsyncMock(return_value=[fake_hit])), \
         patch.object(query_mod.qdrant_store, "search_chunks", AsyncMock(return_value=[])), \
         patch.object(query_mod.encoder, "embed_query", return_value=[0.0] * 4), \
         patch.object(query_mod.neo4j_store, "get_entities_from_chunks", AsyncMock(return_value=[])), \
         patch.object(query_mod.neo4j_store, "bfs_expand", AsyncMock(return_value=[])), \
         patch.object(query_mod, "_hydrate_entities", AsyncMock(return_value=[
             {"eid": "e1", "name": "E1", "type": "T", "access_count": 0, "last_accessed": None}
         ])), \
         patch.object(query_mod.neo4j_store, "touch_entities", side_effect=slow_ents), \
         patch.object(query_mod.neo4j_store, "touch_relations", side_effect=slow_rels):
        await query_mod.retrieve("q", reinforce=True)

    assert abs(times["ents"] - times["rels"]) < 0.03, (
        f"reinforce writes not parallel: ents={times['ents']:.3f}, rels={times['rels']:.3f}"
    )
```

- [ ] **Step 2: Run the test — expect FAIL**

Run: `uv run pytest tests/test_perf_batching.py::test_retrieve_runs_reinforcement_writes_in_parallel -v`
Expected: FAIL (ents and rels ~0.05s apart).

- [ ] **Step 3: Gather the reinforce block**

Edit `src/landscape/retrieval/query.py` lines 263-266:

```python
    if reinforce:
        now_iso = now.isoformat()
        await asyncio.gather(
            neo4j_store.touch_entities(touched_entity_ids, now_iso),
            neo4j_store.touch_relations(touched_edge_ids, now_iso),
        )
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/test_perf_batching.py::test_retrieve_runs_reinforcement_writes_in_parallel -v`
Expected: PASS.

- [ ] **Step 5: Run the reinforcement suite**

Run: `uv run pytest tests/test_reinforcement.py tests/test_retrieval_basic.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/landscape/retrieval/query.py tests/test_perf_batching.py
git commit -m "perf(retrieval): gather touch_entities and touch_relations

The two reinforcement writes are independent — different Neo4j label
sets, no shared-state conflict — and can safely run together."
```

---

## Task 5: Batch chunk encoding + parallel chunk upserts

**Files:**
- Modify: `src/landscape/pipeline.py:56-71`
- Test: `tests/test_perf_batching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_perf_batching.py`:

```python
@pytest.mark.asyncio
async def test_ingest_batch_encodes_chunks_once():
    """A 10-chunk document should trigger exactly one embed_documents call,
    not 10 per-chunk encode() calls."""
    from landscape import pipeline

    fake_chunks = [MagicMock(text=f"chunk {i}", index=i) for i in range(10)]
    with patch.object(pipeline, "chunk_text", return_value=fake_chunks), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.neo4j_store, "create_chunk",
                      AsyncMock(side_effect=[f"ch{i}" for i in range(10)])), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4 for _ in range(10)]) as batch_encode, \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4) as per_call, \
         patch.object(pipeline.qdrant_store, "upsert_chunk", AsyncMock()) as upsert, \
         patch.object(pipeline.llm, "extract",
                      return_value=MagicMock(entities=[], relations=[])):
        await pipeline.ingest(text="x" * 500, title="doc")

    assert batch_encode.call_count == 1
    assert len(batch_encode.call_args.args[0]) == 10
    # The per-chunk encode path should NOT be used for chunks anymore.
    chunk_encode_texts = {c.text for c in fake_chunks}
    per_call_texts = {c.args[0] for c in per_call.call_args_list if c.args}
    assert not chunk_encode_texts & per_call_texts, (
        "chunks are still going through per-call encoder.encode"
    )
    assert upsert.call_count == 10


@pytest.mark.asyncio
async def test_ingest_upserts_chunks_in_parallel():
    from landscape import pipeline

    start = asyncio.get_event_loop().time()
    finish_times: list[float] = []

    async def slow_upsert(*a, **kw):
        await asyncio.sleep(0.05)
        finish_times.append(asyncio.get_event_loop().time() - start)

    fake_chunks = [MagicMock(text=f"chunk {i}", index=i) for i in range(5)]
    with patch.object(pipeline, "chunk_text", return_value=fake_chunks), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.neo4j_store, "create_chunk",
                      AsyncMock(side_effect=[f"ch{i}" for i in range(5)])), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4 for _ in range(5)]), \
         patch.object(pipeline.qdrant_store, "upsert_chunk", side_effect=slow_upsert), \
         patch.object(pipeline.llm, "extract",
                      return_value=MagicMock(entities=[], relations=[])):
        await pipeline.ingest(text="x" * 500, title="doc")

    # Serial: 5 * 0.05 = 0.25s. Parallel: all within 0.06s of each other.
    assert max(finish_times) - min(finish_times) < 0.03
```

- [ ] **Step 2: Run both tests — expect FAIL**

Run: `uv run pytest tests/test_perf_batching.py::test_ingest_batch_encodes_chunks_once tests/test_perf_batching.py::test_ingest_upserts_chunks_in_parallel -v`
Expected: FAIL — `batch_encode.call_count == 0` and serial upserts.

- [ ] **Step 3: Rewrite the chunk-ingest block**

Edit `src/landscape/pipeline.py`. Add `import asyncio` at the top if missing (it is missing — add it).

Replace lines 56-71:

```python
    # Step 2: chunk + embed chunks (batched)
    chunks = chunk_text(text)
    chunks_created = 0
    if chunks:
        chunk_neo4j_ids: list[str] = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            chunk_neo4j_ids.append(
                await neo4j_store.create_chunk(doc_id, chunk.index, chunk.text, chunk_hash)
            )
        chunk_vectors = encoder.embed_documents([c.text for c in chunks])
        await asyncio.gather(
            *(
                qdrant_store.upsert_chunk(
                    chunk_neo4j_id=cid,
                    doc_id=doc_id,
                    source_doc=title,
                    position=chunk.index,
                    text=chunk.text,
                    vector=vec,
                )
                for chunk, cid, vec in zip(chunks, chunk_neo4j_ids, chunk_vectors, strict=True)
            )
        )
        chunks_created = len(chunks)
```

Note: `create_chunk` is kept serial because it writes Neo4j `:Chunk` nodes and a naive parallel call could race-create duplicates within the same doc. The encoder + Qdrant path is the hot loop; that's where batching matters.

- [ ] **Step 4: Run the two new tests — expect PASS**

Run: `uv run pytest tests/test_perf_batching.py::test_ingest_batch_encodes_chunks_once tests/test_perf_batching.py::test_ingest_upserts_chunks_in_parallel -v`
Expected: PASS.

- [ ] **Step 5: Run the full ingest/chunks suite**

Run: `uv run pytest tests/test_ingest.py tests/test_chunks.py tests/test_pipeline_conversation.py tests/test_provenance.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/landscape/pipeline.py tests/test_perf_batching.py
git commit -m "perf(ingest): batch-encode chunks, parallel Qdrant upserts

Replaces a per-chunk encoder.encode + per-chunk upsert_chunk loop with
one embed_documents call and an asyncio.gather of upserts. Neo4j
create_chunk stays serial to avoid duplicate-node races."
```

---

## Task 6: Batch-encode entities + parallel resolution with intra-batch dedupe

**Why tricky:** The current serial loop resolves each extracted entity one at a time. If entity A is created mid-loop, entity B resolving afterward can match A via Qdrant and avoid a duplicate. Naive parallel resolution loses that. Fix: pre-group the extracted entities by `(lowercased name, canonical_type)` before resolution — two mentions of "Alice Smith" in the same doc resolve once and share the canonical id. Cross-name-coreference (e.g., "Alice" ≈ "Alice Smith") is still best-effort as it is today; parallelizing within a single ingest does not meaningfully worsen it because the cases where the old loop caught it required a specific ordering that wasn't guaranteed anyway.

**Files:**
- Modify: `src/landscape/pipeline.py:74-127`
- Test: `tests/test_perf_batching.py`
- Regression: `tests/test_killer_demo.py` (existing, not modified — run as acceptance)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_perf_batching.py`:

```python
@pytest.mark.asyncio
async def test_ingest_batch_encodes_entities_once():
    from landscape import pipeline

    fake_entities = [
        MagicMock(name=f"E{i}", type="Person", confidence=0.9) for i in range(5)
    ]
    # Override the auto-generated `.name` on MagicMock so entity.name returns a string.
    for i, e in enumerate(fake_entities):
        e.name = f"E{i}"
        e.type = "Person"

    extraction = MagicMock(entities=fake_entities, relations=[])

    with patch.object(pipeline, "chunk_text", return_value=[]), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4 for _ in range(5)]) as batch_encode, \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4), \
         patch.object(pipeline.resolver, "resolve_entity",
                      AsyncMock(return_value=(None, True, None))), \
         patch.object(pipeline.neo4j_store, "merge_entity",
                      AsyncMock(side_effect=[f"ent{i}" for i in range(5)])), \
         patch.object(pipeline.qdrant_store, "upsert_entity", AsyncMock()), \
         patch.object(pipeline.llm, "extract", return_value=extraction):
        await pipeline.ingest(text="some doc", title="t")

    # Entity encode should be batched into one call that encodes all 5 names.
    entity_batch_calls = [c for c in batch_encode.call_args_list if len(c.args[0]) == 5]
    assert entity_batch_calls, (
        f"expected one batch of 5 entity texts, got batches: "
        f"{[len(c.args[0]) for c in batch_encode.call_args_list]}"
    )


@pytest.mark.asyncio
async def test_ingest_dedupes_identical_entity_mentions_before_resolving():
    """Two extracted entities with the same (name, type) in one doc should
    resolve once, not twice — the serial loop used to rely on the first
    resolution creating a canonical node that the second would match."""
    from landscape import pipeline

    # Same name twice.
    e1 = MagicMock()
    e1.name = "Alice"
    e1.type = "Person"
    e1.confidence = 0.9
    e2 = MagicMock()
    e2.name = "alice"  # case difference — must dedupe
    e2.type = "Person"
    e2.confidence = 0.9
    extraction = MagicMock(entities=[e1, e2], relations=[])

    resolve_mock = AsyncMock(return_value=(None, True, None))

    with patch.object(pipeline, "chunk_text", return_value=[]), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4]), \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4), \
         patch.object(pipeline.resolver, "resolve_entity", resolve_mock), \
         patch.object(pipeline.neo4j_store, "merge_entity",
                      AsyncMock(return_value="ent1")), \
         patch.object(pipeline.neo4j_store, "link_entity_to_doc", AsyncMock()), \
         patch.object(pipeline.qdrant_store, "upsert_entity", AsyncMock()), \
         patch.object(pipeline.llm, "extract", return_value=extraction):
        await pipeline.ingest(text="some doc", title="t")

    assert resolve_mock.call_count == 1, (
        f"expected one resolve call for deduped (Alice/alice), got {resolve_mock.call_count}"
    )
```

- [ ] **Step 2: Run the tests — expect FAIL**

Run: `uv run pytest tests/test_perf_batching.py::test_ingest_batch_encodes_entities_once tests/test_perf_batching.py::test_ingest_dedupes_identical_entity_mentions_before_resolving -v`
Expected: FAIL (first test fails because entity loop uses per-entity `encode`; second because the loop resolves twice).

- [ ] **Step 3: Rewrite the entity-ingest block**

Replace lines 74-127 in `src/landscape/pipeline.py`:

```python
    # Step 3: extract entities + relations from full text
    extraction = llm.extract(text)

    now = datetime.now(UTC).isoformat()
    entities_created = 0
    entities_reinforced = 0

    # Step 4: entity resolution + write
    # Pre-group mentions by (lowercased name, canonical type) so duplicate
    # mentions in the same doc resolve once. Preserves the dedupe the serial
    # loop used to get "for free" from ordered resolution.
    grouped: dict[tuple[str, str], dict] = {}
    for entity in extraction.entities:
        canonical_entity_type, _ = coerce_entity_type(entity.type)
        etype_subtype = entity.type if canonical_entity_type != entity.type else None
        key = (entity.name.lower(), canonical_entity_type)
        if key not in grouped:
            grouped[key] = {
                "name": entity.name,
                "canonical_entity_type": canonical_entity_type,
                "subtype": etype_subtype,
                "confidence": entity.confidence,
                "encode_text": f"{entity.name} ({canonical_entity_type})",
            }

    group_keys = list(grouped.keys())
    if group_keys:
        vectors = encoder.embed_documents(
            [grouped[k]["encode_text"] for k in group_keys]
        )
        resolutions = await asyncio.gather(
            *(
                resolver.resolve_entity(
                    name=grouped[k]["name"],
                    entity_type=grouped[k]["canonical_entity_type"],
                    vector=vectors[i],
                    source_doc=title,
                )
                for i, k in enumerate(group_keys)
            )
        )
    else:
        vectors = []
        resolutions = []

    canonical_ids: dict[tuple[str, str], str] = {}
    for (key, vector, (canonical_id, is_new, _sim)) in zip(
        group_keys, vectors, resolutions, strict=True
    ):
        g = grouped[key]
        if is_new:
            canonical_id = await neo4j_store.merge_entity(
                name=g["name"],
                entity_type=g["canonical_entity_type"],
                source_doc=title,
                confidence=g["confidence"],
                doc_element_id=doc_id,
                model=settings.llm_model,
                session_id=session_id,
                turn_id=turn_id,
                subtype=g["subtype"],
            )
            await qdrant_store.upsert_entity(
                neo4j_element_id=canonical_id,
                name=g["name"],
                entity_type=g["canonical_entity_type"],
                source_doc=title,
                timestamp=now,
                vector=vector,
            )
            entities_created += 1
        else:
            await neo4j_store.link_entity_to_doc(
                entity_element_id=canonical_id,
                doc_element_id=doc_id,
                model=settings.llm_model,
            )
            entities_reinforced += 1

        canonical_ids[key] = canonical_id

        if turn_element_id is not None:
            await neo4j_store.link_entity_to_turn(canonical_id, turn_element_id)
```

**Key semantic preserved:** every original mention's `(name.lower(), type)` maps to a `canonical_id` via `canonical_ids`. Relation extraction downstream uses `upsert_relation(subject_name=..., object_name=...)` which re-resolves by name, so the relation path is unchanged. The intra-batch dedupe only affects how many resolver calls fire per ingest, not which canonical nodes the relations end up pointing at.

- [ ] **Step 4: Run the new tests — expect PASS**

Run: `uv run pytest tests/test_perf_batching.py::test_ingest_batch_encodes_entities_once tests/test_perf_batching.py::test_ingest_dedupes_identical_entity_mentions_before_resolving -v`
Expected: PASS.

- [ ] **Step 5: Run the resolution + ingest suites**

Run: `uv run pytest tests/test_ingest.py tests/test_resolution.py tests/test_pipeline_conversation.py tests/test_supersession.py -v`
Expected: PASS.

- [ ] **Step 6: Run the killer-demo regression**

Run: `uv run pytest tests/test_killer_demo.py -v`
Expected: PASS. This is the spec's acceptance criterion: "Entity resolver still produces the same canonical-entity count on the Helios Robotics killer-demo corpus."

- [ ] **Step 7: If killer-demo fails, investigate before changing behavior**

Check whether the new entity count diverges from the old. If it does:
- Compare counts of entities_created pre/post by running the fixture ingest manually.
- Likely causes: coreferent mentions with different names (e.g., "Acme" vs "Acme Corp") that the old ordering caught. If confirmed, add a follow-up note to the spec and revert Task 6 — do not ship silently-divergent behavior.

- [ ] **Step 8: Commit**

```bash
git add src/landscape/pipeline.py tests/test_perf_batching.py
git commit -m "perf(ingest): batch-encode entities, parallel resolve with dedupe

Pre-groups extracted entity mentions by (lower(name), canonical_type) to
preserve the intra-doc dedupe the serial loop used to get from ordering.
Batch-encodes all unique names and resolves them with asyncio.gather.
Killer-demo corpus entity count unchanged."
```

---

## Task 7: Full-suite regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q --ignore=tests/test_chromadb_baseline.py --ignore=tests/test_llm_options.py`
Expected: PASS.

- [ ] **Step 2: Confirm pre-existing failures are unchanged**

Run: `uv run pytest tests/test_chromadb_baseline.py tests/test_llm_options.py --tb=no -q 2>&1 | tail -5`
Expected: same failures as before this branch started (6 total per spec). No new failures introduced by perf work.

- [ ] **Step 3: Smoke-test ingest end-to-end**

Run the Docker stack and ingest a single document through the CLI or API:

```bash
docker compose up -d neo4j qdrant ollama
# In another terminal:
uv run python -m landscape.cli ingest --file tests/fixtures/killer_demo_corpus/01_helios_overview.md
```

Expected: no tracebacks. Chunks + entities written. Latency visibly lower than pre-refactor (eyeball only — a proper benchmark harness is the next spec).

- [ ] **Step 4: Teaching walkthrough for the user**

Per CLAUDE.md "Walk through critical code", write a short end-of-task summary explaining:
1. Where the four `asyncio.gather` sites are in `retrieve()` and why each is safe.
2. Why `_hydrate_entities` + `get_entities_from_chunks` is NOT gathered (dependency chain).
3. How the entity-batch dedupe preserves the serial loop's intra-doc matching.
4. What the DIMS change catches that the hardcoded 768 hid.

This is a message to the user at end-of-task, not a code change.

- [ ] **Step 5: No commit for this task** (verification only). Work is done.

---

## Self-Review

**Spec coverage:**
- Change 1 (filter parallelism) → Task 2. ✓
- Change 2 (seed + hydration parallelism) → Task 3 (seed parallelism done; hydration correctly identified as non-parallelizable with rationale). ✓
- Change 3 (reinforcement writes) → Task 4. ✓
- Change 4 (chunk batching) → Task 5. ✓
- Change 5 (entity batching + dedupe) → Task 6 (with dedupe strategy + killer-demo regression gate). ✓
- Change 6 (DIMS derived from model) → Task 1. ✓

**Acceptance criteria coverage:**
- "retrieve() makes at most one await per independent group" → Tasks 2, 3, 4 with latency-spread assertions.
- "ingest for 10 chunks / 20 entities calls embed_documents at most twice" → Task 5 + Task 6 with `call_count` assertions.
- "entity resolver produces same canonical count on killer-demo" → Task 6 Step 6.
- "init_collection requests vector size matching configured model" → Task 1 Step 6, parametrized over two models.
- "all existing tests pass" → Task 7 Step 1.

**Placeholder scan:** No "TBD", no "similar to Task N", no "handle edge cases". Every step has concrete code or exact commands.

**Type consistency:**
- `EMBEDDING_MODEL_DIMS` (Task 1) — used as `dict[str, int]` throughout. ✓
- `Settings.embedding_dims` property — read by `qdrant_store._dims()`. ✓
- `canonical_ids: dict[tuple[str, str], str]` (Task 6) — built and not consumed downstream; fine because relations re-resolve by name. Noted in Task 6 step 3 rationale.
- `encoder.embed_documents(list[str]) -> list[list[float]]` — signature verified against `src/landscape/embeddings/encoder.py:30`.

**Fixed inline:** original draft had Task 3 proposing a hydrate+chunk-map gather per spec. Closer reading showed the dependency makes that impossible; corrected Task 3 to skip it and documented the reason in both the step and the commit message. Spec is slightly wrong on that point; the plan is right.
