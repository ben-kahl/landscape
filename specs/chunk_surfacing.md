# Chunk Surfacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface raw Qdrant chunk hits alongside entity results in hybrid retrieval, and stop discarding chunk-seed similarity scores when they feed graph expansion (Approach A3 from brainstorm).

**Architecture:** Two orthogonal fixes in `retrieval/query.py`:
1. **Bug fix:** when a chunk hit produces canonical entity seeds via `EXTRACTED_FROM`, propagate the chunk's Qdrant similarity score into `seed_sims` (currently set to `0.0`, throwing away real signal).
2. **New output surface:** retain the top-k chunk hits (text, provenance, score) as a parallel list on `RetrievalResult`. Apply the existing `session_id` / `since` filters to chunks using new Neo4j helpers that mirror the entity filter pattern via `Chunk -[:PART_OF]-> Document -[:INGESTED_IN]-> Turn`. Return chunks through the FastAPI `/query` and MCP `search` tool.

**Tech Stack:** FastAPI, Qdrant (chunks collection), Neo4j (filter expansion), Python dataclasses, pytest.

---

## File Structure

**Modify:**
- `src/landscape/retrieval/query.py` — fix seed-sim propagation, add chunk collection + filter integration, extend `RetrievalResult`.
- `src/landscape/storage/neo4j_store.py` — add `get_chunks_in_conversation` and `get_chunks_since` helpers.
- `src/landscape/api/query.py` — add `chunk_limit` request field, `chunks` on response.
- `src/landscape/mcp_server.py` — extend `search` tool payload with `chunks`.

**Create:**
- `tests/test_chunk_surfacing.py` — covers seed-sim propagation, chunk return path, filter application.

---

## Task 1: Fix chunk-seed similarity propagation

Currently `query.py:72-73` drops the chunk Qdrant score by calling `seed_sims.setdefault(eid, 0.0)` on chunk-derived entities. This means entities that were only seeded via a chunk hit enter graph expansion with zero inherited similarity, neutering the whole chunk-seed path. Fix it to propagate the chunk's score.

**Files:**
- Modify: `src/landscape/retrieval/query.py:64-73`
- Test: `tests/test_chunk_surfacing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunk_surfacing.py` with:

```python
"""Chunk surfacing and chunk-seed similarity propagation tests."""
from unittest.mock import AsyncMock, patch

import pytest

BASIC_DOC = (
    "Diego Ortega is a senior engineer on the Vision Team. "
    "The Vision Team is based in Austin, Texas. "
    "Diego works on the Sentinel project."
)
TITLE = "chunk-surfacing-test"


async def _clear(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (c:Chunk)-[:PART_OF]->(d:Document {title: $t}) DETACH DELETE c",
            t=title,
        )
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) DETACH DELETE e",
            t=title,
        )
        await session.run("MATCH (d:Document {title: $t}) DETACH DELETE d", t=title)


@pytest.mark.asyncio
async def test_chunk_seed_propagates_similarity(http_client, neo4j_driver):
    """A chunk-only seed (no direct entity match) should enter graph expansion
    with nonzero vector_sim inherited from the chunk hit."""
    await _clear(neo4j_driver, TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "where is Diego located", "hops": 2, "limit": 10},
    )
    assert q.status_code == 200
    body = q.json()
    # Some result must have vector_sim > 0; if the chunk-seed bug is present
    # and no entity-name matches the query tokens, all sims come back 0.
    assert any(r["vector_sim"] > 0.0 for r in body["results"]), (
        "chunk-seeded entities should inherit chunk similarity, not 0.0"
    )
```

- [ ] **Step 2: Run test to verify it fails (or establish baseline)**

Run: `pytest tests/test_chunk_surfacing.py::test_chunk_seed_propagates_similarity -v`

Expected: may pass if the query happens to match an entity name directly (e.g. "Diego"). To force it to exercise the chunk-seed path, assert stricter — add the chunk text assertion in Task 3. For now this serves as a regression guard.

- [ ] **Step 3: Fix the seed-sim propagation**

In `src/landscape/retrieval/query.py`, replace lines 64-73:

```python
    # 2. Chunk seeds → walk back to canonical entities via EXTRACTED_FROM.
    chunk_hits = await qdrant_store.search_chunks(query_vector, limit=5)
    chunk_ids = [
        h.payload["chunk_neo4j_id"]
        for h in chunk_hits
        if h.payload and h.payload.get("chunk_neo4j_id")
    ]
    chunk_entities = await neo4j_store.get_entities_from_chunks(chunk_ids)
    for ent in chunk_entities:
        seed_sims.setdefault(ent["eid"], 0.0)
```

With:

```python
    # 2. Chunk seeds → walk back to canonical entities via EXTRACTED_FROM.
    #    Propagate the chunk's similarity score so entities reached only via
    #    chunk seeding still enter expansion with real similarity signal.
    chunk_hits = await qdrant_store.search_chunks(query_vector, limit=5)
    chunk_ids: list[str] = []
    chunk_score_by_id: dict[str, float] = {}
    for hit in chunk_hits:
        payload = hit.payload or {}
        cid = payload.get("chunk_neo4j_id")
        if not cid:
            continue
        chunk_ids.append(cid)
        chunk_score_by_id[cid] = float(hit.score)

    chunk_entities = await neo4j_store.get_entities_from_chunks(chunk_ids)
    for ent in chunk_entities:
        eid = ent["eid"]
        # A chunk-derived entity inherits the max similarity across any
        # chunks it was extracted from (chunk_eids optional — graceful
        # fallback if the helper doesn't return them yet).
        src_chunk_ids = ent.get("chunk_eids") or chunk_ids
        best = max(
            (chunk_score_by_id.get(cid, 0.0) for cid in src_chunk_ids),
            default=0.0,
        )
        seed_sims[eid] = max(seed_sims.get(eid, 0.0), best)
```

- [ ] **Step 4: Update `get_entities_from_chunks` to return the chunk elementIds it traversed**

In `src/landscape/storage/neo4j_store.py`, replace the `get_entities_from_chunks` body (lines 703-723):

```python
async def get_entities_from_chunks(chunk_element_ids: list[str]) -> list[dict[str, Any]]:
    """For a set of :Chunk elementIds, return the canonical :Entity nodes
    extracted from the parent :Document of each chunk, together with the
    list of seed chunk elementIds each entity was reached via (so the caller
    can propagate chunk-hit similarity scores to the right entities)."""
    if not chunk_element_ids:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)<-[:EXTRACTED_FROM]-(e:Entity)
            WHERE elementId(c) IN $chunk_ids AND e.canonical = true
            WITH e, collect(DISTINCT elementId(c)) AS chunk_eids
            RETURN
                elementId(e) AS eid,
                e.name AS name,
                e.type AS type,
                coalesce(e.access_count, 0) AS access_count,
                e.last_accessed AS last_accessed,
                chunk_eids
            """,
            chunk_ids=chunk_element_ids,
        )
        return [dict(record) async for record in result]
```

- [ ] **Step 5: Run the full retrieval test suite to verify no regression**

Run: `pytest tests/test_retrieval_basic.py tests/test_chunk_surfacing.py -v`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/landscape/retrieval/query.py src/landscape/storage/neo4j_store.py tests/test_chunk_surfacing.py
git commit -m "fix(retrieval): propagate chunk-hit similarity into seed_sims

Chunk-derived entity seeds were being inserted with vector_sim=0.0,
which neutered graph expansion from chunks-only seeds. Now each
chunk-seeded entity inherits the max Qdrant score across the chunks
it was extracted from."
```

---

## Task 2: Add `RetrievedChunk` dataclass and extend `RetrievalResult`

Add the output surface that will carry chunk text alongside entity results.

**Files:**
- Modify: `src/landscape/retrieval/query.py:14-34`

- [ ] **Step 1: Add the dataclass and extend `RetrievalResult`**

In `src/landscape/retrieval/query.py`, add after the `RetrievedEntity` dataclass:

```python
@dataclass
class RetrievedChunk:
    chunk_neo4j_id: str
    text: str
    doc_id: str
    source_doc: str
    position: int
    score: float
```

And extend `RetrievalResult` with a `chunks` field:

```python
@dataclass
class RetrievalResult:
    query: str
    results: list[RetrievedEntity]
    touched_entity_ids: list[str]
    touched_edge_ids: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
```

- [ ] **Step 2: Run existing tests to confirm the new optional field breaks nothing**

Run: `pytest tests/test_retrieval_basic.py tests/test_langchain_retriever.py -v`

Expected: pass. Existing call sites construct `RetrievalResult` positionally; the new field has a default so existing code is unaffected.

- [ ] **Step 3: Commit**

```bash
git add src/landscape/retrieval/query.py
git commit -m "feat(retrieval): add RetrievedChunk dataclass + chunks field

Prepares RetrievalResult to carry raw chunk text hits alongside
ranked entities. No wiring yet — empty list by default."
```

---

## Task 3: Populate `chunks` in `retrieve()` + add `chunk_limit` param

Wire the chunk hits into the result so callers see chunk text.

**Files:**
- Modify: `src/landscape/retrieval/query.py:37-46` (signature), chunk-seed block (updated in Task 1)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunk_surfacing.py`:

```python
@pytest.mark.asyncio
async def test_query_returns_chunk_text(http_client, neo4j_driver):
    """A hybrid query should return a chunks array with raw text and score."""
    await _clear(neo4j_driver, TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={
            "text": "where is Diego located",
            "hops": 2,
            "limit": 10,
            "chunk_limit": 3,
        },
    )
    assert q.status_code == 200
    body = q.json()
    chunks = body.get("chunks")
    assert chunks is not None, "response should include a chunks field"
    assert len(chunks) >= 1
    assert len(chunks) <= 3
    # At least one surfaced chunk should mention "Austin" — that's the fact
    # the graph-only path was missing in the Diego scenario.
    assert any("Austin" in c["text"] for c in chunks), (
        f"expected Austin-mentioning chunk in results, got: {[c['text'] for c in chunks]}"
    )
    # Each chunk should carry provenance and score.
    for c in chunks:
        assert "score" in c and c["score"] > 0
        assert "source_doc" in c
        assert "position" in c
```

- [ ] **Step 2: Run — expect it to fail**

Run: `pytest tests/test_chunk_surfacing.py::test_query_returns_chunk_text -v`

Expected: FAIL — `chunks` not in response body.

- [ ] **Step 3: Add `chunk_limit` to `retrieve()` and populate chunks**

In `src/landscape/retrieval/query.py`, update the `retrieve` signature:

```python
async def retrieve(
    query_text: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    weights: ScoringWeights | None = None,
    reinforce: bool = True,
    session_id: str | None = None,
    since: datetime | None = None,
) -> RetrievalResult:
```

Inside the chunk-seed block (the one edited in Task 1), after `chunk_hits` is fetched, build the `retrieved_chunks` list:

```python
    retrieved_chunks: list[RetrievedChunk] = []
    for hit in chunk_hits:
        payload = hit.payload or {}
        cid = payload.get("chunk_neo4j_id")
        if not cid:
            continue
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

At the bottom of `retrieve`, change the two `RetrievalResult(...)` constructions and the final return to include the chunks (trimmed to `chunk_limit` and filtered in Task 4):

```python
    # Early return when no seeds at all — still return any chunk text we have.
    if not seed_sims:
        return RetrievalResult(
            query=query_text,
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=retrieved_chunks[:chunk_limit],
        )
```

```python
    return RetrievalResult(
        query=query_text,
        results=ranked,
        touched_entity_ids=touched_entity_ids,
        touched_edge_ids=touched_edge_ids,
        chunks=retrieved_chunks[:chunk_limit],
    )
```

Also, the `session_id`/`since` empty-allowlist early return should carry chunks too — update it:

```python
        if not allowlist:
            return RetrievalResult(
                query=query_text,
                results=[],
                touched_entity_ids=[],
                touched_edge_ids=[],
                chunks=retrieved_chunks[:chunk_limit],
            )
```

- [ ] **Step 4: Expose `chunk_limit` and `chunks` through FastAPI**

In `src/landscape/api/query.py`:

```python
class QueryRequest(BaseModel):
    text: str
    hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    chunk_limit: int = Field(default=3, ge=0, le=20)
    reinforce: bool = True


class QueryChunkItem(BaseModel):
    chunk_neo4j_id: str
    text: str
    doc_id: str
    source_doc: str
    position: int
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResultItem]
    chunks: list[QueryChunkItem]
    touched_entity_count: int
    touched_edge_count: int
```

And update the endpoint:

```python
@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    result = await query_module.retrieve(
        query_text=req.text,
        hops=req.hops,
        limit=req.limit,
        chunk_limit=req.chunk_limit,
        reinforce=req.reinforce,
    )
    return QueryResponse(
        query=result.query,
        results=[
            QueryResultItem(
                neo4j_id=r.neo4j_id,
                name=r.name,
                type=r.type,
                distance=r.distance,
                vector_sim=r.vector_sim,
                reinforcement=r.reinforcement,
                edge_confidence=r.edge_confidence,
                score=r.score,
                path_edge_types=r.path_edge_types,
            )
            for r in result.results
        ],
        chunks=[
            QueryChunkItem(
                chunk_neo4j_id=c.chunk_neo4j_id,
                text=c.text,
                doc_id=c.doc_id,
                source_doc=c.source_doc,
                position=c.position,
                score=c.score,
            )
            for c in result.chunks
        ],
        touched_entity_count=len(result.touched_entity_ids),
        touched_edge_count=len(result.touched_edge_ids),
    )
```

- [ ] **Step 5: Re-run the test**

Run: `pytest tests/test_chunk_surfacing.py -v`

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/landscape/retrieval/query.py src/landscape/api/query.py tests/test_chunk_surfacing.py
git commit -m "feat(retrieval): return chunk text alongside entity results

Raw Qdrant chunk hits are now surfaced as a parallel list on the
/query response. Adds chunk_limit (default 3) to cap the payload.
Fixes the Diego-location class of failure where the fact lived in
chunk text but not in the extracted graph edges."
```

---

## Task 4: Apply `session_id` / `since` filters to chunks

Chunks must respect the same visibility filters as entities so scoped queries don't leak out-of-scope text.

**Files:**
- Modify: `src/landscape/storage/neo4j_store.py` (add two helpers)
- Modify: `src/landscape/retrieval/query.py` (apply filters)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunk_surfacing.py`:

```python
@pytest.mark.asyncio
async def test_chunk_filter_scopes_to_session(http_client, neo4j_driver):
    """If session_id is supplied, only chunks ingested within that session's
    turns should be returned."""
    await _clear(neo4j_driver, TITLE)
    await _clear(neo4j_driver, TITLE + "-other")

    # Ingest the in-scope doc under session sess-A / turn t-A-1.
    r1 = await http_client.post(
        "/ingest",
        json={
            "text": BASIC_DOC,
            "title": TITLE,
            "session_id": "sess-A",
            "turn_id": "t-A-1",
        },
    )
    assert r1.status_code == 200
    # Ingest an out-of-scope doc under a different session.
    r2 = await http_client.post(
        "/ingest",
        json={
            "text": "Marvin is located in Boston at the east coast office.",
            "title": TITLE + "-other",
            "session_id": "sess-B",
            "turn_id": "t-B-1",
        },
    )
    assert r2.status_code == 200

    q = await http_client.post(
        "/query",
        json={
            "text": "where is everyone located",
            "hops": 2,
            "limit": 10,
            "chunk_limit": 10,
            "session_id": "sess-A",
        },
    )
    assert q.status_code == 200
    body = q.json()
    # No chunk from the other session should appear.
    assert not any("Marvin" in c["text"] for c in body["chunks"])
    # And the in-scope chunk should still be present.
    assert any("Austin" in c["text"] for c in body["chunks"])
```

Also extend `QueryRequest` in `src/landscape/api/query.py` to accept session/since before running this test:

```python
class QueryRequest(BaseModel):
    text: str
    hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    chunk_limit: int = Field(default=3, ge=0, le=20)
    reinforce: bool = True
    session_id: str | None = None
    since_hours: int | None = Field(default=None, ge=1)
```

And pass them through in `query_endpoint`:

```python
    since = (
        datetime.now(UTC) - timedelta(hours=req.since_hours)
        if req.since_hours
        else None
    )
    result = await query_module.retrieve(
        query_text=req.text,
        hops=req.hops,
        limit=req.limit,
        chunk_limit=req.chunk_limit,
        reinforce=req.reinforce,
        session_id=req.session_id,
        since=since,
    )
```

with the needed imports at top of file:

```python
from datetime import UTC, datetime, timedelta
```

- [ ] **Step 2: Run — expect it to fail**

Run: `pytest tests/test_chunk_surfacing.py::test_chunk_filter_scopes_to_session -v`

Expected: FAIL — Marvin chunk appears because filters don't apply to chunks.

- [ ] **Step 3: Add Neo4j helpers to resolve chunk allowlists**

In `src/landscape/storage/neo4j_store.py`, after `get_entities_since` (line ~224), add:

```python
async def get_chunks_in_conversation(session_id: str) -> list[str]:
    """Return elementIds of :Chunk nodes belonging to Documents ingested in
    any Turn of the named Conversation. Empty list if session_id unknown."""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Conversation {id: $session_id})-[:HAS_TURN]->(t:Turn)
                  <-[:INGESTED_IN]-(d:Document)<-[:PART_OF]-(ch:Chunk)
            RETURN DISTINCT elementId(ch) AS cid
            """,
            session_id=session_id,
        )
        return [record["cid"] async for record in result]


async def get_chunks_since(since: datetime) -> list[str]:
    """Return elementIds of :Chunk nodes belonging to Documents ingested in
    Turns with t.timestamp >= since (ISO string compare)."""
    since_iso = since.isoformat()
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (ch:Chunk)-[:PART_OF]->(d:Document)-[:INGESTED_IN]->(t:Turn)
            WHERE t.timestamp >= $since_iso
            RETURN DISTINCT elementId(ch) AS cid
            """,
            since_iso=since_iso,
        )
        return [record["cid"] async for record in result]
```

- [ ] **Step 4: Apply the filters in `retrieve()`**

In `src/landscape/retrieval/query.py`, replace the session/since filter block (lines 178-198) with one that computes a chunk allowlist in parallel and narrows both lists:

```python
    # 5. Session/time allowlist filtering (post-search intersection per spec).
    #    Vector search runs against the full index; we narrow candidates after.
    if session_id is not None or since is not None:
        if session_id is not None and since is not None:
            conv_ids = set(await neo4j_store.get_entities_in_conversation(session_id))
            since_ids = set(await neo4j_store.get_entities_since(since))
            allowlist = conv_ids & since_ids
            conv_cids = set(await neo4j_store.get_chunks_in_conversation(session_id))
            since_cids = set(await neo4j_store.get_chunks_since(since))
            chunk_allowlist = conv_cids & since_cids
        elif session_id is not None:
            allowlist = set(await neo4j_store.get_entities_in_conversation(session_id))
            chunk_allowlist = set(
                await neo4j_store.get_chunks_in_conversation(session_id)
            )
        else:
            assert since is not None
            allowlist = set(await neo4j_store.get_entities_since(since))
            chunk_allowlist = set(await neo4j_store.get_chunks_since(since))

        retrieved_chunks = [
            c for c in retrieved_chunks if c.chunk_neo4j_id in chunk_allowlist
        ]

        if not allowlist:
            return RetrievalResult(
                query=query_text,
                results=[],
                touched_entity_ids=[],
                touched_edge_ids=[],
                chunks=retrieved_chunks[:chunk_limit],
            )

        candidates = {k: v for k, v in candidates.items() if k in allowlist}
```

- [ ] **Step 5: Run the filter test**

Run: `pytest tests/test_chunk_surfacing.py::test_chunk_filter_scopes_to_session -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/landscape/retrieval/query.py src/landscape/api/query.py src/landscape/storage/neo4j_store.py tests/test_chunk_surfacing.py
git commit -m "feat(retrieval): apply session_id/since filters to chunk results

Adds get_chunks_in_conversation / get_chunks_since helpers and wires
them into retrieve() so scoped queries don't leak out-of-scope chunk
text. Parity with entity-filter behaviour."
```

---

## Task 5: Surface chunks in the MCP `search` tool

MCP clients (Claude Code, Cursor) consume search via the MCP server, not the FastAPI layer. Extend the tool payload.

**Files:**
- Modify: `src/landscape/mcp_server.py:63-109`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp_server.py` (or the existing search-related test — check which is conventional). Minimal example:

```python
@pytest.mark.asyncio
async def test_search_tool_returns_chunks(mcp_client):
    """The MCP search tool should include a chunks array in its JSON payload."""
    # Assume ingest fixture has populated the store.
    payload = await mcp_client.call_tool(
        "search",
        {"query": "where is Diego located", "hops": 2, "limit": 10, "chunk_limit": 3},
    )
    data = json.loads(payload)
    assert "chunks" in data
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) <= 3
    for c in data["chunks"]:
        assert {"text", "source_doc", "score"}.issubset(c.keys())
```

(If `tests/test_mcp_server.py` uses a different fixture name, match that convention. Inspect the file first.)

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/test_mcp_server.py -v -k search_tool_returns_chunks`

Expected: FAIL — `chunks` not in payload.

- [ ] **Step 3: Extend the `search` tool signature and payload**

In `src/landscape/mcp_server.py`, change the tool:

```python
@mcp.tool()
async def search(
    query: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    session_id: str | None = None,
    since_hours: int | None = None,
) -> str:
    """Hybrid retrieval over the Landscape knowledge graph.

    Combines vector similarity (Qdrant) with graph BFS expansion (Neo4j) to
    surface entities relevant to *query*, and returns up to *chunk_limit*
    raw document chunks (with text) that matched the query vector —
    filters apply to both.

    Args:
        query:        Natural-language search query.
        hops:         BFS depth for graph expansion (1–3 recommended). Default 2.
        limit:        Maximum number of entity results. Default 10.
        chunk_limit:  Maximum number of raw chunks to include. Default 3.
        session_id:   If supplied, scope retrieval (entities + chunks) to
                      this conversation's turns.
        since_hours:  If supplied (int > 0), exclude facts and chunks older
                      than this many hours.

    Returns:
        JSON object ``{results: [...], chunks: [{text, source_doc, score,
        position, doc_id}], touched_entity_count}``.
    """
    await _ensure_init()
    from landscape.retrieval.query import retrieve

    since = (
        datetime.now(UTC) - timedelta(hours=since_hours)
        if since_hours and since_hours > 0
        else None
    )
    result = await retrieve(
        query,
        hops=hops,
        limit=limit,
        chunk_limit=chunk_limit,
        session_id=session_id,
        since=since,
    )
    output = {
        "results": [
            {
                "name": r.name,
                "type": r.type,
                "score": round(r.score, 6),
                "path_edge_types": r.path_edge_types,
            }
            for r in result.results
        ],
        "chunks": [
            {
                "text": c.text,
                "source_doc": c.source_doc,
                "doc_id": c.doc_id,
                "position": c.position,
                "score": round(c.score, 6),
            }
            for c in result.chunks
        ],
        "touched_entity_count": len(result.touched_entity_ids),
    }
    return json.dumps(output)
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_mcp_server.py -v -k search_tool_returns_chunks`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landscape/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): include chunks in search tool payload

MCP clients now receive raw chunk text alongside the ranked entity
list, matching the FastAPI /query response shape."
```

---

## Task 6: Final regression sweep

Confirm nothing upstream broke.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`

Expected: all pass. If anything unrelated breaks, investigate before declaring done.

- [ ] **Step 2: Manual smoke against the Diego scenario**

With the stack running (`docker compose up -d`), ingest `tests/fixtures/killer_demo_corpus/` (or the specific Vision Team doc that mentions Austin) and run the query that previously failed:

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"text":"where is Diego located","hops":2,"limit":5,"chunk_limit":3}' | jq '.chunks[].text'
```

Expected: at least one chunk mentioning Austin / Vision Team appears.

- [ ] **Step 3: Commit any doc or smoke-script updates**

If a new smoke script was added:

```bash
git add scripts/smoke_chunk_surfacing.sh
git commit -m "chore: add smoke script for chunk-surfacing scenario"
```

---

## Self-review notes

- All task steps reference exact paths and line anchors from the current state of `query.py`, `neo4j_store.py`, and `api/query.py`.
- The `RetrievedChunk` fields match the payload shape written by `qdrant_store.upsert_chunk` (`chunk_neo4j_id`, `doc_id`, `source_doc`, `position`, `text`).
- `get_entities_from_chunks` gains a `chunk_eids` column in Task 1, and Task 1's propagation logic consumes it with a fallback — ordering is safe if the helper is updated first (Step 4 before the seed-sim block would be fine; written this way for narrative clarity).
- Filter parity: the new `get_chunks_in_conversation` / `get_chunks_since` mirror `get_entities_in_conversation` / `get_entities_since` verbatim in structure.
- The MCP tool change (Task 5) adds `chunk_limit` without changing existing params' defaults, so existing agent callers keep working.
