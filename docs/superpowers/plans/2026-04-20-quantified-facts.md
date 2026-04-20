# Quantified Facts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve numeric facts as structured qualifiers on Landscape relationship edges and surface them through retrieval.

**Architecture:** Extend `ExtractedRelation` with optional quantity qualifier fields, write them onto `:RELATES_TO` edges, return them from graph expansion, and expose them through HTTP and LangChain result formatting. Keep the graph shape unchanged; reified fact nodes and numeric comparison query logic are deferred.

**Tech Stack:** Python 3.12+, Pydantic v2, Neo4j AsyncDriver, FastAPI/Pydantic response models, LangChain `Document`, pytest.

---

## Baseline Status

Worktree: `/home/kahlb/Documents/landscape/.worktrees/quantified-facts`

Baseline command:

```bash
uv run --extra dev pytest
```

Baseline result before feature implementation:

- 187 passed
- 1 xfailed
- 6 failed
- Duration: 1049.38s (17:29)

Pre-existing failures before this feature work:

- `tests/test_chromadb_baseline.py::test_chromadb_importable`: `ModuleNotFoundError: No module named 'chromadb'`
- `tests/test_chromadb_baseline.py::test_embed_chunks_dim`: `SystemExit: 1` from missing `chromadb`
- `tests/test_chromadb_baseline.py::test_in_memory_collection_ranking`: `ModuleNotFoundError: No module named 'chromadb'`
- `tests/test_chromadb_baseline.py::test_bench_harness_two_doc_subset`: `SystemExit: 1` from missing `chromadb`
- `tests/test_llm_options.py::test_extract_passes_think_false_for_no_think_profile`: expected `think is False`, got `None`
- `tests/test_llm_options.py::test_extract_passes_think_true_for_thinking_profile`: expected `think is True`, got `None`

The full suite is slow because extraction/integration tests run without GPU acceleration in this environment. Before claiming completion, rerun focused tests from this plan and then either rerun the full suite or clearly report these pre-existing baseline failures separately from feature-introduced failures.

## File Structure

- Modify `src/landscape/extraction/schema.py`: add quantity fields and normalization helper.
- Modify `src/landscape/extraction/llm.py`: teach the extractor to emit quantity fields.
- Modify `src/landscape/pipeline.py`: pass relation quantity fields into storage.
- Modify `src/landscape/storage/neo4j_store.py`: store, reinforce, and return quantity fields.
- Modify `src/landscape/retrieval/query.py`: add path-edge quantity payloads to `RetrievedEntity`.
- Modify `src/landscape/api/query.py`: expose path-edge subtypes and quantities in HTTP responses.
- Modify `src/landscape/retrieval/langchain_retriever.py`: render quantity qualifiers and include chunk documents.
- Modify `src/landscape/mcp_server.py`: include edge qualifiers in search JSON.
- Add or modify tests in `tests/test_supersession.py`, `tests/test_retrieval_basic.py`, `tests/test_langchain_retriever.py`, and `tests/test_ingest.py`.

## Task 1: Schema Contract for Quantified Relations

**Files:**
- Modify: `src/landscape/extraction/schema.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing schema test**

Add this test to `tests/test_ingest.py`:

```python
from landscape.extraction.schema import Extraction


def test_extraction_schema_accepts_quantified_relation_fields():
    extraction = Extraction.model_validate(
        {
            "entities": [
                {
                    "name": "Eric",
                    "type": "PERSON",
                    "confidence": 0.95,
                    "aliases": [],
                },
                {
                    "name": "Netflix",
                    "type": "TECHNOLOGY",
                    "confidence": 0.9,
                    "aliases": [],
                },
            ],
            "relations": [
                {
                    "subject": "Eric",
                    "object": "Netflix",
                    "relation_type": "DISCUSSED",
                    "subtype": "watched",
                    "confidence": 0.9,
                    "quantity_value": 8,
                    "quantity_unit": "hours",
                    "quantity_kind": "duration",
                    "time_scope": "today",
                }
            ],
        }
    )

    relation = extraction.relations[0]
    assert relation.quantity_value == 8
    assert relation.quantity_unit == "hours"
    assert relation.quantity_kind == "duration"
    assert relation.time_scope == "today"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_ingest.py::test_extraction_schema_accepts_quantified_relation_fields -q
```

Expected: FAIL because `ExtractedRelation` does not expose the new fields.

- [ ] **Step 3: Add quantity fields to `ExtractedRelation`**

Modify `src/landscape/extraction/schema.py`:

```python
class ExtractedRelation(BaseModel):
    subject: str
    object: str
    relation_type: str
    confidence: float
    subtype: str | None = None
    quantity_value: float | str | None = None
    quantity_unit: str | None = None
    quantity_kind: str | None = None
    time_scope: str | None = None
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
uv run --extra dev pytest tests/test_ingest.py::test_extraction_schema_accepts_quantified_relation_fields -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/landscape/extraction/schema.py tests/test_ingest.py
git commit -m "feat: add quantified relation schema fields"
```

## Task 2: Persist Quantity Fields on Relationship Edges

**Files:**
- Modify: `src/landscape/storage/neo4j_store.py`
- Test: `tests/test_supersession.py`

- [ ] **Step 1: Write failing storage tests**

Add these tests to `tests/test_supersession.py`:

```python
@pytest.mark.asyncio
async def test_relation_quantity_fields_written_on_create(http_client, neo4j_driver):
    title = "quantity-edge-create"
    await _clear_doc(neo4j_driver, title)

    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-quantity-create", title, "text")
    await neo4j_store.merge_entity("Eric", "PERSON", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Netflix", "TECHNOLOGY", title, 0.9, doc_id, "test")
    await _clear_relation(neo4j_driver, "Eric", "DISCUSSED")

    outcome, _ = await neo4j_store.upsert_relation(
        "Eric",
        "Netflix",
        "DISCUSSED",
        0.9,
        title,
        subtype="watched",
        quantity_value=8,
        quantity_unit="hour",
        quantity_kind="duration",
        time_scope="today",
    )

    assert outcome == "created"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (:Entity {name: 'Eric'})-[r:RELATES_TO {type: 'DISCUSSED'}]->
                  (:Entity {name: 'Netflix'})
            WHERE r.valid_until IS NULL
            RETURN r.quantity_value AS quantity_value,
                   r.quantity_unit AS quantity_unit,
                   r.quantity_kind AS quantity_kind,
                   r.time_scope AS time_scope
            """
        )
        record = await result.single()

    assert record["quantity_value"] == 8
    assert record["quantity_unit"] == "hour"
    assert record["quantity_kind"] == "duration"
    assert record["time_scope"] == "today"


@pytest.mark.asyncio
async def test_relation_quantity_fields_reinforce_with_non_null_wins(http_client, neo4j_driver):
    title1 = "quantity-edge-reinforce-1"
    title2 = "quantity-edge-reinforce-2"
    for title in (title1, title2):
        await _clear_doc(neo4j_driver, title)

    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-quantity-reinforce", title1, "text")
    await neo4j_store.merge_entity("Eric", "PERSON", title1, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Netflix", "TECHNOLOGY", title1, 0.9, doc_id, "test")
    await _clear_relation(neo4j_driver, "Eric", "DISCUSSED")

    created, _ = await neo4j_store.upsert_relation(
        "Eric",
        "Netflix",
        "DISCUSSED",
        0.8,
        title1,
        subtype="watched",
        quantity_value=None,
        quantity_unit=None,
        quantity_kind=None,
        time_scope=None,
    )
    reinforced, _ = await neo4j_store.upsert_relation(
        "Eric",
        "Netflix",
        "DISCUSSED",
        0.9,
        title2,
        subtype=None,
        quantity_value=10,
        quantity_unit="hour",
        quantity_kind="duration",
        time_scope="last_month",
    )

    assert created == "created"
    assert reinforced == "reinforced"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (:Entity {name: 'Eric'})-[r:RELATES_TO {type: 'DISCUSSED'}]->
                  (:Entity {name: 'Netflix'})
            WHERE r.valid_until IS NULL
            RETURN r.subtype AS subtype,
                   r.quantity_value AS quantity_value,
                   r.quantity_unit AS quantity_unit,
                   r.quantity_kind AS quantity_kind,
                   r.time_scope AS time_scope
            """
        )
        record = await result.single()

    assert record["subtype"] == "watched"
    assert record["quantity_value"] == 10
    assert record["quantity_unit"] == "hour"
    assert record["quantity_kind"] == "duration"
    assert record["time_scope"] == "last_month"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_supersession.py::test_relation_quantity_fields_written_on_create tests/test_supersession.py::test_relation_quantity_fields_reinforce_with_non_null_wins -q
```

Expected: FAIL because `upsert_relation()` does not accept quantity keyword arguments.

- [ ] **Step 3: Add storage parameters and create-edge properties**

Modify the `upsert_relation` signature in `src/landscape/storage/neo4j_store.py`:

```python
async def upsert_relation(
    subject_name: str,
    object_name: str,
    relation_type: str,
    confidence: float,
    source_doc: str,
    created_by: str = "ingest",
    session_id: str | None = None,
    turn_id: str | None = None,
    subtype: str | None = None,
    quantity_value: float | str | None = None,
    quantity_unit: str | None = None,
    quantity_kind: str | None = None,
    time_scope: str | None = None,
) -> tuple[str, str | None]:
```

In both `CREATE (s)-[r:RELATES_TO { ... }]` blocks, add:

```cypher
quantity_value: $quantity_value,
quantity_unit: $quantity_unit,
quantity_kind: $quantity_kind,
time_scope: $time_scope,
```

Pass the parameters to both `session.run(...)` calls:

```python
quantity_value=quantity_value,
quantity_unit=quantity_unit,
quantity_kind=quantity_kind,
time_scope=time_scope,
```

- [ ] **Step 4: Add non-null-wins reinforcement updates**

In the Case 1 exact-match query, return the existing quantity fields:

```cypher
r.quantity_value AS quantity_value,
r.quantity_unit AS quantity_unit,
r.quantity_kind AS quantity_kind,
r.time_scope AS time_scope
```

Before the reinforcement update call, compute:

```python
new_quantity_value = quantity_value if quantity_value is not None else exact["quantity_value"]
new_quantity_unit = quantity_unit if quantity_unit is not None else exact["quantity_unit"]
new_quantity_kind = quantity_kind if quantity_kind is not None else exact["quantity_kind"]
new_time_scope = time_scope if time_scope is not None else exact["time_scope"]
```

Extend the `SET` clause:

```cypher
SET r.source_docs = $source_docs,
    r.confidence = $conf,
    r.subtype = $subtype,
    r.quantity_value = $quantity_value,
    r.quantity_unit = $quantity_unit,
    r.quantity_kind = $quantity_kind,
    r.time_scope = $time_scope
```

Pass:

```python
quantity_value=new_quantity_value,
quantity_unit=new_quantity_unit,
quantity_kind=new_quantity_kind,
time_scope=new_time_scope,
```

- [ ] **Step 5: Run focused storage tests**

Run:

```bash
uv run --extra dev pytest tests/test_supersession.py::test_relation_quantity_fields_written_on_create tests/test_supersession.py::test_relation_quantity_fields_reinforce_with_non_null_wins -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/landscape/storage/neo4j_store.py tests/test_supersession.py
git commit -m "feat: persist quantified relation fields"
```

## Task 3: Pass Quantity Fields Through the Ingestion Pipeline

**Files:**
- Modify: `src/landscape/pipeline.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write failing pipeline test**

Add this test to `tests/test_ingest.py`:

```python
@pytest.mark.asyncio
async def test_ingest_passes_relation_quantity_fields(monkeypatch):
    from landscape import pipeline
    from landscape.extraction.schema import ExtractedEntity, ExtractedRelation, Extraction

    captured_relation_kwargs = {}

    async def fake_merge_document(content_hash, title, source_type):
        return "doc-1", True

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return f"chunk-{chunk_index}"

    async def fake_upsert_chunk(**kwargs):
        return None

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        return f"{name}-id", True, 0.0

    async def fake_merge_entity(**kwargs):
        return f"{kwargs['name']}-id"

    async def fake_upsert_entity(**kwargs):
        return None

    async def fake_upsert_relation(**kwargs):
        captured_relation_kwargs.update(kwargs)
        return "created", "rel-1"

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", fake_upsert_chunk)
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "merge_entity", fake_merge_entity)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", fake_upsert_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "upsert_relation", fake_upsert_relation)
    monkeypatch.setattr(pipeline.encoder, "embed_documents", lambda texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(
        pipeline.llm,
        "extract",
        lambda text: Extraction(
            entities=[
                ExtractedEntity(name="Eric", type="PERSON", confidence=0.9),
                ExtractedEntity(name="Netflix", type="TECHNOLOGY", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(
                    subject="Eric",
                    object="Netflix",
                    relation_type="DISCUSSED",
                    subtype="watched",
                    confidence=0.9,
                    quantity_value=10,
                    quantity_unit="hour",
                    quantity_kind="duration",
                    time_scope="last_month",
                )
            ],
        ),
    )

    await pipeline.ingest("Eric watched Netflix.", "quantity-pipeline")

    assert captured_relation_kwargs["quantity_value"] == 10
    assert captured_relation_kwargs["quantity_unit"] == "hour"
    assert captured_relation_kwargs["quantity_kind"] == "duration"
    assert captured_relation_kwargs["time_scope"] == "last_month"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_ingest.py::test_ingest_passes_relation_quantity_fields -q
```

Expected: FAIL because `pipeline.ingest()` does not pass quantity fields to storage.

- [ ] **Step 3: Pass fields into `neo4j_store.upsert_relation`**

In `src/landscape/pipeline.py`, extend the call:

```python
outcome, _ = await neo4j_store.upsert_relation(
    subject_name=relation.subject,
    object_name=relation.object,
    relation_type=canonical_rel_type,
    confidence=relation.confidence,
    source_doc=title,
    session_id=session_id,
    turn_id=turn_id,
    subtype=canonical_subtype,
    quantity_value=relation.quantity_value,
    quantity_unit=relation.quantity_unit,
    quantity_kind=relation.quantity_kind,
    time_scope=relation.time_scope,
)
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
uv run --extra dev pytest tests/test_ingest.py::test_ingest_passes_relation_quantity_fields -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/landscape/pipeline.py tests/test_ingest.py
git commit -m "feat: pass quantified relations through ingest"
```

## Task 4: Return Quantity Fields from Graph Expansion

**Files:**
- Modify: `src/landscape/storage/neo4j_store.py`
- Modify: `src/landscape/retrieval/query.py`
- Test: `tests/test_retrieval_basic.py`

- [ ] **Step 1: Write failing retrieval test**

Add this test to `tests/test_retrieval_basic.py`:

```python
@pytest.mark.asyncio
async def test_retrieval_includes_path_edge_quantities(monkeypatch):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])

    class Hit:
        def __init__(self):
            self.score = 0.9
            self.payload = {"neo4j_node_id": "eric-id"}

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit()]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_bfs_expand(seed_ids, max_hops):
        return [
            {
                "seed_id": "eric-id",
                "target_id": "netflix-id",
                "target_name": "Netflix",
                "target_type": "TECHNOLOGY",
                "target_access_count": 0,
                "target_last_accessed": None,
                "distance": 1,
                "edge_ids": ["rel-1"],
                "edge_types": ["DISCUSSED"],
                "edge_subtypes": ["watched"],
                "edge_quantities": [
                    {
                        "quantity_value": 10,
                        "quantity_unit": "hour",
                        "quantity_kind": "duration",
                        "time_scope": "last_month",
                    }
                ],
                "edge_confidences": [0.9],
                "edge_access_counts": [0],
                "edge_last_accessed": [None],
            }
        ]

    async def fake_touch_entities(ids, now):
        return None

    async def fake_touch_relations(ids, now):
        return None

    monkeypatch.setattr(query.qdrant_store, "search_entities_any_type", fake_search_entities_any_type)
    monkeypatch.setattr(query.qdrant_store, "search_chunks", fake_search_chunks)
    async def fake_hydrate_entities(ids):
        return [
            {
                "eid": "eric-id",
                "name": "Eric",
                "type": "PERSON",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    monkeypatch.setattr(query.neo4j_store, "get_entities_from_chunks", fake_get_entities_from_chunks)
    monkeypatch.setattr(query, "_hydrate_entities", fake_hydrate_entities)
    monkeypatch.setattr(query.neo4j_store, "bfs_expand", fake_bfs_expand)
    monkeypatch.setattr(query.neo4j_store, "touch_entities", fake_touch_entities)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", fake_touch_relations)

    result = await query.retrieve("How many hours on Netflix?", reinforce=False)

    netflix = next(r for r in result.results if r.name == "Netflix")
    assert netflix.path_edge_quantities == [
        {
            "quantity_value": 10,
            "quantity_unit": "hour",
            "quantity_kind": "duration",
            "time_scope": "last_month",
        }
    ]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_retrieval_basic.py::test_retrieval_includes_path_edge_quantities -q
```

Expected: FAIL because `RetrievedEntity` does not have `path_edge_quantities`.

- [ ] **Step 3: Return quantity maps from `bfs_expand`**

In `src/landscape/storage/neo4j_store.py`, add this expression to the `RETURN` list in `bfs_expand`:

```cypher
[r IN rels | {
    quantity_value: r.quantity_value,
    quantity_unit: r.quantity_unit,
    quantity_kind: r.quantity_kind,
    time_scope: r.time_scope
}] AS edge_quantities,
```

- [ ] **Step 4: Add `path_edge_quantities` to retrieval dataclass and assignment**

In `src/landscape/retrieval/query.py`, add the field:

```python
path_edge_quantities: list[dict[str, object | None]] = field(default_factory=list)
```

When constructing expansion candidates, pass:

```python
path_edge_quantities=list(row.get("edge_quantities") or []),
```

- [ ] **Step 5: Run the focused test and verify it passes**

Run:

```bash
uv run --extra dev pytest tests/test_retrieval_basic.py::test_retrieval_includes_path_edge_quantities -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/landscape/storage/neo4j_store.py src/landscape/retrieval/query.py tests/test_retrieval_basic.py
git commit -m "feat: return quantified path edges"
```

## Task 5: Surface Quantities in HTTP and MCP Search

**Files:**
- Modify: `src/landscape/api/query.py`
- Modify: `src/landscape/mcp_server.py`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Add HTTP response fields**

Modify `QueryResultItem` in `src/landscape/api/query.py`:

```python
class QueryResultItem(BaseModel):
    neo4j_id: str
    name: str
    type: str
    distance: int
    vector_sim: float
    reinforcement: float
    edge_confidence: float
    score: float
    path_edge_types: list[str]
    path_edge_subtypes: list[str | None] = Field(default_factory=list)
    path_edge_quantities: list[dict[str, object | None]] = Field(default_factory=list)
```

In the response mapping, add:

```python
path_edge_subtypes=r.path_edge_subtypes,
path_edge_quantities=r.path_edge_quantities,
```

- [ ] **Step 2: Add MCP search fields**

In `src/landscape/mcp_server.py`, extend each search result object:

```python
{
    "name": r.name,
    "type": r.type,
    "score": round(r.score, 6),
    "path_edge_types": r.path_edge_types,
    "path_edge_subtypes": r.path_edge_subtypes,
    "path_edge_quantities": r.path_edge_quantities,
}
```

- [ ] **Step 3: Run existing API/MCP tests**

Run:

```bash
uv run --extra dev pytest tests/test_mcp_server.py tests/test_chunk_surfacing.py -q
```

Expected: PASS. Existing clients should tolerate additive JSON fields.

- [ ] **Step 4: Commit**

Run:

```bash
git add src/landscape/api/query.py src/landscape/mcp_server.py
git commit -m "feat: expose quantified path edges"
```

## Task 6: Render Quantities and Include Chunks in LangChain Retriever

**Files:**
- Modify: `src/landscape/retrieval/langchain_retriever.py`
- Test: `tests/test_langchain_retriever.py`

- [ ] **Step 1: Write failing LangChain formatting test**

Add this test to `tests/test_langchain_retriever.py`:

```python
def test_entity_document_renders_quantity_qualifiers():
    from landscape.retrieval.langchain_retriever import _entity_to_document
    from landscape.retrieval.query import RetrievedEntity

    doc = _entity_to_document(
        RetrievedEntity(
            neo4j_id="netflix-id",
            name="Netflix",
            type="TECHNOLOGY",
            distance=1,
            vector_sim=0.9,
            reinforcement=0.0,
            edge_confidence=0.9,
            score=1.0,
            path_edge_ids=["rel-1"],
            path_edge_types=["DISCUSSED"],
            path_edge_subtypes=["watched"],
            path_edge_quantities=[
                {
                    "quantity_value": 10,
                    "quantity_unit": "hour",
                    "quantity_kind": "duration",
                    "time_scope": "last_month",
                }
            ],
        )
    )

    assert "DISCUSSED[watched]" in doc.page_content
    assert "duration=10 hour" in doc.page_content
    assert "scope=last_month" in doc.page_content
    assert doc.metadata["path_edge_quantities"][0]["quantity_value"] == 10
```

- [ ] **Step 2: Write failing chunk surfacing test**

Add this async test to `tests/test_langchain_retriever.py`:

```python
@pytest.mark.asyncio
async def test_langchain_retriever_returns_chunk_documents(monkeypatch):
    from landscape.retrieval import langchain_retriever
    from landscape.retrieval.langchain_retriever import LandscapeRetriever
    from landscape.retrieval.query import RetrievalResult, RetrievedChunk

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            query=kwargs["query_text"],
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=[
                RetrievedChunk(
                    chunk_neo4j_id="chunk-1",
                    text="I spent 10 hours last month watching documentaries on Netflix.",
                    doc_id="doc-1",
                    source_doc="longmemeval:test",
                    position=0,
                    score=0.88,
                )
            ],
        )

    monkeypatch.setattr(langchain_retriever, "retrieve", fake_retrieve)

    retriever = LandscapeRetriever(chunk_limit=1)
    docs = await retriever.ainvoke("How many hours on Netflix?")

    assert len(docs) == 1
    assert docs[0].page_content == "I spent 10 hours last month watching documentaries on Netflix."
    assert docs[0].metadata["kind"] == "chunk"
    assert docs[0].metadata["chunk_neo4j_id"] == "chunk-1"
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run --extra dev pytest tests/test_langchain_retriever.py::test_entity_document_renders_quantity_qualifiers tests/test_langchain_retriever.py::test_langchain_retriever_returns_chunk_documents -q
```

Expected: FAIL because quantity rendering and chunk documents are not implemented.

- [ ] **Step 4: Add chunk limit field and pass it to retrieval**

Modify `LandscapeRetriever`:

```python
chunk_limit: int = 3
```

Extend `_aget_relevant_documents`:

```python
result = await retrieve(
    query_text=query,
    hops=self.hops,
    limit=self.limit,
    chunk_limit=self.chunk_limit,
    weights=self.weights,
    reinforce=self.reinforce,
    session_id=self.session_id,
    since=self.since,
)
return [_entity_to_document(e) for e in result.results] + [
    _chunk_to_document(c) for c in result.chunks
]
```

- [ ] **Step 5: Add quantity formatting and chunk conversion**

In `src/landscape/retrieval/langchain_retriever.py`, add:

```python
def _format_quantity(quantity: dict[str, object | None]) -> str:
    value = quantity.get("quantity_value")
    unit = quantity.get("quantity_unit")
    kind = quantity.get("quantity_kind")
    scope = quantity.get("time_scope")
    parts = []
    if value is not None:
        label = str(kind) if kind else "quantity"
        rendered = f"{label}={value}"
        if unit:
            rendered = f"{rendered} {unit}"
        parts.append(rendered)
    if scope:
        parts.append(f"scope={scope}")
    return ", ".join(parts)
```

Update `_entity_to_document` so each edge can include a quantity suffix:

```python
quantities = entity.path_edge_quantities or [{} for _ in entity.path_edge_types]
if len(quantities) < len(entity.path_edge_types):
    quantities = list(quantities) + [{} for _ in range(len(entity.path_edge_types) - len(quantities))]

path_parts = []
for rel_type, subtype, quantity in zip(entity.path_edge_types, subtypes, quantities):
    edge = _format_edge(rel_type, subtype)
    rendered_quantity = _format_quantity(quantity)
    if rendered_quantity:
        edge = f"{edge} {{{rendered_quantity}}}"
    path_parts.append(edge)
```

Add `path_edge_quantities` to metadata:

```python
"path_edge_quantities": entity.path_edge_quantities,
```

Add:

```python
def _chunk_to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "kind": "chunk",
            "chunk_neo4j_id": chunk.chunk_neo4j_id,
            "doc_id": chunk.doc_id,
            "source_doc": chunk.source_doc,
            "position": chunk.position,
            "score": chunk.score,
        },
    )
```

Import `RetrievedChunk` from `landscape.retrieval.query`.

- [ ] **Step 6: Run LangChain tests**

Run:

```bash
uv run --extra dev pytest tests/test_langchain_retriever.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/landscape/retrieval/langchain_retriever.py tests/test_langchain_retriever.py
git commit -m "feat: surface quantified facts in langchain retrieval"
```

## Task 7: Teach the Extraction Prompt About Quantities

**Files:**
- Modify: `src/landscape/extraction/llm.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Add prompt regression test**

Add this test to `tests/test_ingest.py`:

```python
def test_extraction_prompt_mentions_quantity_fields():
    from landscape.extraction import llm

    prompt = llm._SYSTEM_PROMPT

    assert "quantity_value" in prompt
    assert "quantity_unit" in prompt
    assert "quantity_kind" in prompt
    assert "time_scope" in prompt
    assert "10 hours" in prompt
    assert "three bikes" in prompt
```

- [ ] **Step 2: Run the prompt regression test and verify it fails**

Run:

```bash
uv run --extra dev pytest tests/test_ingest.py::test_extraction_prompt_mentions_quantity_fields -q
```

Expected: FAIL because the prompt does not mention quantity fields.

- [ ] **Step 3: Update prompt rules**

In `src/landscape/extraction/llm.py`, update the relationship description near the top:

```python
"2. Relationships between those entities as (subject, relation_type, object)\n"
"   triples, with optional `subtype` and optional numeric qualifiers:\n"
"   `quantity_value`, `quantity_unit`, `quantity_kind`, and `time_scope`.\n"
```

Add a critical rule after the subtype rule:

```python
"- When a relationship includes a count, duration, frequency, price, distance,\n"
"  percentage, rating, or measurement, preserve it on the relation using:\n"
"  `quantity_value`, `quantity_unit`, `quantity_kind`, and `time_scope`.\n"
"  Examples: 10 hours → quantity_value=10, quantity_unit=hour,\n"
"  quantity_kind=duration; three bikes → quantity_value=3,\n"
"  quantity_unit=bike, quantity_kind=count. Keep relation_type in the closed\n"
"  vocabulary; quantities are edge qualifiers, not new relation types.\n"
```

Add a worked example before `"Now extract..."`:

```python
"\n"
"--- EXAMPLE 6 (quantified facts) ---\n"
'Input: "Eric spent 10 hours last month watching documentaries on Netflix. '\
'He owns three bikes."\n'
"Output:\n"
"{\n"
'  "entities": [\n'
'    {"name": "Eric", "type": "PERSON", "confidence": 0.95, "aliases": []},\n'
'    {"name": "Netflix", "type": "TECHNOLOGY", "confidence": 0.9, "aliases": []},\n'
'    {"name": "Bike", "type": "CONCEPT", "confidence": 0.85, "aliases": ["bikes"]}\n'
"  ],\n"
'  "relations": [\n'
'    {"subject": "Eric", "object": "Netflix", "relation_type": "DISCUSSED", '\
'"subtype": "watched_documentaries", "confidence": 0.9, '\
'"quantity_value": 10, "quantity_unit": "hour", '\
'"quantity_kind": "duration", "time_scope": "last_month"},\n'
'    {"subject": "Eric", "object": "Bike", "relation_type": "HAS_ATTRIBUTE", '\
'"subtype": "owned_count", "confidence": 0.9, '\
'"quantity_value": 3, "quantity_unit": "bike", "quantity_kind": "count"}\n'
"  ]\n"
"}\n"
```

- [ ] **Step 4: Run prompt regression test**

Run:

```bash
uv run --extra dev pytest tests/test_ingest.py::test_extraction_prompt_mentions_quantity_fields -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/landscape/extraction/llm.py tests/test_ingest.py
git commit -m "feat: prompt extractor for quantified facts"
```

## Task 8: Focused Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --extra dev pytest \
  tests/test_ingest.py::test_extraction_schema_accepts_quantified_relation_fields \
  tests/test_ingest.py::test_ingest_passes_relation_quantity_fields \
  tests/test_ingest.py::test_extraction_prompt_mentions_quantity_fields \
  tests/test_supersession.py::test_relation_quantity_fields_written_on_create \
  tests/test_supersession.py::test_relation_quantity_fields_reinforce_with_non_null_wins \
  tests/test_retrieval_basic.py::test_retrieval_includes_path_edge_quantities \
  tests/test_langchain_retriever.py \
  tests/test_mcp_server.py \
  tests/test_chunk_surfacing.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run lint**

Run:

```bash
uv run --extra dev ruff check src tests
```

Expected: PASS.

- [ ] **Step 3: Run or resume full baseline suite**

Run:

```bash
uv run --extra dev pytest
```

Expected: PASS except any pre-existing baseline failures already documented before feature implementation. If failures remain, capture exact failing test names and whether they are pre-existing or introduced by this feature branch.

- [ ] **Step 4: Commit any verification-only doc update**

If the final result requires documenting pre-existing baseline failures, update this plan's Baseline Status section and commit:

```bash
git add docs/superpowers/plans/2026-04-20-quantified-facts.md
git commit -m "docs: record quantified facts verification status"
```
