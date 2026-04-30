import json
import logging

import pytest
import pytest_asyncio
from qdrant_client.models import FieldCondition, Filter, MatchValue

from landscape.extraction.schema import Extraction
from landscape.storage import qdrant_store

TEST_DOC = "Alice leads Project Atlas at Acme Corp. Project Atlas uses PostgreSQL for storage."
TEST_TITLE = "test-doc-integration"


@pytest_asyncio.fixture(autouse=True)
async def _ensure_qdrant_collections(request):
    if request.node.get_closest_marker("unit") or request.node.get_closest_marker("smoke"):
        return
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

@pytest.mark.unit
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

@pytest.mark.unit
def test_extraction_prompt_mentions_quantity_fields():
    from landscape.extraction import llm

    prompt = llm._SYSTEM_PROMPT

    assert "quantity_value" in prompt
    assert "quantity_unit" in prompt
    assert "quantity_kind" in prompt
    assert "time_scope" in prompt
    assert "10 hours" in prompt
    assert "three bikes" in prompt

@pytest.mark.asyncio
@pytest.mark.integration
async def test_ingest_creates_graph_and_vectors(http_client, neo4j_driver, qdrant_client):
    # Clear any prior state for this test title
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (d:Document {title: $title})<-[:EXTRACTED_FROM]-(e:Entity) DETACH DELETE e, d",
            title=TEST_TITLE,
        )
        await session.run("MATCH (d:Document {title: $title}) DETACH DELETE d", title=TEST_TITLE)

    response = await http_client.post(
        "/ingest",
        json={"text": TEST_DOC, "title": TEST_TITLE},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["already_existed"] is False
    # With resolution enabled, some entities may be reinforced rather than created
    assert body["entities_created"] + body["entities_reinforced"] >= 3
    assert body["relations_created"] + body["relations_reinforced"] >= 2
    # New fields present in response
    assert "entities_reinforced" in body
    assert "relations_reinforced" in body
    assert "relations_superseded" in body
    assert "chunks_created" in body
    assert body["chunks_created"] >= 1

    # Verify Neo4j graph
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $title})"
            " RETURN count(e) AS cnt",
            title=TEST_TITLE,
        )
        record = await result.single()
        # Resolved entities don't get a new EXTRACTED_FROM edge; at least entities_created will
        assert record["cnt"] >= body["entities_created"]

        result = await session.run(
            """
            MATCH (d:Document {title: $title})-[:ASSERTS]->(a:Assertion)
                  -[:SUPPORTS]->(f:MemoryFact)
            WHERE f.valid_until IS NULL
            RETURN count(*) AS cnt
            """,
            title=TEST_TITLE,
        )
        record = await result.single()
        assert record["cnt"] >= 2

        result = await session.run(
            "MATCH (d:Document {title: $title}) RETURN count(d) AS cnt",
            title=TEST_TITLE,
        )
        record = await result.single()
        assert record["cnt"] == 1

    # Verify Qdrant: at least entities_created vectors with this source_doc exist,
    # each with a valid entity_id. (Stale points from prior runs may inflate the count.)
    points, _ = await qdrant_client.scroll(
        collection_name="entities",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=TEST_TITLE))]
        ),
        with_payload=True,
        limit=100,
    )
    assert len(points) >= body["entities_created"]
    for point in points:
        assert point.payload.get("entity_id"), "Missing entity_id in Qdrant payload"

    # Verify chunks collection has entries for this doc
    chunk_points, _ = await qdrant_client.scroll(
        collection_name="chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=TEST_TITLE))]
        ),
        with_payload=True,
        limit=100,
    )
    assert len(chunk_points) >= body["chunks_created"]
    for cp in chunk_points:
        assert cp.payload.get("chunk_id"), "Missing chunk_id in Qdrant payload"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ingest_promotes_additive_family_into_memory_rel(http_client, neo4j_driver):
    from landscape import pipeline
    from landscape.extraction.chunker import Chunk
    from landscape.extraction.schema import ExtractedEntity, ExtractedRelation, Extraction

    title = "memory-rel-additive-integration"
    subject_name = "Alice Example"
    object_name = "Project Atlas Example"

    async with neo4j_driver.session() as session:
        await session.run("MATCH (d:Document {title: $title}) DETACH DELETE d", title=title)
        await session.run("MATCH (e:Entity {name: $name}) DETACH DELETE e", name=subject_name)
        await session.run("MATCH (e:Entity {name: $name}) DETACH DELETE e", name=object_name)

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        return None, True, None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)

    async def noop_upsert(**kwargs):
        return None

    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", noop_upsert)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", noop_upsert)
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text=text)])
    monkeypatch.setattr(
        pipeline.llm,
        "extract",
        lambda text: Extraction(
            entities=[
                ExtractedEntity(name=subject_name, type="PERSON", confidence=0.95),
                ExtractedEntity(name=object_name, type="PROJECT", confidence=0.95),
            ],
            relations=[
                ExtractedRelation(
                    subject=subject_name,
                    object=object_name,
                    relation_type="owns",
                    confidence=0.91,
                    quantity_value=1,
                    quantity_unit="instance",
                    quantity_kind="count",
                    time_scope="current",
                )
            ],
        ),
    )

    try:
        response = await http_client.post(
            "/ingest",
            json={"text": f"{subject_name} owns {object_name}.", "title": title},
        )
    finally:
        monkeypatch.undo()

    assert response.status_code == 200
    body = response.json()
    assert body["relations_created"] == 1
    assert body["relations_reinforced"] == 0
    assert body["relations_superseded"] == 0

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {title: $title})-[:ASSERTS]->(a:Assertion)
                  -[:SUBJECT_ENTITY]->(s:Entity {name: $subject}),
                  (a)-[:OBJECT_ENTITY]->(o:Entity {name: $object}),
                  (a)-[:SUPPORTS]->(f:MemoryFact),
                  (s)-[r:MEMORY_REL]->(o)
            WHERE f.valid_until IS NULL
              AND r.valid_until IS NULL
            RETURN a.subtype AS subtype,
                   a.quantity_value AS quantity_value,
                   a.quantity_unit AS quantity_unit,
                   a.quantity_kind AS quantity_kind,
                   a.time_scope AS time_scope,
                   count(r) AS memory_rel_count,
                   count(f) AS fact_count
            """,
            title=title,
            subject=subject_name,
            object=object_name,
        )
        record = await result.single()

    assert record is not None
    assert record["subtype"] == "owns"
    assert record["quantity_value"] == 1
    assert record["quantity_unit"] == "instance"
    assert record["quantity_kind"] == "count"
    assert record["time_scope"] == "current"
    assert record["memory_rel_count"] == 1
    assert record["fact_count"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ingest_idempotent(http_client):
    # First ingest
    r1 = await http_client.post("/ingest", json={"text": TEST_DOC, "title": TEST_TITLE})
    assert r1.status_code == 200

    # Second ingest with same text — should short-circuit
    r2 = await http_client.post("/ingest", json={"text": TEST_DOC, "title": TEST_TITLE})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["already_existed"] is True
    assert body2["entities_created"] == 0
    assert body2["relations_created"] == 0
    assert body2["entities_reinforced"] == 0
    assert body2["relations_reinforced"] == 0
    assert body2["relations_superseded"] == 0
    assert body2["chunks_created"] == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingest_passes_relation_quantity_fields(monkeypatch):
    from landscape import pipeline
    from landscape.extraction.schema import ExtractedEntity, ExtractedRelation
    from landscape.memory_graph.service import PersistenceResult

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

    async def fake_persist_assertion_and_maybe_promote(
        payload,
        *,
        source_node_id,
        source_kind,
        subject_entity_id,
        object_entity_id,
        chunk_ids,
    ):
        captured_relation_kwargs.update(
            {
                "payload": payload,
                "source_node_id": source_node_id,
                "source_kind": source_kind,
                "subject_entity_id": subject_entity_id,
                "object_entity_id": object_entity_id,
                "chunk_ids": chunk_ids,
            }
        )
        return PersistenceResult(
            assertion_id="assertion-1",
            fact_id="fact-1",
            outcome="created",
        )

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", fake_upsert_chunk)
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "merge_entity", fake_merge_entity)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", fake_upsert_entity)
    monkeypatch.setattr(
        pipeline,
        "persist_assertion_and_maybe_promote",
        fake_persist_assertion_and_maybe_promote,
    )
    monkeypatch.setattr(pipeline, "coerce_rel_type", lambda rel_type: (rel_type, 1.0))
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
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

    assert captured_relation_kwargs["payload"].quantity_value == 10
    assert captured_relation_kwargs["payload"].quantity_unit == "hour"
    assert captured_relation_kwargs["payload"].quantity_kind == "duration"
    assert captured_relation_kwargs["payload"].time_scope == "last_month"
    assert captured_relation_kwargs["subject_entity_id"] == "Eric-id"
    assert captured_relation_kwargs["object_entity_id"] == "Netflix-id"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingest_emits_summary_logs_by_default(monkeypatch, caplog):
    from landscape import pipeline
    from landscape.extraction.chunker import Chunk
    from landscape.extraction.schema import ExtractedEntity, ExtractedRelation
    from landscape.memory_graph.service import PersistenceResult

    async def fake_merge_document(content_hash, title, source_type):
        return "doc-1", True

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return f"chunk-{chunk_index}"

    async def fake_upsert_chunk(**kwargs):
        return None

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        return "entity-1", True, 0.0

    async def fake_merge_entity(**kwargs):
        return "entity-1"

    async def fake_upsert_entity(**kwargs):
        return None

    async def fake_persist_assertion_and_maybe_promote(
        payload,
        *,
        source_node_id,
        source_kind,
        subject_entity_id,
        object_entity_id,
        chunk_ids,
    ):
        return PersistenceResult(
            assertion_id="assertion-1",
            fact_id="fact-1",
            outcome="created",
        )

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", fake_upsert_chunk)
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "merge_entity", fake_merge_entity)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", fake_upsert_entity)
    monkeypatch.setattr(
        pipeline,
        "persist_assertion_and_maybe_promote",
        fake_persist_assertion_and_maybe_promote,
    )
    monkeypatch.setattr(pipeline, "coerce_rel_type", lambda rel_type: (rel_type, 1.0))
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text="chunk one")])
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline.llm,
        "extract",
        lambda text: Extraction(
            entities=[
                ExtractedEntity(name="Alice", type="PERSON", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(
                    subject="Alice",
                    object="Beacon Labs",
                    relation_type="WORKS_FOR",
                    confidence=0.9,
                ),
            ],
        ),
    )

    caplog.set_level(logging.INFO, logger="landscape.ingest")

    await pipeline.ingest("Alice joined Beacon Labs.", "log-summary-doc")

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "landscape.ingest"
    ]

    assert [event["event"] for event in events] == [
        "ingest_started",
        "ingest_completed",
    ]
    assert events[-1]["chunks_created"] == 1
    assert events[-1]["entities_created"] == 1
    assert "duration_ms" in events[-1]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingest_emits_debug_stage_logs_when_requested(monkeypatch, caplog):
    from landscape import pipeline
    from landscape.extraction.chunker import Chunk
    from landscape.extraction.schema import ExtractedEntity, ExtractedRelation
    from landscape.memory_graph.service import PersistenceResult

    async def fake_merge_document(content_hash, title, source_type):
        return "doc-2", True

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return f"chunk-{chunk_index}"

    async def fake_upsert_chunk(**kwargs):
        return None

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        return "entity-2", True, 0.0

    async def fake_merge_entity(**kwargs):
        return "entity-2"

    async def fake_upsert_entity(**kwargs):
        return None

    async def fake_persist_assertion_and_maybe_promote(
        payload,
        *,
        source_node_id,
        source_kind,
        subject_entity_id,
        object_entity_id,
        chunk_ids,
    ):
        return PersistenceResult(
            assertion_id="assertion-2",
            fact_id="fact-2",
            outcome="created",
        )

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", fake_upsert_chunk)
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "merge_entity", fake_merge_entity)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", fake_upsert_entity)
    monkeypatch.setattr(
        pipeline,
        "persist_assertion_and_maybe_promote",
        fake_persist_assertion_and_maybe_promote,
    )
    monkeypatch.setattr(pipeline, "coerce_rel_type", lambda rel_type: (rel_type, 1.0))
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text="chunk two")])
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline.llm,
        "extract",
        lambda text: Extraction(
            entities=[
                ExtractedEntity(name="Alice", type="PERSON", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(
                    subject="Alice",
                    object="Beacon Labs",
                    relation_type="WORKS_FOR",
                    confidence=0.9,
                ),
            ],
        ),
    )

    caplog.set_level(logging.INFO, logger="landscape.ingest")

    await pipeline.ingest("Alice joined Beacon Labs.", "log-debug-doc", debug=True)

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "landscape.ingest"
    ]
    event_names = {event["event"] for event in events}

    assert {
        "ingest_started",
        "document_merged",
        "chunking_completed",
        "chunk_embeddings_completed",
        "chunk_upserts_completed",
        "extraction_completed",
        "entity_grouping_completed",
        "entity_resolution_completed",
        "entity_writes_completed",
        "relation_upserts_completed",
        "ingest_completed",
    } <= event_names
    assert all(event["ingest_id"] == events[0]["ingest_id"] for event in events)
    assert all(event["debug"] is True for event in events)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingest_logs_failure_with_failed_stage(monkeypatch, caplog):
    from landscape import pipeline
    from landscape.extraction.chunker import Chunk
    from landscape.memory_graph.service import PersistenceResult

    async def fake_merge_document(content_hash, title, source_type):
        return "doc-3", True

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return f"chunk-{chunk_index}"

    async def boom_upsert_chunk(**kwargs):
        raise RuntimeError("chunk upsert exploded")

    async def fake_persist_assertion_and_maybe_promote(
        payload,
        *,
        source_node_id,
        source_kind,
        subject_entity_id,
        object_entity_id,
        chunk_ids,
    ):
        return PersistenceResult(
            assertion_id="assertion-3",
            fact_id="fact-3",
            outcome="created",
        )

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", boom_upsert_chunk)
    monkeypatch.setattr(
        pipeline,
        "persist_assertion_and_maybe_promote",
        fake_persist_assertion_and_maybe_promote,
    )
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text="chunk three")])
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )

    caplog.set_level(logging.INFO, logger="landscape.ingest")

    with pytest.raises(RuntimeError, match="chunk upsert exploded"):
        await pipeline.ingest("Alice joined Beacon Labs.", "log-failure-doc", debug=True)

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "landscape.ingest"
    ]

    assert events[-1]["event"] == "ingest_failed"
    assert events[-1]["failed_stage"] == "chunk_upserts_completed"
    assert "chunk upsert exploded" in events[-1]["error"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingest_api_threads_debug_flag(monkeypatch, http_client):
    from landscape.api import ingest as ingest_api
    from landscape.pipeline import IngestResult

    calls = []

    async def fake_ingest(
        text,
        title,
        source_type="text",
        session_id=None,
        turn_id=None,
        debug=False,
    ):
        calls.append(
            {
                "text": text,
                "title": title,
                "source_type": source_type,
                "session_id": session_id,
                "turn_id": turn_id,
                "debug": debug,
            }
        )
        return IngestResult(
            doc_id="doc-api-debug",
            already_existed=False,
            entities_created=0,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=0,
        )

    monkeypatch.setattr(ingest_api.pipeline, "ingest", fake_ingest)

    response = await http_client.post(
        "/ingest",
        json={
            "text": "Alice joined Beacon Labs.",
            "title": "debug-api-doc",
            "debug": True,
        },
    )

    assert response.status_code == 200
    assert calls == [
        {
            "text": "Alice joined Beacon Labs.",
            "title": "debug-api-doc",
            "source_type": "text",
            "session_id": None,
            "turn_id": None,
            "debug": True,
        }
    ]


@pytest.mark.unit
def test_ingest_log_sink_writes_jsonl_to_process_scoped_file(tmp_path):
    from landscape.observability.ingest_logging import (
        create_ingest_log_context,
        ensure_ingest_log_sink,
    )

    log_dir = tmp_path / "logs" / "ingest"
    log_path = ensure_ingest_log_sink(log_dir, force=True)
    second_path = ensure_ingest_log_sink(log_dir)

    ctx = create_ingest_log_context(
        title="sink-doc",
        source_type="text",
        debug=False,
    )
    ctx.emit_started(content_hash="abc123", text_length=42)
    ctx.emit_completed(
        doc_id="doc-1",
        already_existed=False,
        entities_created=1,
        entities_reinforced=0,
        relations_created=1,
        relations_reinforced=0,
        relations_superseded=0,
        chunks_created=1,
    )

    assert second_path == log_path
    assert log_path.parent == log_dir
    assert log_path.name.startswith("ingest-")
    assert log_path.suffix == ".jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event"] == "ingest_started"
    assert second["event"] == "ingest_completed"
    assert second["title"] == "sink-doc"
