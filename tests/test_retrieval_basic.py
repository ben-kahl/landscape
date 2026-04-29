"""Basic retrieval integration tests — single-hop and temporal filter.

The multi-hop killer demo lives in test_retrieval_multihop.py under
the 'retrieval' marker. Tests here run by default."""
import json
import logging

import pytest

BASIC_DOC = (
    "Alice leads Project Atlas. Project Atlas uses PostgreSQL for storage. "
    "Sarah Chen approved the PostgreSQL migration. Sarah Chen is on the Platform Team."
)
BASIC_TITLE = "retrieval-basic-test"


async def _clear(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) DETACH DELETE e",
            t=title,
        )
        await session.run("MATCH (d:Document {title: $t}) DETACH DELETE d", t=title)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_query_returns_seeded_entity(http_client, neo4j_driver):
    """A query for an entity name should return that entity as a top result."""
    await _clear(neo4j_driver, BASIC_TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": BASIC_TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "What does Project Atlas use?", "hops": 2, "limit": 10},
    )
    assert q.status_code == 200
    body = q.json()
    assert body["results"], "query should return at least one result"
    names = {r["name"] for r in body["results"]}
    assert "Project Atlas" in names or "PostgreSQL" in names


@pytest.mark.asyncio
@pytest.mark.integration
async def test_query_finds_multihop_target(http_client, neo4j_driver):
    """2-hop expansion: ask about Atlas, the answer path includes Sarah."""
    await _clear(neo4j_driver, BASIC_TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": BASIC_TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "Project Atlas database approval", "hops": 3, "limit": 10},
    )
    assert q.status_code == 200
    body = q.json()
    names = {r["name"] for r in body["results"]}
    # Sarah should be reachable via Atlas -> PostgreSQL -> Sarah (2 hops)
    assert "Sarah Chen" in names or any("Sarah" in n for n in names), (
        f"Expected Sarah in results via graph expansion, got: {names}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_query_reinforces_touched_entities(http_client, neo4j_driver):
    """After a query, the touched entities themselves should have access_count > 0."""
    title = "retrieval-reinforce-test"
    await _clear(neo4j_driver, title)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": title})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "Project Atlas", "hops": 2, "limit": 5, "reinforce": True},
    )
    assert q.status_code == 200
    body = q.json()
    assert body["touched_entity_count"] > 0
    assert body["results"]

    # The response reports which neo4j IDs were touched; check them directly.
    touched_ids = [r["neo4j_id"] for r in body["results"]]
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) IN $ids
            RETURN count(e) AS total,
                   sum(CASE WHEN e.access_count > 0 THEN 1 ELSE 0 END) AS reinforced
            """,
            ids=touched_ids,
        )
        record = await result.single()
    assert record["total"] > 0, "touched ids should resolve to entities"
    assert record["reinforced"] > 0, (
        f"expected at least one touched entity to have access_count > 0, "
        f"got {record['reinforced']}/{record['total']}"
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_temporal_filter_excludes_superseded(neo4j_driver):
    """Construct a superseded/valid MemoryFact pair and verify that
    bfs_expand_memory_rel — the temporal filter at the heart of graph
    retrieval — only returns the currently-valid target. Isolated from LLM
    extraction and Qdrant seeding so the test measures exactly the temporal
    filter."""
    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store

    subj = "TempAlice"
    old_obj = "TempAcmeCo"
    new_obj = "TempZylosInc"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, old_obj, new_obj],
        )
    subject_id = await neo4j_store.merge_entity(subj, "PERSON", "retrieval-temporal-test", 0.9)
    old_object_id = await neo4j_store.merge_entity(
        old_obj, "ORGANIZATION", "retrieval-temporal-test", 0.9
    )
    new_object_id = await neo4j_store.merge_entity(
        new_obj, "ORGANIZATION", "retrieval-temporal-test", 0.9
    )

    old_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="retrieval-temporal-old",
            raw_subject_text=subj,
            raw_relation_text="works for",
            raw_object_text=old_obj,
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    new_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="retrieval-temporal-new",
            raw_subject_text=subj,
            raw_relation_text="works for",
            raw_object_text=new_obj,
            confidence=0.95,
            family_candidate="WORKS_FOR",
        )
    )

    old_fact = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=subject_id,
        object_entity_id=old_object_id,
        subtype=None,
        confidence=0.9,
        assertion_id=old_assertion,
    )
    await neo4j_store.materialize_memory_rel(old_fact)

    new_fact = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=subject_id,
        object_entity_id=new_object_id,
        subtype=None,
        confidence=0.95,
        assertion_id=new_assertion,
    )

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity) WHERE elementId(s) = $subject_id
            MATCH (s)-[r:MEMORY_REL {family: 'WORKS_FOR'}]->(o:Entity)
            RETURN o.name AS target, r.current AS current, r.memory_fact_id AS fact_id
            ORDER BY o.name
            """,
            subject_id=subject_id,
        )
        records = await result.data()

        old_fact_record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fact_id}) RETURN f.current AS current",
                fact_id=old_fact,
            )
        ).single()
        new_fact_record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fact_id}) RETURN f.current AS current",
                fact_id=new_fact,
            )
        ).single()

    expansions = await neo4j_store.bfs_expand_memory_rel([subject_id], max_hops=2)
    target_names = {row["target_name"] for row in expansions}

    assert new_obj in target_names, (
        f"Live target {new_obj} should be reachable, got: {target_names}"
    )
    assert old_obj not in target_names, (
        f"Superseded target {old_obj} should be filtered out, got: {target_names}"
    )
    assert old_fact_record is not None and old_fact_record["current"] is False
    assert new_fact_record is not None and new_fact_record["current"] is True
    assert {record["target"]: record["current"] for record in records} == {
        old_obj: False,
        new_obj: True,
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieval_hydrates_memory_facts_and_supporting_assertions(monkeypatch):
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

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops):
        return [
            {
                "seed_id": "eric-id",
                "target_id": "netflix-id",
                "target_name": "Netflix",
                "target_type": "TECHNOLOGY",
                "distance": 1,
                "path_memory_fact_ids": ["fact-1"],
                "edge_families": ["DISCUSSION"],
            }
        ]

    async def fake_touch_entities(ids, now):
        return None

    async def fake_touch_relations(ids, now):
        return None

    async def fake_hydrate_path_memory_facts(memory_fact_ids):
        assert memory_fact_ids == ["fact-1"]
        return (
            [
                {
                    "memory_fact_id": "fact-1",
                    "family": "DISCUSSION",
                    "current": True,
                    "fact_key": "fact-key",
                    "slot_key": "slot-key",
                    "subtype": None,
                    "support_count": 1,
                    "confidence_agg": 0.9,
                    "subject_entity_id": "eric-id",
                    "subject_name": "Eric",
                    "subject_type": "PERSON",
                    "object_entity_id": "netflix-id",
                    "object_name": "Netflix",
                    "object_type": "TECHNOLOGY",
                    "memory_rel_current": True,
                }
            ],
            [
                {
                    "memory_fact_id": "fact-1",
                    "assertion_id": "assert-1",
                    "source_kind": "document",
                    "source_id": "doc-1",
                    "raw_subject_text": "Eric",
                    "raw_relation_text": "discussed",
                    "raw_object_text": "Netflix",
                    "family_candidate": "DISCUSSION",
                    "confidence": 0.9,
                    "subtype": None,
                    "quantity_value": 10,
                    "quantity_unit": "hour",
                    "quantity_kind": "duration",
                    "time_scope": "last_month",
                    "status": "active",
                    "created_at": "2026-04-29T00:00:00Z",
                }
            ],
        )

    monkeypatch.setattr(
        query.qdrant_store,
        "search_entities_any_type",
        fake_search_entities_any_type,
    )
    monkeypatch.setattr(query.qdrant_store, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(
        query.neo4j_store,
        "get_entities_from_chunks",
        fake_get_entities_from_chunks,
    )
    monkeypatch.setattr(query, "_hydrate_entities", fake_hydrate_entities)
    monkeypatch.setattr(
        query.neo4j_store, "bfs_expand_memory_rel", fake_bfs_expand_memory_rel
    )
    monkeypatch.setattr(query.neo4j_store, "touch_entities", fake_touch_entities)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", fake_touch_relations)
    monkeypatch.setattr(query, "_hydrate_memory_path_details", fake_hydrate_path_memory_facts)

    result = await query.retrieve("How many hours on Netflix?", reinforce=False)

    netflix = next(r for r in result.results if r.name == "Netflix")
    assert netflix.path_memory_fact_ids == ["fact-1"]
    assert netflix.path_edge_types == ["DISCUSSION"]
    assert netflix.memory_facts == [
        {
            "memory_fact_id": "fact-1",
            "family": "DISCUSSION",
            "current": True,
            "fact_key": "fact-key",
            "slot_key": "slot-key",
            "subtype": None,
            "support_count": 1,
            "confidence_agg": 0.9,
            "subject_entity_id": "eric-id",
            "subject_name": "Eric",
            "subject_type": "PERSON",
            "object_entity_id": "netflix-id",
            "object_name": "Netflix",
            "object_type": "TECHNOLOGY",
            "memory_rel_current": True,
        }
    ]
    assert netflix.supporting_assertions == [
        {
            "memory_fact_id": "fact-1",
            "assertion_id": "assert-1",
            "source_kind": "document",
            "source_id": "doc-1",
            "raw_subject_text": "Eric",
            "raw_relation_text": "discussed",
            "raw_object_text": "Netflix",
            "family_candidate": "DISCUSSION",
            "confidence": 0.9,
            "subtype": None,
            "quantity_value": 10,
            "quantity_unit": "hour",
            "quantity_kind": "duration",
            "time_scope": "last_month",
            "status": "active",
            "created_at": "2026-04-29T00:00:00Z",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_emits_summary_logs_by_default(monkeypatch, caplog):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])

    class Hit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit(0.9, {"neo4j_node_id": "atlas-id"})]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(ids):
        return [
            {
                "eid": "atlas-id",
                "name": "Project Atlas",
                "type": "PROJECT",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops):
        return []

    async def noop_touch(*args, **kwargs):
        return None

    monkeypatch.setattr(
        query.qdrant_store,
        "search_entities_any_type",
        fake_search_entities_any_type,
    )
    monkeypatch.setattr(query.qdrant_store, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(
        query.neo4j_store,
        "get_entities_from_chunks",
        fake_get_entities_from_chunks,
    )
    monkeypatch.setattr(query, "_hydrate_entities", fake_hydrate_entities)
    monkeypatch.setattr(
        query.neo4j_store, "bfs_expand_memory_rel", fake_bfs_expand_memory_rel
    )
    monkeypatch.setattr(query.neo4j_store, "touch_entities", noop_touch)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", noop_touch)

    caplog.set_level(logging.INFO, logger="landscape.retrieval")

    await query.retrieve("What does Project Atlas use?")

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "landscape.retrieval"
    ]

    assert [event["event"] for event in events] == [
        "retrieval_started",
        "retrieval_completed",
    ]
    assert events[-1]["result_count"] == 1
    assert events[-1]["top_results"] == [
        {
            "name": "Project Atlas",
            "type": "PROJECT",
            "score": 1.7,
            "distance": 0,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieval_uses_memory_rel_traversal(monkeypatch):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])

    class Hit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit(0.9, {"neo4j_node_id": "atlas-id"})]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(ids):
        return [
            {
                "eid": "atlas-id",
                "name": "Project Atlas",
                "type": "PROJECT",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops):
        assert seed_ids == ["atlas-id"]
        assert max_hops == 2
        return [
            {
                "seed_id": "atlas-id",
                "target_id": "postgres-id",
                "target_name": "PostgreSQL",
                "target_type": "DATABASE",
                "distance": 1,
                "path_memory_fact_ids": ["fact-1"],
                "edge_families": ["USES"],
            }
        ]

    async def noop_touch(*args, **kwargs):
        return None

    async def fail_if_legacy_bfs(*args, **kwargs):
        raise AssertionError("legacy bfs_expand should not be used")

    async def noop_hydrate(memory_fact_ids):
        return ([], [])

    monkeypatch.setattr(
        query.qdrant_store,
        "search_entities_any_type",
        fake_search_entities_any_type,
    )
    monkeypatch.setattr(query.qdrant_store, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(
        query.neo4j_store,
        "get_entities_from_chunks",
        fake_get_entities_from_chunks,
    )
    monkeypatch.setattr(query, "_hydrate_entities", fake_hydrate_entities)
    monkeypatch.setattr(
        query.neo4j_store, "bfs_expand_memory_rel", fake_bfs_expand_memory_rel
    )
    monkeypatch.setattr(query.neo4j_store, "bfs_expand", fail_if_legacy_bfs)
    monkeypatch.setattr(query.neo4j_store, "touch_entities", noop_touch)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", noop_touch)
    monkeypatch.setattr(query, "_hydrate_memory_path_details", noop_hydrate)

    result = await query.retrieve("What does Project Atlas use?", reinforce=False)

    assert [item.name for item in result.results] == ["Project Atlas", "PostgreSQL"]
    postgres = next(item for item in result.results if item.name == "PostgreSQL")
    assert postgres.path_memory_fact_ids == ["fact-1"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_emits_debug_stage_logs_when_requested(monkeypatch, caplog):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])

    class Hit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit(0.9, {"neo4j_node_id": "atlas-id"})]

    async def fake_search_chunks(vector, limit=10):
        return [
            Hit(
                0.7,
                {
                    "chunk_neo4j_id": "chunk-1",
                    "text": "Project Atlas uses PostgreSQL.",
                    "doc_id": "doc-1",
                    "source_doc": "atlas-doc",
                    "position": 0,
                },
            )
        ]

    async def fake_get_entities_from_chunks(chunk_ids):
        return [{"eid": "atlas-id", "chunk_eids": chunk_ids}]

    async def fake_hydrate_entities(ids):
        return [
            {
                "eid": "atlas-id",
                "name": "Project Atlas",
                "type": "PROJECT",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops):
        return []

    async def noop_touch(*args, **kwargs):
        return None

    monkeypatch.setattr(
        query.qdrant_store,
        "search_entities_any_type",
        fake_search_entities_any_type,
    )
    monkeypatch.setattr(query.qdrant_store, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(
        query.neo4j_store,
        "get_entities_from_chunks",
        fake_get_entities_from_chunks,
    )
    monkeypatch.setattr(query, "_hydrate_entities", fake_hydrate_entities)
    monkeypatch.setattr(
        query.neo4j_store, "bfs_expand_memory_rel", fake_bfs_expand_memory_rel
    )
    monkeypatch.setattr(query.neo4j_store, "touch_entities", noop_touch)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", noop_touch)

    caplog.set_level(logging.INFO, logger="landscape.retrieval")

    await query.retrieve("What does Project Atlas use?", debug=True)

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "landscape.retrieval"
    ]
    names = {event["event"] for event in events}

    assert {
        "retrieval_started",
        "query_embedding_completed",
        "seed_search_completed",
        "chunk_entity_propagation_completed",
        "seed_hydration_completed",
        "graph_expansion_completed",
        "filter_completed",
        "ranking_completed",
        "reinforcement_completed",
        "retrieval_completed",
    } <= names
    assert all(event["retrieval_id"] == events[0]["retrieval_id"] for event in events)
    assert all(event["debug"] is True for event in events)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_query_api_threads_debug_flag(monkeypatch, http_client):
    from landscape.api import query as query_api
    from landscape.retrieval.query import RetrievalResult

    calls = []

    async def fake_retrieve(
        query_text,
        hops=2,
        limit=10,
        chunk_limit=3,
        weights=None,
        reinforce=True,
        session_id=None,
        since=None,
        debug=False,
        include_historical=False,
        log_context=None,
    ):
        calls.append(
            {
                "query_text": query_text,
                "hops": hops,
                "limit": limit,
                "chunk_limit": chunk_limit,
                "reinforce": reinforce,
                "session_id": session_id,
                "debug": debug,
                "include_historical": include_historical,
            }
        )
        return RetrievalResult(
            query=query_text,
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=[],
        )

    monkeypatch.setattr(query_api.query_module, "retrieve", fake_retrieve)

    response = await http_client.post(
        "/query",
        json={"text": "Project Atlas", "debug": True, "include_historical": True},
    )

    assert response.status_code == 200
    assert calls == [
        {
            "query_text": "Project Atlas",
            "hops": 2,
            "limit": 10,
            "chunk_limit": 3,
            "reinforce": True,
            "session_id": None,
            "debug": True,
            "include_historical": True,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_query_cli_threads_include_historical_flag(monkeypatch, capsys):
    from argparse import Namespace

    from landscape.cli import query as query_cli
    from landscape.retrieval.query import RetrievalResult, RetrievedEntity

    calls = []

    class FakeEncoder:
        def load_model(self):
            return None

    class FakeStore:
        async def init_collection(self):
            return None

        async def init_chunks_collection(self):
            return None

    async def fake_retrieve(
        query_text,
        hops=2,
        limit=10,
        chunk_limit=3,
        weights=None,
        reinforce=True,
        session_id=None,
        since=None,
        debug=False,
        include_historical=False,
        log_context=None,
    ):
        calls.append(
            {
                "query_text": query_text,
                "include_historical": include_historical,
                "debug": debug,
            }
        )
        return RetrievalResult(
            query=query_text,
            results=[
                RetrievedEntity(
                    neo4j_id="atlas-id",
                    name="Project Atlas",
                    type="PROJECT",
                    distance=0,
                    vector_sim=0.9,
                    reinforcement=0.0,
                    edge_confidence=0.0,
                    score=1.0,
                )
            ],
            touched_entity_ids=["atlas-id"],
            touched_edge_ids=[],
            chunks=[],
        )

    async def noop_close_runtime(*args, **kwargs):
        return None

    monkeypatch.setattr(
        query_cli,
        "_get_runtime",
        lambda: (FakeEncoder(), fake_retrieve, FakeStore(), FakeStore()),
    )
    monkeypatch.setattr(query_cli, "close_runtime", noop_close_runtime)

    exit_code = await query_cli.handle_query(
        Namespace(
            text="Project Atlas",
            hops=2,
            limit=10,
            no_reinforce=False,
            debug=False,
            include_historical=True,
        )
    )

    assert exit_code == 0
    assert calls == [
        {
            "query_text": "Project Atlas",
            "include_historical": True,
            "debug": False,
        }
    ]
    assert "1. Project Atlas [PROJECT]" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.integration
async def test_alias_resolved_relation_traversable_from_canonical(neo4j_driver):
    """Regression: a relation written via add_relation('Bob', ...) where Bob is an
    alias for Robert must be traversable via bfs_expand starting from Robert's
    canonical node id -- not only from the alias stub's id.

    This is the retrieval counterpart to the writeback alias regression:
    verifies that the corrected relation endpoint lands on the canonical node
    and is therefore reachable during graph expansion.
    """
    from datetime import UTC, datetime

    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store
    from landscape.writeback import add_relation

    # Ensure Qdrant collections exist (lifespan not triggered without http_client).
    existing = await qdrant_store.get_client().get_collections()
    names = {c.name for c in existing.collections}
    if qdrant_store.COLLECTION not in names:
        await qdrant_store.init_collection()
    if qdrant_store.CHUNKS_COLLECTION not in names:
        await qdrant_store.init_chunks_collection()
    encoder.load_model()

    # Seed Robert in Neo4j + Qdrant using the "Bob (Person)" vector so that
    # the resolver finds Robert when add_entity("Bob") queries Qdrant.
    bob_vector = encoder.encode("Bob (Person)")
    doc_id, _ = await neo4j_store.merge_document(
        "hash-ret-alias-robert", "ret-alias-robert-doc", "text"
    )
    robert_id = await neo4j_store.merge_entity(
        "Robert", "Person", "ret-alias-robert-doc", 0.9, doc_id, "test"
    )
    await qdrant_store.upsert_entity(
        neo4j_element_id=robert_id,
        name="Robert",
        entity_type="Person",
        source_doc="ret-alias-robert-doc",
        timestamp=datetime.now(UTC).isoformat(),
        vector=bob_vector,
    )
    # Register "Bob" as alias stub for Robert in Neo4j.
    await neo4j_store.add_alias(robert_id, "Bob", "test-alias", 0.95)

    # Seed Acme in Neo4j + Qdrant.
    doc_id2, _ = await neo4j_store.merge_document(
        "hash-ret-alias-acme", "ret-alias-acme-doc", "text"
    )
    acme_id = await neo4j_store.merge_entity(
        "AcmeCorp", "Organization", "ret-alias-acme-doc", 0.9, doc_id2, "test"
    )
    await qdrant_store.upsert_entity(
        neo4j_element_id=acme_id,
        name="AcmeCorp",
        entity_type="Organization",
        source_doc="ret-alias-acme-doc",
        timestamp=datetime.now(UTC).isoformat(),
        vector=encoder.encode("AcmeCorp (Organization)"),
    )

    # Write the relation via the writeback path using alias name "Bob".
    result = await add_relation(
        "Bob",
        "Person",
        "AcmeCorp",
        "Organization",
        "WORKS_FOR",
        source="agent:ret-alias-test:1",
        session_id="s-ret-alias",
        turn_id="t-ret-alias",
    )
    assert result.outcome == "memory_fact"
    assert result.memory_fact_id is not None

    # bfs_expand_memory_rel from Robert's canonical id must reach AcmeCorp.
    expansions = await neo4j_store.bfs_expand_memory_rel([robert_id], max_hops=1)
    target_names = {row["target_name"] for row in expansions}

    assert "AcmeCorp" in target_names, (
        f"AcmeCorp should be reachable from Robert (canonical) via bfs_expand_memory_rel, "
        f"got: {target_names}. The relation may have been written to the alias stub."
    )

    # Confirm the alias stub has no MEMORY_REL edges (the canonical node owns the edge).
    async with neo4j_driver.session() as session:
        stub_edges = await (
            await session.run(
                "MATCH (stub:Entity {name: 'Bob', canonical: false})"
                "-[r:MEMORY_REL]->() RETURN count(r) AS cnt"
            )
        ).single()

    assert stub_edges["cnt"] == 0, (
        f"Alias stub 'Bob' must not have any MEMORY_REL edges; the canonical 'Robert' "
        f"node owns the relation. Got {stub_edges['cnt']} edge(s) on stub."
    )


@pytest.mark.unit
def test_retrieval_log_sink_writes_jsonl_to_process_scoped_file(tmp_path):
    from landscape.observability.retrieval_logging import (
        create_retrieval_log_context,
        ensure_retrieval_log_sink,
    )

    log_dir = tmp_path / "logs" / "retrieval"
    log_path = ensure_retrieval_log_sink(log_dir, force=True)
    second_path = ensure_retrieval_log_sink(log_dir)

    ctx = create_retrieval_log_context(
        query_text="Project Atlas",
        hops=2,
        limit=10,
        chunk_limit=3,
        reinforce=True,
        debug=False,
    )
    ctx.emit_started()
    ctx.emit_completed(
        result_count=1,
        touched_entity_count=1,
        touched_edge_count=0,
        chunk_count=0,
    )

    assert second_path == log_path
    assert log_path.parent == log_dir
    assert log_path.name.startswith("retrieval-")
    assert log_path.suffix == ".jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event"] == "retrieval_started"
    assert second["event"] == "retrieval_completed"
    assert second["top_results"] == []
