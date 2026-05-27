"""Basic retrieval integration tests — single-hop and temporal filter.

The multi-hop killer demo lives in test_retrieval_multihop.py under
the 'retrieval' marker. Tests here run by default."""
import json
import logging
from unittest.mock import AsyncMock

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

    # The response reports which stable entity ids were touched; check them directly.
    touched_ids = [r["entity_id"] for r in body["results"]]
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE e.id IN $ids
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
            MATCH (s:Entity) WHERE s.id = $subject_id
            MATCH (s)-[r:MEMORY_REL {family: 'WORKS_FOR'}]->(o:Entity)
            RETURN o.name AS target, r.system_until AS system_until,
                   (r.system_until IS NULL) AS current, r.memory_fact_id AS fact_id
            ORDER BY o.name
            """,
            subject_id=subject_id,
        )
        records = await result.data()

        old_fact_record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fact_id}) RETURN f.system_until AS system_until, "
                "(f.system_until IS NULL) AS current",
                fact_id=old_fact,
            )
        ).single()
        new_fact_record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fact_id}) RETURN f.system_until AS system_until, "
                "(f.system_until IS NULL) AS current",
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
    assert old_fact_record is not None and old_fact_record["system_until"] is not None
    assert new_fact_record is not None and new_fact_record["system_until"] is None
    assert {record["target"]: record["system_until"] is None for record in records} == {
        old_obj: False,
        new_obj: True,
    }

    historical = await neo4j_store.bfs_expand_memory_rel(
        [subject_id], max_hops=2, include_historical=True
    )
    historical_targets = {row["target_name"] for row in historical}
    assert {old_obj, new_obj}.issubset(historical_targets), (
        f"include_historical should surface both superseded and live targets, "
        f"got: {historical_targets}"
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieval_hydrates_memory_facts_and_supporting_assertions(monkeypatch):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(query.neo4j_store, "resolve_seed_entity_ids", AsyncMock(return_value=[]))

    class Hit:
        def __init__(self):
            self.score = 0.9
            self.payload = {"entity_id": "eric-id"}

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit()]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(ids, include_historical=False, as_of=None):
        return [
            {
                "entity_id": "eric-id",
                "name": "Eric",
                "type": "PERSON",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops, include_historical=False, as_of=None):
        return [
            {
                "seed_id": "eric-id",
                "seed_name": "Eric",
                "seed_type": "PERSON",
                "target_id": "netflix-id",
                "target_name": "Netflix",
                "target_type": "TECHNOLOGY",
                "distance": 1,
                "path_memory_fact_ids": ["fact-1"],
                "path_edge_types": ["DISCUSSION"],
                "path_edge_negated": [False],
                "edge_subtypes": [None],
                "edge_ids": ["rel-1"],
                "path_node_names": ["Eric", "Netflix"],
                "path_node_types": ["PERSON", "TECHNOLOGY"],
                "edge_confidences": [0.9],
                "edge_access_counts": [0],
                "edge_last_accessed": [None],
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
                    "system_until": None,
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
                    "memory_rel_system_until": None,
                    "memory_rel_current": True,
                    "value_text": None,
                    "value_number": None,
                    "value_unit": None,
                    "value_kind": None,
                    "value_time": None,
                    "quantity_value": 10,
                    "quantity_unit": "hour",
                    "quantity_kind": "duration",
                    "time_scope": "last_month",
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
    monkeypatch.setattr(
        query, "_hydrate_current_non_traversable_entity_memory", AsyncMock(return_value=([], []))
    )

    result = await query.retrieve("How many hours on Netflix?", reinforce=False)

    netflix = next(r for r in result.results if r.name == "Netflix")
    assert netflix.path.edges[0].memory_fact_id == "fact-1"
    assert netflix.path.edges[0].type == "DISCUSSION"
    assert netflix.path.nodes[0].name == "Eric"
    assert netflix.path.nodes[1].name == "Netflix"
    assert netflix.retrieval_mode == "graph"
    assert netflix.path.edges[0].quantities == {
        "value_text": None,
        "value_number": None,
        "value_unit": None,
        "value_kind": None,
        "value_time": None,
        "quantity_value": 10,
        "quantity_unit": "hour",
        "quantity_kind": "duration",
        "time_scope": "last_month",
    }
    assert netflix.memory_facts == [
        {
            "memory_fact_id": "fact-1",
            "family": "DISCUSSION",
            "system_until": None,
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
            "memory_rel_system_until": None,
            "memory_rel_current": True,
            "value_text": None,
            "value_number": None,
            "value_unit": None,
            "value_kind": None,
            "value_time": None,
            "quantity_value": 10,
            "quantity_unit": "hour",
            "quantity_kind": "duration",
            "time_scope": "last_month",
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
async def test_retrieval_hydrates_direct_current_memory_for_seed_entities(monkeypatch):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(query.neo4j_store, "resolve_seed_entity_ids", AsyncMock(return_value=[]))

    class Hit:
        def __init__(self, entity_id, score):
            self.score = score
            self.payload = {"entity_id": entity_id}

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit("travel-id", 0.9), Hit("cube-id", 0.88)]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(ids, include_historical=False, as_of=None):
        assert ids == ["travel-id", "cube-id"]
        return [
            {
                "entity_id": "travel-id",
                "name": "travel",
                "type": "n",
                "access_count": 0,
                "last_accessed": None,
            },
            {
                "entity_id": "cube-id",
                "name": "packing cube",
                "type": "n",
                "access_count": 0,
                "last_accessed": None,
            },
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops, include_historical=False, as_of=None):
        return []

    async def fake_touch_entities(ids, now):
        return None

    async def fake_touch_relations(ids, now):
        return None

    async def fake_hydrate_current_entity_memory(entity_ids, as_of=None):
        assert entity_ids == ["travel-id", "cube-id"]
        return (
            [
                {
                    "memory_fact_id": "fact-1",
                    "family": "HAS_ATTRIBUTE",
                    "system_until": None,
                    "current": True,
                    "fact_key": "fact-key",
                    "slot_key": "slot-key",
                    "subtype": "has",
                    "support_count": 1,
                    "confidence_agg": 1.0,
                    "value_text": None,
                    "value_number": None,
                    "value_unit": None,
                    "value_kind": None,
                    "value_time": None,
                    "quantity_value": None,
                    "quantity_unit": None,
                    "quantity_kind": None,
                    "time_scope": None,
                    "subject_entity_id": "travel-id",
                    "subject_name": "travel",
                    "subject_type": "n",
                    "object_entity_id": "cube-id",
                    "object_name": "packing cube",
                    "object_type": "n",
                    "memory_rel_system_until": None,
                    "memory_rel_current": True,
                }
            ],
            [
                {
                    "memory_fact_id": "fact-1",
                    "assertion_id": "assert-1",
                    "source_kind": "document",
                    "source_id": "doc-1",
                    "raw_subject_text": "travel",
                    "raw_relation_text": "HAS_ATTRIBUTE",
                    "raw_object_text": "packing cube",
                    "family_candidate": "HAS_ATTRIBUTE",
                    "confidence": 1.0,
                    "subtype": "has",
                    "value_text": None,
                    "value_number": None,
                    "value_unit": None,
                    "value_kind": None,
                    "value_time": None,
                    "quantity_value": None,
                    "quantity_unit": None,
                    "quantity_kind": None,
                    "time_scope": None,
                    "status": "active",
                    "created_at": "2026-05-01T00:00:00Z",
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
    monkeypatch.setattr(
        query,
        "_hydrate_current_non_traversable_entity_memory",
        AsyncMock(side_effect=AssertionError("legacy non-traversable hydrator should not run")),
    )
    monkeypatch.setattr(
        query,
        "_hydrate_current_entity_memory",
        fake_hydrate_current_entity_memory,
        raising=False,
    )

    result = await query.retrieve("What travel gear did I mention?", reinforce=False)

    travel = next(r for r in result.results if r.name == "travel")
    cube = next(r for r in result.results if r.name == "packing cube")
    expected_fact = {
        "memory_fact_id": "fact-1",
        "family": "HAS_ATTRIBUTE",
        "system_until": None,
        "current": True,
        "fact_key": "fact-key",
        "slot_key": "slot-key",
        "subtype": "has",
        "support_count": 1,
        "confidence_agg": 1.0,
        "value_text": None,
        "value_number": None,
        "value_unit": None,
        "value_kind": None,
        "value_time": None,
        "quantity_value": None,
        "quantity_unit": None,
        "quantity_kind": None,
        "time_scope": None,
        "subject_entity_id": "travel-id",
        "subject_name": "travel",
        "subject_type": "n",
        "object_entity_id": "cube-id",
        "object_name": "packing cube",
        "object_type": "n",
        "memory_rel_system_until": None,
        "memory_rel_current": True,
    }
    expected_assertion = {
        "memory_fact_id": "fact-1",
        "assertion_id": "assert-1",
        "source_kind": "document",
        "source_id": "doc-1",
        "raw_subject_text": "travel",
        "raw_relation_text": "HAS_ATTRIBUTE",
        "raw_object_text": "packing cube",
        "family_candidate": "HAS_ATTRIBUTE",
        "confidence": 1.0,
        "subtype": "has",
        "value_text": None,
        "value_number": None,
        "value_unit": None,
        "value_kind": None,
        "value_time": None,
        "quantity_value": None,
        "quantity_unit": None,
        "quantity_kind": None,
        "time_scope": None,
        "status": "active",
        "created_at": "2026-05-01T00:00:00Z",
    }
    assert travel.memory_facts == [expected_fact]
    assert travel.supporting_assertions == [expected_assertion]
    assert cube.memory_facts == [expected_fact]
    assert cube.supporting_assertions == [expected_assertion]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_emits_summary_logs_by_default(monkeypatch, caplog):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(query.neo4j_store, "resolve_seed_entity_ids", AsyncMock(return_value=[]))

    class Hit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit(0.9, {"entity_id": "atlas-id"})]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(ids, include_historical=False, as_of=None):
        return [
            {
                "entity_id": "atlas-id",
                "name": "Project Atlas",
                "type": "PROJECT",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops, include_historical=False, as_of=None):
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
    monkeypatch.setattr(
        query, "_hydrate_current_non_traversable_entity_memory", AsyncMock(return_value=([], []))
    )

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
    monkeypatch.setattr(query.neo4j_store, "resolve_seed_entity_ids", AsyncMock(return_value=[]))

    class Hit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit(0.9, {"entity_id": "atlas-id"})]

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(ids, include_historical=False, as_of=None):
        return [
            {
                "entity_id": "atlas-id",
                "name": "Project Atlas",
                "type": "PROJECT",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops, include_historical=False, as_of=None):
        assert seed_ids == ["atlas-id"]
        assert max_hops == 2
        return [
            {
                "seed_id": "atlas-id",
                "seed_name": "Project Atlas",
                "seed_type": "PROJECT",
                "target_id": "postgres-id",
                "target_name": "PostgreSQL",
                "target_type": "DATABASE",
                "distance": 1,
                "path_memory_fact_ids": ["fact-1"],
                "path_edge_types": ["USES"],
                "path_edge_negated": [False],
                "edge_subtypes": [None],
                "edge_ids": ["rel-1"],
                "path_node_names": ["Project Atlas", "PostgreSQL"],
                "path_node_types": ["PROJECT", "DATABASE"],
                "edge_confidences": [0.9],
                "edge_access_counts": [0],
                "edge_last_accessed": [None],
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
    monkeypatch.setattr(query.neo4j_store, "bfs_expand", fail_if_legacy_bfs, raising=False)
    monkeypatch.setattr(query.neo4j_store, "touch_entities", noop_touch)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", noop_touch)
    monkeypatch.setattr(query, "_hydrate_memory_path_details", noop_hydrate)
    monkeypatch.setattr(
        query, "_hydrate_current_non_traversable_entity_memory", AsyncMock(return_value=([], []))
    )

    result = await query.retrieve("What does Project Atlas use?", reinforce=False)

    assert [item.name for item in result.results] == ["Project Atlas", "PostgreSQL"]
    postgres = next(item for item in result.results if item.name == "PostgreSQL")
    assert postgres.path.edges[0].memory_fact_id == "fact-1"
    assert postgres.retrieval_mode == "graph"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_emits_debug_stage_logs_when_requested(monkeypatch, caplog):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])
    monkeypatch.setattr(query.neo4j_store, "resolve_seed_entity_ids", AsyncMock(return_value=[]))

    class Hit:
        def __init__(self, score, payload):
            self.score = score
            self.payload = payload

    async def fake_search_entities_any_type(vector, limit=10):
        return [Hit(0.9, {"entity_id": "atlas-id"})]

    async def fake_search_chunks(vector, limit=10):
        return [
            Hit(
                0.7,
                {
                    "chunk_id": "chunk-1",
                    "text": "Project Atlas uses PostgreSQL.",
                    "doc_id": "doc-1",
                    "source_doc": "atlas-doc",
                    "position": 0,
                },
            )
        ]

    async def fake_get_entities_from_chunks(chunk_ids):
        return [{"entity_id": "atlas-id", "chunk_eids": chunk_ids}]

    async def fake_hydrate_entities(ids, include_historical=False, as_of=None):
        return [
            {
                "entity_id": "atlas-id",
                "name": "Project Atlas",
                "type": "PROJECT",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops, include_historical=False, as_of=None):
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
    monkeypatch.setattr(
        query, "_hydrate_current_non_traversable_entity_memory", AsyncMock(return_value=([], []))
    )

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
        as_of=None,
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
        as_of=None,
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
                    entity_id="atlas-id",
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
            as_of=None,
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
@pytest.mark.unit
async def test_query_cli_threads_as_of_flag(monkeypatch, capsys):
    from argparse import Namespace

    from landscape.cli import query as query_cli
    from landscape.retrieval.query import RetrievalResult

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
        as_of=None,
        log_context=None,
    ):
        calls.append({"as_of": as_of})
        return RetrievalResult(
            query=query_text,
            results=[],
            touched_entity_ids=[],
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
            text="q",
            hops=2,
            limit=10,
            no_reinforce=False,
            debug=False,
            include_historical=False,
            as_of="2023-06-01T00:00:00+00:00",
        )
    )

    assert exit_code == 0
    assert calls == [{"as_of": "2023-06-01T00:00:00+00:00"}]
    capsys.readouterr()


def test_cli_as_of_argparse_rejects_garbage(capsys):
    import argparse

    from landscape.cli import query as query_cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    query_cli.register(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["query", "hello", "--as-of", "yesterday"])
    err = capsys.readouterr().err
    assert "--as-of" in err and ("ISO" in err or "iso" in err.lower())


def test_cli_as_of_argparse_normalizes_naive_to_utc():
    import argparse

    from landscape.cli import query as query_cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    query_cli.register(subparsers)

    args = parser.parse_args(["query", "hello", "--as-of", "2023-06-01T00:00:00"])
    assert args.as_of == "2023-06-01T00:00:00+00:00"


def test_api_normalize_as_of_naive_becomes_utc():
    from datetime import datetime

    from landscape.api.query import _normalize_as_of
    naive = datetime(2023, 6, 1, 0, 0, 0)
    assert _normalize_as_of(naive) == "2023-06-01T00:00:00+00:00"


def test_api_normalize_as_of_aware_converted_to_utc():
    from datetime import datetime, timedelta, timezone

    from landscape.api.query import _normalize_as_of
    aware_pst = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
    assert _normalize_as_of(aware_pst) == "2023-06-01T08:00:00+00:00"


def test_api_normalize_as_of_none_passes_through():
    from landscape.api.query import _normalize_as_of
    assert _normalize_as_of(None) is None


@pytest.mark.asyncio
async def test_mcp_search_rejects_garbage_as_of(monkeypatch):
    import landscape.mcp_app as mcp_mod
    from landscape.mcp_app import search

    async def fake_retrieve(*a, **kw):
        raise AssertionError("retrieve should not be called for garbage as_of")

    monkeypatch.setattr(mcp_mod, "require_current_scope", lambda *a, **kw: None)
    monkeypatch.setattr("landscape.retrieval.query.retrieve", fake_retrieve)

    with pytest.raises(ValueError, match="ISO-8601"):
        await search(query="hello", as_of="yesterday")


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
        entity_id=robert_id,
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
        entity_id=acme_id,
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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_seeds_canonical_entity_from_alias_resolution(monkeypatch):
    from landscape.retrieval import query

    monkeypatch.setattr(query.encoder, "embed_query", lambda text: [0.1, 0.2])

    async def fake_resolve_seed_entity_ids(query_text):
        assert query_text == "What is Bob working on?"
        return ["robert-id"]

    async def fake_search_entities_any_type(vector, limit=10):
        return []

    async def fake_search_chunks(vector, limit=10):
        return []

    async def fake_get_entities_from_chunks(chunk_ids):
        return []

    async def fake_hydrate_entities(entity_ids, include_historical=False, as_of=None):
        assert entity_ids == ["robert-id"]
        return [
            {
                "entity_id": "robert-id",
                "name": "Robert",
                "type": "PERSON",
                "access_count": 0,
                "last_accessed": None,
            }
        ]

    async def fake_bfs_expand_memory_rel(seed_ids, max_hops, include_historical=False, as_of=None):
        assert seed_ids == ["robert-id"]
        return []

    async def noop_touch(*args, **kwargs):
        return None

    monkeypatch.setattr(
        query.neo4j_store,
        "resolve_seed_entity_ids",
        fake_resolve_seed_entity_ids,
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
    monkeypatch.setattr(query.neo4j_store, "touch_entities", noop_touch)
    monkeypatch.setattr(query.neo4j_store, "touch_relations", noop_touch)
    monkeypatch.setattr(
        query, "_hydrate_current_non_traversable_entity_memory", AsyncMock(return_value=([], []))
    )

    result = await query.retrieve("What is Bob working on?", reinforce=False)

    assert [item.entity_id for item in result.results] == ["robert-id"]
    assert result.results[0].name == "Robert"


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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bfs_expand_memory_rel_filters_by_as_of(neo4j_driver):
    """Build three edges with effective ranges 2020-2022, 2022-2024, 2024-NULL.
    as_of=2023-01-01 returns the middle one only; as_of=None returns all live;
    as_of=2025 returns the last."""
    from landscape.storage import neo4j_store

    subj = "AsOfAlice"
    objs = ["AsOfAcmeA", "AsOfAcmeB", "AsOfAcmeC"]
    ranges = [
        ("2020-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00"),
        ("2022-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
        ("2024-01-01T00:00:00+00:00", None),
    ]

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj] + objs,
        )
        subj_id = (
            await (
                await session.run(
                    "CREATE (e:Entity {id: $id, name: $n, type: 'PERSON', canonical: true}) "
                    "RETURN e.id AS id",
                    id="entity:asof-subj", n=subj,
                )
            ).single()
        )["id"]
        for i, (name, (ef, eu)) in enumerate(zip(objs, ranges, strict=True)):
            obj_id = (
                await (
                    await session.run(
                        "CREATE (e:Entity {id: $id, name: $n, type: 'ORG', canonical: true}) "
                        "RETURN e.id AS id",
                        id=f"entity:asof-obj-{i}", n=name,
                    )
                ).single()
            )["id"]
            await session.run(
                """
                MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid})
                CREATE (s)-[:MEMORY_REL {
                    memory_fact_id: $fid, family: 'WORKS_FOR',
                    ingested_at: '2026-05-25T00:00:00+00:00',
                    system_until: null,
                    effective_from: $ef, effective_until: $eu,
                    confidence_agg: 0.9, subject_entity_id: $sid, object_entity_id: $oid,
                    access_count: 0, last_accessed: null, updated_at: '2026-05-25T00:00:00+00:00',
                    negated: false
                }]->(o)
                """,
                sid=subj_id, oid=obj_id, fid=f"fact:asof-{i}", ef=ef, eu=eu,
            )

    rows = await neo4j_store.bfs_expand_memory_rel(
        [subj_id], max_hops=1, as_of="2023-01-01T00:00:00+00:00"
    )
    assert {r["target_name"] for r in rows} == {"AsOfAcmeB"}

    rows = await neo4j_store.bfs_expand_memory_rel([subj_id], max_hops=1)
    assert {r["target_name"] for r in rows} == set(objs)

    rows = await neo4j_store.bfs_expand_memory_rel(
        [subj_id], max_hops=1, as_of="2025-06-01T00:00:00+00:00"
    )
    assert {r["target_name"] for r in rows} == {"AsOfAcmeC"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bfs_expand_memory_rel_as_of_includes_null_effective_from(neo4j_driver):
    """Edges with NULL effective_from must surface for any as_of (permissive)."""
    from landscape.storage import neo4j_store

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run("MATCH (e:Entity {id: 'entity:null-eff-s'}) DETACH DELETE e")
        await session.run("MATCH (e:Entity {id: 'entity:null-eff-o'}) DETACH DELETE e")
        await session.run(
            "CREATE (s:Entity {id: 'entity:null-eff-s', name: 'NullEffSubj', "
            "  type: 'PERSON', canonical: true}), "
            "       (o:Entity {id: 'entity:null-eff-o', name: 'NullEffObj', "
            "  type: 'ORG', canonical: true}) "
            "CREATE (s)-[:MEMORY_REL {memory_fact_id: 'fact:null-eff', family: 'WORKS_FOR', "
            "  ingested_at: '2026-01-01T00:00:00+00:00', system_until: null, "
            "  effective_from: null, effective_until: null, confidence_agg: 0.9, "
            "  subject_entity_id: 'entity:null-eff-s', object_entity_id: 'entity:null-eff-o', "
            "  access_count: 0, last_accessed: null, updated_at: '2026-01-01T00:00:00+00:00', "
            "  negated: false}]->(o)"
        )
    rows = await neo4j_store.bfs_expand_memory_rel(
        ["entity:null-eff-s"], max_hops=1, as_of="2010-01-01T00:00:00+00:00"
    )
    assert {r["target_name"] for r in rows} == {"NullEffObj"}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_rankable_entities_as_of_surfaces_superseded_entities(neo4j_driver):
    """An entity whose only edges are system-superseded but event-time-valid at
    as_of must still be rankable. Supersession here represents 'world moved on',
    not 'we were wrong' — the historical belief is still current."""
    from landscape.storage import neo4j_store

    subj = "AsOfHydAlice"
    obj = "AsOfHydAcme"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, obj],
        )
        subj_row = await (
            await session.run(
                "CREATE (e:Entity {id: $id, name: $n, type: 'PERSON', canonical: true}) "
                "RETURN e.id AS id",
                id="entity:asof-hyd-subj", n=subj,
            )
        ).single()
        obj_row = await (
            await session.run(
                "CREATE (e:Entity {id: $id, name: $n, type: 'ORG', canonical: true}) "
                "RETURN e.id AS id",
                id="entity:asof-hyd-obj", n=obj,
            )
        ).single()
        # Edge is system-superseded (system_until set) but event-time
        # covers 2020-06.
        await session.run(
            """
            MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid})
            CREATE (s)-[:MEMORY_REL {
                memory_fact_id: 'fact:asof-hyd', family: 'WORKS_FOR',
                ingested_at: '2026-01-01T00:00:00+00:00',
                system_until: '2026-02-01T00:00:00+00:00',
                effective_from: '2018-01-01T00:00:00+00:00',
                effective_until: '2021-03-01T00:00:00+00:00',
                confidence_agg: 0.9, subject_entity_id: $sid, object_entity_id: $oid,
                access_count: 0, last_accessed: null,
                updated_at: '2026-01-01T00:00:00+00:00', negated: false
            }]->(o)
            """,
            sid=subj_row["id"], oid=obj_row["id"],
        )

    # as_of within the superseded edge's effective window — Acme should be
    # rankable even with include_historical=False (the as_of itself should
    # override system-time gating).
    rows = await neo4j_store.get_rankable_entities(
        [subj_row["id"], obj_row["id"]],
        as_of="2020-06-01T00:00:00+00:00",
    )
    names = {r["name"] for r in rows}
    assert names == {subj, obj}

    # as_of outside the window — both endpoints drop. Their only edge is
    # event-time-invalid at 2025, and total_edges counts that edge for both,
    # so neither passes total_edges=0 OR effective_at_as_of>0.
    rows = await neo4j_store.get_rankable_entities(
        [subj_row["id"], obj_row["id"]],
        as_of="2025-06-01T00:00:00+00:00",
    )
    names = {r["name"] for r in rows}
    assert names == set()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_bfs_expand_memory_rel_as_of_surfaces_superseded_edges(neo4j_driver):
    """A superseded MEMORY_REL whose effective range covers as_of must still
    surface in BFS traversal. as_of overrides system-time gating."""
    from landscape.storage import neo4j_store

    subj_name = "BfsAsOfAlice"
    obj_name = "BfsAsOfAcme"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj_name, obj_name],
        )
        subj_id = (
            await (
                await session.run(
                    "CREATE (e:Entity {id: $id, name: $n, type: 'PERSON', canonical: true}) "
                    "RETURN e.id AS id",
                    id="entity:bfs-asof-subj", n=subj_name,
                )
            ).single()
        )["id"]
        obj_id = (
            await (
                await session.run(
                    "CREATE (e:Entity {id: $id, name: $n, type: 'ORG', canonical: true}) "
                    "RETURN e.id AS id",
                    id="entity:bfs-asof-obj", n=obj_name,
                )
            ).single()
        )["id"]
        await session.run(
            """
            MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid})
            CREATE (s)-[:MEMORY_REL {
                memory_fact_id: 'fact:bfs-asof', family: 'WORKS_FOR',
                ingested_at: '2026-01-01T00:00:00+00:00',
                system_until: '2026-02-01T00:00:00+00:00',
                effective_from: '2018-01-01T00:00:00+00:00',
                effective_until: '2021-03-01T00:00:00+00:00',
                confidence_agg: 0.9, subject_entity_id: $sid, object_entity_id: $oid,
                access_count: 0, last_accessed: null,
                updated_at: '2026-01-01T00:00:00+00:00', negated: false
            }]->(o)
            """,
            sid=subj_id, oid=obj_id,
        )

    rows = await neo4j_store.bfs_expand_memory_rel(
        [subj_id], max_hops=1, as_of="2020-06-01T00:00:00+00:00"
    )
    assert {r["target_name"] for r in rows} == {obj_name}

    rows = await neo4j_store.bfs_expand_memory_rel(
        [subj_id], max_hops=1, as_of="2025-06-01T00:00:00+00:00"
    )
    assert rows == []

    # Without as_of, the superseded edge stays hidden by default.
    rows = await neo4j_store.bfs_expand_memory_rel([subj_id], max_hops=1)
    assert rows == []


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_current_fact_details_filters_by_as_of(neo4j_driver):
    """Per-entity hydration must respect as_of so the response payload's
    facts list reflects the queried moment, not the system-current state."""
    from landscape.storage import neo4j_store

    subj = "EntHydAlice"
    old_obj = "EntHydAcme"
    new_obj = "EntHydGamma"
    old_eff_from = "2018-01-01T00:00:00+00:00"
    old_eff_until = "2023-11-01T00:00:00+00:00"
    new_eff_from = "2023-11-01T00:00:00+00:00"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        # Clean slate
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, old_obj, new_obj],
        )
        await session.run(
            "MATCH (f:MemoryFact) WHERE f.id IN $ids DETACH DELETE f",
            ids=["fact:ent-hyd-old", "fact:ent-hyd-new"],
        )

        # Entities
        create_entity = (
            "CREATE (e:Entity {id: $id, name: $n, type: $t, canonical: true}) "
            "RETURN e.id AS id"
        )
        subj_id = (await (await session.run(
            create_entity, id="entity:ent-hyd-subj", n=subj, t="PERSON",
        )).single())["id"]
        old_obj_id = (await (await session.run(
            create_entity, id="entity:ent-hyd-old-obj", n=old_obj, t="ORG",
        )).single())["id"]
        new_obj_id = (await (await session.run(
            create_entity, id="entity:ent-hyd-new-obj", n=new_obj, t="ORG",
        )).single())["id"]

        # Old MemoryFact (system-superseded, event-time 2018-2023)
        await session.run(
            """
            MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid})
            CREATE (f:MemoryFact {
                id: 'fact:ent-hyd-old', family: 'WORKS_FOR',
                ingested_at: '2026-01-01T00:00:00+00:00',
                system_until: '2026-02-01T00:00:00+00:00',
                effective_from: $ef, effective_until: $eu,
                confidence_agg: 0.9, support_count: 1, negated: false,
                fact_key: 'fk:old', slot_key: 'sk:old',
                subtype: null, value_text: null, value_number: null, value_unit: null,
                value_kind: null, value_time: null, quantity_value: null,
                quantity_unit: null, quantity_kind: null, time_scope: null
            })
            CREATE (s)-[:AS_SUBJECT]->(f)
            CREATE (f)-[:AS_OBJECT]->(o)
            CREATE (s)-[:MEMORY_REL {memory_fact_id: 'fact:ent-hyd-old', family: 'WORKS_FOR',
                ingested_at: '2026-01-01T00:00:00+00:00',
                system_until: '2026-02-01T00:00:00+00:00',
                effective_from: $ef, effective_until: $eu,
                confidence_agg: 0.9, subject_entity_id: $sid, object_entity_id: $oid,
                access_count: 0, last_accessed: null,
                updated_at: '2026-01-01T00:00:00+00:00', negated: false}]->(o)
            """,
            sid=subj_id, oid=old_obj_id, ef=old_eff_from, eu=old_eff_until,
        )
        # New MemoryFact (current)
        await session.run(
            """
            MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid})
            CREATE (f:MemoryFact {
                id: 'fact:ent-hyd-new', family: 'WORKS_FOR',
                ingested_at: '2026-02-01T00:00:00+00:00',
                system_until: null,
                effective_from: $ef, effective_until: null,
                confidence_agg: 0.95, support_count: 1, negated: false,
                fact_key: 'fk:new', slot_key: 'sk:new',
                subtype: null, value_text: null, value_number: null, value_unit: null,
                value_kind: null, value_time: null, quantity_value: null,
                quantity_unit: null, quantity_kind: null, time_scope: null
            })
            CREATE (s)-[:AS_SUBJECT]->(f)
            CREATE (f)-[:AS_OBJECT]->(o)
            CREATE (s)-[:MEMORY_REL {memory_fact_id: 'fact:ent-hyd-new', family: 'WORKS_FOR',
                ingested_at: '2026-02-01T00:00:00+00:00',
                system_until: null,
                effective_from: $ef, effective_until: null,
                confidence_agg: 0.95, subject_entity_id: $sid, object_entity_id: $oid,
                access_count: 0, last_accessed: null,
                updated_at: '2026-02-01T00:00:00+00:00', negated: false}]->(o)
            """,
            sid=subj_id, oid=new_obj_id, ef=new_eff_from,
        )

    # Without as_of: only the system-current new fact.
    facts, _ = await neo4j_store.get_current_fact_details_for_entities([subj_id])
    fact_ids = {f["memory_fact_id"] for f in facts}
    assert fact_ids == {"fact:ent-hyd-new"}

    # as_of inside old window: only the old fact (event-time-valid), even though
    # it is system-superseded.
    facts, _ = await neo4j_store.get_current_fact_details_for_entities(
        [subj_id], as_of="2020-06-01T00:00:00+00:00"
    )
    fact_ids = {f["memory_fact_id"] for f in facts}
    assert fact_ids == {"fact:ent-hyd-old"}

    # as_of after both effective_from boundaries: only the new fact.
    facts, _ = await neo4j_store.get_current_fact_details_for_entities(
        [subj_id], as_of="2025-01-01T00:00:00+00:00"
    )
    fact_ids = {f["memory_fact_id"] for f in facts}
    assert fact_ids == {"fact:ent-hyd-new"}
