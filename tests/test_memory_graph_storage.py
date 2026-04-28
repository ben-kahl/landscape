import pytest

from landscape.memory_graph import AssertionPayload, FAMILY_REGISTRY, slot_key
from landscape.storage import neo4j_store


async def _entity_app_id(neo4j_driver, entity_element_id: str) -> str:
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)
            WHERE elementId(e) = $eid
            RETURN e.id AS entity_id
            """,
            eid=entity_element_id,
        )
        record = await result.single()
        assert record is not None
        return record["entity_id"]


@pytest.mark.asyncio
async def test_merge_assertion_is_idempotent(neo4j_driver):
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-test",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        confidence=0.9,
        family_candidate="WORKS_FOR",
    )
    first = await neo4j_store.merge_assertion(payload)
    second = await neo4j_store.merge_assertion(payload)
    assert first == second
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Assertion {id: $assertion_id})
            RETURN count(a) AS count
            """,
            assertion_id=first,
        )
        record = await result.single()
        assert record is not None
        assert record["count"] == 1


@pytest.mark.asyncio
async def test_superseding_single_current_fact_replaces_memory_rel(neo4j_driver):
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    beacon = await neo4j_store.merge_entity("Beacon", "Organization", "doc-b", 0.9)
    alice_id = await _entity_app_id(neo4j_driver, alice)
    acme_id = await _entity_app_id(neo4j_driver, acme)
    beacon_id = await _entity_app_id(neo4j_driver, beacon)
    first = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=acme_id,
        subtype=None,
        confidence=0.9,
        assertion_id="assertion:1",
    )
    await neo4j_store.materialize_memory_rel(first)
    family_cfg = FAMILY_REGISTRY["WORKS_FOR"]
    slot = slot_key(family_cfg, alice_id, acme_id, None)
    second = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=beacon_id,
        subtype=None,
        confidence=0.95,
        assertion_id="assertion:2",
    )
    explanation = await neo4j_store.get_memory_fact_explanation(second)
    assert explanation["family"] == "WORKS_FOR"
    assert explanation["current"] is True
    assert explanation["object_name"] == "Beacon"
    async with neo4j_driver.session() as session:
        old_fact_result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fact_id})
            RETURN f.current AS current
            """,
            fact_id=first,
        )
        old_fact = await old_fact_result.single()
        assert old_fact is not None
        assert old_fact["current"] is False

        old_rel_result = await session.run(
            """
            MATCH (:Entity {id: $subject_id})-[r:MEMORY_REL {memory_fact_id: $fact_id}]->()
            RETURN r.current AS current
            """,
            subject_id=alice_id,
            fact_id=first,
        )
        old_rel = await old_rel_result.single()
        assert old_rel is not None
        assert old_rel["current"] is False

        current_count_result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key, current: true})
            RETURN count(f) AS count
            """,
            slot_key=slot,
        )
        current_count = await current_count_result.single()
        assert current_count is not None
        assert current_count["count"] == 1


@pytest.mark.asyncio
async def test_bfs_expand_memory_rel_uses_current_edges_only(neo4j_driver):
    rows = await neo4j_store.bfs_expand_memory_rel([], max_hops=2)
    assert rows == []
