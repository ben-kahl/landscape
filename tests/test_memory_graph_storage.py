import pytest

from landscape.memory_graph import AssertionPayload
from landscape.storage import neo4j_store


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


@pytest.mark.asyncio
async def test_superseding_single_current_fact_replaces_memory_rel(neo4j_driver):
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    beacon = await neo4j_store.merge_entity("Beacon", "Organization", "doc-b", 0.9)
    first = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice,
        object_entity_id=acme,
        subtype=None,
        confidence=0.9,
        assertion_id="assertion:1",
    )
    await neo4j_store.materialize_memory_rel(first)
    second = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice,
        object_entity_id=beacon,
        subtype=None,
        confidence=0.95,
        assertion_id="assertion:2",
    )
    explanation = await neo4j_store.get_memory_fact_explanation(second)
    assert explanation["family"] == "WORKS_FOR"
    assert explanation["current"] is True
    assert explanation["object_name"] == "Beacon"


@pytest.mark.asyncio
async def test_bfs_expand_memory_rel_uses_current_edges_only(neo4j_driver):
    rows = await neo4j_store.bfs_expand_memory_rel([], max_hops=2)
    assert rows == []
