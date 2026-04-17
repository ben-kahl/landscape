"""Tests for get_entities_in_conversation and get_entities_since storage primitives."""
from datetime import UTC, datetime, timedelta

import pytest


@pytest.mark.asyncio
async def test_get_entities_in_conversation_returns_only_that_conversations_entities(
    http_client, neo4j_driver
):
    from landscape.storage import neo4j_store

    # Conversation A
    turn_a_eid, _ = await neo4j_store.merge_turn("conv-csr-a", "t1")
    eid_a = await neo4j_store.merge_entity(
        "EntityConvA", "PERSON", "doc-csr-a", 0.9, doc_element_id=None, model="test"
    )
    await neo4j_store.link_entity_to_turn(eid_a, turn_a_eid)

    # Conversation B
    turn_b_eid, _ = await neo4j_store.merge_turn("conv-csr-b", "t1")
    eid_b = await neo4j_store.merge_entity(
        "EntityConvB", "PERSON", "doc-csr-b", 0.9, doc_element_id=None, model="test"
    )
    await neo4j_store.link_entity_to_turn(eid_b, turn_b_eid)

    result_a = await neo4j_store.get_entities_in_conversation("conv-csr-a")
    result_b = await neo4j_store.get_entities_in_conversation("conv-csr-b")

    assert eid_a in result_a
    assert eid_b not in result_a

    assert eid_b in result_b
    assert eid_a not in result_b


@pytest.mark.asyncio
async def test_get_entities_in_conversation_unknown_session_returns_empty(
    http_client, neo4j_driver
):
    from landscape.storage import neo4j_store

    result = await neo4j_store.get_entities_in_conversation("conv-csr-nonexistent-999")
    assert result == []


@pytest.mark.asyncio
async def test_get_entities_since_filters_by_timestamp(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    # Create two turns and two entities
    turn_old_eid, _ = await neo4j_store.merge_turn("conv-csr-since", "t-old")
    turn_new_eid, _ = await neo4j_store.merge_turn("conv-csr-since", "t-new")

    eid_old = await neo4j_store.merge_entity(
        "EntityOld", "PERSON", "doc-csr-since", 0.9, doc_element_id=None, model="test"
    )
    eid_new = await neo4j_store.merge_entity(
        "EntityNew", "PERSON", "doc-csr-since", 0.9, doc_element_id=None, model="test"
    )

    await neo4j_store.link_entity_to_turn(eid_old, turn_old_eid)
    await neo4j_store.link_entity_to_turn(eid_new, turn_new_eid)

    now = datetime.now(UTC)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()
    just_now = now.isoformat()

    # Directly set timestamps via Cypher to control the values precisely
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (t:Turn {id: 'conv-csr-since:t-old'}) SET t.timestamp = $iso",
            iso=two_hours_ago,
        )
        await session.run(
            "MATCH (t:Turn {id: 'conv-csr-since:t-new'}) SET t.timestamp = $iso",
            iso=just_now,
        )

    since = now - timedelta(hours=1)
    result = await neo4j_store.get_entities_since(since)

    assert eid_new in result
    assert eid_old not in result
