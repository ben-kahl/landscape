"""Tests for get_entities_in_conversation, get_entities_since storage primitives,
and the retrieve() filter wiring that uses them."""
from datetime import UTC, datetime, timedelta

import pytest

pytestmark = pytest.mark.integration


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


# ---------------------------------------------------------------------------
# Retrieval-layer filter tests
# ---------------------------------------------------------------------------


async def _seed_entity_to_qdrant(
    neo4j_store,
    qdrant_store,
    encoder,
    name: str,
    entity_type: str,
    source_doc: str,
    session_id: str,
    turn_id: str,
) -> str:
    """Create an Entity in Neo4j, link it to a Conversation turn, and upsert
    its embedding into Qdrant so it is reachable via vector search seeds."""
    turn_eid, _ = await neo4j_store.merge_turn(session_id, turn_id)
    eid = await neo4j_store.merge_entity(
        name, entity_type, source_doc, 0.9, doc_element_id=None, model="test"
    )
    await neo4j_store.link_entity_to_turn(eid, turn_eid)
    vector = encoder.embed_query(name)
    await qdrant_store.upsert_entity(
        neo4j_element_id=eid,
        name=name,
        entity_type=entity_type,
        source_doc=source_doc,
        timestamp=datetime.now(UTC).isoformat(),
        vector=vector,
    )
    return eid


@pytest.mark.asyncio
async def test_retrieve_filtered_by_session_id_returns_only_that_conversations_entities(
    http_client, neo4j_driver
):
    """retrieve(session_id=A) should only surface entities from conversation A."""
    from landscape.embeddings import encoder
    from landscape.retrieval.query import retrieve
    from landscape.storage import neo4j_store, qdrant_store

    eid_a = await _seed_entity_to_qdrant(
        neo4j_store, qdrant_store, encoder,
        "SessionFilterAlice", "PERSON", "doc-sf-a",
        session_id="conv-sf-a", turn_id="t1",
    )
    eid_b = await _seed_entity_to_qdrant(
        neo4j_store, qdrant_store, encoder,
        "SessionFilterBob", "PERSON", "doc-sf-b",
        session_id="conv-sf-b", turn_id="t1",
    )

    result_a = await retrieve(
        "SessionFilterAlice",
        hops=1,
        limit=10,
        reinforce=False,
        session_id="conv-sf-a",
    )
    result_b = await retrieve(
        "SessionFilterBob",
        hops=1,
        limit=10,
        reinforce=False,
        session_id="conv-sf-b",
    )

    ids_a = {r.neo4j_id for r in result_a.results}
    ids_b = {r.neo4j_id for r in result_b.results}

    assert eid_a in ids_a, f"EntityA should be in conv-sf-a results, got: {ids_a}"
    assert eid_b not in ids_a, "EntityB must not bleed into conv-sf-a results"

    assert eid_b in ids_b, f"EntityB should be in conv-sf-b results, got: {ids_b}"
    assert eid_a not in ids_b, "EntityA must not bleed into conv-sf-b results"


@pytest.mark.asyncio
async def test_retrieve_filtered_by_since_excludes_old_turn(
    http_client, neo4j_driver
):
    """retrieve(since=...) should exclude entities whose turn timestamp predates the cutoff."""
    from landscape.embeddings import encoder
    from landscape.retrieval.query import retrieve
    from landscape.storage import neo4j_store, qdrant_store

    now = datetime.now(UTC)
    two_hours_ago_iso = (now - timedelta(hours=2)).isoformat()
    just_now_iso = now.isoformat()

    # Old entity (turn 2h ago)
    eid_old = await _seed_entity_to_qdrant(
        neo4j_store, qdrant_store, encoder,
        "SinceFilterOldEntity", "PERSON", "doc-since-old",
        session_id="conv-since-filter", turn_id="t-old",
    )
    # New entity (turn just now)
    eid_new = await _seed_entity_to_qdrant(
        neo4j_store, qdrant_store, encoder,
        "SinceFilterNewEntity", "PERSON", "doc-since-new",
        session_id="conv-since-filter", turn_id="t-new",
    )

    # Backdate the old turn directly via Cypher
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (t:Turn {id: 'conv-since-filter:t-old'}) SET t.timestamp = $iso",
            iso=two_hours_ago_iso,
        )
        await session.run(
            "MATCH (t:Turn {id: 'conv-since-filter:t-new'}) SET t.timestamp = $iso",
            iso=just_now_iso,
        )

    since_cutoff = now - timedelta(hours=1)
    result = await retrieve(
        "SinceFilterNewEntity", hops=1, limit=10, reinforce=False, since=since_cutoff
    )
    result_ids = {r.neo4j_id for r in result.results}

    assert eid_new in result_ids, f"Recent entity should surface, got: {result_ids}"
    assert eid_old not in result_ids, (
        f"Old entity must be excluded by since filter, got: {result_ids}"
    )


@pytest.mark.asyncio
async def test_retrieve_with_empty_allowlist_returns_empty(
    http_client, neo4j_driver
):
    """retrieve(session_id=<unknown>) should return an empty result, not fall back to unfiltered."""
    from landscape.retrieval.query import retrieve

    result = await retrieve(
        "anything", hops=1, limit=10, reinforce=False,
        session_id="conv-nonexistent-allowlist-999",
    )
    assert result.results == [], (
        f"Expected empty results for unknown session_id, got: {result.results}"
    )


@pytest.mark.asyncio
async def test_retrieve_unfiltered_returns_both(
    http_client, neo4j_driver
):
    """retrieve() with no session_id/since kwargs should reach entities from both conversations."""
    from landscape.embeddings import encoder
    from landscape.retrieval.query import retrieve
    from landscape.storage import neo4j_store, qdrant_store

    eid_a = await _seed_entity_to_qdrant(
        neo4j_store, qdrant_store, encoder,
        "UnfilteredEntityAlpha", "PERSON", "doc-uf-a",
        session_id="conv-uf-a", turn_id="t1",
    )
    eid_b = await _seed_entity_to_qdrant(
        neo4j_store, qdrant_store, encoder,
        "UnfilteredEntityBeta", "PERSON", "doc-uf-b",
        session_id="conv-uf-b", turn_id="t1",
    )

    result_a = await retrieve("UnfilteredEntityAlpha", hops=1, limit=10, reinforce=False)
    result_b = await retrieve("UnfilteredEntityBeta", hops=1, limit=10, reinforce=False)

    ids_a = {r.neo4j_id for r in result_a.results}
    ids_b = {r.neo4j_id for r in result_b.results}

    assert eid_a in ids_a, f"EntityAlpha must be reachable without filter, got: {ids_a}"
    assert eid_b in ids_b, f"EntityBeta must be reachable without filter, got: {ids_b}"
