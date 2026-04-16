"""Tests for src/landscape/writeback.py.

Uses the same per-test isolation fixture as the rest of the suite
(conftest._isolated_test): Neo4j is wiped and Qdrant collections dropped
before each test, so tests are fully independent.
"""
import pytest


@pytest.mark.asyncio
async def test_add_entity_creates_when_no_match(http_client, neo4j_driver):
    """Fresh DB: add_entity should create the node and return resolved_to_existing=False."""
    from landscape.writeback import add_entity

    result = await add_entity(
        "Acme Corp",
        "Organization",
        source="agent:test:1",
        session_id="s1",
        turn_id="t1",
    )

    assert result.resolved_to_existing is False
    assert result.canonical_name == "Acme Corp"
    assert result.entity_id  # non-empty string

    # Verify node was written with created_by="agent"
    async with neo4j_driver.session() as session:
        rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Acme Corp'}) "
                "RETURN e.created_by AS cb, e.session_id AS sid, e.turn_id AS tid"
            )
        ).single()

    assert rec is not None
    assert rec["cb"] == "agent"
    assert rec["sid"] == "s1"
    assert rec["tid"] == "t1"


@pytest.mark.asyncio
async def test_add_entity_resolves_to_existing(http_client, neo4j_driver):
    """Pre-create 'Acme Corporation'; add_entity('Acme Corp') should resolve to it."""
    from landscape.storage import neo4j_store, qdrant_store
    from landscape.embeddings import encoder
    from datetime import UTC, datetime
    from landscape.writeback import add_entity

    # Pre-create the canonical entity exactly as the pipeline does
    doc_id, _ = await neo4j_store.merge_document("hash-wb-r2", "wb-doc-2", "text")
    entity_id = await neo4j_store.merge_entity(
        "Acme Corporation", "Organization", "wb-doc-2", 0.9, doc_id, "test"
    )
    vector = encoder.encode("Acme Corporation (Organization)")
    await qdrant_store.upsert_entity(
        neo4j_element_id=entity_id,
        name="Acme Corporation",
        entity_type="Organization",
        source_doc="wb-doc-2",
        timestamp=datetime.now(UTC).isoformat(),
        vector=vector,
    )

    # Now call add_entity with a slightly different (alias) name
    result = await add_entity(
        "Acme Corp",
        "Organization",
        source="agent:test:2",
    )

    assert result.resolved_to_existing is True
    assert result.entity_id == entity_id
    # canonical_name should reflect the stored entity's name
    assert result.canonical_name == "Acme Corporation"


@pytest.mark.asyncio
async def test_add_relation_creates_endpoints_if_missing(http_client, neo4j_driver):
    """Fresh DB: add_relation should auto-create both endpoints with created_by='agent'."""
    from landscape.writeback import add_relation

    result = await add_relation(
        "Diana Prince",
        "Themyscira Corp",
        "WORKS_FOR",
        source="agent:test:3",
        session_id="s3",
        turn_id="t3",
    )

    assert result.outcome in ("created", "reinforced", "superseded")
    assert result.subject_id
    assert result.object_id

    async with neo4j_driver.session() as session:
        # Both endpoints should exist and be agent-created
        subj_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Diana Prince'}) RETURN e.created_by AS cb"
            )
        ).single()
        obj_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Themyscira Corp'}) RETURN e.created_by AS cb"
            )
        ).single()
        edge_rec = await (
            await session.run(
                "MATCH (s:Entity {name: 'Diana Prince'})-[r:RELATES_TO]->(o:Entity {name: 'Themyscira Corp'}) "
                "WHERE r.valid_until IS NULL "
                "RETURN r.created_by AS cb, r.session_id AS sid"
            )
        ).single()

    assert subj_rec is not None and subj_rec["cb"] == "agent"
    assert obj_rec is not None and obj_rec["cb"] == "agent"
    assert edge_rec is not None and edge_rec["cb"] == "agent"
    assert edge_rec["sid"] == "s3"


@pytest.mark.asyncio
async def test_add_relation_normalizes_rel_type(http_client, neo4j_driver):
    """EMPLOYED_BY synonym should be stored as WORKS_FOR."""
    from landscape.writeback import add_relation

    result = await add_relation(
        "Clark Kent",
        "Daily Planet",
        "EMPLOYED_BY",          # synonym → WORKS_FOR
        source="agent:test:4",
    )

    assert result.outcome in ("created", "reinforced", "superseded")

    async with neo4j_driver.session() as session:
        edge_rec = await (
            await session.run(
                "MATCH (:Entity {name: 'Clark Kent'})-[r:RELATES_TO]->(:Entity {name: 'Daily Planet'}) "
                "WHERE r.valid_until IS NULL "
                "RETURN r.type AS rel_type"
            )
        ).single()

    assert edge_rec is not None
    assert edge_rec["rel_type"] == "WORKS_FOR"


@pytest.mark.asyncio
async def test_add_relation_supersedes_when_functional(http_client, neo4j_driver):
    """Functional rel (WORKS_FOR): second add_relation with different object supersedes."""
    from landscape.storage import neo4j_store
    from landscape.writeback import add_relation

    # Pre-create entities and the original edge via neo4j_store directly
    doc_id, _ = await neo4j_store.merge_document("hash-wb-sup", "wb-sup-doc", "text")
    await neo4j_store.merge_entity("Alice", "Person", "wb-sup-doc", 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Acme", "Organization", "wb-sup-doc", 0.9, doc_id, "test")
    outcome1, _ = await neo4j_store.upsert_relation(
        "Alice", "Acme", "WORKS_FOR", 0.9, "wb-sup-doc"
    )
    assert outcome1 == "created"

    # Agent says Alice now works for Beacon — should supersede the old edge
    result = await add_relation(
        "Alice",
        "Beacon",
        "WORKS_FOR",
        source="agent:test:5",
        session_id="s5",
        turn_id="t5",
    )

    assert result.outcome == "superseded"
    assert result.relation_id

    async with neo4j_driver.session() as session:
        edges = await (
            await session.run(
                "MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO {type: 'WORKS_FOR'}]->(o:Entity) "
                "RETURN o.name AS target, r.valid_until AS vu, r.created_by AS cb "
                "ORDER BY r.valid_from"
            )
        ).data()

    old_edge = next((e for e in edges if e["target"] == "Acme"), None)
    new_edge = next((e for e in edges if e["target"] == "Beacon"), None)

    assert old_edge is not None, "Old Acme edge must still exist"
    assert old_edge["vu"] is not None, "Old edge must be superseded (valid_until set)"

    assert new_edge is not None, "New Beacon edge must be created"
    assert new_edge["vu"] is None, "New edge must be live"
    assert new_edge["cb"] == "agent"


@pytest.mark.asyncio
async def test_status_summary_shape(http_client, neo4j_driver):
    """After a few writes, status_summary returns correctly shaped StatusSummary."""
    from landscape.writeback import add_entity, add_relation, status_summary

    await add_entity("Bruce Wayne", "Person", source="agent:test:6", session_id="s6", turn_id="t1")
    await add_entity("Wayne Enterprises", "Organization", source="agent:test:6")
    await add_relation("Bruce Wayne", "Wayne Enterprises", "WORKS_FOR",
                       source="agent:test:6", session_id="s6", turn_id="t2")
    await add_relation("Bruce Wayne", "Gotham City", "LOCATED_IN",
                       source="agent:test:6", session_id="s6", turn_id="t3")

    summary = await status_summary()

    assert summary.entity_count >= 2
    assert summary.document_count >= 1
    assert summary.relation_count >= 1

    # top_entities entries must have the required keys
    for entry in summary.top_entities:
        assert "name" in entry
        assert "type" in entry
        assert "reinforcement" in entry

    # recent_agent_writes must have required keys
    for entry in summary.recent_agent_writes:
        assert "subject" in entry
        assert "rel_type" in entry
        assert "object" in entry
        # session_id / turn_id / when may be None but must be present
        assert "session_id" in entry
        assert "turn_id" in entry
        assert "when" in entry

    # At least one recent agent write should be present
    assert len(summary.recent_agent_writes) >= 1
