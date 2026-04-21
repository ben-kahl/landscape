"""Provenance field tests for merge_entity and upsert_relation."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_merge_entity_default_provenance(http_client, neo4j_driver):
    """No provenance kwargs → created_by='ingest', session_id/turn_id null."""
    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-prov-e1", "prov-doc-1", "text")
    await neo4j_store.merge_entity("ProvEntity1", "PERSON", "prov-doc-1", 0.9, doc_id, "test")

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'ProvEntity1'}) "
            "RETURN e.created_by AS cb, e.session_id AS sid, e.turn_id AS tid"
        )
        record = await result.single()

    assert record is not None
    assert record["cb"] == "ingest"
    assert record["sid"] is None
    assert record["tid"] is None


@pytest.mark.asyncio
async def test_merge_entity_agent_provenance(http_client, neo4j_driver):
    """Agent provenance kwargs recorded on node."""
    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-prov-e2", "prov-doc-2", "text")
    await neo4j_store.merge_entity(
        "ProvEntity2", "PERSON", "prov-doc-2", 0.9, doc_id, "test",
        created_by="agent", session_id="s1", turn_id="t1",
    )

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'ProvEntity2'}) "
            "RETURN e.created_by AS cb, e.session_id AS sid, e.turn_id AS tid"
        )
        record = await result.single()

    assert record is not None
    assert record["cb"] == "agent"
    assert record["sid"] == "s1"
    assert record["tid"] == "t1"


@pytest.mark.asyncio
async def test_merge_entity_reinforce_preserves_creator(http_client, neo4j_driver):
    """Create as 'ingest', reinforce as 'agent' → created_by stays 'ingest'."""
    from landscape.storage import neo4j_store

    doc_id1, _ = await neo4j_store.merge_document("hash-prov-e3a", "prov-doc-3a", "text")
    doc_id2, _ = await neo4j_store.merge_document("hash-prov-e3b", "prov-doc-3b", "text")

    # First creation: ingest
    await neo4j_store.merge_entity(
        "ProvEntity3", "PERSON", "prov-doc-3a", 0.9, doc_id1, "test",
        created_by="ingest",
    )
    # Reinforce: agent
    await neo4j_store.merge_entity(
        "ProvEntity3", "PERSON", "prov-doc-3b", 0.9, doc_id2, "test",
        created_by="agent", session_id="s2", turn_id="t2",
    )

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'ProvEntity3'}) "
            "RETURN e.created_by AS cb, e.session_id AS sid, e.turn_id AS tid"
        )
        record = await result.single()

    assert record is not None
    assert record["cb"] == "ingest", "ON CREATE SET must not be overwritten on reinforce"
    assert record["sid"] is None
    assert record["tid"] is None


@pytest.mark.asyncio
async def test_upsert_relation_default_provenance(http_client, neo4j_driver):
    """No provenance kwargs → edge has created_by='ingest'."""
    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-prov-r1", "prov-rel-doc-1", "text")
    await neo4j_store.merge_entity("RelSubj1", "PERSON", "prov-rel-doc-1", 0.9, doc_id, "test")
    await neo4j_store.merge_entity("RelObj1", "PROJECT", "prov-rel-doc-1", 0.9, doc_id, "test")

    await neo4j_store.upsert_relation("RelSubj1", "RelObj1", "LEADS", 0.9, "prov-rel-doc-1")

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (s:Entity {name: 'RelSubj1'})-[r:RELATES_TO {type: 'LEADS'}]->(o:Entity) "
            "WHERE r.valid_until IS NULL "
            "RETURN r.created_by AS cb, r.session_id AS sid, r.turn_id AS tid"
        )
        record = await result.single()

    assert record is not None
    assert record["cb"] == "ingest"
    assert record["sid"] is None
    assert record["tid"] is None


@pytest.mark.asyncio
async def test_upsert_relation_agent_appends_source_doc(http_client, neo4j_driver):
    """Agent call appends 'agent:s1:t1' to source_docs list."""
    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-prov-r2", "prov-rel-doc-2", "text")
    await neo4j_store.merge_entity("RelSubj2", "PERSON", "prov-rel-doc-2", 0.9, doc_id, "test")
    await neo4j_store.merge_entity("RelObj2", "PROJECT", "prov-rel-doc-2", 0.9, doc_id, "test")

    await neo4j_store.upsert_relation(
        "RelSubj2", "RelObj2", "LEADS", 0.9, "prov-rel-doc-2",
        created_by="agent", session_id="s1", turn_id="t1",
    )

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (s:Entity {name: 'RelSubj2'})-[r:RELATES_TO {type: 'LEADS'}]->(o:Entity) "
            "WHERE r.valid_until IS NULL "
            "RETURN r.source_docs AS sd, r.created_by AS cb"
        )
        record = await result.single()

    assert record is not None
    assert record["cb"] == "agent"
    assert "agent:s1:t1" in (record["sd"] or [])


@pytest.mark.asyncio
async def test_upsert_relation_supersession_new_edge_has_agent_provenance(
    http_client, neo4j_driver
):
    """Functional supersession by agent → new edge has agent provenance, old edge unchanged."""
    from landscape.storage import neo4j_store

    doc_id1, _ = await neo4j_store.merge_document("hash-prov-r3a", "prov-sup-doc-1", "text")
    doc_id2, _ = await neo4j_store.merge_document("hash-prov-r3b", "prov-sup-doc-2", "text")
    await neo4j_store.merge_entity("RelSubj3", "PERSON", "prov-sup-doc-1", 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("OldOrg", "ORGANIZATION", "prov-sup-doc-1", 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("NewOrg", "ORGANIZATION", "prov-sup-doc-2", 0.9, doc_id2, "test")

    # Original ingest edge
    outcome1, _ = await neo4j_store.upsert_relation(
        "RelSubj3", "OldOrg", "WORKS_FOR", 0.9, "prov-sup-doc-1",
        created_by="ingest",
    )
    assert outcome1 == "created"

    # Agent-triggered supersession
    outcome2, new_rid = await neo4j_store.upsert_relation(
        "RelSubj3", "NewOrg", "WORKS_FOR", 0.9, "prov-sup-doc-2",
        created_by="agent", session_id="s3", turn_id="t3",
    )
    assert outcome2 == "superseded"

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (s:Entity {name: 'RelSubj3'})-[r:RELATES_TO {type: 'WORKS_FOR'}]->(o:Entity) "
            "RETURN o.name AS target, r.valid_until AS vu, r.created_by AS cb, "
            "r.session_id AS sid, r.turn_id AS tid "
            "ORDER BY r.valid_from"
        )
        records = await result.data()

    old_edge = next((r for r in records if r["target"] == "OldOrg"), None)
    new_edge = next((r for r in records if r["target"] == "NewOrg"), None)

    assert old_edge is not None
    assert new_edge is not None

    # Old edge: original provenance preserved
    assert old_edge["cb"] == "ingest"
    assert old_edge["vu"] is not None, "Old edge must be superseded (valid_until set)"

    # New edge: agent provenance
    assert new_edge["cb"] == "agent"
    assert new_edge["sid"] == "s3"
    assert new_edge["tid"] == "t3"
    assert new_edge["vu"] is None, "New edge must be live (valid_until null)"


@pytest.mark.asyncio
async def test_invalid_created_by_raises(http_client, neo4j_driver):
    """created_by with invalid value raises ValueError."""
    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-prov-inv1", "prov-inv-doc-1", "text")

    with pytest.raises(ValueError, match="created_by"):
        await neo4j_store.merge_entity(
            "InvEntity", "PERSON", "prov-inv-doc-1", 0.9, doc_id, "test",
            created_by="bogus",
        )

    with pytest.raises(ValueError, match="created_by"):
        await neo4j_store.upsert_relation(
            "InvSubj", "InvObj", "LEADS", 0.9, "prov-inv-doc-1",
            created_by="bogus",
        )
