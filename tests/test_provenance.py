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
async def test_add_relation_default_provenance(http_client, neo4j_driver):
    """Agent writeback should anchor the assertion to the Turn and materialize a live MemoryFact."""
    from landscape.writeback import add_relation

    result = await add_relation(
        "RelSubj1",
        "PERSON",
        "RelObj1",
        "PROJECT",
        "LEADS",
        source="turn:prov-1",
        session_id="s1",
        turn_id="t1",
    )

    assert result.outcome == "memory_fact"
    assert result.assertion_id
    assert result.memory_fact_id

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (:Turn {id: 's1:t1'})-[:ASSERTS]->(a:Assertion {id: $aid})
            OPTIONAL MATCH (a)-[:SUPPORTS]->(f:MemoryFact {id: $fid, family: 'LEADS'})
            OPTIONAL MATCH (s:Entity {name: 'RelSubj1'})
                          -[r:MEMORY_REL {memory_fact_id: $fid, family: 'LEADS'}]->
                          (o:Entity {name: 'RelObj1'})
            RETURN a.source_kind AS source_kind,
                   a.source_id AS source_id,
                   f.current AS fact_current,
                   r.current AS rel_current,
                   count(r) AS rel_count,
                   count(f) AS fact_count
            """,
            aid=result.assertion_id,
            fid=result.memory_fact_id,
        )
        record = await result.single()

    assert record is not None
    assert record["source_kind"] == "turn"
    assert record["source_id"] == "s1:t1"
    assert record["fact_current"] is True
    assert record["rel_current"] is True
    assert record["fact_count"] == 1
    assert record["rel_count"] == 1


@pytest.mark.asyncio
async def test_add_relation_turn_assertion_links_are_stable(http_client, neo4j_driver):
    """Repeated agent writeback should still keep the assertion anchored to the Turn."""
    from landscape.writeback import add_relation

    first = await add_relation(
        "RelSubj2",
        "PERSON",
        "RelObj2",
        "PROJECT",
        "LEADS",
        source="agent:s1:t1",
        session_id="s1",
        turn_id="t1",
    )
    second = await add_relation(
        "RelSubj2",
        "PERSON",
        "RelObj2",
        "PROJECT",
        "LEADS",
        source="agent:s1:t1",
        session_id="s1",
        turn_id="t1",
    )

    assert first.assertion_id == second.assertion_id
    assert first.memory_fact_id == second.memory_fact_id

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (:Turn {id: 's1:t1'})-[:ASSERTS]->(a:Assertion {id: $aid})
            OPTIONAL MATCH (a)-[:SUPPORTS]->(f:MemoryFact {id: $fid, family: 'LEADS'})
            OPTIONAL MATCH (s:Entity {name: 'RelSubj2'})
                          -[r:MEMORY_REL {memory_fact_id: $fid, family: 'LEADS'}]->
                          (o:Entity {name: 'RelObj2'})
            RETURN count(a) AS assertion_count,
                   count(f) AS fact_count,
                   count(r) AS rel_count,
                   max(f.current) AS fact_current,
                   max(r.current) AS rel_current
            """,
            aid=first.assertion_id,
            fid=first.memory_fact_id,
        )
        record = await result.single()

    assert record is not None
    assert record["assertion_count"] == 1
    assert record["fact_count"] == 1
    assert record["rel_count"] == 1
    assert record["fact_current"] is True
    assert record["rel_current"] is True


@pytest.mark.asyncio
async def test_upsert_relation_supersession_new_edge_has_agent_provenance(
    http_client, neo4j_driver
):
    """Functional supersession should update the live MemoryFact and keep Turn provenance."""
    from datetime import UTC, datetime

    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store
    from landscape.writeback import add_relation
    old_org_name = "AtlasAnalytics"
    new_org_name = "BeaconDynamics"

    subject_doc_id, _ = await neo4j_store.merge_document(
        "hash-prov-sup-1", "prov-sup-doc-1", "text"
    )
    old_doc_id, _ = await neo4j_store.merge_document(
        "hash-prov-sup-2", "prov-sup-doc-2", "text"
    )
    new_doc_id, _ = await neo4j_store.merge_document(
        "hash-prov-sup-3", "prov-sup-doc-3", "text"
    )

    subject_id = await neo4j_store.merge_entity(
        "RelSubj3",
        "PERSON",
        "prov-sup-doc-1",
        0.9,
        subject_doc_id,
        "test",
    )
    old_object_id = await neo4j_store.merge_entity(
        old_org_name,
        "ORGANIZATION",
        "prov-sup-doc-2",
        0.9,
        old_doc_id,
        "test",
    )
    new_object_id = await neo4j_store.merge_entity(
        new_org_name,
        "ORGANIZATION",
        "prov-sup-doc-3",
        0.9,
        new_doc_id,
        "test",
    )

    await qdrant_store.upsert_entity(
        neo4j_element_id=subject_id,
        name="RelSubj3",
        entity_type="PERSON",
        source_doc="prov-sup-doc-1",
        timestamp=datetime.now(UTC).isoformat(),
        vector=encoder.encode("RelSubj3 (PERSON)"),
    )
    await qdrant_store.upsert_entity(
        neo4j_element_id=old_object_id,
        name=old_org_name,
        entity_type="ORGANIZATION",
        source_doc="prov-sup-doc-2",
        timestamp=datetime.now(UTC).isoformat(),
        vector=encoder.encode(f"{old_org_name} (ORGANIZATION)"),
    )
    await qdrant_store.upsert_entity(
        neo4j_element_id=new_object_id,
        name=new_org_name,
        entity_type="ORGANIZATION",
        source_doc="prov-sup-doc-3",
        timestamp=datetime.now(UTC).isoformat(),
        vector=encoder.encode(f"{new_org_name} (ORGANIZATION)"),
    )

    first = await add_relation(
        "RelSubj3",
        "PERSON",
        old_org_name,
        "ORGANIZATION",
        "WORKS_FOR",
        source="agent:s3:t3a",
        session_id="s3",
        turn_id="t3a",
        confidence=0.9,
    )
    second = await add_relation(
        "RelSubj3",
        "PERSON",
        new_org_name,
        "ORGANIZATION",
        "WORKS_FOR",
        source="agent:s3:t3b",
        session_id="s3",
        turn_id="t3b",
        confidence=0.95,
    )

    async with neo4j_driver.session() as session:
        turn_rec = await (
            await session.run(
                """
            MATCH (:Turn {id: $turn_id})-[:ASSERTS]->(a:Assertion {id: $assertion_id})
            OPTIONAL MATCH (a)-[:SUBJECT_ENTITY]->(subject:Entity)
            OPTIONAL MATCH (a)-[:OBJECT_ENTITY]->(object:Entity)
                RETURN count(a) AS assertion_count,
                       max(a.source_kind) AS source_kind,
                       max(a.source_id) AS source_id,
                       max(subject.name) AS subject_name,
                       max(object.name) AS object_name
                """,
                turn_id="s3:t3b",
                assertion_id=second.assertion_id,
            )
        ).single()
        old_fact_rec = await (
            await session.run(
                """
                MATCH (f:MemoryFact {id: $fid})
                OPTIONAL MATCH (s:Entity)-[r:MEMORY_REL {memory_fact_id: $fid, family: 'WORKS_FOR'}]->(o:Entity)
                RETURN f.current AS fact_current,
                       o.name AS target,
                       r.current AS rel_current
                """,
                fid=first.memory_fact_id,
            )
        ).single()
        new_fact_rec = await (
            await session.run(
                """
                MATCH (f:MemoryFact {id: $fid})
                OPTIONAL MATCH (s:Entity)-[r:MEMORY_REL {memory_fact_id: $fid, family: 'WORKS_FOR'}]->(o:Entity)
                RETURN f.current AS fact_current,
                       o.name AS target,
                       r.current AS rel_current
                """,
                fid=second.memory_fact_id,
            )
        ).single()
        result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fid})
            RETURN f.current AS current
            """,
            fid=first.memory_fact_id,
        )
        old_fact = await result.single()

    assert turn_rec is not None
    assert turn_rec["assertion_count"] == 1
    assert turn_rec["source_kind"] == "turn"
    assert turn_rec["source_id"] == "s3:t3b"
    assert turn_rec["subject_name"] == "RelSubj3"
    assert turn_rec["object_name"] == new_org_name
    assert old_fact_rec is not None
    assert new_fact_rec is not None
    assert old_fact_rec["target"] == old_org_name
    assert new_fact_rec["target"] == new_org_name
    assert old_fact_rec["fact_current"] is False
    assert new_fact_rec["fact_current"] is True
    assert old_fact_rec["rel_current"] is False
    assert new_fact_rec["rel_current"] is True
    assert old_fact is not None and old_fact["current"] is False


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
