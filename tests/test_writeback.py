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
async def test_add_relation_no_duplicate_subject_entity(http_client, neo4j_driver):
    """Regression: when add_relation is called with a subject whose name already
    exists as a different type (PERSON), the Unknown-type resolution path must
    find the existing node — not create a second Alice (Unknown) duplicate.

    Failure mode before fix: resolver.resolve_entity called
    qdrant_store.search_similar_entities(..., entity_type="Unknown", ...) which
    found no match because Qdrant filtered by type="Unknown".  A second node
    "Alice (Unknown)" was created.  upsert_relation then matched BOTH nodes,
    creating N cartesian-product edges.
    """
    from datetime import UTC, datetime

    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store
    from landscape.writeback import add_relation

    # Seed Alice as a PERSON entity (simulating prior ingest pipeline output)
    doc_id, _ = await neo4j_store.merge_document("hash-wb-dup", "wb-dup-doc", "text")
    alice_id = await neo4j_store.merge_entity(
        "Alice", "Person", "wb-dup-doc", 0.9, doc_id, "test"
    )
    vector = encoder.encode("Alice (Person)")
    await qdrant_store.upsert_entity(
        neo4j_element_id=alice_id,
        name="Alice",
        entity_type="Person",
        source_doc="wb-dup-doc",
        timestamp=datetime.now(UTC).isoformat(),
        vector=vector,
    )

    # Agent writes a relation with subject "Alice" — type not supplied, defaults to Unknown
    result = await add_relation(
        "Alice",
        "Beacon",
        "WORKS_FOR",
        source="agent:test:dup",
        session_id="s-dup",
        turn_id="t-dup",
    )

    assert result.outcome in ("created", "reinforced", "superseded")

    # Assert: exactly one Entity node named 'Alice' (no Unknown duplicate)
    async with neo4j_driver.session() as session:
        alice_count_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Alice'}) RETURN count(e) AS cnt"
            )
        ).single()
        edge_count_rec = await (
            await session.run(
                "MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO]->(o:Entity {name: 'Beacon'}) "
                "WHERE r.valid_until IS NULL "
                "RETURN count(r) AS cnt"
            )
        ).single()

    assert alice_count_rec["cnt"] == 1, (
        f"Expected exactly 1 Alice node, got {alice_count_rec['cnt']} — "
        "Unknown-type cross-type resolution likely failed, creating a duplicate."
    )
    assert edge_count_rec["cnt"] == 1, (
        f"Expected exactly 1 Alice->Beacon edge, got {edge_count_rec['cnt']} — "
        "cartesian product in upsert_relation likely occurred."
    )


@pytest.mark.asyncio
async def test_add_entity_with_session_turn_creates_conversation_graph(http_client, neo4j_driver):
    """add_entity with session_id+turn_id must create Conversation/Turn nodes and
    :MENTIONED_IN edge, and must NOT create a synthetic Document for this write."""
    from landscape.writeback import add_entity

    result = await add_entity(
        "Bob",
        "Person",
        source="agent:s1:t1",
        session_id="s1",
        turn_id="t1",
    )

    assert result.resolved_to_existing is False
    assert result.entity_id

    async with neo4j_driver.session() as session:
        # Conversation node must exist
        conv_rec = await (
            await session.run(
                "MATCH (c:Conversation {id: 's1'}) RETURN elementId(c) AS cid"
            )
        ).single()
        assert conv_rec is not None, ":Conversation {id: 's1'} not found"

        # Turn node must exist with composite id
        turn_rec = await (
            await session.run(
                "MATCH (t:Turn {id: 's1:t1'}) RETURN elementId(t) AS tid"
            )
        ).single()
        assert turn_rec is not None, ":Turn {id: 's1:t1'} not found"

        # HAS_TURN edge must exist
        edge_rec = await (
            await session.run(
                "MATCH (:Conversation {id: 's1'})-[:HAS_TURN]->(:Turn {id: 's1:t1'}) "
                "RETURN count(*) AS cnt"
            )
        ).single()
        assert edge_rec["cnt"] == 1, ":HAS_TURN edge missing"

        # MENTIONED_IN edge must exist
        mention_rec = await (
            await session.run(
                "MATCH (:Entity {name: 'Bob'})-[:MENTIONED_IN]->(:Turn {id: 's1:t1'}) "
                "RETURN count(*) AS cnt"
            )
        ).single()
        assert mention_rec["cnt"] == 1, ":MENTIONED_IN edge missing"

        # No synthetic Document should have been created for this write
        doc_rec = await (
            await session.run(
                "MATCH (d:Document {title: 'agent:s1:t1'}) RETURN count(d) AS cnt"
            )
        ).single()
        assert doc_rec["cnt"] == 0, "Synthetic :Document was created despite session+turn being provided"


@pytest.mark.asyncio
async def test_add_entity_without_session_turn_falls_back_to_document(http_client, neo4j_driver):
    """add_entity WITHOUT session_id/turn_id should fall back to synthetic Document
    and must NOT create any Turn nodes."""
    from landscape.writeback import add_entity

    result = await add_entity(
        "Carol",
        "Person",
        source="agent:no-session",
    )

    assert result.resolved_to_existing is False

    async with neo4j_driver.session() as session:
        # Synthetic Document must exist keyed by source
        doc_rec = await (
            await session.run(
                "MATCH (d:Document {title: 'agent:no-session'}) RETURN count(d) AS cnt"
            )
        ).single()
        assert doc_rec["cnt"] == 1, "Synthetic :Document not created for no-session path"

        # EXTRACTED_FROM edge must exist
        ef_rec = await (
            await session.run(
                "MATCH (:Entity {name: 'Carol'})-[:EXTRACTED_FROM]->(:Document {title: 'agent:no-session'}) "
                "RETURN count(*) AS cnt"
            )
        ).single()
        assert ef_rec["cnt"] == 1, ":EXTRACTED_FROM edge missing"

        # No Turn nodes should exist
        turn_rec = await (
            await session.run("MATCH (t:Turn) RETURN count(t) AS cnt")
        ).single()
        assert turn_rec["cnt"] == 0, ":Turn node was created despite no session+turn"


@pytest.mark.asyncio
async def test_existing_entity_mention_in_new_turn_creates_mentioned_in(http_client, neo4j_driver):
    """An existing entity mentioned again in a later turn should get a new
    :MENTIONED_IN edge for that turn without creating a duplicate entity node."""
    from landscape.writeback import add_entity

    # Seed Alice in turn t1
    r1 = await add_entity("Alice", "Person", source="seed", session_id="s2", turn_id="t1")
    assert r1.resolved_to_existing is False

    # Mention Alice again in turn t2 — should resolve to existing
    r2 = await add_entity("Alice", "Person", source="agent:s2:t2", session_id="s2", turn_id="t2")
    assert r2.resolved_to_existing is True
    assert r2.entity_id == r1.entity_id

    async with neo4j_driver.session() as session:
        # Exactly one Alice node
        node_cnt = await (
            await session.run(
                "MATCH (e:Entity {name: 'Alice'}) RETURN count(e) AS cnt"
            )
        ).single()
        assert node_cnt["cnt"] == 1, f"Expected 1 Alice node, got {node_cnt['cnt']}"

        # Two :MENTIONED_IN edges (one per turn)
        mention_cnt = await (
            await session.run(
                "MATCH (:Entity {name: 'Alice'})-[:MENTIONED_IN]->(:Turn) RETURN count(*) AS cnt"
            )
        ).single()
        assert mention_cnt["cnt"] == 2, (
            f"Expected 2 :MENTIONED_IN edges (t1 + t2), got {mention_cnt['cnt']}"
        )


@pytest.mark.asyncio
async def test_add_relation_with_session_turn_links_entities_to_turn(http_client, neo4j_driver):
    """add_relation with session+turn should link both endpoints via :MENTIONED_IN
    and the RELATES_TO edge should carry session_id/turn_id/created_by properties."""
    from landscape.writeback import add_relation

    result = await add_relation(
        "Dave",
        "Initech",
        "WORKS_FOR",
        source="agent:s3:t1",
        session_id="s3",
        turn_id="t1",
    )

    assert result.outcome in ("created", "reinforced", "superseded")

    async with neo4j_driver.session() as session:
        # Both Dave and Initech must be linked to the turn
        dave_rec = await (
            await session.run(
                "MATCH (:Entity {name: 'Dave'})-[:MENTIONED_IN]->(:Turn {id: 's3:t1'}) "
                "RETURN count(*) AS cnt"
            )
        ).single()
        assert dave_rec["cnt"] == 1, "Dave missing :MENTIONED_IN -> Turn s3:t1"

        initech_rec = await (
            await session.run(
                "MATCH (:Entity {name: 'Initech'})-[:MENTIONED_IN]->(:Turn {id: 's3:t1'}) "
                "RETURN count(*) AS cnt"
            )
        ).single()
        assert initech_rec["cnt"] == 1, "Initech missing :MENTIONED_IN -> Turn s3:t1"

        # RELATES_TO edge must carry provenance properties
        edge_rec = await (
            await session.run(
                "MATCH (:Entity {name: 'Dave'})-[r:RELATES_TO {type: 'WORKS_FOR'}]->(:Entity {name: 'Initech'}) "
                "WHERE r.valid_until IS NULL "
                "RETURN r.session_id AS sid, r.turn_id AS tid, r.created_by AS cb"
            )
        ).single()
        assert edge_rec is not None, "RELATES_TO edge not found"
        assert edge_rec["sid"] == "s3"
        assert edge_rec["tid"] == "t1"
        assert edge_rec["cb"] == "agent"


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
