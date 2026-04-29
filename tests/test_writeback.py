"""Tests for src/landscape/writeback.py.

Uses the same per-test isolation fixture as the rest of the suite
(conftest._isolated_test): Neo4j is wiped and Qdrant collections dropped
before each test, so tests are fully independent.
"""
import pytest

pytestmark = pytest.mark.integration


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
    from datetime import UTC, datetime

    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store
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
        session_id="s2",
        turn_id="t1",
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
        "Person",
        "Themyscira Corp",
        "Organization",
        "WORKS_FOR",
        source="agent:test:3",
        session_id="s3",
        turn_id="t3",
    )

    assert result.outcome == "memory_fact"
    assert result.assertion_id
    assert result.memory_fact_id
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
        fact_rec = await (
            await session.run(
                "MATCH (:Assertion {id: $aid})-[:SUPPORTS]->"
                "(f:MemoryFact {family: 'WORKS_FOR'}) "
                "RETURN count(f) AS cnt",
                aid=result.assertion_id,
            )
        ).single()

    assert subj_rec is not None and subj_rec["cb"] == "agent"
    assert obj_rec is not None and obj_rec["cb"] == "agent"
    assert fact_rec["cnt"] == 1


@pytest.mark.asyncio
async def test_add_relation_normalizes_rel_type(http_client, neo4j_driver):
    """EMPLOYED_BY synonym should be stored as WORKS_FOR."""
    from landscape.writeback import add_relation

    result = await add_relation(
        "Clark Kent",
        "Person",
        "Daily Planet",
        "Organization",
        "EMPLOYED_BY",          # synonym → WORKS_FOR
        source="agent:test:4",
        session_id="s4",
        turn_id="t1",
    )

    assert result.outcome == "memory_fact"
    assert result.assertion_id
    assert result.memory_fact_id

    async with neo4j_driver.session() as session:
        fact_rec = await (
            await session.run(
                "MATCH (:Assertion {id: $aid})-[:SUPPORTS]->"
                "(f:MemoryFact {family: 'WORKS_FOR'}) "
                "RETURN count(f) AS cnt",
                aid=result.assertion_id,
            )
        ).single()

    assert fact_rec["cnt"] == 1


@pytest.mark.asyncio
async def test_add_relation_supersedes_when_functional(http_client, neo4j_driver):
    """Functional rel (WORKS_FOR): second add_relation with different object supersedes."""
    from landscape.writeback import add_relation

    # First write creates the current fact.
    first = await add_relation(
        "Alice",
        "Person",
        "Acme",
        "Organization",
        "WORKS_FOR",
        source="agent:test:5a",
        session_id="s5",
        turn_id="t5a",
    )
    assert first.outcome == "memory_fact"

    # Agent says Alice now works for Beacon — should supersede the old fact.
    result = await add_relation(
        "Alice",
        "Person",
        "Beacon",
        "Organization",
        "WORKS_FOR",
        source="agent:test:5b",
        session_id="s5",
        turn_id="t5b",
    )

    assert result.outcome == "memory_fact"
    assert result.assertion_id
    assert result.memory_fact_id

    async with neo4j_driver.session() as session:
        facts = await (
            await session.run(
                "MATCH (:Assertion)-[:SUPPORTS]->(f:MemoryFact {family: 'WORKS_FOR'}) "
                "OPTIONAL MATCH (f)-[:AS_OBJECT]->(o:Entity) "
                "RETURN o.name AS target, f.current AS current, "
                "f.created_at AS created_at, f.updated_at AS updated_at "
                "ORDER BY f.created_at"
            )
        ).data()

    old_edge = next((e for e in facts if e["target"] == "Acme"), None)
    new_edge = next((e for e in facts if e["target"] == "Beacon"), None)

    assert old_edge is not None, "Old Acme fact must still exist"
    assert old_edge["current"] is False, "Old fact must be superseded"

    assert new_edge is not None, "New Beacon fact must be created"
    assert new_edge["current"] is True, "New fact must be live"


@pytest.mark.asyncio
async def test_add_relation_no_duplicate_subject_entity(http_client, neo4j_driver):
    """Regression: when add_relation is called with a subject whose name already
    exists as a PERSON node, the resolver must find the existing node — not
    create a second duplicate.

    Also exercises the Unknown-type cross-type resolution path by passing
    subject_type="Unknown" explicitly (still a valid value; the resolver falls
    back to cross-type search when the type is "Unknown").

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

    # Agent writes a relation with subject_type="Unknown" — exercises the cross-type
    # resolution path that finds the existing Person node despite the type mismatch.
    result = await add_relation(
        "Alice",
        "Unknown",
        "Beacon",
        "Organization",
        "WORKS_FOR",
        source="agent:test:dup",
        session_id="s-dup",
        turn_id="t-dup",
    )

    assert result.outcome == "memory_fact"

    # Assert: exactly one Entity node named 'Alice' (no Unknown duplicate)
    async with neo4j_driver.session() as session:
        alice_count_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Alice'}) RETURN count(e) AS cnt"
            )
        ).single()
        edge_count_rec = await (
            await session.run(
                "MATCH (s:Entity {name: 'Alice'})-[r:MEMORY_REL {family: 'WORKS_FOR'}]->"
                "(o:Entity {name: 'Beacon'}) "
                "WHERE r.current = true "
                "RETURN count(r) AS cnt"
            )
        ).single()

    assert alice_count_rec["cnt"] == 1, (
        f"Expected exactly 1 Alice node, got {alice_count_rec['cnt']} — "
        "Unknown-type cross-type resolution likely failed, creating a duplicate."
    )
    assert edge_count_rec["cnt"] == 1, (
        f"Expected exactly 1 Alice->Beacon memory rel, got {edge_count_rec['cnt']} — "
        "entity resolution likely failed."
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
        assert doc_rec["cnt"] == 0, (
            "Synthetic :Document was created despite session+turn being provided"
        )


@pytest.mark.asyncio
async def test_add_entity_without_session_turn_raises(http_client):
    """add_entity WITHOUT session_id/turn_id must raise ValueError — synthetic
    Document provenance has been removed."""
    from landscape.writeback import add_entity

    with pytest.raises(ValueError, match="session_id and turn_id are required"):
        await add_entity(
            "Carol",
            "Person",
            source="agent:no-session",
        )


@pytest.mark.asyncio
async def test_add_relation_without_session_turn_raises(http_client):
    """add_relation WITHOUT session_id/turn_id must raise ValueError — synthetic
    Document provenance has been removed."""
    from landscape.writeback import add_relation

    with pytest.raises(ValueError, match="session_id and turn_id are required"):
        await add_relation(
            "Alice",
            "Person",
            "Beacon",
            "Organization",
            "WORKS_FOR",
            source="agent:no-session",
        )


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
    and the assertion/memory fact should be linked to the current turn."""
    from landscape.writeback import add_relation

    result = await add_relation(
        "Dave",
        "Person",
        "Initech",
        "Organization",
        "WORKS_FOR",
        source="agent:s3:t1",
        session_id="s3",
        turn_id="t1",
    )

    assert result.outcome == "memory_fact"
    assert result.assertion_id
    assert result.memory_fact_id

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

        # Assertion/MemoryFact provenance must exist
        edge_rec = await (
            await session.run(
                "MATCH (:Turn {id: 's3:t1'})-[:ASSERTS]->"
                "(:Assertion {id: $aid})-[:SUPPORTS]->"
                "(f:MemoryFact {id: $fid, family: 'WORKS_FOR'}) "
                "RETURN count(f) AS cnt",
                aid=result.assertion_id,
                fid=result.memory_fact_id,
            )
        ).single()
        assert edge_rec["cnt"] == 1, "Assertion->MemoryFact link not found"


@pytest.mark.asyncio
async def test_status_summary_shape(http_client, neo4j_driver):
    """After a few writes, status_summary returns correctly shaped StatusSummary."""
    from landscape.writeback import add_entity, add_relation, status_summary

    await add_entity(
        "Bruce Wayne",
        "Person",
        source="agent:test:6",
        session_id="s6",
        turn_id="t1",
    )
    await add_entity(
        "Wayne Enterprises",
        "Organization",
        source="agent:test:6",
        session_id="s6",
        turn_id="t1",
    )
    await add_relation(
        "Bruce Wayne",
        "Person",
        "Wayne Enterprises",
        "Organization",
        "WORKS_FOR",
        source="agent:test:6",
        session_id="s6",
        turn_id="t2",
    )
    await add_relation(
        "Bruce Wayne",
        "Person",
        "Gotham City",
        "Location",
        "LOCATED_IN",
        source="agent:test:6",
        session_id="s6",
        turn_id="t3",
    )

    summary = await status_summary()

    assert summary.entity_count >= 2
    # Agent write-back no longer creates synthetic Documents — document_count
    # may be 0 in a pure agent-write test; just assert it is non-negative.
    assert summary.document_count >= 0
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

    # New in Task 14: conversation/turn counts
    assert summary.conversation_count >= 1, (
        "At least one Conversation node should exist after add_entity/add_relation calls"
    )
    assert summary.turn_count >= 1, (
        "At least one Turn node should exist after add_entity/add_relation calls"
    )

    # recent_conversations: entries must have the required shape
    for entry in summary.recent_conversations:
        assert "id" in entry
        assert "title" in entry
        assert "turn_count" in entry
        assert "last_active_at" in entry

    # The conversation used in this test (s6) must appear in recent_conversations
    conv_ids = [c["id"] for c in summary.recent_conversations]
    assert "s6" in conv_ids, (
        f"Session 's6' should be in recent_conversations, got: {conv_ids}"
    )
    s6 = next(c for c in summary.recent_conversations if c["id"] == "s6")
    assert s6["turn_count"] >= 1, "s6 should have at least 1 turn"

    # Defense-in-depth: recent_agent_writes must not contain duplicates
    # (same subject/rel_type/object appearing more than once).
    write_keys = [
        (w["subject"], w["rel_type"], w["object"])
        for w in summary.recent_agent_writes
    ]
    assert len(write_keys) == len(set(write_keys)), (
        f"Duplicate entries in recent_agent_writes: {write_keys}"
    )


@pytest.mark.asyncio
async def test_add_relation_missing_types_raises():
    """add_relation without subject_type or object_type must raise TypeError
    (missing required positional arguments)."""
    from landscape.writeback import add_relation

    # Missing both subject_type and object_type
    with pytest.raises(TypeError):
        await add_relation(
            "Alice",
            "Beacon",      # positional — this lands on subject_type
            # object_type, rel_type, source are now missing → TypeError
        )


@pytest.mark.asyncio
async def test_add_relation_endpoints_have_correct_types(http_client, neo4j_driver):
    """Endpoints created via add_relation must have the declared types — not 'Unknown'."""
    from landscape.writeback import add_relation

    await add_relation(
        "Eve",
        "Person",
        "Nexus Corp",
        "Organization",
        "WORKS_FOR",
        source="agent:test:types",
        session_id="s-types",
        turn_id="t1",
    )

    async with neo4j_driver.session() as session:
        subj_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Eve'}) RETURN e.type AS t"
            )
        ).single()
        obj_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Nexus Corp'}) RETURN e.type AS t"
            )
        ).single()

    assert subj_rec is not None
    assert subj_rec["t"] == "Person", (
        f"Expected subject type 'Person', got {subj_rec['t']!r} — "
        "endpoint was likely auto-created as 'Unknown'"
    )
    assert obj_rec is not None
    assert obj_rec["t"] == "Organization", (
        f"Expected object type 'Organization', got {obj_rec['t']!r} — "
        "endpoint was likely auto-created as 'Unknown'"
    )


@pytest.mark.asyncio
async def test_add_entity_coerces_type_and_stores_subtype(http_client, neo4j_driver):
    """Non-canonical entity_type is coerced to canonical; original stored as subtype."""
    from landscape.writeback import add_entity

    result = await add_entity(
        "Alice Mercer",
        "SoftwareEngineer",
        source="agent:test:coerce",
        session_id="s-coerce",
        turn_id="t1",
    )

    assert result.resolved_to_existing is False
    assert result.entity_id

    async with neo4j_driver.session() as session:
        rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Alice Mercer'}) "
                "RETURN e.type AS t, e.subtype AS st"
            )
        ).single()

    assert rec is not None
    assert rec["t"] == "Person", (
        f"Expected canonical type 'Person', got {rec['t']!r}"
    )
    assert rec["st"] == "SoftwareEngineer", (
        f"Expected subtype 'SoftwareEngineer', got {rec['st']!r}"
    )


@pytest.mark.asyncio
async def test_add_entity_canonical_skips_subtype(http_client, neo4j_driver):
    """When entity_type is already canonical, subtype should be null/absent."""
    from landscape.writeback import add_entity

    result = await add_entity(
        "Bob Canonical",
        "Person",
        source="agent:test:notype",
        session_id="s-notype",
        turn_id="t1",
    )

    assert result.resolved_to_existing is False

    async with neo4j_driver.session() as session:
        rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Bob Canonical'}) "
                "RETURN e.type AS t, e.subtype AS st"
            )
        ).single()

    assert rec is not None
    assert rec["t"] == "Person"
    assert rec["st"] is None, (
        f"Expected subtype to be null for canonical input, got {rec['st']!r}"
    )


@pytest.mark.asyncio
async def test_add_relation_endpoint_types_coerce(http_client, neo4j_driver):
    """Non-canonical endpoint types passed to add_relation are coerced to canonical."""
    from landscape.writeback import add_relation

    await add_relation(
        "Frank Engineer",
        "SoftwareEngineer",   # non-canonical → Person
        "TechCorp Solutions",
        "Company",             # non-canonical → Organization
        "WORKS_FOR",
        source="agent:test:coerce-rel",
        session_id="s-coerce-rel",
        turn_id="t1",
    )

    async with neo4j_driver.session() as session:
        subj_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'Frank Engineer'}) RETURN e.type AS t, e.subtype AS st"
            )
        ).single()
        obj_rec = await (
            await session.run(
                "MATCH (e:Entity {name: 'TechCorp Solutions'}) RETURN e.type AS t, e.subtype AS st"
            )
        ).single()

    assert subj_rec is not None
    assert subj_rec["t"] == "Person", (
        f"Expected 'Person' for SoftwareEngineer, got {subj_rec['t']!r}"
    )
    assert subj_rec["st"] == "SoftwareEngineer", (
        f"Expected subtype 'SoftwareEngineer', got {subj_rec['st']!r}"
    )

    assert obj_rec is not None
    assert obj_rec["t"] == "Organization", (
        f"Expected 'Organization' for Company, got {obj_rec['t']!r}"
    )
    assert obj_rec["st"] == "Company", (
        f"Expected subtype 'Company', got {obj_rec['st']!r}"
    )


@pytest.mark.asyncio
async def test_add_relation_creates_assertion_and_memory_fact(http_client):
    from landscape.writeback import add_relation
    from landscape.storage import neo4j_store

    result = await add_relation(
        subject="Alice",
        subject_type="Person",
        object_="Acme",
        object_type="Organization",
        rel_type="WORKS_FOR",
        source="wb-doc",
        confidence=0.9,
        session_id="s1",
        turn_id="t1",
    )

    assert result.assertion_id
    assert result.memory_fact_id

    rows = await neo4j_store.run_cypher_readonly(
        "MATCH (:Assertion)-[:SUPPORTS]->(f:MemoryFact {family: 'WORKS_FOR'}) "
        "RETURN count(f) AS count"
    )
    assert rows[0]["count"] == 1


@pytest.mark.asyncio
async def test_alias_writeback_creates_alias_not_stub_entity(http_client):
    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store
    from landscape.writeback import add_entity

    robert_id = await _seed_entity_with_vector(
        "Robert",
        "Person",
        vector=encoder.encode("Bob (Person)"),
    )

    result = await add_entity(
        "Bob",
        "Person",
        source="wb-doc",
        confidence=0.9,
        session_id="s1",
        turn_id="t1",
    )

    assert result.resolved_to_existing is True
    assert result.entity_id == robert_id

    repeat = await add_entity(
        "Bob",
        "Person",
        source="wb-doc-2",
        confidence=0.9,
        session_id="s1",
        turn_id="t2",
    )

    assert repeat.resolved_to_existing is True
    assert repeat.entity_id == robert_id

    rows = await neo4j_store.run_cypher_readonly(
        "MATCH (a:Alias)-[:SAME_AS]->(:Entity {name: 'Robert'}) RETURN count(a) AS count"
    )
    assert rows[0]["count"] == 1

    stub_rows = await neo4j_store.run_cypher_readonly(
        "MATCH (e:Entity {name: 'Bob', canonical: false}) RETURN count(e) AS count"
    )
    assert stub_rows[0]["count"] == 0

    alias_rows = await neo4j_store.run_cypher_readonly(
        "MATCH (a:Alias)-[:SAME_AS]->(:Entity {name: 'Robert'}) RETURN count(a) AS count"
    )
    assert alias_rows[0]["count"] == 1


@pytest.mark.asyncio
async def test_assertion_only_writeback_is_reported_in_status_summary(http_client):
    from landscape.writeback import add_relation, status_summary

    result = await add_relation(
        "Alice",
        "Person",
        "Acme",
        "Organization",
        "RELATED_TO",
        source="wb-doc",
        confidence=0.9,
        session_id="s1",
        turn_id="t3",
    )

    assert result.outcome == "assertion_only"
    assert result.assertion_id
    assert result.memory_fact_id is None

    summary = await status_summary()
    assert any(
        entry["subject"] == "Alice" and entry["rel_type"] == "RELATED_TO" and entry["object"] == "Acme"
        for entry in summary.recent_agent_writes
    )


# ---------------------------------------------------------------------------
# Helpers for alias / homonym regression tests
# ---------------------------------------------------------------------------


async def _seed_entity_with_vector(
    name: str,
    entity_type: str,
    *,
    vector: list[float] | None = None,
) -> str:
    """Create Entity in Neo4j + Qdrant. Returns element id.

    If vector is None, encodes ``name (entity_type)`` using the project encoder.
    Callers that want alias resolution to fire from a different surface name
    should pass the vector for *that* surface name so the resolver finds this
    canonical node when queried with the alias.
    """
    from datetime import UTC, datetime

    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store

    doc_id, _ = await neo4j_store.merge_document(
        f"hash-seed-{name.lower().replace(' ', '-')}",
        f"seed-doc-{name.lower().replace(' ', '-')}",
        "text",
    )
    entity_id = await neo4j_store.merge_entity(
        name, entity_type, f"seed-doc-{name.lower().replace(' ', '-')}", 0.9, doc_id, "test"
    )
    v = vector if vector is not None else encoder.encode(f"{name} ({entity_type})")
    await qdrant_store.upsert_entity(
        neo4j_element_id=entity_id,
        name=name,
        entity_type=entity_type,
        source_doc=f"seed-doc-{name.lower().replace(' ', '-')}",
        timestamp=datetime.now(UTC).isoformat(),
        vector=v,
    )
    return entity_id


async def _seed_alias(
    canonical_entity_id: str,
    alias_name: str,
) -> None:
    """Register an alias for an existing canonical entity in Neo4j.

    Uses the Alias node model instead of creating a stub Entity node.
    """
    from landscape.storage import neo4j_store

    await neo4j_store.merge_alias(canonical_entity_id, alias_name, "test-alias", 0.95)


@pytest.mark.asyncio
async def test_add_relation_uses_resolved_alias_target(http_client, neo4j_driver):
    """Regression: when 'Bob' is an alias for 'Robert', add_relation('Bob', ...)
    must attach the edge to the canonical 'Robert' node, not to the alias stub.

    Before the fix: upsert_relation was called with subject_name='Bob' which
    matched the stub node (Entity {name: 'Bob'}).
    After the fix: upsert_relation is called with subject_node_id=robert_id so
    the Cypher MATCH uses elementId() and binds to the canonical node.
    """
    from landscape.embeddings import encoder
    from landscape.writeback import add_relation

    # Seed Robert using the 'Bob (Person)' vector so the resolver finds Robert
    # when add_entity("Bob") searches Qdrant -- this simulates "Bob" being a
    # known alias whose vector representation maps to Robert's canonical entry.
    bob_vector = encoder.encode("Bob (Person)")
    robert_id = await _seed_entity_with_vector("Robert", "Person", vector=bob_vector)

    # Register "Bob" as an alias in Neo4j without creating a stub Entity node.
    await _seed_alias(robert_id, "Bob")

    # Seed Acme (plain -- no alias trickery needed here).
    acme_id = await _seed_entity_with_vector("Acme", "Organization")

    result = await add_relation(
        "Bob",
        "Person",
        "Acme",
        "Organization",
        "WORKS_FOR",
        source="agent:alias-test:1",
        session_id="s-alias",
        turn_id="t-alias",
    )

    assert result.outcome == "memory_fact"

    # The edge must be attached to Robert (canonical), not to the alias stub.
    assert result.subject_id == robert_id, (
        f"Expected edge subject_id to be Robert ({robert_id}), "
        f"got {result.subject_id!r} -- alias stub may have stolen the relation."
    )
    assert result.object_id == acme_id, (
        f"Expected edge object_id to be Acme ({acme_id}), got {result.object_id!r}"
    )

    # Double-check via the graph: Robert should have the MEMORY_REL edge.
    async with neo4j_driver.session() as session:
        edge_on_canonical = await (
            await session.run(
                "MATCH (s:Entity)-[r:MEMORY_REL {family: 'WORKS_FOR'}]->(o:Entity) "
                "WHERE elementId(s) = $sid AND r.current = true "
                "RETURN count(r) AS cnt",
                sid=robert_id,
            )
        ).single()
        edge_on_stub = await (
            await session.run(
                "MATCH (stub:Entity {name: 'Bob', canonical: false})"
                "-[r:MEMORY_REL {family: 'WORKS_FOR'}]->() "
                "WHERE r.current = true "
                "RETURN count(r) AS cnt"
            )
        ).single()

    assert edge_on_canonical["cnt"] == 1, (
        "Expected exactly 1 live WORKS_FOR edge starting from Robert (canonical)"
    )
    assert edge_on_stub["cnt"] == 0, (
        "Alias stub 'Bob' must not have received the WORKS_FOR edge"
    )


@pytest.mark.asyncio
async def test_add_relation_does_not_cross_link_same_surface_name(http_client, neo4j_driver):
    """Regression: when two distinct entities share the name 'Alex' (one Person,
    one Project), add_relation with subject_type='Person' must attach the edge
    to the Person node only -- not to both nodes (cartesian MATCH {name: 'Alex'}).

    Before the fix: upsert_relation matched MATCH (s:Entity {name: 'Alex'}) which
    returned both nodes and created N cartesian-product edges.
    After the fix: the resolved entity_id from add_entity is passed as
    subject_node_id so elementId() match binds exactly one canonical node.
    """
    from landscape.embeddings import encoder
    from landscape.writeback import add_relation

    # Seed Alex Person and Alex Project with their respective type-scoped vectors.
    alex_person_vector = encoder.encode("Alex (Person)")
    alex_project_vector = encoder.encode("Alex (Project)")
    alex_person_id = await _seed_entity_with_vector("Alex", "Person", vector=alex_person_vector)
    # Project named Alex -- same surface name, different type.
    await _seed_entity_with_vector("Alex", "Project", vector=alex_project_vector)

    northwind_id = await _seed_entity_with_vector("Northwind", "Organization")

    result = await add_relation(
        "Alex",
        "Person",           # subject_type disambiguates to the Person node
        "Northwind",
        "Organization",
        "WORKS_FOR",
        source="agent:homonym-test:1",
        session_id="s-homonym",
        turn_id="t-homonym",
    )

    assert result.outcome == "memory_fact"
    assert result.subject_id == alex_person_id, (
        f"Expected edge to land on Alex-Person ({alex_person_id}), "
        f"got {result.subject_id!r} -- homonym cross-link may have occurred."
    )
    assert result.object_id == northwind_id, (
        f"Expected edge object to be Northwind ({northwind_id}), got {result.object_id!r}"
    )

    # Exactly one live WORKS_FOR edge should exist, and it must start from Alex Person.
    async with neo4j_driver.session() as session:
        all_edges = await (
            await session.run(
                "MATCH (s:Entity {name: 'Alex'})-[r:MEMORY_REL {family: 'WORKS_FOR'}]->(o:Entity) "
                "WHERE r.current = true "
                "RETURN elementId(s) AS sid, s.type AS stype, count(r) AS cnt"
            )
        ).data()

    assert len(all_edges) == 1, (
        f"Expected exactly 1 WORKS_FOR edge from any 'Alex' node, got {len(all_edges)}: {all_edges}"
    )
    assert all_edges[0]["stype"] == "Person", (
        f"The single edge should start from Alex-Person, got type {all_edges[0]['stype']!r}"
    )


# ---------------------------------------------------------------------------
# Unit-level guard tests — no DB connection required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_relation_raises_when_only_subject_node_id_given():
    """Passing only subject_node_id (without object_node_id) must raise ValueError
    immediately — before any Cypher is executed — so partial ids cannot silently
    degrade to name-based matching.
    """
    from landscape.storage import neo4j_store

    with pytest.raises(ValueError, match="both be provided or both omitted"):
        await neo4j_store.upsert_relation(
            subject_node_id="4:abc:1",
            subject_name="Alice",
            object_name="Acme",
            relation_type="WORKS_FOR",
            confidence=0.9,
            source_doc="test-doc",
        )


@pytest.mark.asyncio
async def test_upsert_relation_raises_when_only_object_node_id_given():
    """Passing only object_node_id (without subject_node_id) must raise ValueError
    immediately — the symmetric partner of the subject-only test.
    """
    from landscape.storage import neo4j_store

    with pytest.raises(ValueError, match="both be provided or both omitted"):
        await neo4j_store.upsert_relation(
            object_node_id="4:abc:2",
            subject_name="Alice",
            object_name="Acme",
            relation_type="WORKS_FOR",
            confidence=0.9,
            source_doc="test-doc",
        )


@pytest.mark.asyncio
async def test_upsert_relation_raises_when_relation_type_empty():
    """An empty relation_type string must raise ValueError immediately."""
    from landscape.storage import neo4j_store

    with pytest.raises(ValueError, match="relation_type must be a non-empty string"):
        await neo4j_store.upsert_relation(
            subject_name="Alice",
            object_name="Acme",
            relation_type="",
            confidence=0.9,
            source_doc="test-doc",
        )
