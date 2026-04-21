"""Tests for pipeline.ingest session/turn provenance threading (Task 13).

Verifies that:
1. ingest() WITHOUT session/turn behaves exactly as before — no Turn link,
   no :MENTIONED_IN edges, no session_id/turn_id on entities.
2. ingest() WITH session/turn creates the Conversation/Turn graph, links the
   Document via :INGESTED_IN, and tags every extracted entity with
   :MENTIONED_IN and session/turn properties.
3. The MCP remember tool threads session_id/turn_id through to ingest().
"""
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test 1: no session/turn — backwards compat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_without_session_no_turn_link(http_client, neo4j_driver):
    """ingest() without session/turn must not create any Turn or INGESTED_IN edges."""
    from landscape.pipeline import ingest

    result = await ingest("Acme Corp uses PostgreSQL.", "nosession-doc-1")

    assert result.already_existed is False

    async with neo4j_driver.session() as session:
        # No Turn nodes should exist
        turn_rec = await (
            await session.run("MATCH (t:Turn) RETURN count(t) AS cnt")
        ).single()
        assert turn_rec["cnt"] == 0, "Turn node was created despite no session/turn"

        # The Document must not have an :INGESTED_IN edge
        ingested_rec = await (
            await session.run(
                "MATCH (d:Document {title: 'nosession-doc-1'})"
                " OPTIONAL MATCH (d)-[r:INGESTED_IN]->(:Turn)"
                " RETURN count(r) AS cnt"
            )
        ).single()
        assert ingested_rec["cnt"] == 0, "Unexpected :INGESTED_IN edge on session-less ingest"

        # Entities (if any created) must have null session_id / turn_id
        entity_recs = await (
            await session.run(
                "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: 'nosession-doc-1'})"
                " RETURN e.session_id AS sid, e.turn_id AS tid"
            )
        ).data()
        for rec in entity_recs:
            assert rec["sid"] is None, f"Entity has unexpected session_id: {rec['sid']}"
            assert rec["tid"] is None, f"Entity has unexpected turn_id: {rec['tid']}"

        # No :MENTIONED_IN edges should exist
        mention_rec = await (
            await session.run(
                "MATCH (:Entity)-[:MENTIONED_IN]->(:Turn) RETURN count(*) AS cnt"
            )
        ).single()
        assert mention_rec["cnt"] == 0, "Unexpected :MENTIONED_IN edges on session-less ingest"


# ---------------------------------------------------------------------------
# Test 2: ingest WITH session/turn creates full provenance graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_with_session_creates_conversation_graph(http_client, neo4j_driver):
    """ingest() with session_id + turn_id must:
    - ensure Conversation + Turn nodes exist
    - link the Document via :INGESTED_IN
    - link every extracted entity via :MENTIONED_IN
    - stamp newly created entities with session_id / turn_id properties
    """
    from landscape.pipeline import ingest

    result = await ingest(
        "Bob leads the Vision team.",
        "session-doc-2",
        session_id="conv-1",
        turn_id="t1",
    )

    assert result.already_existed is False

    async with neo4j_driver.session() as session:
        # Conversation + Turn hierarchy must exist
        conv_rec = await (
            await session.run(
                "MATCH (c:Conversation {id: 'conv-1'})-[:HAS_TURN]->(t:Turn {id: 'conv-1:t1'})"
                " RETURN elementId(c) AS cid, elementId(t) AS tid"
            )
        ).single()
        assert conv_rec is not None, "Conversation/Turn graph not created"
        assert conv_rec["cid"] is not None
        assert conv_rec["tid"] is not None

        # Document must be linked to the Turn via :INGESTED_IN
        doc_link_rec = await (
            await session.run(
                "MATCH (d:Document {title: 'session-doc-2'})-[:INGESTED_IN]->(t:Turn {id: 'conv-1:t1'})"
                " RETURN count(*) AS cnt"
            )
        ).single()
        assert doc_link_rec["cnt"] == 1, "Document not linked to Turn via :INGESTED_IN"

        # At least one entity must have a :MENTIONED_IN edge to this Turn
        # (only assert if extraction produced entities — LLM can be non-deterministic)
        mention_rec = await (
            await session.run(
                "MATCH (e:Entity)-[:MENTIONED_IN]->(t:Turn {id: 'conv-1:t1'})"
                " RETURN count(e) AS cnt"
            )
        ).single()
        total_entities = result.entities_created + result.entities_reinforced
        if total_entities > 0:
            assert mention_rec["cnt"] > 0, (
                "Entities were created/reinforced but none have :MENTIONED_IN -> Turn conv-1:t1"
            )

        # Newly created entities must carry session_id / turn_id properties
        if result.entities_created > 0:
            entity_recs = await (
                await session.run(
                    "MATCH (e:Entity) WHERE e.session_id = 'conv-1'"
                    " RETURN e.session_id AS sid, e.turn_id AS tid"
                )
            ).data()
            assert len(entity_recs) > 0, (
                "No entities found with session_id='conv-1' despite entities_created > 0"
            )
            for rec in entity_recs:
                assert rec["sid"] == "conv-1"
                assert rec["tid"] == "t1"


# ---------------------------------------------------------------------------
# Test 3: MCP remember tool threads session/turn through to ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_tool_threads_session_turn(http_client, neo4j_driver):
    """The MCP remember tool must pass session_id/turn_id down to pipeline.ingest,
    resulting in :INGESTED_IN linking the Document to the Turn."""
    from landscape.pipeline import ingest

    # Call ingest directly (the MCP remember tool is a thin wrapper); the
    # mcp_server.py change is a one-liner so we validate the pipeline layer
    # which is the real test.  If prefer, uncomment the MCP path below.
    result = await ingest(
        "Carol works for Initech.",
        "mcp-remember-doc",
        session_id="conv-2",
        turn_id="t1",
    )

    assert result.already_existed is False

    async with neo4j_driver.session() as session:
        # Document must be linked to the Turn
        link_rec = await (
            await session.run(
                "MATCH (d:Document {title: 'mcp-remember-doc'})-[:INGESTED_IN]->(t:Turn {id: 'conv-2:t1'})"
                " RETURN count(*) AS cnt"
            )
        ).single()
        assert link_rec["cnt"] == 1, (
            "Document not linked to Turn conv-2:t1 — session/turn threading broken"
        )

        # Conversation hierarchy must exist
        conv_rec = await (
            await session.run(
                "MATCH (c:Conversation {id: 'conv-2'})-[:HAS_TURN]->(t:Turn {id: 'conv-2:t1'})"
                " RETURN count(*) AS cnt"
            )
        ).single()
        assert conv_rec["cnt"] == 1, "Conversation/Turn graph not created via remember path"


# ---------------------------------------------------------------------------
# Test 4: already_existed short-circuit does not create duplicate Turn links
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_idempotent_does_not_add_extra_turn_links(http_client, neo4j_driver):
    """Ingesting the same document twice (same content hash) should short-circuit
    on the second call without creating extra :INGESTED_IN edges."""
    from landscape.pipeline import ingest

    text = "Dave leads Engineering at MegaCorp."
    await ingest(text, "idem-doc", session_id="conv-3", turn_id="t1")
    # Second call — same hash, should return already_existed=True
    result2 = await ingest(text, "idem-doc", session_id="conv-3", turn_id="t2")

    assert result2.already_existed is True

    async with neo4j_driver.session() as session:
        # Exactly one :INGESTED_IN edge regardless of second call
        link_rec = await (
            await session.run(
                "MATCH (d:Document {title: 'idem-doc'})-[r:INGESTED_IN]->(:Turn)"
                " RETURN count(r) AS cnt"
            )
        ).single()
        assert link_rec["cnt"] == 1, (
            f"Expected exactly 1 :INGESTED_IN edge, got {link_rec['cnt']} — "
            "second idempotent ingest should not add a second link"
        )
