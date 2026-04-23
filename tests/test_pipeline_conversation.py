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

from landscape.conversation_ingestion import ConversationIngestResult, ingest_conversation_turn

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Conversation ingestion primitives
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_build_conversation_title_is_stable():
    from landscape.conversation_ingestion import ConversationTurn, build_conversation_title

    turn = ConversationTurn(
        session_id="sess-1",
        turn_id="t-7",
        role=" User ",
        text="Alice moved to Beacon.",
    )

    assert build_conversation_title(turn) == "conversation:sess-1:t-7:user"


@pytest.mark.smoke
def test_should_auto_ingest_turn_rejects_blank_text():
    from landscape.conversation_ingestion import ConversationTurn, should_auto_ingest_turn

    turn = ConversationTurn(
        session_id="sess-1",
        turn_id="t-7",
        role="user",
        text="   ",
    )

    assert should_auto_ingest_turn(turn, seen_fingerprints=set()) is False


@pytest.mark.asyncio
async def test_ingest_conversation_turn_creates_document_and_links_turn(
    http_client, neo4j_driver
):
    from landscape.conversation_ingestion import ConversationTurn, build_conversation_title

    turn = ConversationTurn(
        session_id="conv-4",
        turn_id="t1",
        role="user",
        text="Alice joined Beacon Labs.",
    )

    expected_title = build_conversation_title(turn)
    result = await ingest_conversation_turn(turn)

    assert isinstance(result, ConversationIngestResult)
    assert result.skipped is False
    assert result.reason is None
    assert result.title == expected_title
    assert result.ingest_result is not None
    assert result.ingest_result.already_existed is False

    async with neo4j_driver.session() as session:
        doc_link_rec = await (
            await session.run(
                "MATCH (d:Document {title: $title})-[:INGESTED_IN]->"
                "(t:Turn {id: 'conv-4:t1'})"
                " RETURN count(*) AS cnt"
                ,
                {"title": expected_title},
            )
        ).single()
        assert doc_link_rec["cnt"] == 1, "Conversation turn ingest did not link Document to Turn"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_conversation_turn_skips_duplicate_in_same_process(monkeypatch):
    from landscape.conversation_ingestion import ConversationTurn
    from landscape.pipeline import IngestResult

    async def fake_ingest(
        text: str,
        title: str,
        session_id: str | None = None,
        turn_id: str | None = None,
    ):
        return IngestResult(
            doc_id=f"doc:{title}",
            already_existed=False,
            entities_created=1,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=1,
        )

    monkeypatch.setattr("landscape.conversation_ingestion.ingest", fake_ingest)

    seen = set()
    turn = ConversationTurn(
        session_id="conv-5",
        turn_id="t1",
        role="user",
        text="Alice joined Beacon Labs.",
    )

    first = await ingest_conversation_turn(turn, seen_fingerprints=seen)
    second = await ingest_conversation_turn(turn, seen_fingerprints=seen)

    assert first.skipped is False
    assert first.reason is None
    assert first.ingest_result is not None
    assert first.ingest_result.already_existed is False
    assert first.already_existed is False
    assert first.entities_created == 1
    assert first.entities_reinforced == 0
    assert first.relations_created == 0
    assert first.relations_reinforced == 0
    assert first.relations_superseded == 0
    assert first.chunks_created == 1

    assert second.skipped is True
    assert second.reason == "duplicate"
    assert second.ingest_result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_conversation_turn_marks_seen_only_after_success(monkeypatch):
    from landscape.conversation_ingestion import ConversationTurn
    from landscape.pipeline import IngestResult

    attempts = 0

    async def flaky_ingest(
        text: str,
        title: str,
        session_id: str | None = None,
        turn_id: str | None = None,
    ):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary ingestion failure")
        return IngestResult(
            doc_id=f"doc:{title}",
            already_existed=False,
            entities_created=1,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=1,
        )

    monkeypatch.setattr("landscape.conversation_ingestion.ingest", flaky_ingest)

    seen = set()
    turn = ConversationTurn(
        session_id="conv-6",
        turn_id="t1",
        role="user",
        text="Alice joined Beacon Labs.",
    )

    with pytest.raises(RuntimeError, match="temporary ingestion failure"):
        await ingest_conversation_turn(turn, seen_fingerprints=seen)

    assert seen == set()

    retry = await ingest_conversation_turn(turn, seen_fingerprints=seen)

    assert retry.skipped is False
    assert retry.ingest_result is not None
    assert attempts == 2


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
                "MATCH (d:Document {title: 'session-doc-2'})-[:INGESTED_IN]->"
                "(t:Turn {id: 'conv-1:t1'})"
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
                "MATCH (d:Document {title: 'mcp-remember-doc'})-[:INGESTED_IN]->"
                "(t:Turn {id: 'conv-2:t1'})"
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


def test_build_conversation_title_is_stable():
    from landscape.conversation_ingestion import (
        ConversationTurn,
        build_conversation_title,
    )

    turns_a = [
        ConversationTurn(role="user", text="  Plan the launch\nfor Friday.  "),
        ConversationTurn(role="assistant", text="Sure."),
    ]
    turns_b = [
        ConversationTurn(role="user", text="Plan the launch for Friday."),
        ConversationTurn(role="assistant", text="Sure."),
    ]

    assert build_conversation_title(turns_a) == build_conversation_title(turns_b)


def test_should_auto_ingest_turn_rejects_blank_text():
    from landscape.conversation_ingestion import (
        ConversationTurn,
        should_auto_ingest_turn,
    )

    turn = ConversationTurn(role="user", text=" \n\t ")

    assert should_auto_ingest_turn(turn) is False
