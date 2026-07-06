"""Tests for :Conversation / :Turn node types and their linking edges."""
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.unit
def test_sanitize_id_segment_replaces_colons_with_underscores():
    from landscape.storage.neo4j_documents import _sanitize_id_segment

    # Synthetic hook turn ids use ':' as a separator (client:event:digest),
    # which collides with the Turn.id composite delimiter.
    assert (
        _sanitize_id_segment("claude-code:UserPromptSubmit:eaa0a109074a")
        == "claude-code_UserPromptSubmit_eaa0a109074a"
    )
    assert _sanitize_id_segment("no-colons-here") == "no-colons-here"


@pytest.mark.unit
def test_build_default_conversation_title_includes_agent_timestamp_and_hash():
    from landscape.storage.neo4j_store import build_default_conversation_title

    title = build_default_conversation_title(
        "default",
        agent_id="codex",
        started_at="2026-04-23T04:42:00+00:00",
    )

    assert title.startswith("codex:20260423T044200Z:")
    assert title != "default"
    assert len(title.rsplit(":", maxsplit=1)[-1]) == 8


@pytest.mark.asyncio
async def test_merge_conversation_creates(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    eid, created = await neo4j_store.merge_conversation("conv-tc1", title="Test Conv")
    assert created is True

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (c:Conversation {id: 'conv-tc1'}) "
            "RETURN c.started_at AS sa, c.last_active_at AS laa, c.title AS title"
        )
        record = await result.single()

    assert record is not None
    assert record["title"] == "Test Conv"
    assert record["sa"] == record["laa"], "started_at should equal last_active_at on fresh create"


@pytest.mark.asyncio
async def test_merge_conversation_is_idempotent(http_client, neo4j_driver):
    import asyncio

    from landscape.storage import neo4j_store

    eid1, created1 = await neo4j_store.merge_conversation("conv-tc2")
    assert created1 is True

    # Small sleep so last_active_at timestamp differs
    await asyncio.sleep(0.01)
    eid2, created2 = await neo4j_store.merge_conversation("conv-tc2")

    assert eid1 == eid2
    assert created2 is False

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (c:Conversation {id: 'conv-tc2'}) "
            "RETURN c.started_at AS sa, c.last_active_at AS laa"
        )
        record = await result.single()

    assert record["laa"] >= record["sa"], "last_active_at must be >= started_at after second call"


@pytest.mark.asyncio
async def test_merge_turn_creates_parent_conversation(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    turn_eid, created = await neo4j_store.merge_turn("conv-tc3", "t1")
    assert created is True

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (c:Conversation {id: 'conv-tc3'})-[:HAS_TURN]->(t:Turn {id: 'conv-tc3:t1'}) "
            "RETURN elementId(c) AS cid, elementId(t) AS tid"
        )
        record = await result.single()

    assert record is not None
    assert record["cid"] is not None
    assert record["tid"] is not None


@pytest.mark.asyncio
async def test_merge_turn_parent_conversation_gets_identifiable_default_title(
    http_client, neo4j_driver
):
    from landscape.storage import neo4j_store

    await neo4j_store.merge_turn("default", "t1")

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (c:Conversation {id: 'default'}) RETURN c.title AS title"
        )
        record = await result.single()

    assert record is not None
    assert record["title"].startswith("agent:")
    assert record["title"] != "default"
    assert len(record["title"].rsplit(":", maxsplit=1)[-1]) == 8


@pytest.mark.asyncio
async def test_merge_turn_id_format(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    await neo4j_store.merge_turn("conv-tc4", "t1")

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (t:Turn {session_id: 'conv-tc4', turn_id: 't1'}) RETURN t.id AS tid"
        )
        record = await result.single()

    assert record is not None
    assert record["tid"] == "conv-tc4:t1"


@pytest.mark.asyncio
async def test_merge_turn_rejects_colon_in_session_id(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    # session_id is the left side of the Turn.id composite delimiter, so a
    # colon there would make the composite ambiguous: reject it loudly.
    with pytest.raises(ValueError, match="session_id"):
        await neo4j_store.merge_turn("foo:bar", "t1")


@pytest.mark.asyncio
async def test_merge_turn_sanitizes_colon_in_turn_id(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    # Synthetic hook turn ids contain ':' (client:event:digest). Storage must
    # sanitize them for the composite key rather than raising, while keeping
    # the raw turn_id as a property for provenance.
    raw_turn_id = "claude-code:UserPromptSubmit:eaa0a109074a"
    turn_eid, created = await neo4j_store.merge_turn("conv-tc5", raw_turn_id)
    assert created is True

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (t:Turn) WHERE elementId(t) = $eid "
            "RETURN t.id AS id, t.turn_id AS turn_id, t.session_id AS session_id",
            eid=turn_eid,
        )
        record = await result.single()

    assert record is not None
    assert record["id"] == "conv-tc5:claude-code_UserPromptSubmit_eaa0a109074a"
    assert record["turn_id"] == raw_turn_id, "raw turn_id preserved for provenance"
    assert record["session_id"] == "conv-tc5"


@pytest.mark.asyncio
async def test_merge_turn_is_idempotent(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    eid1, created1 = await neo4j_store.merge_turn("conv-tc6", "t1")
    assert created1 is True

    eid2, created2 = await neo4j_store.merge_turn("conv-tc6", "t1")
    assert eid1 == eid2
    assert created2 is False

    # Only one :HAS_TURN edge should exist
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (c:Conversation {id: 'conv-tc6'})-[r:HAS_TURN]->(:Turn) "
            "RETURN count(r) AS cnt"
        )
        record = await result.single()
    assert record["cnt"] == 1


@pytest.mark.asyncio
async def test_merge_turn_next_edge(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    await neo4j_store.merge_turn("conv-tc7", "t1", turn_number=1)
    await neo4j_store.merge_turn("conv-tc7", "t2", turn_number=2)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (t1:Turn {id: 'conv-tc7:t1'})-[:NEXT]->(t2:Turn {id: 'conv-tc7:t2'}) "
            "RETURN count(*) AS cnt"
        )
        record = await result.single()

    assert record["cnt"] == 1


@pytest.mark.asyncio
async def test_link_entity_to_turn(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    doc_id, _, _ = await neo4j_store.merge_document("hash-elt-tc8", "elt-doc-tc8", "text")
    eid = await neo4j_store.merge_entity("EltEntity8", "PERSON", "elt-doc-tc8", 0.9, doc_id, "test")
    turn_eid, _ = await neo4j_store.merge_turn("conv-tc8", "t1")

    await neo4j_store.link_entity_to_turn(eid, turn_eid, confidence=0.8)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'EltEntity8'})-[r:MENTIONED_IN]->(t:Turn {id: 'conv-tc8:t1'}) "
            "RETURN r.confidence AS conf, r.created_at AS cat"
        )
        record = await result.single()

    assert record is not None
    assert abs(record["conf"] - 0.8) < 1e-6
    assert record["cat"] is not None


@pytest.mark.asyncio
async def test_link_entity_to_turn_is_idempotent_and_bumps_confidence(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    doc_id, _, _ = await neo4j_store.merge_document("hash-elt-tc9", "elt-doc-tc9", "text")
    eid = await neo4j_store.merge_entity("EltEntity9", "PERSON", "elt-doc-tc9", 0.9, doc_id, "test")
    turn_eid, _ = await neo4j_store.merge_turn("conv-tc9", "t1")

    await neo4j_store.link_entity_to_turn(eid, turn_eid, confidence=0.5)
    await neo4j_store.link_entity_to_turn(eid, turn_eid, confidence=0.9)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'EltEntity9'})-[r:MENTIONED_IN]->(t:Turn {id: 'conv-tc9:t1'}) "
            "RETURN count(r) AS cnt, r.confidence AS conf"
        )
        record = await result.single()

    assert record["cnt"] == 1, "Should be exactly one :MENTIONED_IN edge"
    assert abs(record["conf"] - 0.9) < 1e-6, "Confidence should be bumped to higher value"


@pytest.mark.asyncio
async def test_link_document_to_turn(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    doc_id, _, _ = await neo4j_store.merge_document("hash-ldt-tc10", "ldt-doc-tc10", "text")
    turn_eid, _ = await neo4j_store.merge_turn("conv-tc10", "t1")

    await neo4j_store.link_document_to_turn(doc_id, turn_eid)
    # Second call should be idempotent
    await neo4j_store.link_document_to_turn(doc_id, turn_eid)

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {content_hash: 'hash-ldt-tc10'})-[r:INGESTED_IN]->"
            "(t:Turn {id: 'conv-tc10:t1'}) RETURN count(r) AS cnt"
        )
        record = await result.single()

    assert record["cnt"] == 1


@pytest.mark.asyncio
async def test_merge_entity_without_doc_skips_extracted_from(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    eid = await neo4j_store.merge_entity(
        "DoclessEntity11", "PERSON", "no-doc", 0.9, doc_element_id=None, model="test"
    )
    assert eid is not None

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'DoclessEntity11'}) "
            "OPTIONAL MATCH (e)-[r:EXTRACTED_FROM]->(:Document) "
            "RETURN count(r) AS cnt"
        )
        record = await result.single()

    assert record["cnt"] == 0, "No :EXTRACTED_FROM edge when doc_element_id=None"


@pytest.mark.asyncio
async def test_merge_entity_with_doc_creates_extracted_from(http_client, neo4j_driver):
    from landscape.storage import neo4j_store

    doc_id, _, _ = await neo4j_store.merge_document("hash-ef-tc12", "ef-doc-tc12", "text")
    await neo4j_store.merge_entity(
        "DocEntity12", "PERSON", "ef-doc-tc12", 0.9, doc_element_id=doc_id, model="test"
    )

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: 'DocEntity12'})-[r:EXTRACTED_FROM]->(d:Document) "
            "RETURN count(r) AS cnt"
        )
        record = await result.single()

    assert record["cnt"] == 1, ":EXTRACTED_FROM edge must exist when doc_element_id provided"
