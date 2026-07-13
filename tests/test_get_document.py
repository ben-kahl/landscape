"""Integration tests for document fetch storage reads (get_document feature).

DB isolation via the autouse `_isolated_test` conftest fixture."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def _seed_doc_with_chunks() -> str:
    from landscape.storage.neo4j_driver import get_driver

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            CREATE (d:Document {content_hash: 'h1', title: 'Ticket 119987',
                                source_type: 'text',
                                ingested_at: '2026-07-12T00:00:00+00:00',
                                ingest_completed_at: '2026-07-12T00:00:00+00:00'})
            CREATE (c1:Chunk {chunk_id: 'd:1:h1', chunk_index: 1, position: 1,
                              text: 'second chunk'})
            CREATE (c0:Chunk {chunk_id: 'd:0:h1', chunk_index: 0, position: 0,
                              text: 'first chunk'})
            CREATE (c0)-[:PART_OF]->(d)
            CREATE (c1)-[:PART_OF]->(d)
            CREATE (t:Turn {id: 'sess-1:t1', session_id: 'sess-1', turn_id: 't1'})
            CREATE (d)-[:INGESTED_IN]->(t)
            CREATE (a:Assertion {id: 'assert-1'})
            CREATE (f:MemoryFact {id: 'fact-1', family: 'USES'})
            CREATE (d)-[:ASSERTS]->(a)
            CREATE (a)-[:SUPPORTS]->(f)
            RETURN elementId(d) AS doc_id
            """
        )
        record = await result.single()
        return record["doc_id"]


@pytest.mark.asyncio
async def test_get_document_with_chunks_orders_and_includes_sessions():
    from landscape.storage.neo4j_documents import get_document_with_chunks

    doc_id = await _seed_doc_with_chunks()
    doc = await get_document_with_chunks(doc_id)

    assert doc is not None
    assert doc["title"] == "Ticket 119987"
    assert doc["source_type"] == "text"
    assert [c["text"] for c in doc["chunks"]] == ["first chunk", "second chunk"]
    assert [c["position"] for c in doc["chunks"]] == [0, 1]
    assert doc["sessions"] == ["sess-1"]


@pytest.mark.asyncio
async def test_get_document_with_chunks_missing_returns_none():
    from landscape.storage.neo4j_documents import get_document_with_chunks

    assert await get_document_with_chunks("4:deadbeef:999") is None


@pytest.mark.asyncio
async def test_find_doc_id_for_chunk():
    from landscape.storage.neo4j_documents import find_doc_id_for_chunk

    doc_id = await _seed_doc_with_chunks()
    assert await find_doc_id_for_chunk("d:0:h1") == doc_id
    assert await find_doc_id_for_chunk("nope") is None


@pytest.mark.asyncio
async def test_find_doc_ids_for_memory_fact():
    from landscape.storage.neo4j_documents import find_doc_ids_for_memory_fact

    doc_id = await _seed_doc_with_chunks()
    assert await find_doc_ids_for_memory_fact("fact-1") == [doc_id]
    assert await find_doc_ids_for_memory_fact("missing-fact") == []
