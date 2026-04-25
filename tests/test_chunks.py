"""Chunk creation integration tests."""
import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

pytestmark = pytest.mark.integration

SHORT_DOC = "Alice leads Project Atlas. The project uses PostgreSQL."
LONG_DOC = " ".join(
    [
        "Sentence number {} is part of a long document that needs to be chunked properly.".format(i)
        for i in range(1, 60)  # ~60 sentences × ~15 tokens = ~900 tokens → multiple chunks
    ]
)

CHUNK_TITLE_SHORT = "chunks-test-short"
CHUNK_TITLE_LONG = "chunks-test-long"


async def _clear_doc(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run("MATCH (d:Document {title: $title}) DETACH DELETE d", title=title)


@pytest.mark.asyncio
async def test_single_short_doc_produces_one_chunk(http_client, neo4j_driver, qdrant_client):
    await _clear_doc(neo4j_driver, CHUNK_TITLE_SHORT)

    r = await http_client.post("/ingest", json={"text": SHORT_DOC, "title": CHUNK_TITLE_SHORT})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_created"] == 1

    # Verify :Chunk node exists in Neo4j
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {title: $title})
            MATCH (c:Chunk {doc_id: elementId(d)})
            RETURN count(c) AS cnt
            """,
            title=CHUNK_TITLE_SHORT,
        )
        record = await result.single()
    assert record["cnt"] == 1

    # Verify Qdrant chunk entry
    points, _ = await qdrant_client.scroll(
        collection_name="chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=CHUNK_TITLE_SHORT))]
        ),
        with_payload=True,
        limit=10,
    )
    assert len(points) == 1


@pytest.mark.asyncio
async def test_long_doc_produces_multiple_chunks(http_client, neo4j_driver, qdrant_client):
    await _clear_doc(neo4j_driver, CHUNK_TITLE_LONG)

    r = await http_client.post("/ingest", json={"text": LONG_DOC, "title": CHUNK_TITLE_LONG})
    assert r.status_code == 200
    body = r.json()
    assert body["chunks_created"] >= 3, f"Expected ≥3 chunks, got {body['chunks_created']}"


@pytest.mark.asyncio
async def test_chunks_linked_to_document(http_client, neo4j_driver):
    await _clear_doc(neo4j_driver, CHUNK_TITLE_SHORT)

    r = await http_client.post("/ingest", json={"text": SHORT_DOC, "title": CHUNK_TITLE_SHORT})
    assert r.status_code == 200

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {title: $title})
            MATCH (c:Chunk {doc_id: elementId(d)})
            RETURN count(c) AS cnt
            """,
            title=CHUNK_TITLE_SHORT,
        )
        record = await result.single()
    assert record["cnt"] >= 1, "All chunks must have a [:PART_OF] edge to their Document"


@pytest.mark.asyncio
async def test_chunk_vectors_have_cross_ref(http_client, neo4j_driver, qdrant_client):
    await _clear_doc(neo4j_driver, CHUNK_TITLE_SHORT)

    r = await http_client.post("/ingest", json={"text": SHORT_DOC, "title": CHUNK_TITLE_SHORT})
    assert r.status_code == 200

    points, _ = await qdrant_client.scroll(
        collection_name="chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=CHUNK_TITLE_SHORT))]
        ),
        with_payload=True,
        limit=10,
    )
    assert points, "Expected chunk vectors in Qdrant"

    for point in points:
        chunk_id = point.payload.get("chunk_id")
        assert chunk_id, f"Missing chunk_id in payload: {point.payload}"

        # Verify the id resolves to an actual :Chunk node with persisted metadata.
        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Chunk {chunk_id: $cid})
                RETURN c.doc_id AS doc_id,
                       c.chunk_index AS idx,
                       c.position AS position,
                       c.content_hash AS content_hash
                """,
                cid=chunk_id,
            )
            record = await result.single()
        assert record is not None, (
            f"chunk_id {chunk_id} does not resolve to a :Chunk node"
        )
        assert record["idx"] == record["position"]
        assert record["content_hash"]


@pytest.mark.asyncio
async def test_identical_chunk_text_across_documents_stays_separate(
    http_client, neo4j_driver, qdrant_client, monkeypatch
):
    from landscape.extraction.chunker import Chunk

    titles = ["chunks-doc-a", "chunks-doc-b"]
    texts = [
        "Document A has a unique intro. Shared chunk text.",
        "Document B has a different intro. Shared chunk text.",
    ]

    for title in titles:
        await _clear_doc(neo4j_driver, title)

    monkeypatch.setattr(
        "landscape.pipeline.chunk_text",
        lambda text: [Chunk(index=0, text="Shared chunk text.")],
    )

    for title, text in zip(titles, texts, strict=True):
        r = await http_client.post("/ingest", json={"text": text, "title": title})
        assert r.status_code == 200
        assert r.json()["chunks_created"] == 1

    points_a, _ = await qdrant_client.scroll(
        collection_name="chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=titles[0]))]
        ),
        with_payload=True,
        limit=10,
    )
    points_b, _ = await qdrant_client.scroll(
        collection_name="chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=titles[1]))]
        ),
        with_payload=True,
        limit=10,
    )

    assert len(points_a) == 1
    assert len(points_b) == 1
    assert points_a[0].payload["chunk_id"] != points_b[0].payload["chunk_id"]

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document)
            WHERE d.title IN $titles
            MATCH (c:Chunk {doc_id: elementId(d)})
            RETURN count(c) AS cnt, collect(DISTINCT c.chunk_id) AS chunk_ids
            """,
            titles=titles,
        )
        record = await result.single()

    assert record["cnt"] == 2
    assert len(record["chunk_ids"]) == 2
