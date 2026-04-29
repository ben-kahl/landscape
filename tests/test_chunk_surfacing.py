"""Chunk surfacing and chunk-seed similarity propagation tests."""

from unittest.mock import AsyncMock, patch

import pytest

from landscape.retrieval.query import retrieve

pytestmark = pytest.mark.integration

BASIC_DOC = (
    "Diego Ortega is a senior engineer on the Vision Team. "
    "The Vision Team is based in Austin, Texas. "
    "Diego works on the Sentinel project."
)
TITLE = "chunk-surfacing-test"


async def _clear(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        # Remove chunks + entities tied to this document title.
        await session.run(
            "MATCH (c:Chunk)-[:PART_OF]->(d:Document {title: $t}) DETACH DELETE c",
            t=title,
        )
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) DETACH DELETE e",
            t=title,
        )
        # Remove the Turn nodes that ingested this document, plus Conversation
        # nodes that have no other live turns. Otherwise session_id-keyed tests
        # rerun against stale session state.
        await session.run(
            """
            MATCH (d:Document {title: $t})-[:INGESTED_IN]->(t:Turn)
            OPTIONAL MATCH (c:Conversation)-[:HAS_TURN]->(t)
            DETACH DELETE t
            WITH DISTINCT c
            WHERE c IS NOT NULL
              AND NOT EXISTS { MATCH (c)-[:HAS_TURN]->(:Turn) }
            DETACH DELETE c
            """,
            t=title,
        )
        await session.run("MATCH (d:Document {title: $t}) DETACH DELETE d", t=title)


@pytest.mark.asyncio
async def test_chunk_seed_propagates_similarity(http_client, neo4j_driver):
    """A chunk-only seed (no direct entity match) should enter graph expansion
    with nonzero vector_sim inherited from the chunk hit."""
    await _clear(neo4j_driver, TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "where is Diego located", "hops": 2, "limit": 10},
    )
    assert q.status_code == 200
    body = q.json()
    assert any(item["vector_sim"] > 0.0 for item in body["results"]), (
        "chunk-seeded entities should inherit chunk similarity, not 0.0"
    )


@pytest.mark.asyncio
async def test_chunk_only_seed_propagates_score(http_client, neo4j_driver):
    """Isolate the chunk-seed path: force the entity-vector search to return
    no hits, so the only seeds that reach graph expansion come from chunks.
    At least one returned entity must carry a nonzero vector_sim — proving
    the chunk's similarity score was propagated, not silently set to 0.0."""
    await _clear(neo4j_driver, TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": TITLE})
    assert r.status_code == 200

    # Patch the entity-vector search at the site retrieve() imports it.
    with patch(
        "landscape.retrieval.query.qdrant_store.search_entities_any_type",
        new=AsyncMock(return_value=[]),
    ):
        result = await retrieve("location vision team office", hops=2, limit=10)

    assert result.results, "chunk-only seeding should still produce results"
    assert any(c.vector_sim > 0.0 for c in result.results), (
        "chunk-seeded entities should inherit the chunk's Qdrant similarity score "
        "— if this is 0 for every result, the seed_sims bug has returned"
    )


@pytest.mark.asyncio
async def test_query_returns_chunk_text(http_client, neo4j_driver):
    """A hybrid query should return a chunks array with raw text and score."""
    await _clear(neo4j_driver, TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={
            "text": "where is Diego located",
            "hops": 2,
            "limit": 10,
            "chunk_limit": 3,
        },
    )
    assert q.status_code == 200
    body = q.json()
    chunks = body.get("chunks")
    assert chunks is not None, "response should include a chunks field"
    assert len(chunks) >= 1
    assert len(chunks) <= 3
    # At least one surfaced chunk should mention "Austin" — that's the fact
    # the graph-only path was missing in the original Diego scenario.
    assert any("Austin" in c["text"] for c in chunks), (
        f"expected Austin-mentioning chunk, got: {[c['text'] for c in chunks]}"
    )
    for c in chunks:
        assert "score" in c and c["score"] > 0
        assert "source_doc" in c
        assert "position" in c


@pytest.mark.asyncio
async def test_chunk_filter_scopes_to_session(http_client, neo4j_driver):
    """If session_id is supplied, only chunks ingested within that session's
    turns should be returned."""
    await _clear(neo4j_driver, TITLE)
    await _clear(neo4j_driver, TITLE + "-other")

    # In-scope doc under session sess-A / turn t-A-1.
    r1 = await http_client.post(
        "/ingest",
        json={
            "text": BASIC_DOC,
            "title": TITLE,
            "session_id": "sess-A",
            "turn_id": "t-A-1",
        },
    )
    assert r1.status_code == 200
    # Out-of-scope doc under a different session.
    r2 = await http_client.post(
        "/ingest",
        json={
            "text": "Marvin is located in Boston at the east coast office.",
            "title": TITLE + "-other",
            "session_id": "sess-B",
            "turn_id": "t-B-1",
        },
    )
    assert r2.status_code == 200

    q = await http_client.post(
        "/query",
        json={
            "text": "where is everyone located",
            "hops": 2,
            "limit": 10,
            "chunk_limit": 10,
            "session_id": "sess-A",
        },
    )
    assert q.status_code == 200
    body = q.json()
    # No chunk from the other session should appear.
    assert not any("Marvin" in c["text"] for c in body["chunks"])
    # The in-scope Austin chunk should still be present.
    assert any("Austin" in c["text"] for c in body["chunks"])
