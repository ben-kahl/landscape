"""Chunk surfacing and chunk-seed similarity propagation tests."""

from unittest.mock import AsyncMock, patch

import pytest

from landscape.retrieval.query import retrieve

BASIC_DOC = (
    "Diego Ortega is a senior engineer on the Vision Team. "
    "The Vision Team is based in Austin, Texas. "
    "Diego works on the Sentinel project."
)
TITLE = "chunk-surfacing-test"


async def _clear(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (c:Chunk)-[:PART_OF]->(d:Document {title: $t}) DETACH DELETE c",
            t=title,
        )
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) DETACH DELETE e",
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
