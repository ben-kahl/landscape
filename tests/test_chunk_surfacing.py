"""Chunk surfacing and chunk-seed similarity propagation tests."""

import pytest

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
    assert any(r["vector_sim"] > 0.0 for r in body["results"]), (
        "chunk-seeded entities should inherit chunk similarity, not 0.0"
    )
