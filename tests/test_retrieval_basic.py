"""Basic retrieval integration tests — single-hop and temporal filter.

The multi-hop killer demo lives in test_retrieval_multihop.py under
the 'retrieval' marker. Tests here run by default."""
import pytest

BASIC_DOC = (
    "Alice leads Project Atlas. Project Atlas uses PostgreSQL for storage. "
    "Sarah Chen approved the PostgreSQL migration. Sarah Chen is on the Platform Team."
)
BASIC_TITLE = "retrieval-basic-test"


async def _clear(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) DETACH DELETE e",
            t=title,
        )
        await session.run("MATCH (d:Document {title: $t}) DETACH DELETE d", t=title)


@pytest.mark.asyncio
async def test_query_returns_seeded_entity(http_client, neo4j_driver):
    """A query for an entity name should return that entity as a top result."""
    await _clear(neo4j_driver, BASIC_TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": BASIC_TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "What does Project Atlas use?", "hops": 2, "limit": 10},
    )
    assert q.status_code == 200
    body = q.json()
    assert body["results"], "query should return at least one result"
    names = {r["name"] for r in body["results"]}
    assert "Project Atlas" in names or "PostgreSQL" in names


@pytest.mark.asyncio
async def test_query_finds_multihop_target(http_client, neo4j_driver):
    """2-hop expansion: ask about Atlas, the answer path includes Sarah."""
    await _clear(neo4j_driver, BASIC_TITLE)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": BASIC_TITLE})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "Project Atlas database approval", "hops": 3, "limit": 10},
    )
    assert q.status_code == 200
    body = q.json()
    names = {r["name"] for r in body["results"]}
    # Sarah should be reachable via Atlas -> PostgreSQL -> Sarah (2 hops)
    assert "Sarah Chen" in names or any("Sarah" in n for n in names), (
        f"Expected Sarah in results via graph expansion, got: {names}"
    )


@pytest.mark.asyncio
async def test_query_reinforces_touched_entities(http_client, neo4j_driver):
    """After a query, the touched entities themselves should have access_count > 0."""
    title = "retrieval-reinforce-test"
    await _clear(neo4j_driver, title)
    r = await http_client.post("/ingest", json={"text": BASIC_DOC, "title": title})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={"text": "Project Atlas", "hops": 2, "limit": 5, "reinforce": True},
    )
    assert q.status_code == 200
    body = q.json()
    assert body["touched_entity_count"] > 0
    assert body["results"]

    # The response reports which neo4j IDs were touched; check them directly.
    touched_ids = [r["neo4j_id"] for r in body["results"]]
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) IN $ids
            RETURN count(e) AS total,
                   sum(CASE WHEN e.access_count > 0 THEN 1 ELSE 0 END) AS reinforced
            """,
            ids=touched_ids,
        )
        record = await result.single()
    assert record["total"] > 0, "touched ids should resolve to entities"
    assert record["reinforced"] > 0, (
        f"expected at least one touched entity to have access_count > 0, "
        f"got {record['reinforced']}/{record['total']}"
    )


@pytest.mark.asyncio
async def test_temporal_filter_excludes_superseded(neo4j_driver):
    """Directly construct a superseded/valid edge pair in Neo4j and verify
    that bfs_expand — the temporal filter at the heart of graph retrieval —
    only returns the currently-valid target. Isolated from LLM extraction
    and Qdrant seeding so the test measures exactly the temporal filter."""
    from landscape.storage import neo4j_store

    subj = "TempAlice"
    old_obj = "TempAcmeCo"
    new_obj = "TempZylosInc"

    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, old_obj, new_obj],
        )
        result = await session.run(
            """
            CREATE (s:Entity {
                name: $subj, type: "PERSON", canonical: true,
                aliases: [], access_count: 0, last_accessed: null,
                source_doc: "retrieval-temporal-test", confidence: 1.0,
                timestamp: datetime()
            })
            CREATE (oldO:Entity {
                name: $old_obj, type: "ORGANIZATION", canonical: true,
                aliases: [], access_count: 0, last_accessed: null,
                source_doc: "retrieval-temporal-test", confidence: 1.0,
                timestamp: datetime()
            })
            CREATE (newO:Entity {
                name: $new_obj, type: "ORGANIZATION", canonical: true,
                aliases: [], access_count: 0, last_accessed: null,
                source_doc: "retrieval-temporal-test", confidence: 1.0,
                timestamp: datetime()
            })
            CREATE (s)-[:RELATES_TO {
                type: "WORKS_FOR", confidence: 0.9,
                source_docs: ["temporal-doc1"],
                valid_from: datetime() - duration({days: 10}),
                valid_until: datetime() - duration({days: 1}),
                superseded_by_doc: "temporal-doc2",
                access_count: 0, last_accessed: null
            }]->(oldO)
            CREATE (s)-[:RELATES_TO {
                type: "WORKS_FOR", confidence: 0.9,
                source_docs: ["temporal-doc2"],
                valid_from: datetime() - duration({days: 1}),
                valid_until: null,
                access_count: 0, last_accessed: null
            }]->(newO)
            RETURN elementId(s) AS sid
            """,
            subj=subj,
            old_obj=old_obj,
            new_obj=new_obj,
        )
        record = await result.single()
        seed_id = record["sid"]

    expansions = await neo4j_store.bfs_expand([seed_id], max_hops=2)
    target_names = {row["target_name"] for row in expansions}

    assert new_obj in target_names, (
        f"Live target {new_obj} should be reachable, got: {target_names}"
    )
    assert old_obj not in target_names, (
        f"Superseded target {old_obj} should be filtered out, got: {target_names}"
    )
