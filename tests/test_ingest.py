import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from landscape.extraction.schema import Extraction

TEST_DOC = "Alice leads Project Atlas at Acme Corp. Project Atlas uses PostgreSQL for storage."
TEST_TITLE = "test-doc-integration"


def test_extraction_schema_accepts_quantified_relation_fields():
    extraction = Extraction.model_validate(
        {
            "entities": [
                {
                    "name": "Eric",
                    "type": "PERSON",
                    "confidence": 0.95,
                    "aliases": [],
                },
                {
                    "name": "Netflix",
                    "type": "TECHNOLOGY",
                    "confidence": 0.9,
                    "aliases": [],
                },
            ],
            "relations": [
                {
                    "subject": "Eric",
                    "object": "Netflix",
                    "relation_type": "DISCUSSED",
                    "subtype": "watched",
                    "confidence": 0.9,
                    "quantity_value": 8,
                    "quantity_unit": "hours",
                    "quantity_kind": "duration",
                    "time_scope": "today",
                }
            ],
        }
    )

    relation = extraction.relations[0]
    assert relation.quantity_value == 8
    assert relation.quantity_unit == "hours"
    assert relation.quantity_kind == "duration"
    assert relation.time_scope == "today"


@pytest.mark.asyncio
async def test_ingest_creates_graph_and_vectors(http_client, neo4j_driver, qdrant_client):
    # Clear any prior state for this test title
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (d:Document {title: $title})<-[:EXTRACTED_FROM]-(e:Entity) DETACH DELETE e, d",
            title=TEST_TITLE,
        )
        await session.run("MATCH (d:Document {title: $title}) DETACH DELETE d", title=TEST_TITLE)

    response = await http_client.post(
        "/ingest",
        json={"text": TEST_DOC, "title": TEST_TITLE},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["already_existed"] is False
    # With resolution enabled, some entities may be reinforced rather than created
    assert body["entities_created"] + body["entities_reinforced"] >= 3
    assert body["relations_created"] + body["relations_reinforced"] >= 2
    # New fields present in response
    assert "entities_reinforced" in body
    assert "relations_reinforced" in body
    assert "relations_superseded" in body
    assert "chunks_created" in body
    assert body["chunks_created"] >= 1

    # Verify Neo4j graph
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $title})"
            " RETURN count(e) AS cnt",
            title=TEST_TITLE,
        )
        record = await result.single()
        # Resolved entities don't get a new EXTRACTED_FROM edge; at least entities_created will
        assert record["cnt"] >= body["entities_created"]

        result = await session.run(
            """
            MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $title})
            WITH collect(e) AS entities
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE a IN entities AND b IN entities AND r.valid_until IS NULL
            RETURN count(*) AS cnt
            """,
            title=TEST_TITLE,
        )
        record = await result.single()
        assert record["cnt"] >= 2

        result = await session.run(
            "MATCH (d:Document {title: $title}) RETURN count(d) AS cnt",
            title=TEST_TITLE,
        )
        record = await result.single()
        assert record["cnt"] == 1

    # Verify Qdrant: at least entities_created vectors with this source_doc exist,
    # each with a valid neo4j_node_id. (Stale points from prior runs may inflate the count.)
    points, _ = await qdrant_client.scroll(
        collection_name="entities",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=TEST_TITLE))]
        ),
        with_payload=True,
        limit=100,
    )
    assert len(points) >= body["entities_created"]
    for point in points:
        assert point.payload.get("neo4j_node_id"), "Missing neo4j_node_id in Qdrant payload"

    # Verify chunks collection has entries for this doc
    chunk_points, _ = await qdrant_client.scroll(
        collection_name="chunks",
        scroll_filter=Filter(
            must=[FieldCondition(key="source_doc", match=MatchValue(value=TEST_TITLE))]
        ),
        with_payload=True,
        limit=100,
    )
    assert len(chunk_points) >= body["chunks_created"]
    for cp in chunk_points:
        assert cp.payload.get("chunk_neo4j_id"), "Missing chunk_neo4j_id in Qdrant payload"


@pytest.mark.asyncio
async def test_ingest_idempotent(http_client):
    # First ingest
    r1 = await http_client.post("/ingest", json={"text": TEST_DOC, "title": TEST_TITLE})
    assert r1.status_code == 200

    # Second ingest with same text — should short-circuit
    r2 = await http_client.post("/ingest", json={"text": TEST_DOC, "title": TEST_TITLE})
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["already_existed"] is True
    assert body2["entities_created"] == 0
    assert body2["relations_created"] == 0
    assert body2["entities_reinforced"] == 0
    assert body2["relations_reinforced"] == 0
    assert body2["relations_superseded"] == 0
    assert body2["chunks_created"] == 0
