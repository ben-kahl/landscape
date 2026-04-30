"""Regression tests for LLM extraction quality. Marked 'extraction' for selective skipping."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external]

ACCEPTED_APPROVED_FAMILIES = {"APPROVED"}
ACCEPTED_LEADS_FAMILIES = {"LEADS"}
ACCEPTED_ALICE_ACME_FAMILIES = {"WORKS_FOR", "BELONGS_TO", "MEMBER_OF"}
ACCEPTED_ATLAS_STORAGE_FAMILIES = {"USES", "DEPENDS_ON"}
ACCEPTED_SARAH_TEAM_FAMILIES = {"MEMBER_OF", "WORKS_FOR", "BELONGS_TO"}

SARAH_DOC = "Sarah approved the PostgreSQL migration."
ALICE_DOC = "Alice leads Project Atlas at Acme Corp."
FULL_3HOP_DOC = (
    "Alice leads Project Atlas at Acme Corp. "
    "Sarah approved the PostgreSQL migration. "
    "Project Atlas uses PostgreSQL for storage. "
    "Sarah is on the Platform Team."
)


async def _assert_current_memory_relation(
    neo4j_driver,
    *,
    subject_text: str,
    accepted_families: set[str],
    object_text: str,
    title: str,
) -> None:
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {title: $title})-[:ASSERTS]->(a:Assertion)
                  -[:SUPPORTS]->(f:MemoryFact)
            WHERE f.valid_until IS NULL
              AND a.raw_subject_text = $subject_text
              AND a.raw_object_text = $object_text
            OPTIONAL MATCH (a)-[:SUBJECT_ENTITY]->(s:Entity)
            OPTIONAL MATCH (a)-[:OBJECT_ENTITY]->(o:Entity)
            OPTIONAL MATCH (s)-[r:MEMORY_REL {memory_fact_id: f.id}]->(o)
            WHERE r.valid_until IS NULL
            RETURN DISTINCT f.family AS family,
                            f.subtype AS fact_subtype,
                            a.raw_relation_text AS raw_relation_text,
                            a.subtype AS assertion_subtype,
                            s.name AS subject_name,
                            o.name AS object_name,
                            r.family AS rel_family
            """,
            title=title,
            subject_text=subject_text,
            object_text=object_text,
        )
        records = await result.data()
        found_families = {rec["family"] for rec in records if rec["rel_family"] == rec["family"]}
        assert found_families & accepted_families, (
            f"Expected live assertion-backed relation ({subject_text!r}) -[{accepted_families}]-> "
            f"({object_text!r}) in doc {title!r}. Found families: {found_families}. Records: {records}"
        )


async def _ingest_fresh(http_client, neo4j_driver, text: str, title: str) -> dict:
    await _clear_title_graph(neo4j_driver, title)
    response = await http_client.post("/ingest", json={"text": text, "title": title})
    assert response.status_code == 200
    return response.json()


async def _clear_title_graph(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {title: $title})-[:ASSERTS]->(a:Assertion)
            OPTIONAL MATCH (a)-[:SUPPORTS]->(f:MemoryFact)
            RETURN collect(DISTINCT f.id) AS fact_ids
            """,
            title=title,
        )
        record = await result.single()
        fact_ids = [fact_id for fact_id in (record["fact_ids"] if record is not None else []) if fact_id]

        if fact_ids:
            await session.run(
                """
                UNWIND $fact_ids AS fact_id
                MATCH ()-[r:MEMORY_REL {memory_fact_id: fact_id}]-()
                DELETE r
                """,
                fact_ids=fact_ids,
            )
            await session.run(
                """
                UNWIND $fact_ids AS fact_id
                MATCH (f:MemoryFact {id: fact_id})
                DETACH DELETE f
                """,
                fact_ids=fact_ids,
            )

        await session.run(
            """
            MATCH (d:Document {title: $title})-[:ASSERTS]->(a:Assertion)
            DETACH DELETE a
            """,
            title=title,
        )
        await session.run(
            "MATCH (d:Document {title: $title}) DETACH DELETE d",
            title=title,
        )


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_sarah_approved_postgresql(http_client, neo4j_driver):
    title = "extraction-test-sarah-approved"
    await _ingest_fresh(http_client, neo4j_driver, SARAH_DOC, title)
    await _assert_current_memory_relation(
        neo4j_driver,
        subject_text="Sarah",
        accepted_families=ACCEPTED_APPROVED_FAMILIES,
        object_text="PostgreSQL",
        title=title,
    )


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_alice_leads_project_atlas(http_client, neo4j_driver):
    title = "extraction-test-alice-leads"
    await _ingest_fresh(http_client, neo4j_driver, ALICE_DOC, title)
    await _assert_current_memory_relation(
        neo4j_driver,
        subject_text="Alice",
        accepted_families=ACCEPTED_LEADS_FAMILIES,
        object_text="Project Atlas",
        title=title,
    )


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_alice_at_acme_org_relation(http_client, neo4j_driver):
    """At least one of: Alice WORKS_FOR Acme OR Project Atlas BELONGS_TO Acme."""
    title = "extraction-test-alice-acme"
    await _ingest_fresh(http_client, neo4j_driver, ALICE_DOC, title)

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {title: $title})-[:ASSERTS]->(a:Assertion)
                  -[:SUPPORTS]->(f:MemoryFact)
            WHERE f.valid_until IS NULL
              AND a.raw_object_text = $acme
              AND a.raw_subject_text IN [$alice, $atlas]
            OPTIONAL MATCH (a)-[:SUBJECT_ENTITY]->(s:Entity)
            OPTIONAL MATCH (a)-[:OBJECT_ENTITY]->(o:Entity)
            OPTIONAL MATCH (s)-[r:MEMORY_REL {memory_fact_id: f.id}]->(o)
            WHERE r.valid_until IS NULL
            RETURN DISTINCT a.raw_subject_text AS subj,
                            f.family AS family,
                            s.name AS subject_name,
                            o.name AS object_name,
                            r.family AS rel_family
            """,
            title=title,
            acme="Acme Corp",
            alice="Alice",
            atlas="Project Atlas",
        )
        records = await result.data()

    found_families = {
        record["family"] for record in records if record["rel_family"] == record["family"]
    }
    assert found_families & ACCEPTED_ALICE_ACME_FAMILIES, (
        "Expected at least one live assertion-backed org relation to 'Acme Corp' "
        f"from Alice or Project Atlas. Found families: {found_families}. Records: {records}"
    )


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_full_3hop_subgraph(http_client, neo4j_driver):
    """Full 3-hop demo text must produce the complete subgraph for killer demo."""
    title = "extraction-test-3hop"
    await _clear_title_graph(neo4j_driver, title)

    response = await http_client.post("/ingest", json={"text": FULL_3HOP_DOC, "title": title})
    assert response.status_code == 200

    await _assert_current_memory_relation(
        neo4j_driver,
        subject_text="Project Atlas",
        accepted_families=ACCEPTED_ATLAS_STORAGE_FAMILIES,
        object_text="PostgreSQL",
        title=title,
    )
    await _assert_current_memory_relation(
        neo4j_driver,
        subject_text="Sarah",
        accepted_families=ACCEPTED_APPROVED_FAMILIES,
        object_text="PostgreSQL",
        title=title,
    )
    await _assert_current_memory_relation(
        neo4j_driver,
        subject_text="Sarah",
        accepted_families=ACCEPTED_SARAH_TEAM_FAMILIES,
        object_text="Platform Team",
        title=title,
    )
