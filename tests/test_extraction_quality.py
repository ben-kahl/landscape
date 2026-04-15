"""Regression tests for LLM extraction quality. Marked 'extraction' for selective skipping."""
import pytest

ACCEPTED_APPROVED_SYNONYMS = {"APPROVED", "APPROVED_BY", "AUTHORIZED", "SANCTIONED", "SIGNED_OFF"}
ACCEPTED_LEADS_SYNONYMS = {"LEADS", "LED_BY", "MANAGES", "MANAGED_BY", "OWNS", "HEADED_BY"}
ACCEPTED_WORKS_FOR_SYNONYMS = {
    "WORKS_FOR", "EMPLOYED_BY", "MEMBER_OF", "PART_OF", "BELONGS_TO", "AFFILIATED_WITH"
}
ACCEPTED_BELONGS_TO_SYNONYMS = {
    "BELONGS_TO", "PART_OF", "OWNED_BY", "UNDER", "WITHIN", "AFFILIATED_WITH"
}

SARAH_DOC = "Sarah approved the PostgreSQL migration."
ALICE_DOC = "Alice leads Project Atlas at Acme Corp."
FULL_3HOP_DOC = (
    "Alice leads Project Atlas at Acme Corp. "
    "Sarah approved the PostgreSQL migration. "
    "Project Atlas uses PostgreSQL for storage. "
    "Sarah is on the Platform Team."
)


async def _assert_edge(
    neo4j_driver, subject: str, accepted_types: set[str], obj: str, title: str
) -> None:
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: $subject})-[r:RELATES_TO]->(o:Entity {name: $object})
            WHERE r.valid_until IS NULL
            RETURN r.type AS rel_type
            """,
            subject=subject,
            object=obj,
        )
        records = await result.data()
        found_types = {rec["rel_type"] for rec in records}
        assert found_types & accepted_types, (
            f"Expected edge ({subject!r}) -[{accepted_types}]-> ({obj!r}) in doc {title!r}. "
            f"Found relation types: {found_types}"
        )


async def _ingest_fresh(http_client, neo4j_driver, text: str, title: str) -> dict:
    # Clear prior state for this title
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (d:Document {title: $title}) DETACH DELETE d",
            title=title,
        )
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) DETACH DELETE e",
            t=title,
        )
    # Also delete orphan entities from prior runs that may interfere
    async with neo4j_driver.session() as session:
        await session.run(
            """
            MATCH (s:Entity {name: $s})-[r:RELATES_TO]->(o:Entity)
            WHERE r.valid_until IS NULL
            DELETE r
            """,
            s="Sarah",
        )

    response = await http_client.post("/ingest", json={"text": text, "title": title})
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_sarah_approved_postgresql(http_client, neo4j_driver):
    title = "extraction-test-sarah-approved"
    await _ingest_fresh(http_client, neo4j_driver, SARAH_DOC, title)
    await _assert_edge(neo4j_driver, "Sarah", ACCEPTED_APPROVED_SYNONYMS, "PostgreSQL", title)


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_alice_leads_project_atlas(http_client, neo4j_driver):
    title = "extraction-test-alice-leads"
    await _ingest_fresh(http_client, neo4j_driver, ALICE_DOC, title)
    await _assert_edge(neo4j_driver, "Alice", ACCEPTED_LEADS_SYNONYMS, "Project Atlas", title)


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_alice_at_acme_org_relation(http_client, neo4j_driver):
    """At least one of: Alice WORKS_FOR Acme OR Project Atlas BELONGS_TO Acme."""
    title = "extraction-test-alice-acme"
    await _ingest_fresh(http_client, neo4j_driver, ALICE_DOC, title)

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity {name: $acme})
            WHERE r.valid_until IS NULL AND s.name IN [$alice, $atlas]
            RETURN s.name AS subj, r.type AS rel_type
            """,
            acme="Acme Corp",
            alice="Alice",
            atlas="Project Atlas",
        )
        records = await result.data()

    assert records, (
        "Expected at least one relation to 'Acme Corp' from Alice or Project Atlas. Got none."
    )


@pytest.mark.asyncio
@pytest.mark.extraction
async def test_full_3hop_subgraph(http_client, neo4j_driver):
    """Full 3-hop demo text must produce the complete subgraph for killer demo."""
    title = "extraction-test-3hop"
    # Clear any stale relations that could pollute traversal
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (d:Document {title: $title}) DETACH DELETE d",
            title=title,
        )

    response = await http_client.post("/ingest", json={"text": FULL_3HOP_DOC, "title": title})
    assert response.status_code == 200

    # Atlas USES PostgreSQL
    await _assert_edge(
        neo4j_driver,
        "Project Atlas",
        {"USES", "UTILIZES", "STORES_WITH", "POWERED_BY"},
        "PostgreSQL",
        title,
    )
    # Sarah APPROVED PostgreSQL
    await _assert_edge(neo4j_driver, "Sarah", ACCEPTED_APPROVED_SYNONYMS, "PostgreSQL", title)
    # Sarah on Platform Team
    sarah_team_synonyms = {"MEMBER_OF", "PART_OF", "ON", "WORKS_FOR", "BELONGS_TO", "EMPLOYED_BY"}
    await _assert_edge(neo4j_driver, "Sarah", sarah_team_synonyms, "Platform Team", title)
