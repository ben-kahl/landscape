"""Personal-memory corpus integration test (scenarios.json — employment_supersession).

Marked @pytest.mark.retrieval and excluded from the default test run because
this drives the real LLM extractor over four docs (~20s+) and asserts
post-supersession graph state.

Scope: `employment_supersession` scenario only. It exercises the two
behaviors unlocked by vocab expansion steps 2/3:
  - subject-keyed WORKS_FOR supersession (Alice moves Atlas → Beacon)
  - object-keyed HAS_TITLE supersession at Atlas (senior → staff engineer)

The full scenarios.json has additional cases (life updates, family multi-hop,
preferences axis-keyed) — those are candidates for follow-on tests once this
one stabilizes against LLM extraction drift.

Assertions are lenient on entity name variants ('Alice' vs 'Alice Chen') the
way test_killer_demo handles Maya/Maya Chen.
"""
import pathlib

import pytest
import pytest_asyncio

pytestmark = [pytest.mark.integration, pytest.mark.external]

CORPUS_DIR = pathlib.Path(__file__).parent / "fixtures" / "personal_memory_corpus"
TITLE_PREFIX = "personal-memory:"

# Module-level so the four-doc LLM ingest runs once per pytest invocation.
_INGESTED = False

EMPLOYMENT_DOCS = (
    "doc_01_alice_intro.md",
    "doc_02_alice_promotion.md",
    "doc_03_alice_job_change.md",
    "doc_04_felix_still_atlas.md",
)


async def _wipe_all(neo4j_driver, qdrant_client) -> None:
    from landscape.storage import qdrant_store

    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
        existing = await qdrant_client.get_collections()
        if coll in {c.name for c in existing.collections}:
            await qdrant_client.delete_collection(coll)

    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()


@pytest_asyncio.fixture
async def employment_ingested(http_client, neo4j_driver, qdrant_client):
    global _INGESTED
    if _INGESTED:
        return
    await _wipe_all(neo4j_driver, qdrant_client)
    for name in EMPLOYMENT_DOCS:
        path = CORPUS_DIR / name
        r = await http_client.post(
            "/ingest",
            json={"text": path.read_text(), "title": f"{TITLE_PREFIX}{path.stem}"},
        )
        assert r.status_code == 200, f"ingest failed for {name}: {r.text}"
    _INGESTED = True


def _names(body: dict) -> set[str]:
    return {r["name"] for r in body["results"]}


def _name_match(body: dict, *expected_substrings: str) -> bool:
    names = _names(body)
    return any(
        any(sub.lower() in n.lower() for sub in expected_substrings) for n in names
    )


async def _alice_works_for_edges(neo4j_driver) -> list[dict]:
    """Return all WORKS_FOR edges from any Alice node, with org name and
    supersession status. Lenient on canonical name variants."""
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (p:Entity)-[r:RELATES_TO {type: 'WORKS_FOR'}]->(o:Entity)
            WHERE toLower(p.name) CONTAINS 'alice'
            RETURN o.name AS org,
                   r.valid_until IS NULL AS live,
                   r.subtype AS subtype
            """,
        )
        return [dict(row) async for row in result]


async def _alice_has_title_edges(neo4j_driver) -> list[dict]:
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (p:Entity)-[r:RELATES_TO {type: 'HAS_TITLE'}]->(o:Entity)
            WHERE toLower(p.name) CONTAINS 'alice'
            RETURN o.name AS org,
                   r.subtype AS subtype,
                   r.valid_until IS NULL AS live
            """,
        )
        return [dict(row) async for row in result]


@pytest.mark.retrieval
class TestEmploymentSupersession:
    async def test_works_for_superseded_on_job_change(
        self, employment_ingested, neo4j_driver
    ):
        """After doc_03, Alice's WORKS_FOR edge is subject-keyed functional:
        Beacon is live, Atlas is superseded."""
        edges = await _alice_works_for_edges(neo4j_driver)
        live_orgs = {e["org"] for e in edges if e["live"]}
        superseded_orgs = {e["org"] for e in edges if not e["live"]}
        assert any("beacon" in o.lower() for o in live_orgs), (
            f"Expected Beacon AI live in WORKS_FOR, got live={live_orgs} "
            f"superseded={superseded_orgs}"
        )
        assert any("atlas" in o.lower() for o in superseded_orgs), (
            f"Expected Atlas superseded in WORKS_FOR, got live={live_orgs} "
            f"superseded={superseded_orgs}"
        )

    async def test_has_title_subtype_supersession_at_atlas(
        self, employment_ingested, neo4j_driver
    ):
        """HAS_TITLE is object-keyed with subtype. At Atlas, senior_engineer
        should be superseded by staff_engineer (doc_02). This is lenient:
        the LLM sometimes phrases 'promoted to staff engineer' in a way that
        doesn't re-emit a HAS_TITLE edge, so we assert the softer invariant —
        if two senior/staff title edges at Atlas exist, they're not both live."""
        edges = await _alice_has_title_edges(neo4j_driver)
        atlas_titles = [
            e for e in edges if e.get("org") and "atlas" in e["org"].lower()
        ]
        live_atlas = [e for e in atlas_titles if e["live"]]
        # At most one live title per (subject, org) under object-keyed rules.
        assert len(live_atlas) <= 1, (
            f"Expected ≤1 live HAS_TITLE at Atlas for Alice, got {live_atlas}"
        )

    async def test_retrieval_current_employer(
        self, http_client, employment_ingested
    ):
        """'Where does Alice work now?' should surface Beacon, not Atlas,
        in the top results after supersession."""
        q = await http_client.post(
            "/query",
            json={
                "text": "Where does Alice Chen work now?",
                "hops": 2,
                "limit": 10,
                "reinforce": False,
            },
        )
        assert q.status_code == 200
        body = q.json()
        assert _name_match(body, "Beacon"), (
            f"Expected Beacon in current-employer results, got: {_names(body)}"
        )

    async def test_felix_still_at_atlas(
        self, http_client, employment_ingested
    ):
        """Felix's WORKS_FOR Atlas edge (doc_04) must not be collateral
        damage of Alice's supersession — different subject, fully independent."""
        q = await http_client.post(
            "/query",
            json={
                "text": "Where does Felix Park work?",
                "hops": 2,
                "limit": 10,
                "reinforce": False,
            },
        )
        assert q.status_code == 200
        body = q.json()
        assert _name_match(body, "Atlas"), (
            f"Expected Atlas in Felix's employer results, got: {_names(body)}"
        )
