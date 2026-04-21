"""Multi-hop retrieval tests against the Helios Robotics killer-demo corpus.

Marked @pytest.mark.retrieval and excluded from the default test run because
(a) full corpus ingest takes ~7 LLM extraction calls and (b) results depend
on local LLM extraction quality. Run with:

    pytest -m retrieval

This is the central Phase 2 proof: hybrid retrieval should answer 1-, 2-,
and 3-hop questions over a corpus where the answer requires traversing
multiple documents."""
import pathlib

import pytest
import pytest_asyncio

pytestmark = [pytest.mark.integration, pytest.mark.external]

CORPUS_DIR = pathlib.Path(__file__).parent / "fixtures" / "killer_demo_corpus"
TITLE_PREFIX = "killer-demo:"

# Module-level flag so the seven-doc LLM ingest only runs once per pytest
# invocation, even though the fixture itself is function-scoped (required by
# the autouse driver-reset fixture in conftest.py).
_INGESTED = False


def _doc_title(path: pathlib.Path) -> str:
    return f"{TITLE_PREFIX}{path.stem}"


async def _wipe_all(neo4j_driver, qdrant_client) -> None:
    """Full wipe of Neo4j + Qdrant so the killer-demo corpus is the only
    data visible during the test run. Safe because -m retrieval only runs
    this module; other test modules set up their own data in their own
    fixtures."""
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
async def killer_demo_ingested(http_client, neo4j_driver, qdrant_client):
    """Ingest the corpus exactly once per pytest invocation; subsequent tests
    reuse the populated graph state. Function-scoped because module-scoped
    async fixtures fight with the autouse driver-reset in conftest.py."""
    global _INGESTED
    if _INGESTED:
        return
    await _wipe_all(neo4j_driver, qdrant_client)
    for path in sorted(CORPUS_DIR.glob("*.md")):
        r = await http_client.post(
            "/ingest",
            json={"text": path.read_text(), "title": _doc_title(path)},
        )
        assert r.status_code == 200, f"ingest failed for {path.name}: {r.text}"
    _INGESTED = True


def _names(body: dict) -> set[str]:
    return {r["name"] for r in body["results"]}


def _name_match(body: dict, *expected_substrings: str) -> bool:
    """Lenient: any returned name contains any of the expected substrings.
    Tolerates LLM extraction variation like 'Maya Chen' vs 'Maya'."""
    names = _names(body)
    return any(
        any(sub.lower() in n.lower() for sub in expected_substrings) for n in names
    )


@pytest.mark.retrieval
class TestKillerDemo:
    async def test_one_hop_vision_team_lead(self, http_client, killer_demo_ingested):
        """1-hop: 'Who leads the Vision Team?' — Diego is one edge away."""
        q = await http_client.post(
            "/query",
            json={
                "text": "Who leads the Vision Team?",
                "hops": 2,
                "limit": 10,
                "reinforce": False,
            },
        )
        assert q.status_code == 200
        body = q.json()
        assert _name_match(body, "Diego"), (
            f"Expected Diego in results, got: {_names(body)}"
        )

    async def test_one_hop_sentinel_vision_tech(
        self, http_client, killer_demo_ingested
    ):
        """1-hop: 'What does Sentinel use for vision?' — PyTorch is one edge away."""
        q = await http_client.post(
            "/query",
            json={
                "text": "What does Project Sentinel use for computer vision?",
                "hops": 2,
                "limit": 10,
                "reinforce": False,
            },
        )
        assert q.status_code == 200
        body = q.json()
        assert _name_match(body, "PyTorch"), (
            f"Expected PyTorch in results, got: {_names(body)}"
        )

    async def test_two_hop_aurora_db_approver(
        self, http_client, killer_demo_ingested
    ):
        """2-hop: 'Who approved the database for Aurora?'

        Path: Aurora --USES--> PostgreSQL <--APPROVED-- Maya Chen.
        The two facts live in separate documents (03 and 04)."""
        q = await http_client.post(
            "/query",
            json={
                "text": "Who approved the database for Project Aurora?",
                "hops": 3,
                "limit": 15,
                "reinforce": False,
            },
        )
        assert q.status_code == 200
        body = q.json()
        assert _name_match(body, "Maya"), (
            f"Expected Maya in 2-hop result via Aurora->PostgreSQL->Maya, "
            f"got: {_names(body)}"
        )

    async def test_three_hop_team_of_aurora_db_approver(
        self, http_client, killer_demo_ingested
    ):
        """3-hop marquee: 'What team does the person who approved Aurora's
        database lead?'

        Path: Aurora --USES--> PostgreSQL <--APPROVED-- Maya Chen --LEADS-->
        Platform Team. The three facts live in three different documents
        (03, 04, 01); no single chunk contains the chain. This is the
        central Phase 2 proof — vector-only retrieval cannot answer it."""
        q = await http_client.post(
            "/query",
            json={
                "text": (
                    "What team does the person who approved Aurora's database lead?"
                ),
                "hops": 3,
                "limit": 20,
                "reinforce": False,
            },
        )
        assert q.status_code == 200
        body = q.json()
        assert _name_match(body, "Platform Team", "Platform"), (
            f"Expected 'Platform Team' in 3-hop result, got: {_names(body)}"
        )
