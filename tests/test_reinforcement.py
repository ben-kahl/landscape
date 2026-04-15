"""Longitudinal reinforcement integration tests against the live stack.

test_scoring.py exercises the pure-Python scoring math in isolation.
This file verifies the end-to-end feedback loop: repeated /query calls
on the same topic should monotonically increase the reinforcement
contribution of touched entities, and the total should stay bounded by
settings.reinforcement_cap — the rumination guard.

Uses a tiny hand-authored corpus with novel entity names (Aurora Labs,
Jamie, Aurora Labs telemetry) so the test is robust against cross-test
contamination from other modules that ingest 'Project Atlas' / 'Alice'."""
import pytest

from landscape.config import settings

REINF_DOC = (
    "Aurora Labs is a robotics startup headquartered in Portland. "
    "Aurora Labs uses PostgreSQL as its telemetry datastore. "
    "Jamie Fox leads Aurora Labs."
)


async def _clear(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity)-[:EXTRACTED_FROM]->(d:Document {title: $t}) "
            "DETACH DELETE e",
            t=title,
        )
        await session.run(
            "MATCH (d:Document {title: $t}) DETACH DELETE d", t=title
        )


async def _query(client, text: str) -> dict:
    r = await client.post(
        "/query",
        json={"text": text, "hops": 2, "limit": 10, "reinforce": True},
    )
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_reinforcement_grows_monotonically_with_repeated_queries(
    http_client, neo4j_driver
):
    """Repeated queries on the same topic should produce a non-decreasing
    reinforcement sequence for the entity being touched. touch_entities
    increments access_count between queries, so the scoring contribution
    log1p(access_count) * decay climbs monotonically until it hits the
    configured cap.

    We track an entity that appears in every query result (it's the one
    being reinforced) rather than asserting starting-at-zero, because
    cross-test contamination can give us entities whose access_count is
    already nonzero from earlier test modules. The invariant we care about
    is non-decrease + bounded."""
    title = "reinforcement-monotonic-test"
    await _clear(neo4j_driver, title)
    r = await http_client.post("/ingest", json={"text": REINF_DOC, "title": title})
    assert r.status_code == 200

    history: list[dict[str, float]] = []
    for _ in range(8):
        body = await _query(http_client, "Aurora Labs")
        history.append({item["name"]: item["reinforcement"] for item in body["results"]})

    # Pick an entity that shows up in every query — it's the one being
    # continuously reinforced across the loop.
    common = set(history[0].keys())
    for snap in history[1:]:
        common &= set(snap.keys())
    assert common, (
        f"No entity shared across all 8 queries: {[sorted(s) for s in history]}"
    )

    cap = settings.reinforcement_cap

    # Every common entity's reinforcement sequence must be non-decreasing
    # and bounded. At least one should grow strictly (proving the feedback
    # loop is active, not just flat-at-cap from prior contamination).
    grew_strictly = False
    for pick in common:
        seq = [snap[pick] for snap in history]
        for i in range(len(seq) - 1):
            assert seq[i + 1] >= seq[i] - 1e-9, (
                f"Reinforcement for {pick} decreased between queries "
                f"{i} and {i + 1}: {seq}"
            )
        for i, val in enumerate(seq):
            assert val <= cap + 1e-9, (
                f"Reinforcement for {pick} at query {i} is {val}, "
                f"exceeds cap {cap}"
            )
        if seq[-1] > seq[0] + 1e-9:
            grew_strictly = True

    assert grew_strictly, (
        f"Expected at least one entity's reinforcement to strictly grow "
        f"across 8 queries (active feedback loop), but all remained flat. "
        f"Sequences: "
        f"{ {p: [snap[p] for snap in history] for p in common} }"
    )


@pytest.mark.asyncio
async def test_touch_writes_access_count_and_last_accessed(http_client, neo4j_driver):
    """A single query with reinforce=True should leave access_count > 0
    and a non-null last_accessed on every touched entity."""
    title = "reinforcement-touch-test"
    await _clear(neo4j_driver, title)
    r = await http_client.post("/ingest", json={"text": REINF_DOC, "title": title})
    assert r.status_code == 200

    q = await http_client.post(
        "/query",
        json={
            "text": "Aurora Labs telemetry datastore",
            "hops": 2,
            "limit": 10,
            "reinforce": True,
        },
    )
    assert q.status_code == 200
    touched_ids = [item["neo4j_id"] for item in q.json()["results"]]
    assert touched_ids, "query should return at least one result"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) IN $ids
            RETURN e.name AS name,
                   coalesce(e.access_count, 0) AS ac,
                   e.last_accessed AS la
            """,
            ids=touched_ids,
        )
        rows = [dict(row) async for row in result]

    assert rows, "touched ids should resolve to entities"
    for row in rows:
        assert row["ac"] > 0, (
            f"Entity {row['name']} access_count was not incremented: {row}"
        )
        assert row["la"] is not None, (
            f"Entity {row['name']} last_accessed was not set: {row}"
        )


@pytest.mark.asyncio
async def test_reinforcement_bounded_after_many_queries(http_client, neo4j_driver):
    """Rumination guard: after 20 repeated queries on the same topic, the
    reinforcement contribution in the ranked results must still be bounded
    by reinforcement_cap. With cap=2.0 and log1p(n) growth, the raw signal
    crosses the cap at n ≈ e^2 - 1 ≈ 6.4, so 20 iterations exercises the
    min() clamp in reinforcement_score several times over."""
    title = "reinforcement-bound-test"
    await _clear(neo4j_driver, title)
    r = await http_client.post("/ingest", json={"text": REINF_DOC, "title": title})
    assert r.status_code == 200

    cap = settings.reinforcement_cap
    last_body = None
    for _ in range(20):
        last_body = await _query(http_client, "Aurora Labs")

    assert last_body is not None and last_body["results"], (
        "query should return results"
    )
    for item in last_body["results"]:
        assert item["reinforcement"] <= cap + 1e-9, (
            f"Reinforcement {item['reinforcement']} for {item['name']} "
            f"exceeds cap {cap} after 20 queries"
        )

    # At least one result should have saturated close to the cap, otherwise
    # the feedback loop isn't actually cranking. log1p(20) ≈ 3.04 > cap=2.0.
    max_reinf = max(item["reinforcement"] for item in last_body["results"])
    assert max_reinf >= cap - 0.1, (
        f"After 20 reinforcing queries, max observed reinforcement was "
        f"{max_reinf}, expected close to cap={cap}. Feedback loop may be "
        f"inactive."
    )
