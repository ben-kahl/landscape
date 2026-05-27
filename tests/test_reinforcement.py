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

pytestmark = pytest.mark.integration

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
    touched_ids = [item["entity_id"] for item in q.json()["results"]]
    assert touched_ids, "query should return at least one result"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE e.id IN $ids
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


@pytest.mark.asyncio
async def test_write_bumps_entity_access_count(neo4j_driver):
    """Each merge_entity call increments access_count and updates
    last_accessed. A cold-start entity created via merge_entity becomes
    access_count=1 immediately; repeated upserts climb from there."""
    from landscape.storage import neo4j_store

    # Clean slate for this specific entity
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity {name: $n, type: $t}) DETACH DELETE e",
            n="Reinforce Entity Probe",
            t="Concept",
        )

    for _ in range(3):
        await neo4j_store.merge_entity(
            name="Reinforce Entity Probe",
            entity_type="Concept",
            source_doc="phase-3.5-test",
            confidence=0.9,
            model="test",
            created_by="ingest",
        )

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: $n}) RETURN e.access_count AS c, "
            "e.last_accessed AS la",
            n="Reinforce Entity Probe",
        )
        row = await result.single()

    assert row is not None
    assert row["c"] == 3
    assert row["la"] is not None

    # Cleanup
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity {name: $n}) DETACH DELETE e",
            n="Reinforce Entity Probe",
        )


@pytest.mark.asyncio
async def test_write_bumps_relation_access_count(neo4j_driver):
    """Each repeated add_relation call should increment access_count and update
    last_accessed on the live memory edge."""
    from landscape.writeback import add_relation

    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (:Entity {name: $s})-[r:MEMORY_REL {family: $t}]->(:Entity {name: $o}) "
            "DELETE r",
            s="Reinforce Relation Subj",
            t="USES",
            o="Reinforce Relation Obj",
        )
        await session.run(
            "MATCH (e:Entity {name: $n}) DETACH DELETE e",
            n="Reinforce Relation Subj",
        )
        await session.run(
            "MATCH (e:Entity {name: $n}) DETACH DELETE e",
            n="Reinforce Relation Obj",
        )

    from landscape.storage import neo4j_store

    await neo4j_store.merge_entity(
        name="Reinforce Relation Subj",
        entity_type="Concept",
        source_doc="phase-3.5-test",
        confidence=0.9,
        model="test",
        created_by="ingest",
    )
    await neo4j_store.merge_entity(
        name="Reinforce Relation Obj",
        entity_type="Concept",
        source_doc="phase-3.5-test",
        confidence=0.9,
        model="test",
        created_by="ingest",
    )

    for _ in range(3):
        await add_relation(
            "Reinforce Relation Subj",
            "Concept",
            "Reinforce Relation Obj",
            "Concept",
            "USES",
            source="phase-3.5-test",
            session_id="s-rel",
            turn_id="t-rel",
        )

    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (:Entity {name: $s})-[r:MEMORY_REL {family: $t}]->(:Entity {name: $o}) "
            "WHERE r.valid_until IS NULL "
            "RETURN coalesce(r.access_count, 0) AS c, r.last_accessed AS la",
            s="Reinforce Relation Subj",
            t="USES",
            o="Reinforce Relation Obj",
        )
        row = await result.single()

    assert row is not None
    assert row["c"] == 3
    assert row["la"] is not None


@pytest.mark.asyncio
async def test_reasserted_fact_outranks_cold_fact(
    http_client, neo4j_driver, monkeypatch
):
    """A repeatedly reasserted fact should rank above an equivalent cold fact."""
    from landscape.config import settings
    from landscape.embeddings import encoder
    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store, qdrant_store

    names = ["Reinforce Subject", "Warm Stack", "Cold Stack"]
    subject_vector = [1.0] + [0.0] * (settings.embedding_dims - 1)

    monkeypatch.setattr(encoder, "embed_query", lambda _text: subject_vector)

    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=names,
        )

    subject_id = await neo4j_store.merge_entity(
        name="Reinforce Subject",
        entity_type="Concept",
        source_doc="phase-3.5-test",
        confidence=0.9,
        model="test",
        created_by="ingest",
    )
    warm_id = await neo4j_store.merge_entity(
        name="Warm Stack",
        entity_type="Concept",
        source_doc="phase-3.5-test",
        confidence=0.9,
        model="test",
        created_by="ingest",
    )
    cold_id = await neo4j_store.merge_entity(
        name="Cold Stack",
        entity_type="Concept",
        source_doc="phase-3.5-test",
        confidence=0.9,
        model="test",
        created_by="ingest",
    )

    fixed_ts = "2026-04-20T00:00:00+00:00"
    await qdrant_store.upsert_entity(
        entity_id=subject_id,
        name="Reinforce Subject",
        entity_type="Concept",
        source_doc="phase-3.5-test",
        timestamp=fixed_ts,
        vector=subject_vector,
    )

    async with neo4j_driver.session() as session:
        await session.run(
            """
            MATCH (e:Entity)
            WHERE e.name IN ['Reinforce Subject', 'Warm Stack', 'Cold Stack']
            SET e.access_count = 1,
                e.last_accessed = datetime($ts)
            """,
            ts=fixed_ts,
        )

    warm_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="phase-3.5-warm",
            raw_subject_text="Reinforce Subject",
            raw_relation_text="uses",
            raw_object_text="Warm Stack",
            confidence=0.9,
            family_candidate="USES",
        )
    )
    cold_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="phase-3.5-cold",
            raw_subject_text="Reinforce Subject",
            raw_relation_text="uses",
            raw_object_text="Cold Stack",
            confidence=0.9,
            family_candidate="USES",
        )
    )

    await neo4j_store.upsert_memory_fact_from_assertion(
        family="USES",
        subject_entity_id=subject_id,
        object_entity_id=cold_id,
        subtype=None,
        confidence=0.9,
        assertion_id=cold_assertion,
    )
    for _ in range(4):
        await neo4j_store.upsert_memory_fact_from_assertion(
            family="USES",
            subject_entity_id=subject_id,
            object_entity_id=warm_id,
            subtype=None,
            confidence=0.9,
            assertion_id=warm_assertion,
        )

    # Normalize recency so the ranking delta comes from access_count
    # reinforcement, not a fresher last_accessed timestamp on the warm edge.
    async with neo4j_driver.session() as session:
        await session.run(
            """
            MATCH (:Entity {name: 'Reinforce Subject'})
                  -[r:MEMORY_REL {family: 'USES'}]->(:Entity)
            WHERE r.valid_until IS NULL
              AND endNode(r).name IN ['Warm Stack', 'Cold Stack']
            MATCH (f:MemoryFact {id: r.memory_fact_id})
            SET r.last_accessed = datetime($ts),
                f.last_accessed = datetime($ts)
            """,
            ts=fixed_ts,
        )

    response = await http_client.post(
        "/query",
        json={
            "text": "Which fact is more established?",
            "hops": 1,
            "limit": 10,
            "reinforce": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    names_in_order = [r["name"] for r in body["results"]]
    warm = next(r for r in body["results"] if r["name"] == "Warm Stack")
    cold = next(r for r in body["results"] if r["name"] == "Cold Stack")

    assert warm["distance"] == cold["distance"] == 1
    assert abs(warm["vector_sim"] - cold["vector_sim"]) < 1e-9
    assert warm["reinforcement"] > cold["reinforcement"]
    assert warm["score"] > cold["score"]
    assert names_in_order.index("Warm Stack") < names_in_order.index("Cold Stack"), (
        f"Expected warm to rank above cold; got order {names_in_order}"
    )

    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=names,
        )
