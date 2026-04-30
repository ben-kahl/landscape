"""Deterministic supersession scenarios for A/B benchmarking.

Six scenarios that write graph state directly via neo4j_store and
persist_assertion_and_maybe_promote(). No LLM involved.
Verification is via direct Cypher queries.

Usage from bench_ab.py:
    from scripts.fixtures.supersession_scenarios import run_all_scenarios
    results = await run_all_scenarios()
    # results: list[ScenarioResult]
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.service import persist_assertion_and_maybe_promote
from landscape.storage import neo4j_store


@dataclass
class ScenarioResult:
    id: int
    name: str
    passed: bool
    error: str = ""


async def _wipe_entities(*names: str) -> None:
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=list(names),
        )


async def _make_doc(label: str) -> str:
    h = hashlib.sha256(label.encode()).hexdigest()
    doc_id, _ = await neo4j_store.merge_document(h, label, "text")
    return doc_id


async def _make_entity(name: str, entity_type: str) -> str:
    return await neo4j_store.merge_entity(
        name=name,
        entity_type=entity_type,
        source_doc="supersession-bench",
        confidence=0.9,
    )


async def _assert_fact(
    doc_id: str,
    subject_id: str,
    object_id: str,
    subject_name: str,
    rel_type: str,
    object_name: str,
    *,
    subtype: str | None = None,
) -> None:
    payload = AssertionPayload(
        source_kind="document",
        source_id=doc_id,
        raw_subject_text=subject_name,
        raw_relation_text=rel_type,
        raw_object_text=object_name,
        confidence=0.9,
        family_candidate=rel_type,
        subtype=subtype,
    )
    await persist_assertion_and_maybe_promote(
        payload,
        source_node_id=doc_id,
        source_kind="document",
        subject_entity_id=subject_id,
        object_entity_id=object_id,
        chunk_ids=[],
    )


async def _live_objects(subject_id: str, family: str) -> list[str]:
    """Return names of live object entities for subject via a specific family."""
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {id: $sid})-[r:MEMORY_REL]->(o:Entity)
            WHERE r.family = $family AND r.valid_until IS NULL
            RETURN o.name AS name
            """,
            sid=subject_id,
            family=family,
        )
        return [rec["name"] async for rec in result]


async def _stale_objects(subject_id: str, family: str) -> list[str]:
    """Return names of stale object entities for subject via a specific family."""
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {id: $sid})-[r:MEMORY_REL]->(o:Entity)
            WHERE r.family = $family AND r.valid_until IS NOT NULL
            RETURN o.name AS name
            """,
            sid=subject_id,
            family=family,
        )
        return [rec["name"] async for rec in result]


# ---------------------------------------------------------------------------
# Scenario 1: subject-keyed WORKS_FOR supersession
# ---------------------------------------------------------------------------

async def _scenario_1() -> ScenarioResult:
    """Alice works at Acme then moves to Beacon; Acme edge must be stale."""
    names = ["Alice-S1", "Acme Corp-S1", "Beacon Corp-S1"]
    await _wipe_entities(*names)
    try:
        doc = await _make_doc("bench-s1")
        alice = await _make_entity("Alice-S1", "Person")
        acme = await _make_entity("Acme Corp-S1", "Organization")
        beacon = await _make_entity("Beacon Corp-S1", "Organization")

        await _assert_fact(doc, alice, acme, "Alice-S1", "WORKS_FOR", "Acme Corp-S1")
        await _assert_fact(doc, alice, beacon, "Alice-S1", "WORKS_FOR", "Beacon Corp-S1")

        live = await _live_objects(alice, "WORKS_FOR")
        stale = await _stale_objects(alice, "WORKS_FOR")

        passed = "Beacon Corp-S1" in live and "Acme Corp-S1" in stale
        error = "" if passed else f"live={live!r} stale={stale!r}"
        return ScenarioResult(id=1, name="subject_keyed_works_for", passed=passed, error=error)
    finally:
        await _wipe_entities(*names)


# ---------------------------------------------------------------------------
# Scenario 2: object-keyed HAS_TITLE supersession
# ---------------------------------------------------------------------------

async def _scenario_2() -> ScenarioResult:
    """Alice's title at Atlas goes from senior_engineer to principal_engineer."""
    names = ["Alice-S2", "Atlas Corp-S2"]
    await _wipe_entities(*names)
    try:
        doc = await _make_doc("bench-s2")
        alice = await _make_entity("Alice-S2", "Person")
        atlas = await _make_entity("Atlas Corp-S2", "Organization")

        await _assert_fact(
            doc, alice, atlas, "Alice-S2", "HAS_TITLE", "Atlas Corp-S2",
            subtype="senior_engineer",
        )
        await _assert_fact(
            doc, alice, atlas, "Alice-S2", "HAS_TITLE", "Atlas Corp-S2",
            subtype="principal_engineer",
        )

        driver = neo4j_store.get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Entity {id: $sid})-[r:MEMORY_REL]->(o:Entity {id: $oid})
                WHERE r.family = 'HAS_TITLE'
                RETURN r.subtype AS subtype, r.valid_until IS NULL AS live
                """,
                sid=alice,
                oid=atlas,
            )
            rows = [{"subtype": rec["subtype"], "live": rec["live"]} async for rec in result]

        live_subtypes = {r["subtype"] for r in rows if r["live"]}
        stale_subtypes = {r["subtype"] for r in rows if not r["live"]}
        passed = "principal_engineer" in live_subtypes and "senior_engineer" in stale_subtypes
        error = "" if passed else f"live={live_subtypes!r} stale={stale_subtypes!r}"
        return ScenarioResult(id=2, name="object_keyed_has_title", passed=passed, error=error)
    finally:
        await _wipe_entities(*names)


# ---------------------------------------------------------------------------
# Scenario 3: additive LEADS (no supersession)
# ---------------------------------------------------------------------------

async def _scenario_3() -> ScenarioResult:
    """Diego leads both VisionTeam and Sentinel; both edges must stay live."""
    names = ["Diego-S3", "VisionTeam-S3", "Sentinel-S3"]
    await _wipe_entities(*names)
    try:
        doc = await _make_doc("bench-s3")
        diego = await _make_entity("Diego-S3", "Person")
        vision = await _make_entity("VisionTeam-S3", "Organization")
        sentinel = await _make_entity("Sentinel-S3", "Project")

        await _assert_fact(doc, diego, vision, "Diego-S3", "LEADS", "VisionTeam-S3")
        await _assert_fact(doc, diego, sentinel, "Diego-S3", "LEADS", "Sentinel-S3")

        live = await _live_objects(diego, "LEADS")
        stale = await _stale_objects(diego, "LEADS")

        passed = "VisionTeam-S3" in live and "Sentinel-S3" in live and len(stale) == 0
        error = "" if passed else f"live={live!r} stale={stale!r}"
        return ScenarioResult(id=3, name="additive_leads", passed=passed, error=error)
    finally:
        await _wipe_entities(*names)


# ---------------------------------------------------------------------------
# Scenario 4: alias seed resolution
# ---------------------------------------------------------------------------

async def _scenario_4() -> ScenarioResult:
    """'Bob' is an alias for Robert; alias resolves to Robert's canonical node."""
    names = ["Robert-S4"]
    await _wipe_entities(*names)
    try:
        robert = await _make_entity("Robert-S4", "Person")
        await neo4j_store.merge_alias(robert, "Bob-S4", "supersession-bench", 0.95)

        driver = neo4j_store.get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Alias {name: $alias})-[:SAME_AS]->(e:Entity)
                RETURN e.id AS canonical_id
                """,
                alias="Bob-S4",
            )
            rec = await result.single()

        canonical_id = rec["canonical_id"] if rec else None
        passed = rec is not None and canonical_id == robert
        error = "" if passed else f"canonical_id={canonical_id!r}"
        return ScenarioResult(id=4, name="alias_seed_resolution", passed=passed, error=error)
    finally:
        driver = neo4j_store.get_driver()
        try:
            await _wipe_entities(*names)
        finally:
            async with driver.session() as session:
                await session.run("MATCH (a:Alias {name: 'Bob-S4'}) DETACH DELETE a")


# ---------------------------------------------------------------------------
# Scenario 5: interval propagation — MEMORY_REL.valid_until matches MemoryFact
# ---------------------------------------------------------------------------

async def _scenario_5() -> ScenarioResult:
    """Superseding a WORKS_FOR fact sets MEMORY_REL.valid_until to the same
    timestamp as the MemoryFact that was closed."""
    names = ["Carol-S5", "OldCo-S5", "NewCo-S5"]
    await _wipe_entities(*names)
    try:
        doc = await _make_doc("bench-s5")
        carol = await _make_entity("Carol-S5", "Person")
        oldco = await _make_entity("OldCo-S5", "Organization")
        newco = await _make_entity("NewCo-S5", "Organization")

        await _assert_fact(doc, carol, oldco, "Carol-S5", "WORKS_FOR", "OldCo-S5")
        await _assert_fact(doc, carol, newco, "Carol-S5", "WORKS_FOR", "NewCo-S5")

        driver = neo4j_store.get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Entity {id: $sid})-[r:MEMORY_REL]->(o:Entity {id: $oid})
                WHERE r.family = 'WORKS_FOR' AND r.valid_until IS NOT NULL
                OPTIONAL MATCH (f:MemoryFact)
                WHERE f.subject_entity_id = $sid
                    AND f.object_entity_id = $oid
                    AND f.family = 'WORKS_FOR'
                    AND f.valid_until IS NOT NULL
                RETURN r.valid_until AS rel_until, f.valid_until AS fact_until
                """,
                sid=carol,
                oid=oldco,
            )
            rec = await result.single()

        rel_until = rec["rel_until"] if rec else None
        fact_until = rec["fact_until"] if rec else None
        passed = rel_until is not None and fact_until is not None and rel_until == fact_until
        error = "" if passed else f"rel_until={rel_until!r} fact_until={fact_until!r}"
        return ScenarioResult(id=5, name="interval_propagation", passed=passed, error=error)
    finally:
        await _wipe_entities(*names)


# ---------------------------------------------------------------------------
# Scenario 6: multi-hop excludes stale paths
# ---------------------------------------------------------------------------

async def _scenario_6() -> ScenarioResult:
    """Alice → Acme (stale) → PostgreSQL; Alice → Beacon (live) → Qdrant.
    Live graph traversal should reach Qdrant but not PostgreSQL."""
    names = ["Alice-S6", "Acme-S6", "Beacon-S6", "PostgreSQL-S6", "Qdrant-S6"]
    await _wipe_entities(*names)
    try:
        doc = await _make_doc("bench-s6")
        alice = await _make_entity("Alice-S6", "Person")
        acme = await _make_entity("Acme-S6", "Organization")
        beacon = await _make_entity("Beacon-S6", "Organization")
        pg = await _make_entity("PostgreSQL-S6", "Technology")
        qdrant = await _make_entity("Qdrant-S6", "Technology")

        # Alice WORKS_FOR Acme (first), then WORKS_FOR Beacon (supersedes Acme)
        await _assert_fact(doc, alice, acme, "Alice-S6", "WORKS_FOR", "Acme-S6")
        await _assert_fact(doc, alice, beacon, "Alice-S6", "WORKS_FOR", "Beacon-S6")

        # Both orgs use a technology
        await _assert_fact(doc, acme, pg, "Acme-S6", "USES", "PostgreSQL-S6")
        await _assert_fact(doc, beacon, qdrant, "Beacon-S6", "USES", "Qdrant-S6")

        # Two-hop BFS: WORKS_FOR (live only) then USES
        driver = neo4j_store.get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Entity {id: $alice})
                    -[r1:MEMORY_REL {family: 'WORKS_FOR'}]->(org:Entity)
                    -[r2:MEMORY_REL {family: 'USES'}]->(tech:Entity)
                WHERE r1.valid_until IS NULL AND r2.valid_until IS NULL
                RETURN tech.name AS tech_name
                """,
                alice=alice,
            )
            techs = [rec["tech_name"] async for rec in result]

        passed = "Qdrant-S6" in techs and "PostgreSQL-S6" not in techs
        error = "" if passed else f"reachable_techs={techs!r}"
        return ScenarioResult(id=6, name="multi_hop_excludes_stale", passed=passed, error=error)
    finally:
        await _wipe_entities(*names)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_SCENARIOS: list[tuple[int, str, object]] = [
    (1, "subject_keyed_works_for", _scenario_1),
    (2, "object_keyed_has_title", _scenario_2),
    (3, "additive_leads", _scenario_3),
    (4, "alias_seed_resolution", _scenario_4),
    (5, "interval_propagation", _scenario_5),
    (6, "multi_hop_excludes_stale", _scenario_6),
]


async def run_all_scenarios() -> list[ScenarioResult]:
    results = []
    for sid, name, fn in _SCENARIOS:
        try:
            result = await fn()
        except Exception as exc:
            result = ScenarioResult(id=sid, name=name, passed=False, error=str(exc))
        results.append(result)
    return results
