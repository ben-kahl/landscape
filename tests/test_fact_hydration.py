"""Regression tests for adjacent-fact hydration on ranked entities.

The 2026-07-07 W&B support session exposed that entities appearing only as
the *object* of memory facts (Okta, Domain capture) hydrated zero facts —
`fact_ids: []` in the compact search payload — because the Cypher only
matched the AS_SUBJECT side. DB isolation via the autouse `_isolated_test`
conftest fixture."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def _seed_fact_pair() -> None:
    from landscape.storage.neo4j_driver import get_driver

    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            CREATE (s:Entity {id: 'ent-subj', name: 'PhysicsX', type: 'Organization'})
            CREATE (o:Entity {id: 'ent-obj', name: 'Okta', type: 'Technology'})
            CREATE (f:MemoryFact {id: 'fact-1', family: 'USES',
                                  subtype: 'for_sso_configuration',
                                  confidence_agg: 0.9, support_count: 1,
                                  negated: false})
            CREATE (s)-[:AS_SUBJECT]->(f)
            CREATE (f)-[:AS_OBJECT]->(o)
            CREATE (s)-[:MEMORY_REL {memory_fact_id: 'fact-1'}]->(o)
            """
        )


@pytest.mark.asyncio
async def test_object_side_entity_hydrates_fact():
    from landscape.storage.neo4j_memory import get_current_fact_details_for_entities

    await _seed_fact_pair()
    facts, _ = await get_current_fact_details_for_entities(["ent-obj"])

    assert [f["memory_fact_id"] for f in facts] == ["fact-1"]
    assert facts[0]["subject_name"] == "PhysicsX"
    assert facts[0]["object_name"] == "Okta"
    assert facts[0]["subtype"] == "for_sso_configuration"


@pytest.mark.asyncio
async def test_subject_side_still_hydrates_and_no_duplicates():
    from landscape.storage.neo4j_memory import get_current_fact_details_for_entities

    await _seed_fact_pair()
    subj_facts, _ = await get_current_fact_details_for_entities(["ent-subj"])
    both_facts, _ = await get_current_fact_details_for_entities(
        ["ent-subj", "ent-obj"]
    )

    assert [f["memory_fact_id"] for f in subj_facts] == ["fact-1"]
    assert [f["memory_fact_id"] for f in both_facts] == ["fact-1"]


@pytest.mark.asyncio
async def test_superseded_fact_excluded_for_object_side():
    from landscape.storage.neo4j_driver import get_driver
    from landscape.storage.neo4j_memory import get_current_fact_details_for_entities

    await _seed_fact_pair()
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "MATCH (f:MemoryFact {id: 'fact-1'}) "
            "SET f.system_until = '2026-07-01T00:00:00+00:00'"
        )

    facts, _ = await get_current_fact_details_for_entities(["ent-obj"])
    assert facts == []
