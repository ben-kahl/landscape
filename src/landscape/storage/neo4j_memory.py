from __future__ import annotations

from typing import Any

from landscape.storage.neo4j_driver import get_driver
from landscape.storage.neo4j_entities import _resolve_entity_app_ids


async def get_memory_fact_explanation(memory_fact_id: str) -> dict[str, Any] | None:
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (fact:MemoryFact {id: $fact_id})
            OPTIONAL MATCH (subject:Entity)-[:AS_SUBJECT]->(fact)
            OPTIONAL MATCH (fact)-[:AS_OBJECT]->(object:Entity)
            OPTIONAL MATCH (subject)-[rel:MEMORY_REL {memory_fact_id: $fact_id}]->(object)
            RETURN fact.id AS fact_id,
                   fact.family AS family,
                   fact.valid_until AS valid_until,
                   (fact.valid_until IS NULL) AS current,
                   fact.fact_key AS fact_key,
                   fact.slot_key AS slot_key,
                   fact.subtype AS subtype,
                   fact.support_count AS support_count,
                   fact.confidence_agg AS confidence_agg,
                   subject.id AS subject_entity_id,
                   subject.name AS subject_name,
                   subject.type AS subject_type,
                   object.id AS object_entity_id,
                   object.name AS object_name,
                   object.type AS object_type,
                   rel.valid_until AS memory_rel_valid_until,
                   (rel.valid_until IS NULL) AS memory_rel_current
            """,
            fact_id=memory_fact_id,
        )
        record = await result.single()
        return dict(record) if record is not None else None


async def get_memory_fact_details_batch(
    memory_fact_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not memory_fact_ids:
        return [], []

    unique_fact_ids = list(dict.fromkeys(memory_fact_ids))
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (fact:MemoryFact)
            WHERE fact.id IN $fact_ids
            OPTIONAL MATCH (subject:Entity)-[:AS_SUBJECT]->(fact)
            OPTIONAL MATCH (fact)-[:AS_OBJECT]->(object:Entity)
            OPTIONAL MATCH (subject)-[rel:MEMORY_REL {memory_fact_id: fact.id}]->(object)
            OPTIONAL MATCH (a:Assertion)-[:SUPPORTS]->(fact)
            WITH fact, subject, object, rel, collect(DISTINCT a) AS assertions
            RETURN fact.id AS memory_fact_id,
                   fact.family AS family,
                   fact.valid_until AS valid_until,
                   (fact.valid_until IS NULL) AS current,
                   fact.fact_key AS fact_key,
                   fact.slot_key AS slot_key,
                   fact.subtype AS subtype,
                   fact.support_count AS support_count,
                   fact.confidence_agg AS confidence_agg,
                   fact.value_text AS value_text,
                   fact.value_number AS value_number,
                   fact.value_unit AS value_unit,
                   fact.value_kind AS value_kind,
                   fact.value_time AS value_time,
                   fact.quantity_value AS quantity_value,
                   fact.quantity_unit AS quantity_unit,
                   fact.quantity_kind AS quantity_kind,
                   fact.time_scope AS time_scope,
                   subject.id AS subject_entity_id,
                   subject.name AS subject_name,
                   subject.type AS subject_type,
                   object.id AS object_entity_id,
                   object.name AS object_name,
                   object.type AS object_type,
                   rel.valid_until AS memory_rel_valid_until,
                   (rel.valid_until IS NULL) AS memory_rel_current,
                   [a IN assertions WHERE a IS NOT NULL | {
                     assertion_id: a.id,
                     source_kind: a.source_kind,
                     source_id: a.source_id,
                     raw_subject_text: a.raw_subject_text,
                     raw_relation_text: a.raw_relation_text,
                     raw_object_text: a.raw_object_text,
                     family_candidate: a.family_candidate,
                     confidence: a.confidence,
                     subtype: a.subtype,
                     value_text: a.value_text,
                     value_number: a.value_number,
                     value_unit: a.value_unit,
                     value_kind: a.value_kind,
                     value_time: a.value_time,
                     quantity_value: a.quantity_value,
                     quantity_unit: a.quantity_unit,
                     quantity_kind: a.quantity_kind,
                     time_scope: a.time_scope,
                     status: a.status,
                     created_at: a.created_at
                   }] AS supporting_assertions
            ORDER BY memory_fact_id
            """,
            fact_ids=unique_fact_ids,
        )
        rows = [dict(record) async for record in result]

    row_by_fact_id = {row["memory_fact_id"]: row for row in rows}
    memory_facts: list[dict[str, object]] = []
    supporting_assertions: list[dict[str, object]] = []
    for fact_id in unique_fact_ids:
        row = row_by_fact_id.get(fact_id)
        if row is None:
            continue
        assertions = list(row.pop("supporting_assertions") or [])
        memory_facts.append(row)
        supporting_assertions.extend(
            {**assertion, "memory_fact_id": fact_id} for assertion in assertions
        )
    return memory_facts, supporting_assertions


async def get_current_fact_details_for_entities(
    entity_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not entity_ids:
        return [], []

    unique_entity_ids = list(dict.fromkeys(entity_ids))
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (subject:Entity)-[:AS_SUBJECT]->(fact:MemoryFact)
            WHERE subject.id IN $entity_ids
              AND fact.valid_until IS NULL
            OPTIONAL MATCH (fact)-[:AS_OBJECT]->(object:Entity)
            OPTIONAL MATCH (subject)-[rel:MEMORY_REL {memory_fact_id: fact.id}]->(object)
            WITH subject, fact, object, rel
            WHERE rel IS NULL
            OPTIONAL MATCH (a:Assertion)-[:SUPPORTS]->(fact)
            WITH subject, fact, object, collect(DISTINCT a) AS assertions
            RETURN fact.id AS memory_fact_id,
                   fact.family AS family,
                   fact.valid_until AS valid_until,
                   (fact.valid_until IS NULL) AS current,
                   fact.fact_key AS fact_key,
                   fact.slot_key AS slot_key,
                   fact.subtype AS subtype,
                   fact.support_count AS support_count,
                   fact.confidence_agg AS confidence_agg,
                   fact.value_text AS value_text,
                   fact.value_number AS value_number,
                   fact.value_unit AS value_unit,
                   fact.value_kind AS value_kind,
                   fact.value_time AS value_time,
                   fact.quantity_value AS quantity_value,
                   fact.quantity_unit AS quantity_unit,
                   fact.quantity_kind AS quantity_kind,
                   fact.time_scope AS time_scope,
                   subject.id AS subject_entity_id,
                   subject.name AS subject_name,
                   subject.type AS subject_type,
                   object.id AS object_entity_id,
                   object.name AS object_name,
                   object.type AS object_type,
                   null AS memory_rel_valid_until,
                   false AS memory_rel_current,
                   [a IN assertions WHERE a IS NOT NULL | {
                     assertion_id: a.id,
                     source_kind: a.source_kind,
                     source_id: a.source_id,
                     raw_subject_text: a.raw_subject_text,
                     raw_relation_text: a.raw_relation_text,
                     raw_object_text: a.raw_object_text,
                     family_candidate: a.family_candidate,
                     confidence: a.confidence,
                     subtype: a.subtype,
                     value_text: a.value_text,
                     value_number: a.value_number,
                     value_unit: a.value_unit,
                     value_kind: a.value_kind,
                     value_time: a.value_time,
                     quantity_value: a.quantity_value,
                     quantity_unit: a.quantity_unit,
                     quantity_kind: a.quantity_kind,
                     time_scope: a.time_scope,
                     status: a.status,
                     created_at: a.created_at
                   }] AS supporting_assertions
            ORDER BY subject.id, fact.family, fact.id
            """,
            entity_ids=unique_entity_ids,
        )
        rows = [dict(record) async for record in result]

    memory_facts: list[dict[str, object]] = []
    supporting_assertions: list[dict[str, object]] = []
    for row in rows:
        assertions = list(row.pop("supporting_assertions") or [])
        fact_id = row["memory_fact_id"]
        memory_facts.append(row)
        supporting_assertions.extend(
            {**assertion, "memory_fact_id": fact_id} for assertion in assertions
        )
    return memory_facts, supporting_assertions


async def bfs_expand_memory_rel(
    seed_entity_ids: list[str],
    max_hops: int,
) -> list[dict[str, Any]]:
    if not seed_entity_ids:
        return []
    if max_hops < 1 or max_hops > 5:
        raise ValueError(f"max_hops must be 1..5, got {max_hops}")
    seed_ids = await _resolve_entity_app_ids(seed_entity_ids)
    if not seed_ids:
        return []
    query = f"""
    MATCH (seed:Entity) WHERE seed.id IN $seed_ids
    MATCH path = shortestPath((seed)-[rels:MEMORY_REL*1..{max_hops}]-(target:Entity))
    WHERE seed.id <> target.id
      AND target.canonical = true
      AND ALL(r IN rels WHERE r.valid_until IS NULL)
    RETURN
      seed.id AS seed_id,
      target.id AS target_id,
      target.name AS target_name,
      target.type AS target_type,
      coalesce(target.access_count, 0) AS target_access_count,
      target.last_accessed AS target_last_accessed,
      length(path) AS distance,
      [r IN rels | r.memory_fact_id] AS path_memory_fact_ids,
      [r IN rels | elementId(r)] AS edge_ids,
      [r IN rels | r.family] AS path_edge_types,
      [r IN rels | r.subtype] AS edge_subtypes,
      [r IN rels | coalesce(r.confidence_agg, 0.0)] AS edge_confidences,
      [r IN rels | coalesce(r.access_count, 0)] AS edge_access_counts,
      [r IN rels | r.last_accessed] AS edge_last_accessed
    """
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(query, seed_ids=seed_ids)
        return [dict(record) async for record in result]


async def touch_relations(element_ids: list[str], now: str) -> None:
    if not element_ids:
        return
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH ()-[r:MEMORY_REL]->() WHERE elementId(r) IN $ids
            OPTIONAL MATCH (f:MemoryFact {id: r.memory_fact_id})
            SET r.access_count = coalesce(r.access_count, 0) + 1,
                r.last_accessed = $now,
                r.updated_at = $now
            FOREACH (_ IN CASE WHEN f IS NULL THEN [] ELSE [1] END |
                SET f.access_count = coalesce(f.access_count, 0) + 1,
                    f.last_accessed = $now,
                    f.updated_at = $now
            )
            """,
            ids=element_ids,
            now=now,
        )
