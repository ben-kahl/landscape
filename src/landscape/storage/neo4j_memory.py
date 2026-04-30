from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from landscape.memory_graph import AssertionPayload, FAMILY_REGISTRY, fact_key, slot_key
from landscape.storage.neo4j_driver import get_driver
from landscape.storage.neo4j_entities import _resolve_entity_app_id, _resolve_entity_app_ids


def _memory_slot_lock_id(slot: str) -> str:
    return f"slot-lock:{slot}"


async def ensure_memory_graph_schema() -> None:
    driver = get_driver()
    statements = [
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT alias_id_unique IF NOT EXISTS FOR (n:Alias) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS FOR (n:Assertion) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT memory_fact_id_unique IF NOT EXISTS FOR (n:MemoryFact) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT memory_slot_lock_id_unique IF NOT EXISTS FOR (n:MemorySlotLock) REQUIRE n.id IS UNIQUE",
    ]
    async with driver.session() as session:
        for stmt in statements:
            await session.run(stmt)


async def merge_assertion(payload: AssertionPayload) -> str:
    from landscape.memory_graph import assertion_id

    aid = assertion_id(
        source_kind=payload.source_kind,
        source_id=payload.source_id,
        raw_subject_text=payload.raw_subject_text,
        raw_relation_text=payload.raw_relation_text,
        raw_object_text=payload.raw_object_text,
        subtype=payload.subtype,
        qualifier_payload={
            "value_text": payload.value_text,
            "value_number": payload.value_number,
            "value_unit": payload.value_unit,
            "value_kind": payload.value_kind,
            "value_time": payload.value_time,
            "quantity_value": payload.quantity_value,
            "quantity_unit": payload.quantity_unit,
            "quantity_kind": payload.quantity_kind,
            "time_scope": payload.time_scope,
        },
        chunk_refs=payload.chunk_refs,
    )
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MERGE (a:Assertion {id: $assertion_id})
            ON CREATE SET a.source_kind = $source_kind,
                          a.source_id = $source_id,
                          a.raw_subject_text = $raw_subject_text,
                          a.raw_relation_text = $raw_relation_text,
                          a.raw_object_text = $raw_object_text,
                          a.family_candidate = $family_candidate,
                          a.confidence = $confidence,
                          a.subtype = $subtype,
                          a.value_text = $value_text,
                          a.value_number = $value_number,
                          a.value_unit = $value_unit,
                          a.value_kind = $value_kind,
                          a.value_time = $value_time,
                          a.quantity_value = $quantity_value,
                          a.quantity_unit = $quantity_unit,
                          a.quantity_kind = $quantity_kind,
                          a.time_scope = $time_scope,
                          a.status = 'active',
                          a.created_at = $now
            """,
            assertion_id=aid,
            source_kind=payload.source_kind,
            source_id=payload.source_id,
            raw_subject_text=payload.raw_subject_text,
            raw_relation_text=payload.raw_relation_text,
            raw_object_text=payload.raw_object_text,
            family_candidate=payload.family_candidate,
            confidence=payload.confidence,
            subtype=payload.subtype,
            value_text=payload.value_text,
            value_number=payload.value_number,
            value_unit=payload.value_unit,
            value_kind=payload.value_kind,
            value_time=payload.value_time,
            quantity_value=payload.quantity_value,
            quantity_unit=payload.quantity_unit,
            quantity_kind=payload.quantity_kind,
            time_scope=payload.time_scope,
            now=datetime.now(UTC).isoformat(),
        )
    return aid


async def link_source_to_assertion(
    source_kind: str,
    source_node_id: str,
    assertion_id: str,
) -> None:
    driver = get_driver()
    async with driver.session() as session:
        if source_kind == "document":
            await session.run(
                """
                MATCH (source:Document) WHERE elementId(source) = $source_node_id
                MATCH (a:Assertion {id: $assertion_id})
                MERGE (source)-[:ASSERTS]->(a)
                """,
                source_node_id=source_node_id,
                assertion_id=assertion_id,
            )
            return
        if source_kind == "turn":
            await session.run(
                """
                MATCH (source:Turn) WHERE elementId(source) = $source_node_id
                MATCH (a:Assertion {id: $assertion_id})
                MERGE (source)-[:ASSERTS]->(a)
                """,
                source_node_id=source_node_id,
                assertion_id=assertion_id,
            )
            return
        raise ValueError(f"unsupported source_kind: {source_kind!r}")


async def link_assertion_subject(assertion_id: str, subject_entity_id: str) -> None:
    subject_entity_id = await _resolve_entity_app_id(subject_entity_id)
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (a:Assertion {id: $assertion_id})
            MATCH (e:Entity {id: $entity_id})
            MERGE (a)-[:SUBJECT_ENTITY]->(e)
            """,
            assertion_id=assertion_id,
            entity_id=subject_entity_id,
        )


async def link_assertion_object(assertion_id: str, object_entity_id: str) -> None:
    object_entity_id = await _resolve_entity_app_id(object_entity_id)
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (a:Assertion {id: $assertion_id})
            MATCH (e:Entity {id: $entity_id})
            MERGE (a)-[:OBJECT_ENTITY]->(e)
            """,
            assertion_id=assertion_id,
            entity_id=object_entity_id,
        )


async def _create_memory_fact_version_in_tx(
    tx,
    *,
    family: str,
    subject_entity_id: str,
    object_entity_id: str | None,
    subtype: str | None,
    value_text: str | None,
    value_number: float | None,
    value_unit: str | None,
    value_kind: str | None,
    value_time: str | None,
    quantity_value: float | str | None,
    quantity_unit: str | None,
    quantity_kind: str | None,
    time_scope: str | None,
    confidence: float,
    assertion_id: str,
    now: str,
) -> str:
    family_cfg = FAMILY_REGISTRY[family]
    fkey = fact_key(
        family_cfg,
        subject_entity_id,
        object_entity_id,
        subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
    )
    skey = slot_key(
        family_cfg,
        subject_entity_id,
        object_entity_id,
        subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
    )
    fact_id = f"fact:{hashlib.sha256(f'{fkey}:{assertion_id}'.encode()).hexdigest()[:20]}"
    result = await tx.run(
        """
        MATCH (subject:Entity {id: $subject_entity_id})
        MATCH (a:Assertion {id: $assertion_id})
        OPTIONAL MATCH (object:Entity {id: $object_entity_id})
        MERGE (fact:MemoryFact {id: $fact_id})
        ON CREATE SET fact.family = $family,
                      fact.fact_key = $fact_key,
                      fact.slot_key = $slot_key,
                      fact.subtype = $subtype,
                      fact.value_text = $value_text,
                      fact.value_number = $value_number,
                      fact.value_unit = $value_unit,
                      fact.value_kind = $value_kind,
                      fact.value_time = $value_time,
                      fact.quantity_value = $quantity_value,
                      fact.quantity_unit = $quantity_unit,
                      fact.quantity_kind = $quantity_kind,
                      fact.time_scope = $time_scope,
                      fact.valid_from = $now,
                      fact.valid_until = null,
                      fact.support_count = 1,
                      fact.confidence_agg = $confidence,
                      fact.normalization_policy = 'v1_family_rules',
                      fact.created_at = $now,
                      fact.updated_at = $now,
                      fact.access_count = 1,
                      fact.last_accessed = $now,
                      fact.subject_entity_id = $subject_entity_id,
                      fact.object_entity_id = $object_entity_id,
                      fact.assertion_id = $assertion_id
        ON MATCH SET fact.support_count = coalesce(fact.support_count, 0) + 1,
                     fact.confidence_agg = CASE
                         WHEN coalesce(fact.confidence_agg, 0.0) >= $confidence THEN fact.confidence_agg
                         ELSE $confidence
                     END,
                     fact.updated_at = $now,
                     fact.valid_until = null,
                     fact.access_count = coalesce(fact.access_count, 0) + 1,
                     fact.last_accessed = $now
        MERGE (subject)-[:AS_SUBJECT]->(fact)
        FOREACH (_ IN CASE WHEN object IS NULL THEN [] ELSE [1] END |
            MERGE (fact)-[:AS_OBJECT]->(object)
        )
        MERGE (a)-[:SUPPORTS]->(fact)
        RETURN fact.id AS fact_id
        """,
        fact_id=fact_id,
        family=family,
        fact_key=fkey,
        slot_key=skey,
        subtype=subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
        quantity_value=quantity_value,
        quantity_unit=quantity_unit,
        quantity_kind=quantity_kind,
        time_scope=time_scope,
        confidence=confidence,
        now=now,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        assertion_id=assertion_id,
    )
    record = await result.single()
    if record is None:
        raise ValueError(f"missing assertion or subject for fact {fact_id!r}")
    return record["fact_id"]


async def _materialize_memory_rel_in_tx(tx, memory_fact_id: str, now: str) -> None:
    await tx.run(
        """
        MATCH (subject:Entity)-[:AS_SUBJECT]->(fact:MemoryFact {id: $fact_id})
        OPTIONAL MATCH (fact)-[:AS_OBJECT]->(object:Entity)
        FOREACH (_ IN CASE WHEN object IS NULL THEN [] ELSE [1] END |
            MERGE (subject)-[r:MEMORY_REL {memory_fact_id: $fact_id}]->(object)
            SET r.family = fact.family,
                r.valid_from = coalesce(r.valid_from, $now),
                r.valid_until = null,
                r.confidence_agg = fact.confidence_agg,
                r.subtype = fact.subtype,
                r.subject_entity_id = subject.id,
                r.object_entity_id = object.id,
                r.access_count = coalesce(fact.access_count, 0),
                r.last_accessed = fact.last_accessed,
                r.updated_at = $now
        )
        SET fact.valid_until = null,
            fact.updated_at = $now
        """,
        fact_id=memory_fact_id,
        now=now,
    )


async def create_memory_fact_version(
    *,
    family: str,
    subject_entity_id: str,
    object_entity_id: str | None,
    subtype: str | None,
    value_text: str | None = None,
    value_number: float | None = None,
    value_unit: str | None = None,
    value_kind: str | None = None,
    value_time: str | None = None,
    quantity_value: float | str | None = None,
    quantity_unit: str | None = None,
    quantity_kind: str | None = None,
    time_scope: str | None = None,
    confidence: float,
    assertion_id: str,
) -> str:
    subject_entity_id = await _resolve_entity_app_id(subject_entity_id)
    object_entity_id = (
        await _resolve_entity_app_id(object_entity_id)
        if object_entity_id is not None
        else None
    )
    driver = get_driver()
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            fact_id = await _create_memory_fact_version_in_tx(
                tx,
                family=family,
                subject_entity_id=subject_entity_id,
                object_entity_id=object_entity_id,
                subtype=subtype,
                value_text=value_text,
                value_number=value_number,
                value_unit=value_unit,
                value_kind=value_kind,
                value_time=value_time,
                quantity_value=quantity_value,
                quantity_unit=quantity_unit,
                quantity_kind=quantity_kind,
                time_scope=time_scope,
                confidence=confidence,
                assertion_id=assertion_id,
                now=datetime.now(UTC).isoformat(),
            )
            await tx.commit()
            return fact_id
        except Exception:
            await tx.rollback()
            raise


async def upsert_memory_fact_from_assertion(
    *,
    family: str,
    subject_entity_id: str,
    object_entity_id: str | None,
    subtype: str | None,
    value_text: str | None = None,
    value_number: float | None = None,
    value_unit: str | None = None,
    value_kind: str | None = None,
    value_time: str | None = None,
    quantity_value: float | str | None = None,
    quantity_unit: str | None = None,
    quantity_kind: str | None = None,
    time_scope: str | None = None,
    confidence: float,
    assertion_id: str,
) -> tuple[str, str]:
    family_cfg = FAMILY_REGISTRY[family]
    subject_entity_id = await _resolve_entity_app_id(subject_entity_id)
    object_entity_id = (
        await _resolve_entity_app_id(object_entity_id)
        if object_entity_id is not None
        else None
    )
    if family_cfg.slot_mode != "additive":
        family_key = slot_key(family_cfg, subject_entity_id, object_entity_id, subtype)
        driver = get_driver()
        async with driver.session() as session:
            current_result = await session.run(
                """
                MATCH (f:MemoryFact {family: $family, slot_key: $slot_key})
                WHERE f.valid_until IS NULL
                RETURN collect(f.id) AS fact_ids
                """,
                family=family,
                slot_key=family_key,
            )
            current_record = await current_result.single()
            current_fact_ids = current_record["fact_ids"] if current_record is not None else []

        fact_id = await supersede_single_current_fact(
            family=family,
            subject_entity_id=subject_entity_id,
            object_entity_id=object_entity_id,
            subtype=subtype,
            value_text=value_text,
            value_number=value_number,
            value_unit=value_unit,
            value_kind=value_kind,
            value_time=value_time,
            quantity_value=quantity_value,
            quantity_unit=quantity_unit,
            quantity_kind=quantity_kind,
            time_scope=time_scope,
            confidence=confidence,
            assertion_id=assertion_id,
        )
        if fact_id in current_fact_ids:
            return fact_id, "reinforced"
        if current_fact_ids:
            return fact_id, "superseded"
        return fact_id, "created"

    family_key = fact_key(
        family_cfg,
        subject_entity_id,
        object_entity_id,
        subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
    )
    driver = get_driver()
    async with driver.session() as session:
        existing_result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fact_id})
            RETURN f.id AS fact_id
            LIMIT 1
            """,
            fact_id=f"fact:{hashlib.sha256(f'{family_key}:{assertion_id}'.encode()).hexdigest()[:20]}",
        )
        existing_record = await existing_result.single()
    fact_id = await create_memory_fact_version(
        family=family,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        subtype=subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
        quantity_value=quantity_value,
        quantity_unit=quantity_unit,
        quantity_kind=quantity_kind,
        time_scope=time_scope,
        confidence=confidence,
        assertion_id=assertion_id,
    )
    await materialize_memory_rel(fact_id)
    return fact_id, "reinforced" if existing_record is not None else "created"


async def materialize_memory_rel(memory_fact_id: str) -> None:
    now = datetime.now(UTC).isoformat()
    driver = get_driver()
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            await _materialize_memory_rel_in_tx(tx, memory_fact_id, now)
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise


async def supersede_single_current_fact(
    *,
    family: str,
    subject_entity_id: str,
    object_entity_id: str | None,
    subtype: str | None,
    value_text: str | None = None,
    value_number: float | None = None,
    value_unit: str | None = None,
    value_kind: str | None = None,
    value_time: str | None = None,
    quantity_value: float | str | None = None,
    quantity_unit: str | None = None,
    quantity_kind: str | None = None,
    time_scope: str | None = None,
    confidence: float,
    assertion_id: str,
) -> str:
    subject_entity_id = await _resolve_entity_app_id(subject_entity_id)
    object_entity_id = (
        await _resolve_entity_app_id(object_entity_id)
        if object_entity_id is not None
        else None
    )
    family_cfg = FAMILY_REGISTRY[family]
    if family_cfg.slot_mode == "additive":
        raise ValueError(f"family {family!r} is not slot-replacing")
    skey = slot_key(
        family_cfg,
        subject_entity_id,
        object_entity_id,
        subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
    )
    driver = get_driver()
    now = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        tx = await session.begin_transaction()
        try:
            await tx.run(
                """
                MERGE (slot:MemorySlotLock {id: $lock_id})
                ON CREATE SET slot.created_at = $now
                SET slot.updated_at = $now
                """,
                lock_id=_memory_slot_lock_id(skey),
                now=now,
            )
            result = await tx.run(
                """
                MATCH (old:MemoryFact {family: $family, slot_key: $slot_key})
                WHERE old.valid_until IS NULL
                RETURN old.id AS old_fact_id
                LIMIT 1
                """,
                family=family,
                slot_key=skey,
            )
            record = await result.single()
            old_fact_id = record["old_fact_id"] if record is not None else None

            new_fact_id = await _create_memory_fact_version_in_tx(
                tx,
                family=family,
                subject_entity_id=subject_entity_id,
                object_entity_id=object_entity_id,
                subtype=subtype,
                value_text=value_text,
                value_number=value_number,
                value_unit=value_unit,
                value_kind=value_kind,
                value_time=value_time,
                quantity_value=quantity_value,
                quantity_unit=quantity_unit,
                quantity_kind=quantity_kind,
                time_scope=time_scope,
                confidence=confidence,
                assertion_id=assertion_id,
                now=now,
            )

            if old_fact_id == new_fact_id:
                await _materialize_memory_rel_in_tx(tx, new_fact_id, now)
                await tx.commit()
                return new_fact_id

            if old_fact_id is not None:
                await tx.run(
                    """
                    MATCH (old:MemoryFact {id: $old_fact_id})
                    SET old.valid_until = $now,
                        old.updated_at = $now
                    WITH old
                    OPTIONAL MATCH ()-[r:MEMORY_REL {memory_fact_id: $old_fact_id}]-()
                    SET r.valid_until = $now,
                        r.updated_at = $now
                    """,
                    old_fact_id=old_fact_id,
                    now=now,
                )

            await _materialize_memory_rel_in_tx(tx, new_fact_id, now)
            await tx.commit()
            return new_fact_id
        except Exception:
            await tx.rollback()
            raise


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
      elementId(seed) AS seed_element_id,
      target.id AS target_id,
      elementId(target) AS target_element_id,
      target.name AS target_name,
      target.type AS target_type,
      coalesce(target.access_count, 0) AS target_access_count,
      target.last_accessed AS target_last_accessed,
      length(path) AS distance,
      [r IN rels | r.memory_fact_id] AS memory_fact_ids,
      [r IN rels | r.memory_fact_id] AS path_memory_fact_ids,
      [r IN rels | elementId(r)] AS edge_ids,
      [r IN rels | r.family] AS edge_families,
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
