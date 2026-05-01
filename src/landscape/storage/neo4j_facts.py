from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from landscape.memory_graph import FAMILY_REGISTRY, fact_key, slot_key
from landscape.storage.neo4j_driver import get_driver
from landscape.storage.neo4j_entities import _resolve_entity_app_id


def _memory_slot_lock_id(slot: str) -> str:
    return f"slot-lock:{slot}"


async def ensure_memory_graph_schema() -> None:
    driver = get_driver()
    statements = [
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS"
        " FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT alias_id_unique IF NOT EXISTS"
        " FOR (n:Alias) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS"
        " FOR (n:Assertion) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT memory_fact_id_unique IF NOT EXISTS"
        " FOR (n:MemoryFact) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT memory_slot_lock_id_unique IF NOT EXISTS"
        " FOR (n:MemorySlotLock) REQUIRE n.id IS UNIQUE",
    ]
    async with driver.session() as session:
        for stmt in statements:
            await session.run(stmt)


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
                         WHEN coalesce(fact.confidence_agg, 0.0) >= $confidence
                         THEN fact.confidence_agg
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
