from __future__ import annotations

from datetime import UTC, datetime

from landscape.memory_graph import AssertionPayload
from landscape.storage.neo4j_driver import get_driver
from landscape.storage.neo4j_entities import _resolve_entity_app_id


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
