from dataclasses import dataclass

from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.normalization import normalize_assertion
from landscape.storage import neo4j_store


@dataclass(frozen=True)
class PersistenceResult:
    assertion_id: str
    fact_id: str | None
    outcome: str


async def persist_assertion_and_maybe_promote(
    payload: AssertionPayload,
    *,
    source_node_id: str,
    source_kind: str,
    subject_entity_id: str | None,
    object_entity_id: str | None,
    chunk_ids: list[str],
) -> PersistenceResult:
    assertion_id = await neo4j_store.merge_assertion(payload)
    await neo4j_store.link_source_to_assertion(source_kind, source_node_id, assertion_id)
    for chunk_id in chunk_ids:
        await neo4j_store.link_assertion_to_chunk(assertion_id, chunk_id)
    if subject_entity_id is not None:
        await neo4j_store.link_assertion_subject(assertion_id, subject_entity_id)
    if object_entity_id is not None:
        await neo4j_store.link_assertion_object(assertion_id, object_entity_id)

    normalized = normalize_assertion(
        payload,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
    )
    if not normalized.promotable:
        return PersistenceResult(assertion_id=assertion_id, fact_id=None, outcome="created")

    fact_id, outcome = await neo4j_store.upsert_memory_fact_from_assertion(
        family=normalized.family,
        subject_entity_id=normalized.subject_entity_id,
        object_entity_id=normalized.object_entity_id,
        subtype=normalized.subtype,
        value_text=normalized.value_text,
        value_number=normalized.value_number,
        value_unit=normalized.value_unit,
        value_kind=normalized.value_kind,
        value_time=normalized.value_time,
        quantity_value=normalized.quantity_value,
        quantity_unit=normalized.quantity_unit,
        quantity_kind=normalized.quantity_kind,
        time_scope=normalized.time_scope,
        confidence=payload.confidence,
        assertion_id=assertion_id,
    )
    return PersistenceResult(assertion_id=assertion_id, fact_id=fact_id, outcome=outcome)
