from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.normalization import normalize_assertion
from landscape.storage import neo4j_store


async def persist_assertion_and_maybe_promote(
    payload: AssertionPayload,
    *,
    source_node_id: str,
    source_kind: str,
    subject_entity_id: str | None,
    object_entity_id: str | None,
    chunk_ids: list[str],
) -> tuple[str, str | None]:
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
        return assertion_id, None

    fact_id = await neo4j_store.upsert_memory_fact_from_assertion(
        family=normalized.family,
        subject_entity_id=normalized.subject_entity_id,
        object_entity_id=normalized.object_entity_id,
        subtype=normalized.subtype,
        confidence=payload.confidence,
        assertion_id=assertion_id,
    )
    return assertion_id, fact_id
