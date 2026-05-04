from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from landscape.memory_graph import alias_id
from landscape.storage.neo4j_driver import get_driver


async def merge_entity(
    name: str,
    entity_type: str,
    source_doc: str,
    confidence: float,
    doc_element_id: str | None = None,
    model: str = "",
    created_by: str = "ingest",
    session_id: str | None = None,
    turn_id: str | None = None,
    subtype: str | None = None,
) -> str:
    """Returns the stable app id of the entity node."""
    if created_by not in ("ingest", "agent"):
        raise ValueError(f"created_by must be 'ingest' or 'agent', got {created_by!r}")
    driver = get_driver()
    now = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        result = await session.run(
            """
            MERGE (e:Entity {name: $name, type: $type})
            ON CREATE SET e.source_doc = $source_doc,
                          e.confidence = $confidence,
                          e.timestamp = $now,
                          e.canonical = true,
                          e.access_count = 1,
                          e.last_accessed = $now,
                          e.created_by = $created_by,
                          e.session_id = $session_id,
                          e.turn_id = $turn_id,
                          e.subtype = $subtype,
                          e.id = randomUUID()
            ON MATCH SET e.access_count = coalesce(e.access_count, 0) + 1,
                         e.last_accessed = $now,
                         e.id = coalesce(e.id, randomUUID())
            RETURN e.id AS entity_id
            """,
            name=name,
            type=entity_type,
            source_doc=source_doc,
            confidence=confidence,
            now=now,
            created_by=created_by,
            session_id=session_id,
            turn_id=turn_id,
            subtype=subtype,
        )
        record = await result.single()
        entity_id = record["entity_id"]

    if doc_element_id is not None:
        async with driver.session() as session:
            await session.run(
                """
                MATCH (e:Entity {id: $entity_id})
                MATCH (d:Document) WHERE elementId(d) = $doc_id
                MERGE (e)-[:EXTRACTED_FROM {method: "llm", model: $model}]->(d)
                """,
                entity_id=entity_id,
                doc_id=doc_element_id,
                model=model,
            )

    return entity_id


async def _resolve_entity_app_id(entity_ref: str) -> str:
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)
            WHERE e.id = $entity_ref OR elementId(e) = $entity_ref
            WITH e, coalesce(e.id, randomUUID()) AS entity_id
            SET e.id = entity_id
            RETURN entity_id AS entity_id
            LIMIT 1
            """,
            entity_ref=entity_ref,
        )
        record = await result.single()
        if record is None:
            raise ValueError(f"Unknown entity reference: {entity_ref!r}")
        return record["entity_id"]


async def _resolve_entity_app_ids(entity_refs: list[str]) -> list[str]:
    if not entity_refs:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            UNWIND $entity_refs AS entity_ref
            MATCH (e:Entity)
            WHERE e.id = entity_ref OR elementId(e) = entity_ref
            WITH entity_ref, e, coalesce(e.id, randomUUID()) AS entity_id
            SET e.id = entity_id
            RETURN entity_ref, entity_id
            """,
            entity_refs=entity_refs,
        )
        rows = [dict(record) async for record in result]

    by_ref = {row["entity_ref"]: row["entity_id"] for row in rows}
    resolved: list[str] = []
    missing = [entity_ref for entity_ref in entity_refs if entity_ref not in by_ref]
    if missing:
        raise ValueError(f"Unknown entity reference(s): {missing!r}")
    for entity_ref in entity_refs:
        entity_id = by_ref[entity_ref]
        if entity_id not in resolved:
            resolved.append(entity_id)
    return resolved


async def resolve_entity_app_id(entity_ref: str) -> str:
    return await _resolve_entity_app_id(entity_ref)


async def resolve_entity_app_ids(entity_refs: list[str]) -> list[str]:
    return await _resolve_entity_app_ids(entity_refs)


async def merge_alias(
    canonical_entity_id: str,
    alias_text: str,
    source_doc: str,
    confidence: float,
) -> str:
    canonical_entity_id = await _resolve_entity_app_id(canonical_entity_id)
    aid = alias_id(canonical_entity_id, alias_text)
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (canonical:Entity {id: $entity_id})
            MERGE (a:Alias {id: $alias_id})
            ON CREATE SET a.name = $alias_text,
                          a.normalized_name = toLower(trim($alias_text)),
                          a.source_doc = $source_doc,
                          a.confidence = $confidence,
                          a.created_at = $now
            MERGE (a)-[r:SAME_AS]->(canonical)
            ON CREATE SET r.confidence = $confidence,
                          r.source_doc = $source_doc
            """,
            entity_id=canonical_entity_id,
            alias_id=aid,
            alias_text=alias_text,
            confidence=confidence,
            source_doc=source_doc,
            now=datetime.now(UTC).isoformat(),
        )
    return aid


async def find_entity_by_app_id(entity_id: str) -> dict[str, Any] | None:
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (alias:Alias)-[:SAME_AS]->(e)
            RETURN e.id AS entity_id,
                   e.name AS name,
                   e.type AS type,
                   collect(DISTINCT alias.name) AS aliases
            """,
            entity_id=entity_id,
        )
        record = await result.single()
        if record is None:
            return None
        return {
            "entity_id": record["entity_id"],
            "name": record["name"],
            "type": record["type"],
            "aliases": [alias for alias in record["aliases"] or [] if alias is not None],
        }


async def find_entity_by_id(entity_id: str) -> dict[str, Any] | None:
    return await find_entity_by_app_id(entity_id)


async def add_alias(
    canonical_element_id: str,
    alias: str,
    source_doc: str,
    confidence: float,
) -> None:
    await merge_alias(canonical_element_id, alias, source_doc, confidence)


async def resolve_seed_entity_ids(query_text: str) -> list[str]:
    normalized_query = " ".join(query_text.strip().lower().split())
    if not normalized_query:
        return []

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)
            WHERE e.canonical = true
            OPTIONAL MATCH (a:Alias)-[:SAME_AS]->(e)
            WITH e, collect(DISTINCT a.normalized_name) + [toLower(trim(e.name))] AS candidate_names
            UNWIND candidate_names AS candidate_name
            WITH e, trim(candidate_name) AS candidate_name
            WHERE candidate_name <> ""
              AND $normalized_query CONTAINS candidate_name
            RETURN e.id AS entity_id, max(size(candidate_name)) AS match_length
            ORDER BY match_length DESC, entity_id ASC
            """,
            normalized_query=normalized_query,
        )
        return [record["entity_id"] async for record in result]


async def link_entity_to_doc(
    entity_element_id: str,
    doc_element_id: str,
    model: str,
) -> None:
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (e:Entity) WHERE e.id = $eid
            MATCH (d:Document) WHERE elementId(d) = $did
            MERGE (e)-[:EXTRACTED_FROM {method: "llm", model: $model}]->(d)
            """,
            eid=entity_element_id,
            did=doc_element_id,
            model=model,
        )


async def get_entities_from_chunks(chunk_element_ids: list[str]) -> list[dict[str, Any]]:
    if not chunk_element_ids:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)<-[:EXTRACTED_FROM]-(e:Entity)
            WHERE c.chunk_id IN $chunk_ids
              AND e.canonical = true
            WITH e,
                 collect(
                     DISTINCT c.chunk_id
                 ) AS chunk_eids
            RETURN
                e.id AS entity_id,
                e.name AS name,
                e.type AS type,
                coalesce(e.access_count, 0) AS access_count,
                e.last_accessed AS last_accessed,
                chunk_eids
            """,
            chunk_ids=chunk_element_ids,
        )
        return [dict(record) async for record in result]


async def get_rankable_entities(entity_ids: list[str]) -> list[dict[str, Any]]:
    if not entity_ids:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WHERE e.id IN $ids AND e.canonical = true
            OPTIONAL MATCH (e)-[r:MEMORY_REL]-()
            WITH e,
                 count(r) AS total_edges,
                 sum(CASE WHEN r.valid_until IS NULL THEN 1 ELSE 0 END) AS valid_edges
            WHERE total_edges = 0 OR valid_edges > 0
            RETURN e.id AS entity_id,
                   e.name AS name,
                   e.type AS type,
                   coalesce(e.access_count, 0) AS access_count,
                   e.last_accessed AS last_accessed
            """,
            ids=entity_ids,
        )
        return [dict(record) async for record in result]


async def touch_entities(entity_ids: list[str], now: str) -> None:
    if not entity_ids:
        return
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (e:Entity) WHERE e.id IN $ids
            SET e.access_count = coalesce(e.access_count, 0) + 1,
                e.last_accessed = $now
            """,
            ids=entity_ids,
            now=now,
        )
