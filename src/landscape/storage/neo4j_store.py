from datetime import UTC, datetime
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from landscape.config import settings

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def merge_document(content_hash: str, title: str, source_type: str) -> tuple[str, bool]:
    """Returns (doc_id, created). created=False means hash already existed."""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MERGE (d:Document {content_hash: $hash})
            ON CREATE SET d.title = $title,
                          d.source_type = $source_type,
                          d.ingested_at = $now
            RETURN elementId(d) AS doc_id, (d.ingested_at = $now) AS created
            """,
            hash=content_hash,
            title=title,
            source_type=source_type,
            now=datetime.now(UTC).isoformat(),
        )
        record = await result.single()
        return record["doc_id"], record["created"]


async def merge_entity(
    name: str,
    entity_type: str,
    source_doc: str,
    confidence: float,
    doc_element_id: str,
    model: str,
) -> str:
    """Returns the elementId of the entity node."""
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
                          e.aliases = [],
                          e.access_count = 0,
                          e.last_accessed = null
            WITH e
            MATCH (d:Document) WHERE elementId(d) = $doc_id
            MERGE (e)-[:EXTRACTED_FROM {method: "llm", model: $model}]->(d)
            RETURN elementId(e) AS eid
            """,
            name=name,
            type=entity_type,
            source_doc=source_doc,
            confidence=confidence,
            now=now,
            doc_id=doc_element_id,
            model=model,
        )
        record = await result.single()
        return record["eid"]


async def link_entity_to_doc(
    entity_element_id: str,
    doc_element_id: str,
    model: str,
) -> None:
    """Ensure an [:EXTRACTED_FROM] edge exists from the given entity to
    the given doc. Idempotent — used by the ingest pipeline for both
    newly-created AND resolved entities so provenance stays complete."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) = $eid
            MATCH (d:Document) WHERE elementId(d) = $did
            MERGE (e)-[:EXTRACTED_FROM {method: "llm", model: $model}]->(d)
            """,
            eid=entity_element_id,
            did=doc_element_id,
            model=model,
        )


async def find_entity_by_element_id(element_id: str) -> dict[str, Any] | None:
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE elementId(e) = $eid"
            " RETURN e.name AS name, e.type AS type, e.aliases AS aliases",
            eid=element_id,
        )
        record = await result.single()
        if record is None:
            return None
        return {"name": record["name"], "type": record["type"], "aliases": record["aliases"] or []}


async def add_alias(
    canonical_element_id: str,
    alias: str,
    source_doc: str,
    confidence: float,
) -> None:
    driver = get_driver()
    async with driver.session() as session:
        # Append alias to canonical node's aliases list (idempotent via list membership check)
        await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) = $eid
            SET e.aliases = CASE
                WHEN $alias IN coalesce(e.aliases, []) THEN e.aliases
                ELSE coalesce(e.aliases, []) + $alias
            END
            """,
            eid=canonical_element_id,
            alias=alias,
        )
        # Create a stub alias node with SAME_AS edge pointing to canonical
        await session.run(
            """
            MATCH (canonical:Entity) WHERE elementId(canonical) = $eid
            MERGE (stub:Entity {name: $alias, type: canonical.type})
            ON CREATE SET stub.canonical = false,
                          stub.aliases = [],
                          stub.source_doc = $source_doc,
                          stub.access_count = 0,
                          stub.last_accessed = null
            MERGE (stub)-[r:SAME_AS]->(canonical)
            ON CREATE SET r.confidence = $confidence,
                          r.method = "vector_similarity",
                          r.source_doc = $source_doc
            """,
            eid=canonical_element_id,
            alias=alias,
            source_doc=source_doc,
            confidence=confidence,
        )


async def create_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    content_hash: str,
) -> str:
    """MERGE a :Chunk node and [:PART_OF] edge. Returns elementId."""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document) WHERE elementId(d) = $doc_id
            MERGE (c:Chunk {content_hash: $hash})
            ON CREATE SET c.text = $text,
                          c.chunk_index = $chunk_index,
                          c.position = $chunk_index
            MERGE (c)-[:PART_OF]->(d)
            RETURN elementId(c) AS cid
            """,
            doc_id=doc_id,
            hash=content_hash,
            text=text,
            chunk_index=chunk_index,
        )
        record = await result.single()
        return record["cid"]


async def upsert_relation(
    subject_name: str,
    object_name: str,
    relation_type: str,
    confidence: float,
    source_doc: str,
) -> tuple[str, str | None]:
    """
    Returns (outcome, relation_id) where outcome is "created" | "reinforced" | "superseded".
    For "superseded", the new edge id is returned.
    """
    driver = get_driver()
    now = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        # Case 1: exact match — same (s, rel_type, o), still valid
        result = await session.run(
            """
            MATCH (s:Entity {name: $subject})-[r:RELATES_TO {type: $rel_type}]->
                  (o:Entity {name: $object})
            WHERE r.valid_until IS NULL
            RETURN elementId(r) AS rid, r.source_docs AS source_docs, r.confidence AS conf
            """,
            subject=subject_name,
            object=object_name,
            rel_type=relation_type,
        )
        exact = await result.single()

        if exact:
            existing_docs = exact["source_docs"] or []
            new_docs = (
                existing_docs if source_doc in existing_docs else existing_docs + [source_doc]
            )
            new_conf = max(exact["conf"] or confidence, confidence)
            await session.run(
                """
                MATCH (s:Entity {name: $subject})-[r:RELATES_TO {type: $rel_type}]->
                      (o:Entity {name: $object})
                WHERE r.valid_until IS NULL
                SET r.source_docs = $source_docs, r.confidence = $conf
                """,
                subject=subject_name,
                object=object_name,
                rel_type=relation_type,
                source_docs=new_docs,
                conf=new_conf,
            )
            return ("reinforced", exact["rid"])

        # Case 2: same (s, rel_type) with a different object. Supersede the
        # old edge *only* if rel_type is functional (one-object-per-subject).
        # For non-functional rels (LEADS, USES, APPROVED, ...) the new edge
        # is additive — both targets can coexist.
        from landscape.extraction.schema import FUNCTIONAL_RELATION_TYPES

        is_functional = relation_type in FUNCTIONAL_RELATION_TYPES

        result = await session.run(
            """
            MATCH (s:Entity {name: $subject})-[old:RELATES_TO {type: $rel_type}]->(other:Entity)
            WHERE other.name <> $object AND old.valid_until IS NULL
            RETURN elementId(old) AS old_rid, elementId(s) AS sid, elementId(other) AS oid
            LIMIT 1
            """,
            subject=subject_name,
            object=object_name,
            rel_type=relation_type,
        )
        conflict = await result.single()

        if conflict and is_functional:
            # Mark old edge as superseded
            await session.run(
                """
                MATCH (s:Entity {name: $subject})-[old:RELATES_TO {type: $rel_type}]->(other:Entity)
                WHERE other.name <> $object AND old.valid_until IS NULL
                SET old.valid_until = $now, old.superseded_by_doc = $source_doc
                """,
                subject=subject_name,
                object=object_name,
                rel_type=relation_type,
                now=now,
                source_doc=source_doc,
            )
            # Create the new edge
            result2 = await session.run(
                """
                MATCH (s:Entity {name: $subject}), (o:Entity {name: $object})
                CREATE (s)-[r:RELATES_TO {
                    type: $rel_type,
                    confidence: $confidence,
                    source_docs: [$source_doc],
                    valid_from: $now,
                    valid_until: null,
                    supersedes_edge_id: $old_rid,
                    access_count: 0,
                    last_accessed: null
                }]->(o)
                RETURN elementId(r) AS rid
                """,
                subject=subject_name,
                object=object_name,
                rel_type=relation_type,
                confidence=confidence,
                source_doc=source_doc,
                now=now,
                old_rid=conflict["old_rid"],
            )
            new_rec = await result2.single()
            return ("superseded", new_rec["rid"] if new_rec else None)

        # Case 3: fresh relation — no prior edge with this (s, rel_type) exists
        result = await session.run(
            """
            MATCH (s:Entity {name: $subject}), (o:Entity {name: $object})
            CREATE (s)-[r:RELATES_TO {
                type: $rel_type,
                confidence: $confidence,
                source_docs: [$source_doc],
                valid_from: $now,
                valid_until: null,
                access_count: 0,
                last_accessed: null
            }]->(o)
            RETURN elementId(r) AS rid
            """,
            subject=subject_name,
            object=object_name,
            rel_type=relation_type,
            confidence=confidence,
            source_doc=source_doc,
            now=now,
        )
        record = await result.single()
        return ("created", record["rid"] if record else None)


async def get_entities_from_chunks(chunk_element_ids: list[str]) -> list[dict[str, Any]]:
    """For a set of :Chunk elementIds, return the canonical :Entity nodes
    extracted from the parent :Document of each chunk."""
    if not chunk_element_ids:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)<-[:EXTRACTED_FROM]-(e:Entity)
            WHERE elementId(c) IN $chunk_ids AND e.canonical = true
            RETURN DISTINCT
                elementId(e) AS eid,
                e.name AS name,
                e.type AS type,
                coalesce(e.access_count, 0) AS access_count,
                e.last_accessed AS last_accessed
            """,
            chunk_ids=chunk_element_ids,
        )
        return [dict(record) async for record in result]


async def bfs_expand(
    seed_element_ids: list[str],
    max_hops: int,
) -> list[dict[str, Any]]:
    """BFS from seed entities over currently-valid :RELATES_TO edges.
    Uses Cypher shortestPath to dedupe per target. Undirected traversal
    so 'A APPROVED B' is reachable from either A or B.

    Returns one dict per (seed, target) pair with target info, distance,
    and the list of edge stats along the shortest path (for scoring and
    reinforcement touching)."""
    if not seed_element_ids:
        return []
    if max_hops < 1 or max_hops > 5:
        raise ValueError(f"max_hops must be 1..5, got {max_hops}")
    driver = get_driver()
    # max_hops is validated int; safe to interpolate
    query = f"""
    MATCH (seed:Entity) WHERE elementId(seed) IN $seed_ids
    MATCH path = shortestPath(
        (seed)-[rels:RELATES_TO*1..{max_hops}]-(target:Entity)
    )
    WHERE elementId(seed) <> elementId(target)
      AND target.canonical = true
      AND ALL(r IN rels WHERE r.valid_until IS NULL)
    RETURN
        elementId(seed) AS seed_id,
        elementId(target) AS target_id,
        target.name AS target_name,
        target.type AS target_type,
        coalesce(target.access_count, 0) AS target_access_count,
        target.last_accessed AS target_last_accessed,
        length(path) AS distance,
        [r IN rels | elementId(r)] AS edge_ids,
        [r IN rels | r.type] AS edge_types,
        [r IN rels | coalesce(r.confidence, 0.0)] AS edge_confidences,
        [r IN rels | coalesce(r.access_count, 0)] AS edge_access_counts,
        [r IN rels | r.last_accessed] AS edge_last_accessed
    """
    async with driver.session() as session:
        result = await session.run(query, seed_ids=seed_element_ids)
        return [dict(record) async for record in result]


async def touch_entities(element_ids: list[str], now: str) -> None:
    """Increment access_count and set last_accessed on the given entities."""
    if not element_ids:
        return
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) IN $ids
            SET e.access_count = coalesce(e.access_count, 0) + 1,
                e.last_accessed = $now
            """,
            ids=element_ids,
            now=now,
        )


async def touch_relations(element_ids: list[str], now: str) -> None:
    """Increment access_count and set last_accessed on the given edges."""
    if not element_ids:
        return
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH ()-[r:RELATES_TO]->() WHERE elementId(r) IN $ids
            SET r.access_count = coalesce(r.access_count, 0) + 1,
                r.last_accessed = $now
            """,
            ids=element_ids,
            now=now,
        )
