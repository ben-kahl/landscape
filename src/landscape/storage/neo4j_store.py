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


def _validate_id_segment(name: str, value: str) -> None:
    """Reject ':' in session_id / turn_id to avoid Turn.id ambiguity."""
    if ":" in value:
        raise ValueError(f"{name} must not contain ':' (got {value!r})")


async def merge_conversation(
    session_id: str,
    title: str | None = None,
    agent_id: str | None = None,
) -> tuple[str, bool]:
    """MERGE on Conversation.id = session_id. Returns (element_id, created).
    On MATCH: updates last_active_at to now.
    On CREATE: sets started_at + last_active_at to now, plus title/agent_id."""
    _validate_id_segment("session_id", session_id)
    driver = get_driver()
    now = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        result = await session.run(
            """
            MERGE (c:Conversation {id: $session_id})
            ON CREATE SET c.started_at = $now,
                          c.last_active_at = $now,
                          c.title = $title,
                          c.agent_id = $agent_id
            ON MATCH SET  c.last_active_at = $now
            RETURN elementId(c) AS cid, (c.started_at = $now) AS created
            """,
            session_id=session_id,
            now=now,
            title=title,
            agent_id=agent_id,
        )
        record = await result.single()
        return record["cid"], record["created"]


async def merge_turn(
    session_id: str,
    turn_id: str,
    turn_number: int | None = None,
    role: str | None = None,
    summary: str | None = None,
) -> tuple[str, bool]:
    """MERGE on Turn.id = f'{session_id}:{turn_id}'. Returns (element_id, created).
    Ensures parent Conversation exists. Creates :HAS_TURN edge. If turn_number > 1,
    creates :NEXT edge from the prior turn."""
    _validate_id_segment("session_id", session_id)
    _validate_id_segment("turn_id", turn_id)
    # Ensure parent conversation exists (updates last_active_at as side effect)
    await merge_conversation(session_id)
    composite_id = f"{session_id}:{turn_id}"
    driver = get_driver()
    now = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        result = await session.run(
            """
            MERGE (t:Turn {id: $composite_id})
            ON CREATE SET t.session_id = $session_id,
                          t.turn_id = $turn_id,
                          t.turn_number = $turn_number,
                          t.role = $role,
                          t.summary = $summary,
                          t.timestamp = $now
            WITH t
            MATCH (c:Conversation {id: $session_id})
            MERGE (c)-[:HAS_TURN]->(t)
            RETURN elementId(t) AS tid, (t.timestamp = $now) AS created
            """,
            composite_id=composite_id,
            session_id=session_id,
            turn_id=turn_id,
            turn_number=turn_number,
            role=role,
            summary=summary,
            now=now,
        )
        record = await result.single()
        turn_element_id = record["tid"]
        created = record["created"]

    # Wire :NEXT edge from prior turn when turn_number > 1
    if turn_number is not None and turn_number > 1:
        async with driver.session() as session:
            await session.run(
                """
                MATCH (prior:Turn {session_id: $session_id, turn_number: $prior_num})
                MATCH (curr:Turn {id: $composite_id})
                MERGE (prior)-[:NEXT]->(curr)
                """,
                session_id=session_id,
                prior_num=turn_number - 1,
                composite_id=composite_id,
            )

    return turn_element_id, created


async def link_entity_to_turn(
    entity_element_id: str,
    turn_element_id: str,
    confidence: float = 1.0,
) -> None:
    """Idempotent MERGE of :MENTIONED_IN edge from Entity to Turn.
    On repeat calls, keeps the higher confidence value."""
    driver = get_driver()
    now = datetime.now(UTC).isoformat()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (e:Entity) WHERE elementId(e) = $eid
            MATCH (t:Turn)   WHERE elementId(t) = $tid
            MERGE (e)-[r:MENTIONED_IN]->(t)
            ON CREATE SET r.confidence = $confidence,
                          r.created_at = $now
            ON MATCH SET  r.confidence = CASE
                              WHEN $confidence > r.confidence THEN $confidence
                              ELSE r.confidence
                          END
            """,
            eid=entity_element_id,
            tid=turn_element_id,
            confidence=confidence,
            now=now,
        )


async def link_document_to_turn(
    doc_element_id: str,
    turn_element_id: str,
) -> None:
    """Idempotent MERGE of :INGESTED_IN edge from Document to Turn."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (d:Document) WHERE elementId(d) = $did
            MATCH (t:Turn)     WHERE elementId(t) = $tid
            MERGE (d)-[:INGESTED_IN]->(t)
            """,
            did=doc_element_id,
            tid=turn_element_id,
        )


async def get_entities_in_conversation(session_id: str) -> list[str]:
    """Return elementIds of Entity nodes with MENTIONED_IN → Turn → Conversation(id=session_id).
    Inclusive semantics: any entity mentioned in any turn of that conversation, regardless of
    where else it appears. Empty list if session_id unknown."""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Conversation {id: $session_id})-[:HAS_TURN]->(t:Turn)<-[:MENTIONED_IN]-(e:Entity)
            RETURN DISTINCT elementId(e) AS eid
            """,
            session_id=session_id,
        )
        return [record["eid"] async for record in result]


async def get_entities_since(since: datetime) -> list[str]:
    """Return elementIds of Entity nodes mentioned in turns with t.timestamp >= since (ISO string compare).
    Dedup across turns."""
    since_iso = since.isoformat()
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(t:Turn)
            WHERE t.timestamp >= $since_iso
            RETURN DISTINCT elementId(e) AS eid
            """,
            since_iso=since_iso,
        )
        return [record["eid"] async for record in result]


async def get_conversation_detail(session_id: str, turn_limit: int = 10) -> dict:
    """Return {"conversation": {id, title, agent_id, started_at, last_active_at} | None,
             "turns": [{id, turn_id, turn_number, role, summary, timestamp, entities_mentioned: [{eid, name, type}]}]}
    Turns ordered by t.timestamp ASC, capped to turn_limit. If session_id unknown: conversation=None, turns=[]."""
    driver = get_driver()
    async with driver.session() as session:
        conv_result = await session.run(
            """
            MATCH (c:Conversation {id: $session_id})
            RETURN c.id AS id,
                   properties(c).title AS title,
                   properties(c).agent_id AS agent_id,
                   c.started_at AS started_at,
                   c.last_active_at AS last_active_at
            """,
            session_id=session_id,
        )
        conv_record = await conv_result.single()

    if conv_record is None:
        return {"conversation": None, "turns": []}

    conversation = {
        "id": conv_record["id"],
        "title": conv_record["title"],
        "agent_id": conv_record["agent_id"],
        "started_at": conv_record["started_at"],
        "last_active_at": conv_record["last_active_at"],
    }

    async with driver.session() as session:
        turns_result = await session.run(
            """
            MATCH (c:Conversation {id: $session_id})-[:HAS_TURN]->(t:Turn)
            WITH t ORDER BY t.timestamp ASC LIMIT $turn_limit
            OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(t)
            RETURN elementId(t) AS id,
                   t.turn_id AS turn_id,
                   properties(t).turn_number AS turn_number,
                   properties(t).role AS role,
                   properties(t).summary AS summary,
                   t.timestamp AS timestamp,
                   collect(CASE WHEN e IS NOT NULL THEN {eid: elementId(e), name: e.name, type: e.type} END) AS entities_mentioned
            ORDER BY timestamp ASC
            """,
            session_id=session_id,
            turn_limit=turn_limit,
        )
        turns = []
        async for record in turns_result:
            entities = [em for em in record["entities_mentioned"] if em is not None]
            turns.append({
                "id": record["id"],
                "turn_id": record["turn_id"],
                "turn_number": record["turn_number"],
                "role": record["role"],
                "summary": record["summary"],
                "timestamp": record["timestamp"],
                "entities_mentioned": entities,
            })

    return {"conversation": conversation, "turns": turns}


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
    """Returns the elementId of the entity node.

    doc_element_id is optional. When None, the :EXTRACTED_FROM edge is
    skipped (used when provenance comes from a :Turn rather than a :Document).

    subtype is the agent's or LLM's original entity_type string before
    coercion to the canonical vocab. Only written ON CREATE and only when
    subtype is not None. Existing nodes are not updated on match.
    """
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
                          e.aliases = [],
                          e.access_count = 0,
                          e.last_accessed = null,
                          e.created_by = $created_by,
                          e.session_id = $session_id,
                          e.turn_id = $turn_id,
                          e.subtype = $subtype
            RETURN elementId(e) AS eid
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
        eid = record["eid"]

    if doc_element_id is not None:
        async with driver.session() as session:
            await session.run(
                """
                MATCH (e:Entity) WHERE elementId(e) = $eid
                MATCH (d:Document) WHERE elementId(d) = $doc_id
                MERGE (e)-[:EXTRACTED_FROM {method: "llm", model: $model}]->(d)
                """,
                eid=eid,
                doc_id=doc_element_id,
                model=model,
            )

    return eid


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
    created_by: str = "ingest",
    session_id: str | None = None,
    turn_id: str | None = None,
) -> tuple[str, str | None]:
    """
    Returns (outcome, relation_id) where outcome is "created" | "reinforced" | "superseded".
    For "superseded", the new edge id is returned.
    """
    if created_by not in ("ingest", "agent"):
        raise ValueError(f"created_by must be 'ingest' or 'agent', got {created_by!r}")
    driver = get_driver()
    now = datetime.now(UTC).isoformat()

    # Build agent provenance entry to append to source_docs when created_by=="agent"
    agent_entry = f"agent:{session_id or '?'}:{turn_id or '?'}" if created_by == "agent" else None

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
            if agent_entry and agent_entry not in new_docs:
                new_docs = new_docs + [agent_entry]
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
            # Mark old edge as superseded (old edge keeps its original provenance)
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
            # Build source_docs for new edge, appending agent entry if applicable
            new_edge_docs = [source_doc]
            if agent_entry:
                new_edge_docs.append(agent_entry)
            # Create the new edge with provenance
            result2 = await session.run(
                """
                MATCH (s:Entity {name: $subject}) WITH s LIMIT 1
                MATCH (o:Entity {name: $object}) WITH s, o LIMIT 1
                CREATE (s)-[r:RELATES_TO {
                    type: $rel_type,
                    confidence: $confidence,
                    source_docs: $source_docs,
                    valid_from: $now,
                    valid_until: null,
                    supersedes_edge_id: $old_rid,
                    access_count: 0,
                    last_accessed: null,
                    created_by: $created_by,
                    session_id: $session_id,
                    turn_id: $turn_id
                }]->(o)
                RETURN elementId(r) AS rid
                """,
                subject=subject_name,
                object=object_name,
                rel_type=relation_type,
                confidence=confidence,
                source_docs=new_edge_docs,
                now=now,
                old_rid=conflict["old_rid"],
                created_by=created_by,
                session_id=session_id,
                turn_id=turn_id,
            )
            new_rec = await result2.single()
            return ("superseded", new_rec["rid"] if new_rec else None)

        # Case 3: fresh relation — no prior edge with this (s, rel_type) exists
        fresh_docs = [source_doc]
        if agent_entry:
            fresh_docs.append(agent_entry)
        result = await session.run(
            """
            MATCH (s:Entity {name: $subject}) WITH s LIMIT 1
            MATCH (o:Entity {name: $object}) WITH s, o LIMIT 1
            CREATE (s)-[r:RELATES_TO {
                type: $rel_type,
                confidence: $confidence,
                source_docs: $source_docs,
                valid_from: $now,
                valid_until: null,
                access_count: 0,
                last_accessed: null,
                created_by: $created_by,
                session_id: $session_id,
                turn_id: $turn_id
            }]->(o)
            RETURN elementId(r) AS rid
            """,
            subject=subject_name,
            object=object_name,
            rel_type=relation_type,
            confidence=confidence,
            source_docs=fresh_docs,
            now=now,
            created_by=created_by,
            session_id=session_id,
            turn_id=turn_id,
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


async def run_cypher_readonly(
    cypher: str, params: dict | None = None
) -> list[dict]:
    """Execute a read-only Cypher query.

    Validates the query with :func:`~landscape.storage.cypher_guard.assert_read_only`
    first (raises :exc:`CypherWriteAttempted` on any write keyword), then
    executes inside a ``session.execute_read`` transaction for driver-level
    enforcement.  Returns rows as a list of plain Python dicts.
    """
    from landscape.storage.cypher_guard import assert_read_only

    assert_read_only(cypher)

    driver = get_driver()
    params = params or {}

    async def _work(tx: Any) -> list[dict]:
        result = await tx.run(cypher, **params)
        return [dict(record) async for record in result]

    async with driver.session() as session:
        return await session.execute_read(_work)
