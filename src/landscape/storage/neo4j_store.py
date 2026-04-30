import hashlib
from datetime import UTC, datetime
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from landscape.config import settings
from landscape.memory_graph import (
    FAMILY_REGISTRY,
    AssertionPayload,
    alias_id,
    assertion_id,
    fact_key,
    slot_key,
)

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


def build_default_conversation_title(
    session_id: str,
    *,
    agent_id: str | None = None,
    started_at: str | None = None,
) -> str:
    timestamp = started_at or datetime.now(UTC).isoformat()
    compact_timestamp = (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace("+0000", "Z")
        .replace("+00:00", "Z")
    )
    if "." in compact_timestamp:
        compact_timestamp = compact_timestamp.split(".", maxsplit=1)[0] + "Z"
    short_hash = hashlib.sha256(session_id.encode()).hexdigest()[:8]
    label = (agent_id or "agent").strip() or "agent"
    return f"{label}:{compact_timestamp}:{short_hash}"


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
    conversation_title = title or build_default_conversation_title(
        session_id,
        agent_id=agent_id,
        started_at=now,
    )
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
            title=conversation_title,
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
    """Return entities mentioned in the conversation identified by ``session_id``.

    Inclusive semantics: any entity mentioned in any turn of that conversation,
    regardless of where else it appears. Empty list if session_id is unknown.
    """
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Conversation {id: $session_id})-[:HAS_TURN]->(t:Turn)
                  <-[:MENTIONED_IN]-(e:Entity)
            RETURN DISTINCT elementId(e) AS eid
            """,
            session_id=session_id,
        )
        return [record["eid"] async for record in result]


async def get_entities_since(since: datetime) -> list[str]:
    """Return entities mentioned in turns whose timestamp is >= ``since``.

    Uses ISO string comparison and deduplicates across turns.
    """
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


async def get_chunks_in_conversation(session_id: str) -> list[str]:
    """Return chunk ids belonging to Documents ingested in any Turn of the
    named Conversation. Empty list if session_id unknown."""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Conversation {id: $session_id})-[:HAS_TURN]->(t:Turn)
                  <-[:INGESTED_IN]-(d:Document)<-[:PART_OF]-(ch:Chunk)
            RETURN DISTINCT coalesce(ch.chunk_id, elementId(ch)) AS cid
            """,
            session_id=session_id,
        )
        return [record["cid"] async for record in result]


async def get_chunks_since(since: datetime) -> list[str]:
    """Return chunk ids belonging to Documents ingested in Turns with
    t.timestamp >= since (ISO string compare)."""
    since_iso = since.isoformat()
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (ch:Chunk)-[:PART_OF]->(d:Document)-[:INGESTED_IN]->(t:Turn)
            WHERE t.timestamp >= $since_iso
            RETURN DISTINCT coalesce(ch.chunk_id, elementId(ch)) AS cid
            """,
            since_iso=since_iso,
        )
        return [record["cid"] async for record in result]


async def get_conversation_detail(session_id: str, turn_limit: int = 10) -> dict:
    """Return conversation metadata plus the oldest ``turn_limit`` turns.

    Shape:
    ``{"conversation": {...} | None, "turns": [{..., "entities_mentioned":
    [{"eid", "name", "type"}]}]}``

    Turns are ordered by ``t.timestamp`` ascending. If ``session_id`` is
    unknown, returns ``{"conversation": None, "turns": []}``.
    """
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
                   collect(
                       CASE WHEN e IS NOT NULL
                            THEN {
                                eid: elementId(e),
                                name: e.name,
                                type: e.type
                            }
                       END
                   ) AS entities_mentioned
            ORDER BY timestamp ASC
            """,
            session_id=session_id,
            turn_limit=turn_limit,
        )
        turns = []
        async for record in turns_result:
            entities = [em for em in record["entities_mentioned"] if em is not None]
            turns.append(
                {
                    "id": record["id"],
                    "turn_id": record["turn_id"],
                    "turn_number": record["turn_number"],
                    "role": record["role"],
                    "summary": record["summary"],
                    "timestamp": record["timestamp"],
                    "entities_mentioned": entities,
                }
            )

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
    resolved: list[str] = []
    for entity_ref in entity_refs:
        entity_id = await _resolve_entity_app_id(entity_ref)
        if entity_id not in resolved:
            resolved.append(entity_id)
    return resolved


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
                          a.normalized_name = toLower(trim($alias_text))
            MERGE (a)-[r:SAME_AS]->(canonical)
            ON CREATE SET r.confidence = $confidence,
                          r.source_doc = $source_doc
            """,
            entity_id=canonical_entity_id,
            alias_id=aid,
            alias_text=alias_text,
            confidence=confidence,
            source_doc=source_doc,
        )
    return aid


async def merge_assertion(payload: AssertionPayload) -> str:
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


async def link_assertion_to_chunk(assertion_id: str, chunk_id: str) -> None:
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (a:Assertion {id: $assertion_id})
            MATCH (c:Chunk)
            WHERE c.chunk_id = $chunk_id OR elementId(c) = $chunk_id
            MERGE (a)-[:MENTIONS_CHUNK]->(c)
            """,
            assertion_id=assertion_id,
            chunk_id=chunk_id,
        )


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
                      fact.current = true,
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
                r.current = true,
                r.confidence_agg = fact.confidence_agg,
                r.subtype = fact.subtype,
                r.subject_entity_id = subject.id,
                r.object_entity_id = object.id,
                r.access_count = coalesce(fact.access_count, 0),
                r.last_accessed = fact.last_accessed,
                r.updated_at = $now
        )
        SET fact.current = true,
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
                MATCH (f:MemoryFact {family: $family, slot_key: $slot_key, current: true})
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
                MATCH (old:MemoryFact {family: $family, slot_key: $slot_key, current: true})
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
                    SET old.current = false,
                        old.updated_at = $now
                    WITH old
                    OPTIONAL MATCH ()-[r:MEMORY_REL {memory_fact_id: $old_fact_id}]-()
                    SET r.current = false,
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
                   fact.current AS current,
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
                   rel.current AS memory_rel_current
            """,
            fact_id=memory_fact_id,
        )
        record = await result.single()
        return dict(record) if record is not None else None


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
    """MERGE a document-scoped :Chunk node and [:PART_OF] edge.

    The stable chunk identity is ``chunk_id = f"{doc_id}:{chunk_index}:{content_hash}"``.
    Returns that chunk_id so downstream systems can use it as the shared key.
    """
    driver = get_driver()
    chunk_id = f"{doc_id}:{chunk_index}:{content_hash}"
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document) WHERE elementId(d) = $doc_id
            MERGE (c:Chunk {chunk_id: $chunk_id})
            ON CREATE SET c.text = $text,
                          c.doc_id = $doc_id,
                          c.chunk_index = $chunk_index,
                          c.position = $chunk_index,
                          c.content_hash = $content_hash
            ON MATCH SET  c.text = $text,
                          c.doc_id = $doc_id,
                          c.chunk_index = $chunk_index,
                          c.position = $chunk_index,
                          c.content_hash = $content_hash
            MERGE (c)-[:PART_OF]->(d)
            RETURN c.chunk_id AS cid
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=text,
            chunk_index=chunk_index,
            content_hash=content_hash,
        )
        record = await result.single()
        return record["cid"]


async def upsert_relation(
    subject_name: str | None = None,
    object_name: str | None = None,
    relation_type: str = "",
    confidence: float = 1.0,
    source_doc: str = "",
    created_by: str = "ingest",
    session_id: str | None = None,
    turn_id: str | None = None,
    subtype: str | None = None,
    quantity_value: float | str | None = None,
    quantity_unit: str | None = None,
    quantity_kind: str | None = None,
    time_scope: str | None = None,
    subject_node_id: str | None = None,
    object_node_id: str | None = None,
) -> tuple[str, str | None]:
    """
    Returns (outcome, relation_id) where outcome is "created" | "reinforced" | "superseded".
    For "superseded", the new edge id is returned.

    Endpoints are matched either by canonical Neo4j element id (subject_node_id /
    object_node_id) or by name (subject_name / object_name).  Id-based matching is
    preferred: when both ``subject_node_id`` and ``object_node_id`` are provided the
    MATCH clauses use ``elementId()`` so alias stubs and same-surface-name homonyms
    cannot steal the relation from the intended canonical nodes.  Name-based matching
    is kept as a fallback for callers (e.g. the ingest pipeline) that do not have ids
    available.
    """
    if created_by not in ("ingest", "agent"):
        raise ValueError(f"created_by must be 'ingest' or 'agent', got {created_by!r}")
    if bool(subject_node_id) != bool(object_node_id):
        raise ValueError(
            "subject_node_id and object_node_id must both be provided or both omitted"
        )
    if not subject_node_id and not subject_name:
        raise ValueError("Either subject_node_id or subject_name must be provided")
    if not object_node_id and not object_name:
        raise ValueError("Either object_node_id or object_name must be provided")
    if not relation_type:
        raise ValueError("relation_type must be a non-empty string")

    use_ids = bool(subject_node_id and object_node_id)

    driver = get_driver()
    now = datetime.now(UTC).isoformat()

    # Build agent provenance entry to append to source_docs when created_by=="agent"
    agent_entry = f"agent:{session_id or '?'}:{turn_id or '?'}" if created_by == "agent" else None

    # Load the per-rel-type functional key declaration (step 2).
    from landscape.extraction.schema import FUNCTIONAL_KEYS

    key_fields = FUNCTIONAL_KEYS.get(relation_type)
    # Object-keyed rels treat `subtype` as part of edge identity -- a new
    # subtype on the same (s, o) is a distinct fact (promotion/demotion) and
    # triggers Case 2 supersession, not Case 1 reinforce. Non-object-keyed
    # rels ignore subtype in identity; Case 1 reinforces and updates subtype
    # to the newer value (non-null wins).
    object_keyed = key_fields == ("object",)

    async with driver.session() as session:
        # Case 1: exact match on (s, rel_type, o). Null-safe subtype check
        # gates reinforce for object-keyed rels.
        if object_keyed:
            subtype_identity_clause = (
                "AND (r.subtype IS NULL OR $subtype IS NULL OR r.subtype = $subtype)"
            )
        else:
            subtype_identity_clause = ""

        if use_ids:
            result = await session.run(
                f"""
                MATCH (s:Entity)-[r:RELATES_TO {{type: $rel_type}}]->(o:Entity)
                WHERE elementId(s) = $subject_node_id
                  AND elementId(o) = $object_node_id
                  AND r.valid_until IS NULL
                  {subtype_identity_clause}
                RETURN elementId(r) AS rid,
                       r.source_docs AS source_docs,
                       r.confidence AS conf,
                       r.subtype AS subtype,
                       r.quantity_value AS quantity_value,
                       r.quantity_unit AS quantity_unit,
                       r.quantity_kind AS quantity_kind,
                       r.time_scope AS time_scope
                """,
                subject_node_id=subject_node_id,
                object_node_id=object_node_id,
                rel_type=relation_type,
                subtype=subtype,
            )
        else:
            result = await session.run(
                f"""
                MATCH (s:Entity {{name: $subject_name}})-[r:RELATES_TO {{type: $rel_type}}]->
                      (o:Entity {{name: $object_name}})
                WHERE r.valid_until IS NULL
                  {subtype_identity_clause}
                RETURN elementId(r) AS rid,
                       r.source_docs AS source_docs,
                       r.confidence AS conf,
                       r.subtype AS subtype,
                       r.quantity_value AS quantity_value,
                       r.quantity_unit AS quantity_unit,
                       r.quantity_kind AS quantity_kind,
                       r.time_scope AS time_scope
                """,
                subject_name=subject_name,
                object_name=object_name,
                rel_type=relation_type,
                subtype=subtype,
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
            # Subtype: newer non-null value wins; null input preserves existing.
            new_subtype = subtype if subtype is not None else exact["subtype"]
            new_quantity_value = (
                quantity_value if quantity_value is not None else exact["quantity_value"]
            )
            new_quantity_unit = (
                quantity_unit if quantity_unit is not None else exact["quantity_unit"]
            )
            new_quantity_kind = (
                quantity_kind if quantity_kind is not None else exact["quantity_kind"]
            )
            new_time_scope = time_scope if time_scope is not None else exact["time_scope"]
            await session.run(
                """
                MATCH ()-[r:RELATES_TO]->()
                WHERE elementId(r) = $rid
                SET r.source_docs = $source_docs,
                    r.confidence = $conf,
                    r.access_count = coalesce(r.access_count, 0) + 1,
                    r.last_accessed = $now,
                    r.subtype = $subtype,
                    r.quantity_value = $quantity_value,
                    r.quantity_unit = $quantity_unit,
                    r.quantity_kind = $quantity_kind,
                    r.time_scope = $time_scope
                """,
                rid=exact["rid"],
                source_docs=new_docs,
                conf=new_conf,
                subtype=new_subtype,
                now=now,
                quantity_value=new_quantity_value,
                quantity_unit=new_quantity_unit,
                quantity_kind=new_quantity_kind,
                time_scope=new_time_scope,
            )
            return ("reinforced", exact["rid"])

        # Case 2: a live edge already occupies the "slot" the new edge would
        # land in, per FUNCTIONAL_KEYS. Supersede the old edge.
        #
        # Slot identity is (subject, type) plus each extra key field declared
        # in FUNCTIONAL_KEYS[rel_type]. Extra fields ("object", "subtype") are
        # null-safe: if either the incoming or existing edge has NULL for a
        # declared field, treat the slot as non-matching rather than colliding
        # two unknowns -- prevents spurious supersession on missing data.
        conflict = None
        if key_fields is not None:
            # Object-keyed (HAS_TITLE): same (s, rel_type, o), different
            # subtype. Case 1 already rejected null-safe matches.
            # Subject-/subtype-keyed: different object is the conflict signal.
            if "object" in key_fields:
                if use_ids:
                    object_clause = "AND elementId(other) = $object_node_id"
                else:
                    object_clause = "AND other.name = $object_name"
            else:
                if use_ids:
                    object_clause = "AND elementId(other) <> $object_node_id"
                else:
                    object_clause = "AND other.name <> $object_name"

            if "subtype" in key_fields:
                # Subtype-keyed: both sides must carry a matching non-null
                # subtype for the slot to collide.
                subtype_clause = (
                    "AND old.subtype IS NOT NULL AND $subtype IS NOT NULL "
                    "AND old.subtype = $subtype"
                )
            elif "object" in key_fields:
                # Object-keyed: both sides must carry a non-null differing
                # subtype -- Case 1 already absorbed null-safe and equal.
                subtype_clause = (
                    "AND old.subtype IS NOT NULL AND $subtype IS NOT NULL "
                    "AND old.subtype <> $subtype"
                )
            else:
                subtype_clause = ""

            if use_ids:
                result = await session.run(
                    f"""
                    MATCH (s:Entity)-[old:RELATES_TO {{type: $rel_type}}]->(other:Entity)
                    WHERE elementId(s) = $subject_node_id
                      AND old.valid_until IS NULL
                      {object_clause}
                      {subtype_clause}
                    RETURN elementId(old) AS old_rid,
                           elementId(s) AS sid,
                           elementId(other) AS oid
                    LIMIT 1
                    """,
                    subject_node_id=subject_node_id,
                    object_node_id=object_node_id,
                    rel_type=relation_type,
                    subtype=subtype,
                )
            else:
                result = await session.run(
                    f"""
                    MATCH (s:Entity {{name: $subject_name}})
                          -[old:RELATES_TO {{type: $rel_type}}]->(other:Entity)
                    WHERE old.valid_until IS NULL
                      {object_clause}
                      {subtype_clause}
                    RETURN elementId(old) AS old_rid,
                           elementId(s) AS sid,
                           elementId(other) AS oid
                    LIMIT 1
                    """,
                    subject_name=subject_name,
                    object_name=object_name,
                    rel_type=relation_type,
                    subtype=subtype,
                )
            conflict = await result.single()

        if conflict:
            # Mark old edge as superseded (old edge keeps its original provenance)
            await session.run(
                """
                MATCH ()-[old:RELATES_TO]->()
                WHERE elementId(old) = $old_rid
                SET old.valid_until = $now, old.superseded_by_doc = $source_doc
                """,
                old_rid=conflict["old_rid"],
                now=now,
                source_doc=source_doc,
            )
            # Build source_docs for new edge, appending agent entry if applicable
            new_edge_docs = [source_doc]
            if agent_entry:
                new_edge_docs.append(agent_entry)
            # Create the new edge with provenance
            if use_ids:
                result2 = await session.run(
                    """
                    MATCH (s:Entity) WHERE elementId(s) = $subject_node_id
                    MATCH (o:Entity) WHERE elementId(o) = $object_node_id
                    CREATE (s)-[r:RELATES_TO {
                        type: $rel_type,
                        subtype: $subtype,
                        confidence: $confidence,
                        source_docs: $source_docs,
                        valid_from: $now,
                        valid_until: null,
                        supersedes_edge_id: $old_rid,
                        access_count: 1,
                        last_accessed: $now,
                        created_by: $created_by,
                        session_id: $session_id,
                        turn_id: $turn_id,
                        quantity_value: $quantity_value,
                        quantity_unit: $quantity_unit,
                        quantity_kind: $quantity_kind,
                        time_scope: $time_scope
                    }]->(o)
                    RETURN elementId(r) AS rid
                    """,
                    subject_node_id=subject_node_id,
                    object_node_id=object_node_id,
                    rel_type=relation_type,
                    subtype=subtype,
                    confidence=confidence,
                    source_docs=new_edge_docs,
                    now=now,
                    old_rid=conflict["old_rid"],
                    created_by=created_by,
                    session_id=session_id,
                    turn_id=turn_id,
                    quantity_value=quantity_value,
                    quantity_unit=quantity_unit,
                    quantity_kind=quantity_kind,
                    time_scope=time_scope,
                )
            else:
                result2 = await session.run(
                    """
                    MATCH (s:Entity {name: $subject_name}) WITH s LIMIT 1
                    MATCH (o:Entity {name: $object_name}) WITH s, o LIMIT 1
                    CREATE (s)-[r:RELATES_TO {
                        type: $rel_type,
                        subtype: $subtype,
                        confidence: $confidence,
                        source_docs: $source_docs,
                        valid_from: $now,
                        valid_until: null,
                        supersedes_edge_id: $old_rid,
                        access_count: 1,
                        last_accessed: $now,
                        created_by: $created_by,
                        session_id: $session_id,
                        turn_id: $turn_id,
                        quantity_value: $quantity_value,
                        quantity_unit: $quantity_unit,
                        quantity_kind: $quantity_kind,
                        time_scope: $time_scope
                    }]->(o)
                    RETURN elementId(r) AS rid
                    """,
                    subject_name=subject_name,
                    object_name=object_name,
                    rel_type=relation_type,
                    subtype=subtype,
                    confidence=confidence,
                    source_docs=new_edge_docs,
                    now=now,
                    old_rid=conflict["old_rid"],
                    created_by=created_by,
                    session_id=session_id,
                    turn_id=turn_id,
                    quantity_value=quantity_value,
                    quantity_unit=quantity_unit,
                    quantity_kind=quantity_kind,
                    time_scope=time_scope,
                )
            new_rec = await result2.single()
            return ("superseded", new_rec["rid"] if new_rec else None)

        # Case 3: fresh relation -- no prior edge with this (s, rel_type) exists
        fresh_docs = [source_doc]
        if agent_entry:
            fresh_docs.append(agent_entry)
        if use_ids:
            result = await session.run(
                """
                MATCH (s:Entity) WHERE elementId(s) = $subject_node_id
                MATCH (o:Entity) WHERE elementId(o) = $object_node_id
                CREATE (s)-[r:RELATES_TO {
                    type: $rel_type,
                    subtype: $subtype,
                    confidence: $confidence,
                    source_docs: $source_docs,
                    valid_from: $now,
                    valid_until: null,
                    access_count: 1,
                    last_accessed: $now,
                    created_by: $created_by,
                    session_id: $session_id,
                    turn_id: $turn_id,
                    quantity_value: $quantity_value,
                    quantity_unit: $quantity_unit,
                    quantity_kind: $quantity_kind,
                    time_scope: $time_scope
                }]->(o)
                RETURN elementId(r) AS rid
                """,
                subject_node_id=subject_node_id,
                object_node_id=object_node_id,
                rel_type=relation_type,
                subtype=subtype,
                confidence=confidence,
                source_docs=fresh_docs,
                now=now,
                created_by=created_by,
                session_id=session_id,
                turn_id=turn_id,
                quantity_value=quantity_value,
                quantity_unit=quantity_unit,
                quantity_kind=quantity_kind,
                time_scope=time_scope,
            )
        else:
            result = await session.run(
                """
                MATCH (s:Entity {name: $subject_name}) WITH s LIMIT 1
                MATCH (o:Entity {name: $object_name}) WITH s, o LIMIT 1
                CREATE (s)-[r:RELATES_TO {
                    type: $rel_type,
                    subtype: $subtype,
                    confidence: $confidence,
                    source_docs: $source_docs,
                    valid_from: $now,
                    valid_until: null,
                    access_count: 1,
                    last_accessed: $now,
                    created_by: $created_by,
                    session_id: $session_id,
                    turn_id: $turn_id,
                    quantity_value: $quantity_value,
                    quantity_unit: $quantity_unit,
                    quantity_kind: $quantity_kind,
                    time_scope: $time_scope
                }]->(o)
                RETURN elementId(r) AS rid
                """,
                subject_name=subject_name,
                object_name=object_name,
                rel_type=relation_type,
                subtype=subtype,
                confidence=confidence,
                source_docs=fresh_docs,
                now=now,
                created_by=created_by,
                session_id=session_id,
                turn_id=turn_id,
                quantity_value=quantity_value,
                quantity_unit=quantity_unit,
                quantity_kind=quantity_kind,
                time_scope=time_scope,
            )
        record = await result.single()
        return ("created", record["rid"] if record else None)


async def get_entities_from_chunks(chunk_element_ids: list[str]) -> list[dict[str, Any]]:
    """For a set of document-scoped chunk ids, return the canonical :Entity nodes
    extracted from the parent :Document of each chunk, together with the
    list of seed chunk ids each entity was reached via (so the caller
    can propagate chunk-hit similarity scores to the right entities)."""
    if not chunk_element_ids:
        return []
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)<-[:EXTRACTED_FROM]-(e:Entity)
            WHERE (c.chunk_id IN $chunk_ids OR elementId(c) IN $chunk_ids)
              AND e.canonical = true
            WITH e,
                 collect(
                     DISTINCT coalesce(c.chunk_id, elementId(c))
                 ) AS chunk_eids
            RETURN
                elementId(e) AS eid,
                e.name AS name,
                e.type AS type,
                coalesce(e.access_count, 0) AS access_count,
                e.last_accessed AS last_accessed,
                chunk_eids
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
        [r IN rels | r.subtype] AS edge_subtypes,
        [r IN rels | {{
            quantity_value: r.quantity_value,
            quantity_unit: r.quantity_unit,
            quantity_kind: r.quantity_kind,
            time_scope: r.time_scope
        }}] AS edge_quantities,
        [r IN rels | coalesce(r.confidence, 0.0)] AS edge_confidences,
        [r IN rels | coalesce(r.access_count, 0)] AS edge_access_counts,
        [r IN rels | r.last_accessed] AS edge_last_accessed
    """
    async with driver.session() as session:
        result = await session.run(query, seed_ids=seed_element_ids)
        return [dict(record) async for record in result]


async def bfs_expand_memory_rel(
    seed_entity_ids: list[str],
    max_hops: int,
) -> list[dict[str, Any]]:
    """BFS over current MEMORY_REL edges.

    Input seed refs may be Neo4j element ids or stable app ids. Returned rows
    preserve the stable app ids (`seed_id`/`target_id`) for memory-graph
    callers and include Neo4j element ids for retrieval/ranking code that
    keys Qdrant payloads, allowlists, and touch writes by elementId().
    """
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
      AND ALL(r IN rels WHERE r.current = true)
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
