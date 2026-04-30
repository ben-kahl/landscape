from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from landscape.storage.neo4j_driver import get_driver


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
    """MERGE on Conversation.id = session_id. Returns (element_id, created)."""
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
    """MERGE on Turn.id = f'{session_id}:{turn_id}'."""
    _validate_id_segment("session_id", session_id)
    _validate_id_segment("turn_id", turn_id)
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


async def create_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    content_hash: str,
) -> str:
    """MERGE a document-scoped :Chunk node and [:PART_OF] edge."""
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
