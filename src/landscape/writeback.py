"""Agent-facing write-back module for Landscape.

Exposes structured ``add_entity`` and ``add_relation`` functions that the MCP
server (Task 3) calls to let agents persist new facts into the graph + vector
store, and a ``status_summary`` function for the MCP ``status`` tool.

Design note — endpoint types in ``add_relation``:
    Both ``subject_type`` and ``object_type`` are required parameters.  Agents
    must declare what kind of entities they are relating.  This avoids the
    silent graph-quality degradation that occurred when endpoints were
    auto-created with ``entity_type="Unknown"`` — Unknown-typed nodes bypass
    the typed-similarity search path and accumulate as unresolvable cruft.

    The Unknown-type cross-type resolution path in the resolver is still
    available for callers that explicitly pass "Unknown" (e.g. when the type
    is genuinely indeterminate), but this module no longer routes through it
    by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from landscape.embeddings import encoder
from landscape.extraction.schema import normalize_relation_type
from landscape.extraction.entity_type_coercion import coerce_entity_type
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.service import persist_assertion_and_maybe_promote
from landscape.storage import neo4j_store, qdrant_store

# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AddEntityResult:
    entity_id: str          # Neo4j element id of the canonical node
    canonical_name: str     # May differ from input if resolved to existing
    resolved_to_existing: bool


@dataclass
class AddRelationResult:
    assertion_id: str       # Neo4j id of the persisted assertion
    memory_fact_id: str | None
    outcome: str            # "assertion_only" | "memory_fact"
    subject_id: str
    object_id: str


@dataclass
class StatusSummary:
    entity_count: int
    document_count: int
    relation_count: int
    conversation_count: int = 0
    turn_count: int = 0
    top_entities: list[dict] = field(default_factory=list)  # [{name, type, reinforcement}]
    recent_agent_writes: list[dict] = field(
        default_factory=list
    )  # [{subject, rel_type, object, session_id, turn_id, when}]
    recent_conversations: list[dict] = field(
        default_factory=list
    )  # [{id, title, turn_count, last_active_at}]


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------


UNKNOWN_TYPE_THRESHOLD = 0.90


async def _resolve_entity_for_writeback(
    name: str,
    entity_type: str,
    *,
    source: str,
    confidence: float,
    session_id: str,
    turn_id: str,
) -> AddEntityResult:
    canonical_type, _ = coerce_entity_type(entity_type)

    vector = encoder.encode(f"{name} ({canonical_type})")

    if canonical_type == "Unknown":
        candidates = await qdrant_store.search_entities_any_type(vector=vector, limit=5)
        effective_threshold = UNKNOWN_TYPE_THRESHOLD
    else:
        candidates = await qdrant_store.search_similar_entities(
            vector=vector,
            entity_type=canonical_type,
            limit=5,
        )
        effective_threshold = 0.85

    if candidates and candidates[0].score >= effective_threshold:
        best = candidates[0]
        canonical_id = best.payload["neo4j_node_id"]
        existing = await neo4j_store.find_entity_by_element_id(canonical_id)
        if existing is not None:
            canonical_name = existing["name"] if existing["name"] else name
            if name.lower() != canonical_name.lower() and name not in existing["aliases"]:
                await neo4j_store.merge_alias(canonical_id, name, source, best.score)

            turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)
            await neo4j_store.link_entity_to_turn(canonical_id, turn_element_id, confidence=confidence)

            return AddEntityResult(
                entity_id=canonical_id,
                canonical_name=canonical_name,
                resolved_to_existing=True,
            )

    turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)

    entity_id = await neo4j_store.merge_entity(
        name=name,
        entity_type=canonical_type,
        source_doc=source,
        confidence=confidence,
        doc_element_id=None,
        model="agent",
        created_by="agent",
        session_id=session_id,
        turn_id=turn_id,
        subtype=entity_type if canonical_type != entity_type else None,
    )

    await neo4j_store.link_entity_to_turn(entity_id, turn_element_id, confidence=confidence)

    now = datetime.now(UTC).isoformat()
    await qdrant_store.upsert_entity(
        neo4j_element_id=entity_id,
        name=name,
        entity_type=canonical_type,
        source_doc=source,
        timestamp=now,
        vector=vector,
    )

    return AddEntityResult(
        entity_id=entity_id,
        canonical_name=name,
        resolved_to_existing=False,
    )


async def add_entity(
    name: str,
    entity_type: str,
    *,
    source: str,
    confidence: float = 0.8,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> AddEntityResult:
    """Persist an entity authored by an agent.

    Both ``session_id`` and ``turn_id`` are required.  Agent-authored entities
    must be anchored to a real conversation Turn for provenance to be
    meaningful.  Calls without them raise ``ValueError``.

    Resolution flow:
    1. Validate that session_id and turn_id are both non-empty.
    2. Embed ``name`` with the same encoder used by the ingest pipeline.
    3. Search Qdrant for a near-duplicate canonical node using the same typed
       similarity path as ingest.
    4. If resolved → return the canonical id/name, add a :MENTIONED_IN edge to
       anchor the mention to the current Turn, and record the alias as an Alias
       node when the surface form differs.
    5. If not resolved → create a new Neo4j entity node (``created_by="agent"``)
       and upsert its embedding into Qdrant.  Provenance is anchored via a
       :MENTIONED_IN edge to the Turn.
    """
    if not session_id or not turn_id:
        raise ValueError(
            "session_id and turn_id are required for agent write-back; "
            "synthetic-Document provenance has been removed"
        )

    return await _resolve_entity_for_writeback(
        name,
        entity_type,
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
    )


async def add_relation(
    subject: str,
    subject_type: str,
    object_: str,
    object_type: str,
    rel_type: str,
    *,
    source: str,
    confidence: float = 0.8,
    session_id: str | None = None,
    turn_id: str | None = None,
    subtype: str | None = None,
) -> AddRelationResult:
    """Persist a relationship authored by an agent.

    Both ``session_id`` and ``turn_id`` are required.  Agent-authored relations
    must be anchored to a real conversation Turn for provenance to be
    meaningful.  Calls without them raise ``ValueError``.

    Both endpoints are auto-resolved (or auto-created with the declared types)
    if they don't already exist.  ``subject_type`` and ``object_type`` are
    required: agents must declare what kind of entities they are relating so
    that the write-back layer can find existing typed nodes and avoid creating
    Unknown-typed duplicates that degrade graph quality.

    ``rel_type`` is normalised via ``normalize_relation_type`` before the
    assertion is persisted, so callers may pass synonyms like ``"EMPLOYED_BY"``
    and they will be stored under the canonical family.  If the family is
    promotable, the assertion is upgraded to a MemoryFact.

    Args:
        subject:      Name of the subject entity (e.g. "Alice Chen").
        subject_type: Entity type for the subject (e.g. "Person").  Required.
        object_:      Name of the object entity (e.g. "Beacon Corp").
        object_type:  Entity type for the object (e.g. "Organization").  Required.
        rel_type:     Relationship type (e.g. "WORKS_FOR").
        source:       Provenance label.
        confidence:   Extraction confidence in [0, 1].  Default 0.8.
        session_id:   Conversation session identifier.
        turn_id:      Turn identifier within the session.
    """
    if not session_id or not turn_id:
        raise ValueError(
            "session_id and turn_id are required for agent write-back; "
            "synthetic-Document provenance has been removed"
        )

    # Resolve / create both endpoints with the declared types.
    subj_result = await add_entity(
        subject,
        subject_type,
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
    )
    obj_result = await add_entity(
        object_,
        object_type,
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
    )

    turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)

    payload = AssertionPayload(
        source_kind="turn",
        source_id=f"{session_id}:{turn_id}",
        raw_subject_text=subject,
        raw_relation_text=rel_type,
        raw_object_text=object_,
        confidence=confidence,
        family_candidate=normalize_relation_type(rel_type),
        subtype=None,
    )
    persistence = await persist_assertion_and_maybe_promote(
        payload,
        source_node_id=turn_element_id,
        source_kind="turn",
        subject_entity_id=subj_result.entity_id,
        object_entity_id=obj_result.entity_id,
        chunk_ids=[],
    )

    return AddRelationResult(
        assertion_id=persistence.assertion_id,
        memory_fact_id=persistence.fact_id,
        outcome="memory_fact" if persistence.fact_id is not None else "assertion_only",
        subject_id=subj_result.entity_id,
        object_id=obj_result.entity_id,
    )


async def status_summary() -> StatusSummary:
    """Return a compact summary of the graph state suitable for the MCP
    ``status`` tool (~200 tokens).

    Runs three small read-only Cypher queries in a single session:
    a. Counts for Entity / Document / live MemoryFact edges.
    b. Top-5 entities by total incident edge reinforcement (a proxy for
       support).
    c. Most-recent 5 agent-written assertions / memory facts.
    """
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        # (a) Counts — use OPTIONAL MATCH so an empty Document or MemoryFact
        # set does not suppress the whole row via Cypher's eager-MATCH semantics.
        count_result = await session.run(
            """
            OPTIONAL MATCH (e:Entity)
            WITH count(e) AS entity_count
            OPTIONAL MATCH (d:Document)
            WITH entity_count, count(d) AS doc_count
            OPTIONAL MATCH ()-[r:MEMORY_REL]->()
            WHERE r.current = true
            RETURN entity_count, doc_count, count(r) AS rel_count
            """
        )
        count_rec = await count_result.single()
        entity_count = count_rec["entity_count"] if count_rec else 0
        document_count = count_rec["doc_count"] if count_rec else 0
        relation_count = count_rec["rel_count"] if count_rec else 0

        # (b) Top-5 entities by reinforcement (sum of incident edge confidence)
        top_result = await session.run(
            """
            MATCH (e:Entity)-[r:MEMORY_REL]-()
            WHERE r.current = true
            WITH e.name AS name, e.type AS type,
                 sum(coalesce(r.confidence_agg, 0)) AS reinforcement
            ORDER BY reinforcement DESC
            LIMIT 5
            RETURN name, type, reinforcement
            """
        )
        top_entities = []
        async for rec in top_result:
            top_entities.append({
                "name": rec["name"],
                "type": rec["type"],
                "reinforcement": rec["reinforcement"],
            })

        # (c) Recent agent writes — DISTINCT elementId(f) defends against
        # any future duplicate facts from surfacing the same write twice.
        recent_result = await session.run(
            """
            MATCH (t:Turn)-[:ASSERTS]->(a:Assertion)-[:SUPPORTS]->(f:MemoryFact)
            OPTIONAL MATCH (s:Entity)-[:AS_SUBJECT]->(f)
            OPTIONAL MATCH (f)-[:AS_OBJECT]->(o:Entity)
            WITH DISTINCT elementId(f) AS fid, s.name AS subject,
                 f.family AS rel_type, coalesce(o.name, '') AS object,
                 t.session_id AS session_id, t.turn_id AS turn_id,
                 a.created_at AS when
            ORDER BY when DESC
            LIMIT 5
            RETURN fid, subject, rel_type, object, session_id, turn_id, when
            """
        )
        recent_agent_writes = []
        async for rec in recent_result:
            recent_agent_writes.append({
                "subject": rec["subject"],
                "rel_type": rec["rel_type"],
                "object": rec["object"],
                "session_id": rec["session_id"],
                "turn_id": rec["turn_id"],
                "when": rec["when"],
            })

        # (d) Conversation + turn counts, and recent 3 conversations
        conv_result = await session.run(
            """
            OPTIONAL MATCH (c:Conversation)
            WITH count(c) AS conversation_count
            OPTIONAL MATCH (t:Turn)
            RETURN conversation_count, count(t) AS turn_count
            """
        )
        conv_rec = await conv_result.single()
        conversation_count = conv_rec["conversation_count"] if conv_rec else 0
        turn_count = conv_rec["turn_count"] if conv_rec else 0

        recent_conv_result = await session.run(
            """
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WITH c, count(t) AS tc, max(t.timestamp) AS last_active
            ORDER BY last_active DESC
            LIMIT 3
            RETURN c.id AS id, coalesce(c.title, c.id) AS title,
                   tc AS turn_count, last_active AS last_active_at
            """
        )
        recent_conversations = []
        async for rec in recent_conv_result:
            recent_conversations.append({
                "id": rec["id"],
                "title": rec["title"],
                "turn_count": rec["turn_count"],
                "last_active_at": rec["last_active_at"],
            })

    return StatusSummary(
        entity_count=entity_count,
        document_count=document_count,
        relation_count=relation_count,
        conversation_count=conversation_count,
        turn_count=turn_count,
        top_entities=top_entities,
        recent_agent_writes=recent_agent_writes,
        recent_conversations=recent_conversations,
    )
