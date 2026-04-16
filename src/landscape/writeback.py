"""Agent-facing write-back module for Landscape.

Exposes structured ``add_entity`` and ``add_relation`` functions that the MCP
server (Task 3) calls to let agents persist new facts into the graph + vector
store, and a ``status_summary`` function for the MCP ``status`` tool.

Design note — missing endpoints in ``add_relation``:
    If the subject or object name doesn't resolve to an existing entity, we
    auto-create it with ``entity_type="Unknown"``.  This matches the principle
    that agents shouldn't have to pre-declare every entity before writing a
    relation: the agent knows "Alice joined Beacon" — it doesn't know or care
    whether Alice was already indexed.  The Unknown type propagates into the
    graph and can be corrected later by a richer ingestion pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from landscape.embeddings import encoder
from landscape.entities import resolver
from landscape.extraction.schema import normalize_relation_type
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
    relation_id: str        # Neo4j element id of the (new or reinforced) edge
    outcome: str            # "created" | "reinforced" | "superseded"
    subject_id: str
    object_id: str


@dataclass
class StatusSummary:
    entity_count: int
    document_count: int
    relation_count: int
    top_entities: list[dict] = field(default_factory=list)       # [{name, type, reinforcement}]
    recent_agent_writes: list[dict] = field(default_factory=list)  # [{subject, rel_type, object, session_id, turn_id, when}]


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------


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
    3. Call ``resolver.resolve_entity`` to check for a near-duplicate canonical
       node.  The resolver searches Qdrant by type, so the entity_type must
       match for resolution to fire.
    4. If resolved → return the canonical id/name and add a :MENTIONED_IN edge
       to anchor the mention to the current Turn.
    5. If not resolved → create a new Neo4j entity node (``created_by="agent"``)
       and upsert its embedding into Qdrant.  Provenance is anchored via a
       :MENTIONED_IN edge to the Turn.
    """
    if not session_id or not turn_id:
        raise ValueError(
            "session_id and turn_id are required for agent write-back; "
            "synthetic-Document provenance has been removed"
        )

    vector = encoder.encode(f"{name} ({entity_type})")

    canonical_id, is_new, _sim = await resolver.resolve_entity(
        name=name,
        entity_type=entity_type,
        vector=vector,
        source_doc=source,
    )

    if not is_new:
        # Resolved to an existing canonical node — fetch its name for the caller.
        existing = await neo4j_store.find_entity_by_element_id(canonical_id)
        canonical_name = existing["name"] if existing else name

        # Record the mention in this turn.
        turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)
        await neo4j_store.link_entity_to_turn(canonical_id, turn_element_id, confidence=confidence)

        return AddEntityResult(
            entity_id=canonical_id,
            canonical_name=canonical_name,
            resolved_to_existing=True,
        )

    # No match — create the Turn node then create the entity node.
    turn_element_id, _ = await neo4j_store.merge_turn(session_id, turn_id)

    entity_id = await neo4j_store.merge_entity(
        name=name,
        entity_type=entity_type,
        source_doc=source,
        confidence=confidence,
        doc_element_id=None,
        model="agent",
        created_by="agent",
        session_id=session_id,
        turn_id=turn_id,
    )

    await neo4j_store.link_entity_to_turn(entity_id, turn_element_id, confidence=confidence)

    now = datetime.now(UTC).isoformat()
    await qdrant_store.upsert_entity(
        neo4j_element_id=entity_id,
        name=name,
        entity_type=entity_type,
        source_doc=source,
        timestamp=now,
        vector=vector,
    )

    return AddEntityResult(
        entity_id=entity_id,
        canonical_name=name,
        resolved_to_existing=False,
    )


async def add_relation(
    subject: str,
    object_: str,
    rel_type: str,
    *,
    source: str,
    confidence: float = 0.8,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> AddRelationResult:
    """Persist a relationship authored by an agent.

    Both ``session_id`` and ``turn_id`` are required.  Agent-authored relations
    must be anchored to a real conversation Turn for provenance to be
    meaningful.  Calls without them raise ``ValueError``.

    Both endpoints are auto-resolved (or auto-created with type ``"Unknown"``)
    if they don't already exist — the agent shouldn't have to pre-declare
    entities before writing a relation.

    ``rel_type`` is normalised via ``normalize_relation_type`` before the edge
    is written, so callers may pass synonyms like ``"EMPLOYED_BY"`` and they
    will be stored as the canonical ``"WORKS_FOR"``.  Functional-type
    supersession semantics apply automatically.
    """
    if not session_id or not turn_id:
        raise ValueError(
            "session_id and turn_id are required for agent write-back; "
            "synthetic-Document provenance has been removed"
        )

    # Resolve / create both endpoints (type Unknown if caller didn't supply one)
    subj_result = await add_entity(
        subject,
        "Unknown",
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
    )
    obj_result = await add_entity(
        object_,
        "Unknown",
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
    )

    canonical_rel_type = normalize_relation_type(rel_type)

    outcome, relation_id = await neo4j_store.upsert_relation(
        subject_name=subject,
        object_name=object_,
        relation_type=canonical_rel_type,
        confidence=confidence,
        source_doc=source,
        created_by="agent",
        session_id=session_id,
        turn_id=turn_id,
    )

    return AddRelationResult(
        relation_id=relation_id or "",
        outcome=outcome,
        subject_id=subj_result.entity_id,
        object_id=obj_result.entity_id,
    )


async def status_summary() -> StatusSummary:
    """Return a compact summary of the graph state suitable for the MCP
    ``status`` tool (~200 tokens).

    Runs three small read-only Cypher queries in a single session:
    a. Counts for Entity / Document / live RELATES_TO edges.
    b. Top-5 entities by total incident edge access_count (a proxy for
       reinforcement).
    c. Most-recent 5 agent-written RELATES_TO edges.
    """
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        # (a) Counts — use OPTIONAL MATCH so an empty Document or RELATES_TO
        # set does not suppress the whole row via Cypher's eager-MATCH semantics.
        count_result = await session.run(
            """
            OPTIONAL MATCH (e:Entity)
            WITH count(e) AS entity_count
            OPTIONAL MATCH (d:Document)
            WITH entity_count, count(d) AS doc_count
            OPTIONAL MATCH ()-[r:RELATES_TO]->() WHERE r.valid_until IS NULL
            RETURN entity_count, doc_count, count(r) AS rel_count
            """
        )
        count_rec = await count_result.single()
        entity_count = count_rec["entity_count"] if count_rec else 0
        document_count = count_rec["doc_count"] if count_rec else 0
        relation_count = count_rec["rel_count"] if count_rec else 0

        # (b) Top-5 entities by reinforcement (sum of incident edge access_counts)
        top_result = await session.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]-()
            WHERE r.valid_until IS NULL
            WITH e.name AS name, e.type AS type,
                 sum(coalesce(r.access_count, 0)) AS reinforcement
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

        # (c) Recent agent writes
        recent_result = await session.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.created_by = 'agent'
            RETURN s.name AS subject, r.type AS rel_type, o.name AS object,
                   r.session_id AS session_id, r.turn_id AS turn_id,
                   r.valid_from AS when
            ORDER BY r.valid_from DESC
            LIMIT 5
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

    return StatusSummary(
        entity_count=entity_count,
        document_count=document_count,
        relation_count=relation_count,
        top_entities=top_entities,
        recent_agent_writes=recent_agent_writes,
    )
