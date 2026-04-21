"""Landscape MCP server — exposes Landscape as a memory backend to any MCP
client (Claude Code, Cursor, custom).

Run via the ``landscape-mcp`` console entry-point (defined in pyproject.toml):

    uv run landscape-mcp

The server communicates over stdio using JSON-RPC 2.0 as defined by the MCP
specification.  Six tools are exposed:

* ``search``       — hybrid retrieval (vector + graph traversal)
* ``remember``     — ingest a text document into the memory store
* ``add_entity``   — agent-authored entity write-back
* ``add_relation`` — agent-authored relation write-back
* ``graph_query``  — read-only Cypher pass-through
* ``status``       — compact graph summary (~200-token wake-up payload)

All errors are returned as MCP tool-error responses so agents can react to
them rather than receiving an unhandled exception.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("landscape")


# ---------------------------------------------------------------------------
# Helper: ensure encoder + Qdrant collections are ready before any tool runs.
# FastMCP does not have a built-in lifespan hook in this version, so we do a
# lazy initialisation gate inside each tool.
# ---------------------------------------------------------------------------

_initialised = False


async def _ensure_init() -> None:
    global _initialised
    if _initialised:
        return
    from landscape.embeddings import encoder
    from landscape.storage import qdrant_store

    encoder.load_model()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()
    _initialised = True


# ---------------------------------------------------------------------------
# Tool: search
# ---------------------------------------------------------------------------


@mcp.tool()
async def search(
    query: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    session_id: str | None = None,
    since_hours: int | None = None,
) -> str:
    """Hybrid retrieval over the Landscape knowledge graph.

    Combines vector similarity (Qdrant) with graph BFS expansion (Neo4j) to
    surface entities relevant to *query*.  Returns up to *limit* results
    ranked by a composite score (vector_sim × graph_distance × recency).
    Also returns raw chunk text passages up to *chunk_limit* items, providing
    verbatim source context alongside the ranked entity list.

    Args:
        query:       Natural-language search query.
        hops:        BFS depth for graph expansion (1–3 recommended). Default 2.
        limit:       Maximum number of entity results to return. Default 10.
        chunk_limit: Maximum number of raw chunk passages to return. Default 3.
                     Chunks are filtered by *session_id* and *since_hours* the
                     same way entities are.
        session_id:  If supplied, tag retrieval reinforcement to this session.
        since_hours: If supplied (int >= 1), exclude facts older than this many
                     hours.  Values of 0 or below are treated as unset (no
                     temporal filter applied), matching the behaviour of the
                     HTTP /query endpoint.

    Returns:
        JSON object ``{results: [{name, type, score, path_edge_types}],
        touched_entity_count,
        chunks: [{text, source_doc, doc_id, position, score}]}``.
        *chunks* contains verbatim chunk text ordered by score descending.
        Both *results* and *chunks* respect the *session_id* and *since_hours*
        filters when supplied.
    """
    await _ensure_init()
    from landscape.retrieval.query import retrieve

    since = (
        datetime.now(UTC) - timedelta(hours=since_hours)
        if since_hours is not None and since_hours >= 1
        else None
    )
    result = await retrieve(
        query,
        hops=hops,
        limit=limit,
        chunk_limit=chunk_limit,
        session_id=session_id,
        since=since,
    )
    output = {
        "results": [
            {
                "name": r.name,
                "type": r.type,
                "score": round(r.score, 6),
                "path_edge_types": r.path_edge_types,
                "path_edge_subtypes": r.path_edge_subtypes,
                "path_edge_quantities": r.path_edge_quantities,
            }
            for r in result.results
        ],
        "touched_entity_count": len(result.touched_entity_ids),
        "chunks": [
            {
                "text": c.text,
                "source_doc": c.source_doc,
                "doc_id": c.doc_id,
                "position": c.position,
                "score": round(c.score, 6),
            }
            for c in result.chunks
        ],
    }
    return json.dumps(output)


# ---------------------------------------------------------------------------
# Tool: remember
# ---------------------------------------------------------------------------


@mcp.tool()
async def remember(text: str, title: str, session_id: str, turn_id: str) -> str:
    """Ingest a text document into the Landscape memory store.

    Chunks the text, extracts entities and relations via LLM, resolves
    entities against the existing graph, and writes everything to Neo4j +
    Qdrant.

    Both ``session_id`` and ``turn_id`` are required.  Agent ingestion of a
    document only makes sense within a conversation; anchoring to a Turn
    ensures provenance is meaningful and avoids graph cruft.

    Args:
        text:       Full text to ingest (markdown/plain text/prose).
        title:      Document title used as the source provenance label.
        session_id: Conversation session identifier.  The resulting document is
                    linked to the Turn via :INGESTED_IN and extracted entities
                    are tagged with the conversation's session/turn provenance
                    via :MENTIONED_IN edges.
        turn_id:    Turn identifier within the session.

    Returns:
        JSON object ``{doc_id, entities_created, relations_created,
        relations_superseded, already_existed}``.
    """
    await _ensure_init()
    from landscape.pipeline import ingest

    result = await ingest(text, title, session_id=session_id, turn_id=turn_id)
    output = {
        "doc_id": result.doc_id,
        "entities_created": result.entities_created,
        "relations_created": result.relations_created,
        "relations_superseded": result.relations_superseded,
        "already_existed": result.already_existed,
    }
    return json.dumps(output)


# ---------------------------------------------------------------------------
# Tool: add_entity
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_entity(
    name: str,
    entity_type: str,
    source: str,
    session_id: str,
    turn_id: str,
    confidence: float = 0.8,
) -> str:
    """Persist an agent-authored entity into the knowledge graph.

    Resolves the name against existing entities first; if a near-duplicate
    canonical entity exists, returns its id without creating a duplicate.

    Both ``session_id`` and ``turn_id`` are required.  Agent-authored entities
    must be anchored to a real conversation Turn; this ensures provenance is
    meaningful and avoids graph cruft from synthetic Document nodes.

    Args:
        name:        Entity name (e.g. "Alice Chen").
        entity_type: Entity type. Prefer the 8 canonical types: Person,
                     Organization, Project, Technology, Location, Concept,
                     Event, Document. Non-canonical types are coerced to the
                     nearest canonical via embedding similarity (threshold
                     0.55); the original type is preserved as ``subtype`` on
                     the node so nuance is not lost.
        source:      Provenance label (e.g. "agent:session-1:turn-3").
        session_id:  Conversation session identifier.
        turn_id:     Turn identifier within the session.
        confidence:  Extraction confidence in [0, 1].  Default 0.8.

    Returns:
        JSON object ``{entity_id, canonical_name, resolved_to_existing}``.
    """
    await _ensure_init()
    from landscape.writeback import add_entity as _add_entity

    result = await _add_entity(
        name,
        entity_type,
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
    )
    return json.dumps(
        {
            "entity_id": result.entity_id,
            "canonical_name": result.canonical_name,
            "resolved_to_existing": result.resolved_to_existing,
        }
    )


# ---------------------------------------------------------------------------
# Tool: add_relation
# ---------------------------------------------------------------------------


@mcp.tool()
async def add_relation(
    subject: str,
    subject_type: str,
    object: str,
    object_type: str,
    rel_type: str,
    source: str,
    session_id: str,
    turn_id: str,
    confidence: float = 0.8,
    subtype: str | None = None,
) -> str:
    """Persist an agent-authored relationship between two entities.

    Both ``subject_type`` and ``object_type`` are required.  You must declare
    what kind of entities you are relating — this is enforced to prevent
    Unknown-typed nodes from accumulating in the graph and degrading retrieval
    quality.  Common canonical types: Person, Organization, Project,
    Technology, Location, Concept, Event, Document.

    Endpoints are auto-resolved against existing nodes of the declared type,
    or auto-created with the supplied types if no near-duplicate is found.
    ``rel_type`` is normalised to the canonical vocabulary before write:
    first via a string-synonym map, then via embedding-based coercion if the
    supplied type is semantically closer to a different canonical.
    Functional-relation supersession applies automatically: writing
    ``WORKS_FOR`` for an entity that already has a live ``WORKS_FOR`` edge
    will supersede the old edge and record an audit trail.

    Both ``session_id`` and ``turn_id`` are required.  Agent-authored relations
    must be anchored to a real conversation Turn; this ensures provenance is
    meaningful and avoids graph cruft from synthetic Document nodes.

    Args:
        subject:      Name of the subject entity (e.g. "Alice Chen").
        subject_type: Entity type of the subject. Required. Prefer canonical
                      types: Person, Organization, Project, Technology,
                      Location, Concept, Event, Document. Non-canonical types
                      are coerced to the nearest canonical and preserved as
                      ``subtype`` on the node.
        object:       Name of the object entity (e.g. "Beacon Corp").
        object_type:  Entity type of the object. Required. Same canonical-type
                      coercion applies as for subject_type.
        rel_type:     Relationship type (e.g. "WORKS_FOR", "LEADS").
        source:       Provenance label.
        session_id:   Conversation session identifier.
        turn_id:      Turn identifier within the session.
        confidence:   Extraction confidence.  Default 0.8.
        subtype:      Optional snake_case nuance for the edge — preserved as
                      edge metadata. Examples: "senior_engineer" on HAS_TITLE,
                      "daughter" on FAMILY_OF, "favorite_color" on HAS_PREFERENCE.
                      Object-keyed rels (HAS_TITLE) treat subtype as part of
                      the edge's slot identity: writing a new subtype on the
                      same (subject, object) pair supersedes the old edge.

    Returns:
        JSON object ``{relation_id, outcome, subject_id, object_id}`` where
        *outcome* is ``"created"``, ``"reinforced"``, or ``"superseded"``.

    Canonical relation types (synonyms are normalized; semantically-confused
    types are coerced via embedding similarity):
      WORKS_FOR    - employment / org affiliation. Use for "joined",
                     "moved to a job at", "is a Y at X".
      LEADS        - manages or runs. Use for "heads", "directs", "manages".
      MEMBER_OF    - non-employment group membership. Use for "is on the X team", "part of X group".
      REPORTS_TO   - direct manager relationship.
      APPROVED     - sign-off / authorization.
      USES         - technology / dependency.
      BELONGS_TO   - parent-org / division relationship.
      LOCATED_IN   - physical location only. NOT for org changes - use WORKS_FOR.
      CREATED      - authored / built / founded.
      RELATED_TO   - fallback when nothing else fits.

    Functional relations (writing a new object supersedes the prior one):
      WORKS_FOR, REPORTS_TO, BELONGS_TO.
    """
    await _ensure_init()
    from landscape.writeback import add_relation as _add_relation

    result = await _add_relation(
        subject,
        subject_type,
        object,
        object_type,
        rel_type,
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
        subtype=subtype,
    )
    return json.dumps(
        {
            "relation_id": result.relation_id,
            "outcome": result.outcome,
            "subject_id": result.subject_id,
            "object_id": result.object_id,
        }
    )


# ---------------------------------------------------------------------------
# Tool: graph_query
# ---------------------------------------------------------------------------


@mcp.tool()
async def graph_query(cypher: str, params: dict | None = None) -> str:
    """Execute a read-only Cypher query against the Neo4j knowledge graph.

    Write operations (``CREATE``, ``DELETE``, ``MERGE``, ``SET``, ``REMOVE``,
    ``DROP``, etc.) are rejected — including write keywords in subqueries.

    Args:
        cypher: Cypher query string.  Must be read-only.
        params: Optional query parameters dict.

    Returns:
        JSON object ``{rows: [...]}``.  Each element in *rows* is a dict
        mapping column names to serialisable values.

    Raises MCP tool error if the query contains write operations.
    """
    from landscape.storage.cypher_guard import CypherWriteAttempted
    from landscape.storage.neo4j_store import run_cypher_readonly

    try:
        rows = await run_cypher_readonly(cypher, params or {})
    except CypherWriteAttempted as exc:
        # Return as MCP tool error so agents can react gracefully.
        raise ValueError(str(exc)) from exc

    return json.dumps({"rows": rows})


# ---------------------------------------------------------------------------
# Tool: status
# ---------------------------------------------------------------------------


@mcp.tool()
async def status() -> str:
    """Return a compact summary of the Landscape graph state.

    Designed as a low-cost wake-up payload (~200 tokens) an agent can load
    at session start to understand what is already in memory before querying.

    Returns:
        JSON object with:
        - ``entity_count`` — total :Entity nodes
        - ``document_count`` — total :Document nodes
        - ``relation_count`` — live (non-superseded) :RELATES_TO edges
        - ``conversation_count`` — total :Conversation nodes
        - ``turn_count`` — total :Turn nodes across all conversations
        - ``top_entities`` — top-5 entities by reinforcement (incident edge access_count)
        - ``recent_agent_writes`` — last 5 agent-authored relations
        - ``recent_conversations`` — last 3 conversations by last_active_at,
          each with ``{id, title, turn_count, last_active_at}``
    """
    from landscape.writeback import status_summary

    summary = await status_summary()
    return json.dumps(asdict(summary))


# ---------------------------------------------------------------------------
# Tool: conversation_history
# ---------------------------------------------------------------------------


@mcp.tool()
async def conversation_history(session_id: str, limit: int = 10) -> str:
    """Return the turns of a conversation in chronological order with entities mentioned in each.

    Args:
        session_id: Conversation session identifier.
        limit:      Max turns to return (ordered by timestamp ASC). Default 10.

    Returns:
        JSON of {conversation: {...} | null, turns: [{..., entities_mentioned: [...]}]}.
    """
    await _ensure_init()
    from landscape.storage import neo4j_store

    detail = await neo4j_store.get_conversation_detail(session_id, turn_limit=limit)
    return json.dumps(detail, default=str)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio (blocking).

    ``FastMCP.run()`` creates its own event loop internally, so this is a
    simple synchronous wrapper suitable for use as a console script entry-point.
    """
    # Pre-load the encoder so the first tool call is not slow.
    from landscape.embeddings import encoder

    encoder.load_model()

    mcp.run(transport="stdio")
