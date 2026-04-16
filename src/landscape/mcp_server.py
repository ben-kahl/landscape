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
async def search(query: str, hops: int = 2, limit: int = 10) -> str:
    """Hybrid retrieval over the Landscape knowledge graph.

    Combines vector similarity (Qdrant) with graph BFS expansion (Neo4j) to
    surface entities relevant to *query*.  Returns up to *limit* results
    ranked by a composite score (vector_sim × graph_distance × recency).

    Args:
        query: Natural-language search query.
        hops:  BFS depth for graph expansion (1–3 recommended). Default 2.
        limit: Maximum number of results to return. Default 10.

    Returns:
        JSON object ``{results: [{name, type, score, path_edge_types}],
        touched_entity_count}``.
    """
    await _ensure_init()
    from landscape.retrieval.query import retrieve

    result = await retrieve(query, hops=hops, limit=limit)
    output = {
        "results": [
            {
                "name": r.name,
                "type": r.type,
                "score": round(r.score, 6),
                "path_edge_types": r.path_edge_types,
            }
            for r in result.results
        ],
        "touched_entity_count": len(result.touched_entity_ids),
    }
    return json.dumps(output)


# ---------------------------------------------------------------------------
# Tool: remember
# ---------------------------------------------------------------------------


@mcp.tool()
async def remember(text: str, title: str, session_id: str | None = None, turn_id: str | None = None) -> str:
    """Ingest a text document into the Landscape memory store.

    Chunks the text, extracts entities and relations via LLM, resolves
    entities against the existing graph, and writes everything to Neo4j +
    Qdrant.

    Args:
        text:       Full text to ingest (markdown/plain text/prose).
        title:      Document title used as the source provenance label.
        session_id: Optional conversation session identifier.  When provided
                    together with *turn_id*, the resulting document is linked
                    to the Turn via :INGESTED_IN and extracted entities are
                    tagged with the conversation's session/turn provenance via
                    :MENTIONED_IN edges.
        turn_id:    Optional turn identifier within the session.  Only used
                    when *session_id* is also provided.

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
    confidence: float = 0.8,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """Persist an agent-authored entity into the knowledge graph.

    Resolves the name against existing entities first; if a near-duplicate
    canonical entity exists, returns its id without creating a duplicate.

    Args:
        name:        Entity name (e.g. "Alice Chen").
        entity_type: Entity type (e.g. "Person", "Organization").
        source:      Provenance label (e.g. "agent:session-1:turn-3").
        confidence:  Extraction confidence in [0, 1].  Default 0.8.
        session_id:  Optional conversation session identifier.
        turn_id:     Optional turn identifier within the session.

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
    object: str,
    rel_type: str,
    source: str,
    confidence: float = 0.8,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """Persist an agent-authored relationship between two entities.

    Both endpoints are auto-created (type ``Unknown``) if they don't exist.
    ``rel_type`` is normalised to the canonical vocabulary before write.
    Functional-relation supersession applies automatically: writing
    ``WORKS_FOR`` for an entity that already has a live ``WORKS_FOR`` edge
    will supersede the old edge and record an audit trail.

    Args:
        subject:    Name of the subject entity.
        object:     Name of the object entity.
        rel_type:   Relationship type (e.g. "WORKS_FOR", "LEADS").
        source:     Provenance label.
        confidence: Extraction confidence.  Default 0.8.
        session_id: Optional session identifier.
        turn_id:    Optional turn identifier.

    Returns:
        JSON object ``{relation_id, outcome, subject_id, object_id}`` where
        *outcome* is ``"created"``, ``"reinforced"``, or ``"superseded"``.
    """
    await _ensure_init()
    from landscape.writeback import add_relation as _add_relation

    result = await _add_relation(
        subject,
        object,
        rel_type,
        source=source,
        confidence=confidence,
        session_id=session_id,
        turn_id=turn_id,
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
        JSON object with ``entity_count``, ``document_count``,
        ``relation_count``, ``top_entities`` (top-5 by reinforcement), and
        ``recent_agent_writes`` (last 5 agent-authored relations).
    """
    from landscape.writeback import status_summary

    summary = await status_summary()
    return json.dumps(asdict(summary))


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
