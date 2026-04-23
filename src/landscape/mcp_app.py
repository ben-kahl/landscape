"""Landscape MCP app — transport-agnostic tool registration for MCP clients.

This module owns the FastMCP app and all tool definitions. Runtime startup
belongs elsewhere so the same app can be mounted inside FastAPI.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("landscape")
_AUTO_INGEST_SEEN_FINGERPRINTS: set[str] = set()
_EXPLICIT_MEMORY_TURN_KEYS: set[tuple[str, str]] = set()


def _turn_key(session_id: str, turn_id: str) -> tuple[str, str]:
    return (session_id, turn_id)


async def _auto_ingest_turn(
    text: str,
    session_id: str,
    turn_id: str,
    role: str = "user",
    debug: bool = False,
):
    from landscape.conversation_ingestion import ConversationTurn, ingest_conversation_turn

    if _turn_key(session_id, turn_id) in _EXPLICIT_MEMORY_TURN_KEYS:
        return None

    turn = ConversationTurn(session_id=session_id, turn_id=turn_id, role=role, text=text)
    return await ingest_conversation_turn(
        turn,
        seen_fingerprints=_AUTO_INGEST_SEEN_FINGERPRINTS,
        debug=debug,
    )


def _log_auto_ingestion_failure(task: asyncio.Task) -> None:
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Landscape auto-ingestion task failed unexpectedly")
        return

    if exc is not None:
        logger.error(
            "Landscape auto-ingestion task failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _schedule_auto_ingestion(
    text: str,
    session_id: str,
    turn_id: str,
    role: str = "user",
    debug: bool = False,
) -> asyncio.Task:
    task = asyncio.create_task(
        _auto_ingest_turn(text, session_id, turn_id, role=role, debug=debug)
    )
    task.add_done_callback(_log_auto_ingestion_failure)
    return task


@mcp.tool()
async def search(
    query: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    session_id: str | None = None,
    since_hours: int | None = None,
) -> str:
    """Hybrid retrieval over the Landscape knowledge graph."""
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


@mcp.tool()
async def remember(
    text: str,
    title: str,
    session_id: str,
    turn_id: str,
    debug: bool = False,
) -> str:
    """Ingest a text document into the Landscape memory store."""
    from landscape.pipeline import ingest

    result = await ingest(
        text,
        title,
        session_id=session_id,
        turn_id=turn_id,
        debug=debug,
    )
    _EXPLICIT_MEMORY_TURN_KEYS.add(_turn_key(session_id, turn_id))
    output = {
        "doc_id": result.doc_id,
        "entities_created": result.entities_created,
        "relations_created": result.relations_created,
        "relations_superseded": result.relations_superseded,
        "already_existed": result.already_existed,
    }
    return json.dumps(output)


@mcp.tool()
async def capture_turn(
    session_id: str,
    turn_id: str,
    role: str,
    text: str,
    debug: bool = False,
) -> str:
    """Capture an explicit conversation turn boundary for background ingestion."""
    from landscape.conversation_ingestion import ConversationTurn, should_auto_ingest_turn

    turn = ConversationTurn(session_id=session_id, turn_id=turn_id, role=role, text=text)
    if _turn_key(session_id, turn_id) in _EXPLICIT_MEMORY_TURN_KEYS:
        return json.dumps({"accepted": False, "scheduled": False})
    if not should_auto_ingest_turn(turn, seen_fingerprints=set()):
        return json.dumps({"accepted": False, "scheduled": False})

    _schedule_auto_ingestion(text, session_id, turn_id, role=role, debug=debug)
    return json.dumps({"accepted": True, "scheduled": True})


@mcp.tool()
async def add_entity(
    name: str,
    entity_type: str,
    source: str,
    session_id: str,
    turn_id: str,
    confidence: float = 0.8,
) -> str:
    """Persist an agent-authored entity into the knowledge graph."""
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
    """Persist an agent-authored relationship between two entities."""
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


@mcp.tool()
async def graph_query(cypher: str, params: dict | None = None) -> str:
    """Execute a read-only Cypher query against the Neo4j knowledge graph."""
    from landscape.storage.cypher_guard import CypherWriteAttempted
    from landscape.storage.neo4j_store import run_cypher_readonly

    try:
        rows = await run_cypher_readonly(cypher, params or {})
    except CypherWriteAttempted as exc:
        raise ValueError(str(exc)) from exc

    return json.dumps({"rows": rows})


@mcp.tool()
async def status() -> str:
    """Return a compact summary of the Landscape graph state."""
    from landscape.writeback import status_summary

    summary = await status_summary()
    return json.dumps(asdict(summary))


@mcp.tool()
async def conversation_history(session_id: str, limit: int = 10) -> str:
    """Return the turns of a conversation in chronological order."""
    from landscape.storage import neo4j_store

    detail = await neo4j_store.get_conversation_detail(session_id, turn_limit=limit)
    return json.dumps(detail, default=str)
