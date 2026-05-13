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

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl

from landscape.config import settings
from landscape.security import require_current_scope
from landscape.storage.oauth_provider import LandscapeOAuthProvider

logger = logging.getLogger(__name__)

_oauth_provider = LandscapeOAuthProvider()

mcp = FastMCP(
    "landscape",
    auth_server_provider=_oauth_provider,
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(settings.mcp_issuer_url),
        resource_server_url=AnyHttpUrl(settings.mcp_issuer_url),
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["agent", "graph_query"],
            default_scopes=["agent"],
        ),
        revocation_options=RevocationOptions(enabled=True),
    ),
)
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


def _render_path(item) -> str:
    """Mirror the CLI's (seed) -[REL]-> name [Type] rendering."""
    if not item.path_edge_types:
        return f"(seed) {item.name} [{item.type}]"
    negated = item.path_edge_negated or [False] * len(item.path_edge_types)
    subtypes = item.path_edge_subtypes or [None] * len(item.path_edge_types)
    parts: list[str] = []
    for edge, neg, sub in zip(item.path_edge_types, negated, subtypes, strict=False):
        label = edge if not sub else f"{edge}/{sub}"
        if neg:
            label = f"NOT {label}"
        parts.append(f"-[{label}]->")
    parts.append(f"{item.name} [{item.type}]")
    return "(seed) " + " ".join(parts)


def _value_suffix(fact: dict) -> str:
    """Render the populated value/quantity field on a memory fact, if any."""
    if (vt := fact.get("value_text")) not in (None, ""):
        return f' = "{vt}"'
    if (vn := fact.get("value_number")) is not None:
        unit = fact.get("value_unit") or ""
        return f" = {vn}{(' ' + unit) if unit else ''}"
    if (vtime := fact.get("value_time")) not in (None, ""):
        return f" = {vtime}"
    if (qv := fact.get("quantity_value")) is not None:
        unit = fact.get("quantity_unit") or ""
        return f" = {qv}{(' ' + unit) if unit else ''}"
    return ""


def _render_fact(fact: dict) -> str:
    family = fact.get("family") or "RELATES_TO"
    subtype = fact.get("subtype")
    rel = family if not subtype else f"{family}/{subtype}"
    if fact.get("negated"):
        rel = f"NOT {rel}"
    subj_name = fact.get("subject_name") or "?"
    subj_type = fact.get("subject_type") or "?"
    obj_name = fact.get("object_name") or "?"
    obj_type = fact.get("object_type") or "?"
    return (
        f"{subj_name} [{subj_type}] -[{rel}]-> {obj_name} [{obj_type}]"
        + _value_suffix(fact)
    )


def _build_compact_output(result, *, chunk_preview_chars: int = 200) -> dict:
    """Compact, agent-friendly search payload.

    - Dedupes memory_facts across ranked entities (each fact emitted once).
    - Each result references facts by id; full fact text lives in `facts`.
    - Drops supporting_assertions (audit trail; not needed to answer queries).
    - Strips null value/quantity columns; renders populated ones inline.
    """
    facts_by_id: dict[str, dict] = {}
    fact_order: list[str] = []
    # Collapse memory_facts whose rendered text is identical (multiple
    # MemoryFact nodes can express the same assertion at different
    # confidences). First-seen id is canonical; later ids alias to it.
    canonical_by_text: dict[str, str] = {}
    alias_to_canonical: dict[str, str] = {}

    def _absorb(fact: dict) -> str | None:
        fid = fact.get("memory_fact_id")
        if not fid:
            return None
        fid = str(fid)
        if fid in alias_to_canonical:
            return alias_to_canonical[fid]
        if fid in facts_by_id:
            return fid
        text = _render_fact(fact)
        existing = canonical_by_text.get(text)
        if existing is not None:
            alias_to_canonical[fid] = existing
            prev = facts_by_id[existing]
            prev["confidence_agg"] = max(
                float(prev.get("confidence_agg") or 0.0),
                float(fact.get("confidence_agg") or 0.0),
            )
            prev["support_count"] = int(prev.get("support_count") or 1) + int(
                fact.get("support_count") or 1
            )
            return existing
        canonical_by_text[text] = fid
        facts_by_id[fid] = dict(fact)
        fact_order.append(fid)
        return fid

    compact_results = []
    for rank, r in enumerate(result.results, start=1):
        fact_ids: list[str] = []
        seen_local: set[str] = set()
        for fact in r.memory_facts:
            fid = _absorb(fact)
            if fid and fid not in seen_local:
                seen_local.add(fid)
                fact_ids.append(fid)
        compact_results.append(
            {
                "rank": rank,
                "name": r.name,
                "type": r.type,
                "score": round(r.score, 4),
                "distance": r.distance,
                "path": _render_path(r),
                "fact_ids": fact_ids,
            }
        )

    facts_out = []
    for fid in fact_order:
        fact = facts_by_id[fid]
        entry: dict = {
            "id": fid,
            "text": _render_fact(fact),
            "confidence": round(float(fact.get("confidence_agg") or 0.0), 3),
        }
        if fact.get("negated"):
            entry["negated"] = True
        if (sc := fact.get("support_count")) and sc != 1:
            entry["support"] = sc
        facts_out.append(entry)

    chunks_out = []
    for c in result.chunks:
        preview = c.text[:chunk_preview_chars].replace("\n", " ").strip()
        chunks_out.append(
            {
                "source": c.source_doc,
                "preview": preview,
                "score": round(c.score, 4),
            }
        )

    return {
        "query": result.query,
        "results": compact_results,
        "facts": facts_out,
        "chunks": chunks_out,
        "touched_entity_count": len(result.touched_entity_ids),
    }


@mcp.tool()
async def search(
    query: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    session_id: str | None = None,
    since_hours: int | None = None,
    include_historical: bool = False,
    debug: bool = False,
    verbose: bool = False,
) -> str:
    """Hybrid retrieval over the Landscape knowledge graph.

    Returns a compact payload by default: ranked entities with rendered path
    strings reference a deduplicated `facts` table. Set verbose=True for the
    legacy payload with raw memory_facts and supporting_assertions.
    """
    require_current_scope("agent")
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
        debug=debug,
        include_historical=include_historical,
    )
    if verbose:
        output = {
            "results": [
                {
                    "name": r.name,
                    "type": r.type,
                    "score": round(r.score, 6),
                    "path_memory_fact_ids": r.path_memory_fact_ids,
                    "path_edge_types": r.path_edge_types,
                    "path_edge_subtypes": r.path_edge_subtypes,
                    "path_edge_quantities": r.path_edge_quantities,
                    "memory_facts": r.memory_facts,
                    "supporting_assertions": r.supporting_assertions,
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
    else:
        output = _build_compact_output(result)
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
    require_current_scope("agent")
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
    require_current_scope("agent")
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
    require_current_scope("agent")
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
    require_current_scope("agent")
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
            "assertion_id": result.assertion_id,
            "memory_fact_id": result.memory_fact_id,
            "outcome": result.outcome,
            "subject_id": result.subject_id,
            "object_id": result.object_id,
        }
    )


@mcp.tool()
async def graph_query(cypher: str, params: dict | None = None) -> str:
    """Execute a read-only Cypher query against the Neo4j knowledge graph."""
    require_current_scope("graph_query")
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
    require_current_scope("agent")
    from landscape.writeback import status_summary

    summary = await status_summary()
    return json.dumps(asdict(summary))


@mcp.tool()
async def conversation_history(session_id: str, limit: int = 10) -> str:
    """Return the turns of a conversation in chronological order."""
    require_current_scope("agent")
    from landscape.storage import neo4j_store

    detail = await neo4j_store.get_conversation_detail(session_id, turn_limit=limit)
    return json.dumps(detail, default=str)
