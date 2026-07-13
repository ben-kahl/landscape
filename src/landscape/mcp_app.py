"""Landscape MCP app — transport-agnostic tool registration for MCP clients.

This module owns the FastMCP app and all tool definitions. Runtime startup
belongs elsewhere so the same app can be mounted inside FastAPI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl

from landscape.config import settings
from landscape.retrieval.render import (
    build_compact_payload as _build_compact_output,
)
from landscape.retrieval.render import (
    render_fact as _render_fact,
)
from landscape.retrieval.render import (
    render_path as _render_path,
)
from landscape.retrieval.render import (
    value_suffix as _value_suffix,
)
from landscape.security import require_current_scope
from landscape.storage.oauth_provider import LandscapeOAuthProvider

# Re-export for tests and downstream callers that imported these names from
# mcp_app before the helpers moved into landscape.retrieval.render.
__all__ = [
    "_render_path",
    "_render_fact",
    "_value_suffix",
    "_build_compact_output",
]

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




@mcp.tool()
async def search(
    query: str,
    hops: int = 2,
    limit: int = 10,
    chunk_limit: int = 3,
    session_id: str | None = None,
    since_hours: int | None = None,
    include_historical: bool = False,
    as_of: str | None = None,
    debug: bool = False,
    verbose: bool = False,
) -> str:
    """Hybrid retrieval over the Landscape knowledge graph.

    Returns a compact payload by default: ranked entities with rendered path
    strings reference a deduplicated `facts` table. Set verbose=True for the
    legacy payload with raw memory_facts and supporting_assertions.

    Chunks in the payload carry doc_id/chunk_id — pass one to get_document
    to fetch the full source text instead of querying the graph for it.
    """
    require_current_scope("agent")
    from landscape.retrieval.query import retrieve

    if as_of is not None:
        try:
            _dt = datetime.fromisoformat(as_of)
        except ValueError as exc:
            raise ValueError(f"as_of must be ISO-8601; got {as_of!r}") from exc
        if _dt.tzinfo is None:
            _dt = _dt.replace(tzinfo=UTC)
        as_of = _dt.astimezone(UTC).isoformat()

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
        as_of=as_of,
    )
    if verbose:
        output = {
            "results": [
                {
                    "name": r.name,
                    "type": r.type,
                    "score": round(r.score, 6),
                    "retrieval_mode": r.retrieval_mode,
                    "path_str": _render_path(r.path),
                    "path": {
                        "nodes": [
                            {"name": n.name, "type": n.type} for n in r.path.nodes
                        ],
                        "edges": [
                            {
                                "type": e.type,
                                "negated": e.negated,
                                "subtype": e.subtype,
                                "memory_fact_id": e.memory_fact_id,
                                "quantities": e.quantities,
                            }
                            for e in r.path.edges
                        ],
                    },
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
    output = {
        "doc_id": result.doc_id,
        "entities_created": result.entities_created,
        "relations_created": result.relations_created,
        "relations_superseded": result.relations_superseded,
        "already_existed": result.already_existed,
    }
    return json.dumps(output)



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
async def lookup_entity(
    name: str,
    include_historical: bool = False,
    as_of: str | None = None,
) -> str:
    """Look up a single canonical entity by name and return everything Landscape
    knows about it: aliases, live facts (or as-of-time facts when ``as_of`` is
    provided), and entity metadata.

    Returns a compact JSON payload. When the name resolves to a canonical
    entity, the payload includes ``entity`` (id, name, type, aliases) and
    ``facts`` (a list of rendered fact strings such as
    ``"Benjamin L Kahl [Person] -[WORKS_FOR]-> Hackingtons Inc [Organization]"``).
    When no canonical match is found, the payload is
    ``{"match": "miss", "suggestions": [...]}`` listing up to 3 of the
    closest substring-matched entities so the caller can re-issue with a
    more precise name.

    Use ``lookup_entity`` when you have an entity in mind and want a complete
    dump of its current state, as opposed to ``search`` which takes a question
    and returns ranked candidates. ``include_historical=true`` includes
    superseded facts; ``as_of`` (ISO-8601) shifts the system-time filter to
    a past moment.
    """
    require_current_scope("agent")
    from landscape.storage.neo4j_entities import find_canonical_entities_by_name
    from landscape.storage.neo4j_memory import get_current_fact_details_for_entities

    matches = await find_canonical_entities_by_name(name, limit=5)
    if not matches:
        return json.dumps({"match": "miss", "suggestions": []})

    best = matches[0]
    if best["match"] not in {"exact_name", "exact_alias"}:
        return json.dumps(
            {
                "match": "miss",
                "suggestions": [
                    {
                        "entity_id": m["entity_id"],
                        "name": m["name"],
                        "type": m["type"],
                        "match": m["match"],
                    }
                    for m in matches[:3]
                ],
            }
        )

    memory_facts, _ = await get_current_fact_details_for_entities(
        [best["entity_id"]],
        as_of=as_of,
    )
    if not include_historical and as_of is None:
        memory_facts = [f for f in memory_facts if f.get("current")]

    facts_rendered = [
        {
            "id": fact.get("memory_fact_id"),
            "text": _render_fact(fact),
            "confidence": fact.get("confidence_agg"),
            "current": bool(fact.get("current")),
            "negated": bool(fact.get("negated")),
        }
        for fact in memory_facts
    ]

    return json.dumps(
        {
            "match": best["match"],
            "entity": {
                "entity_id": best["entity_id"],
                "name": best["name"],
                "type": best["type"],
                "aliases": best["aliases"],
            },
            "facts": facts_rendered,
            "other_matches": [
                {
                    "entity_id": m["entity_id"],
                    "name": m["name"],
                    "type": m["type"],
                    "match": m["match"],
                }
                for m in matches[1:]
            ],
        }
    )


@mcp.tool()
async def get_document(
    doc_id: str | None = None,
    chunk_id: str | None = None,
    memory_fact_id: str | None = None,
) -> str:
    """Fetch the full source text behind a search result.

    Two-stage pattern: `search` returns compact previews whose chunks carry
    `doc_id`/`chunk_id`; pass one of those here to recover the complete
    document (all chunks, ordered). `memory_fact_id` traverses provenance to
    the source document(s) of a fact (capped at 3).

    Provide exactly one of doc_id, chunk_id, memory_fact_id.
    """
    require_current_scope("agent")
    from landscape.storage import neo4j_store

    provided = [v for v in (doc_id, chunk_id, memory_fact_id) if v]
    if len(provided) != 1:
        raise ValueError(
            "Provide exactly one of doc_id, chunk_id, memory_fact_id"
        )

    if doc_id is not None:
        doc_ids = [doc_id]
    elif chunk_id is not None:
        found = await neo4j_store.find_doc_id_for_chunk(chunk_id)
        doc_ids = [found] if found else []
    else:
        doc_ids = await neo4j_store.find_doc_ids_for_memory_fact(memory_fact_id)

    documents = []
    for did in doc_ids:
        doc = await neo4j_store.get_document_with_chunks(did)
        if doc is not None:
            doc["full_text"] = "\n".join(ch["text"] for ch in doc["chunks"])
            documents.append(doc)
    return json.dumps({"documents": documents})


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
