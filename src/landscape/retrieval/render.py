"""Shared rendering for retrieval results.

The CLI, FastAPI, and MCP surfaces all serve the same `RetrievalResult` to
different audiences. This module centralizes the fact / path / value
rendering so a query produces the same logical answer everywhere, while
each surface still chooses its own packaging (text vs JSON, compact vs
verbose).

The helpers here only read fields off the data classes and plain dict
representations of memory facts — no DB or LLM calls. Pure functions, safe
to unit-test."""

from __future__ import annotations

from typing import Any

from landscape.retrieval.query import EntityPath


def value_suffix(fact: dict) -> str:
    """Render the populated value/quantity field on a memory fact, if any.

    Returns the empty string when no value column is set. The leading
    ` = ` separator is included so callers can concatenate directly.
    Priority: value_text > value_number(+unit) > value_time >
    quantity_value(+unit)."""
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


def render_fact(fact: dict) -> str:
    """One-line rendering of a memory fact.

    Binary: `Subject [Type] -[FAMILY/subtype]-> Object [Type] = value`
    Unary (no object entity): `Subject [Type] -[FAMILY]-> value`
    Negated: `... -[NOT FAMILY]-> ...`"""
    family = fact.get("family") or "RELATES_TO"
    subtype = fact.get("subtype")
    rel = family if not subtype else f"{family}/{subtype}"
    if fact.get("negated"):
        rel = f"NOT {rel}"
    subj_name = fact.get("subject_name") or "?"
    subj_type = fact.get("subject_type") or "?"
    value = value_suffix(fact)
    if fact.get("object_entity_id") is None and fact.get("object_name") is None:
        bare = value.removeprefix(" = ")
        if bare:
            return f"{subj_name} [{subj_type}] -[{rel}]-> {bare}"
        return f"{subj_name} [{subj_type}] -[{rel}]-> (no object)"
    obj_name = fact.get("object_name") or "?"
    obj_type = fact.get("object_type") or "?"
    return f"{subj_name} [{subj_type}] -[{rel}]-> {obj_name} [{obj_type}]{value}"


def render_path(path: EntityPath) -> str:
    """Render an EntityPath as a human-readable string.

    Examples:
        - Empty path: "(unknown)"
        - Direct hit (no edges): "(Qdrant) [direct match]"
        - One hop: "(Eric) -[DISCUSSION]-> Netflix [TECHNOLOGY]"
        - Multi-hop:
          "(Project Aurora) -[USES]-> PostgreSQL [TECHNOLOGY] -[APPROVED_BY]-> Maya Chen [PERSON]"
        - Negated edge: "(Alice) -[NOT WORKS_FOR]-> Acme Corp [ORGANIZATION]"
    """
    if not path.nodes:
        return "(unknown)"
    if not path.edges:
        return f"({path.nodes[0].name}) [direct match]"

    parts = [f"({path.nodes[0].name})"]
    for edge, node in zip(path.edges, path.nodes[1:], strict=False):
        # subtype intentionally omitted; available in structured path
        label = f"NOT {edge.type}" if edge.negated else edge.type
        parts.append(f"-[{label}]->")
        parts.append(f"{node.name} [{node.type}]")
    return " ".join(parts)


def build_compact_payload(result: Any, *, chunk_preview_chars: int = 200) -> dict:
    """Token-efficient JSON payload for agent consumption.

    - Ranked entities reference deduplicated facts by id.
    - Facts with identical rendered text collapse; support_count accumulates
      and the highest confidence wins.
    - Drops supporting_assertions (audit trail; not needed to answer).
    - Strips null value/quantity columns; renders populated ones inline.
    """
    facts_by_id: dict[str, dict] = {}
    fact_order: list[str] = []
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
        text = render_fact(fact)
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
                "path": render_path(r.path),
                "fact_ids": fact_ids,
            }
        )

    facts_out = []
    for fid in fact_order:
        fact = facts_by_id[fid]
        entry: dict = {
            "id": fid,
            "text": render_fact(fact),
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
