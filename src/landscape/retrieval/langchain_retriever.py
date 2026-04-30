"""LangChain BaseRetriever wrapper around Landscape's hybrid retrieve().

This is a thin adapter so any LangChain agent can treat Landscape as a
drop-in retriever. The interesting logic — vector seeds, BFS expansion,
scoring, reinforcement — lives in retrieval/query.py. This file is the
integration surface, not the implementation.

Usage:
    from landscape.retrieval.langchain_retriever import LandscapeRetriever

    retriever = LandscapeRetriever(hops=2, limit=10)
    docs = await retriever.ainvoke("who approved Aurora's db choice?")
"""
from datetime import datetime
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from landscape.retrieval.query import RetrievedChunk, RetrievedEntity, retrieve
from landscape.retrieval.scoring import ScoringWeights


class LandscapeRetriever(BaseRetriever):
    """LangChain retriever backed by Landscape's hybrid engine.

    Each ranked entity becomes one Document. The Document's `page_content`
    is a human-readable line (name + type + path edges) so it's usable in
    a naive LLM prompt without any downstream formatting. All of the
    numeric signals (score, vector_sim, distance, reinforcement,
    edge_confidence, path edges) land in `metadata` for callers that
    want to re-rank or filter."""

    hops: int = 2
    limit: int = 10
    chunk_limit: int = 3
    reinforce: bool = True
    weights: ScoringWeights | None = Field(default=None)
    session_id: str | None = None
    since: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        result = await retrieve(
            query_text=query,
            hops=self.hops,
            limit=self.limit,
            chunk_limit=self.chunk_limit,
            weights=self.weights,
            reinforce=self.reinforce,
            session_id=self.session_id,
            since=self.since,
        )
        return [_entity_to_document(e) for e in result.results] + [
            _chunk_to_document(c) for c in result.chunks
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Landscape's retrieve() is async-only (Neo4j AsyncDriver, Qdrant
        # AsyncClient). A sync wrapper would either block the event loop or
        # spin up a fresh one — both are footguns inside an agent framework.
        # Callers should use `ainvoke` / `aget_relevant_documents`.
        raise NotImplementedError(
            "LandscapeRetriever is async-only. Use ainvoke() / "
            "aget_relevant_documents() from an async context."
        )


def _format_edge(rel_type: str, subtype: str | None) -> str:
    """Render a single edge as ``TYPE`` or ``TYPE[subtype]`` when present."""
    return f"{rel_type}[{subtype}]" if subtype else rel_type


def _format_quantity(quantity: dict[str, object | None]) -> str:
    value = quantity.get("quantity_value")
    unit = quantity.get("quantity_unit")
    kind = quantity.get("quantity_kind")
    scope = quantity.get("time_scope")
    parts = []
    if value is not None:
        label = str(kind) if kind else "quantity"
        rendered = f"{label}={value}"
        if unit:
            rendered = f"{rendered} {unit}"
        parts.append(rendered)
    if scope:
        parts.append(f"scope={scope}")
    return ", ".join(parts)


def _entity_to_document(entity: RetrievedEntity) -> Document:
    # Zip subtype list (may be empty/shorter on old edges) against types so
    # display stays stable when a path includes pre-subtype edges.
    subtypes = entity.path_edge_subtypes or [None] * len(entity.path_edge_types)
    if len(subtypes) < len(entity.path_edge_types):
        subtypes = list(subtypes) + [None] * (len(entity.path_edge_types) - len(subtypes))
    quantities = entity.path_edge_quantities or [{} for _ in entity.path_edge_types]
    if len(quantities) < len(entity.path_edge_types):
        quantities = list(quantities) + [
            {} for _ in range(len(entity.path_edge_types) - len(quantities))
        ]

    path_parts = []
    for rel_type, subtype, quantity in zip(entity.path_edge_types, subtypes, quantities):
        edge = _format_edge(rel_type, subtype)
        rendered_quantity = _format_quantity(quantity)
        if rendered_quantity:
            edge = f"{edge} {{{rendered_quantity}}}"
        path_parts.append(edge)
    path = " → ".join(path_parts) if path_parts else ""

    header = f"{entity.name} ({entity.type})"
    if entity.distance > 0 and path:
        content = f"{header} [{entity.distance} hops via {path}]"
    elif entity.distance > 0:
        content = f"{header} [{entity.distance} hops]"
    else:
        content = f"{header} [seed]"

    metadata: dict[str, Any] = {
        "kind": "entity",
        "entity_id": entity.entity_id,
        "name": entity.name,
        "type": entity.type,
        "distance": entity.distance,
        "score": entity.score,
        "vector_sim": entity.vector_sim,
        "reinforcement": entity.reinforcement,
        "edge_confidence": entity.edge_confidence,
        "path_edge_ids": entity.path_edge_ids,
        "path_edge_types": entity.path_edge_types,
        "path_edge_subtypes": entity.path_edge_subtypes,
        "path_edge_quantities": entity.path_edge_quantities,
    }
    return Document(page_content=content, metadata=metadata)


def _chunk_to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "kind": "chunk",
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_doc": chunk.source_doc,
            "position": chunk.position,
            "score": chunk.score,
        },
    )
