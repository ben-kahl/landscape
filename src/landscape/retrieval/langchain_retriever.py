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

from landscape.retrieval.query import RetrievedEntity, retrieve
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
            weights=self.weights,
            reinforce=self.reinforce,
            session_id=self.session_id,
            since=self.since,
        )
        return [_entity_to_document(e) for e in result.results]

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


def _entity_to_document(entity: RetrievedEntity) -> Document:
    path = " → ".join(entity.path_edge_types) if entity.path_edge_types else ""
    header = f"{entity.name} ({entity.type})"
    if entity.distance > 0 and path:
        content = f"{header} [{entity.distance} hops via {path}]"
    elif entity.distance > 0:
        content = f"{header} [{entity.distance} hops]"
    else:
        content = f"{header} [seed]"

    metadata: dict[str, Any] = {
        "neo4j_id": entity.neo4j_id,
        "name": entity.name,
        "type": entity.type,
        "distance": entity.distance,
        "score": entity.score,
        "vector_sim": entity.vector_sim,
        "reinforcement": entity.reinforcement,
        "edge_confidence": entity.edge_confidence,
        "path_edge_ids": entity.path_edge_ids,
        "path_edge_types": entity.path_edge_types,
    }
    return Document(page_content=content, metadata=metadata)
