from datetime import UTC, datetime, timedelta

from fastapi import APIRouter
from pydantic import BaseModel, Field

from landscape.retrieval import query as query_module

router = APIRouter()


class QueryRequest(BaseModel):
    text: str
    hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    chunk_limit: int = Field(default=3, ge=0, le=20)
    reinforce: bool = True
    session_id: str | None = None
    since_hours: int | None = Field(default=None, ge=1)


class QueryResultItem(BaseModel):
    neo4j_id: str
    name: str
    type: str
    distance: int
    vector_sim: float
    reinforcement: float
    edge_confidence: float
    score: float
    path_edge_types: list[str]


class QueryChunkItem(BaseModel):
    chunk_neo4j_id: str
    text: str
    doc_id: str
    source_doc: str
    position: int
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResultItem]
    chunks: list[QueryChunkItem]
    touched_entity_count: int
    touched_edge_count: int


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    since = (
        datetime.now(UTC) - timedelta(hours=req.since_hours)
        if req.since_hours
        else None
    )
    result = await query_module.retrieve(
        query_text=req.text,
        hops=req.hops,
        limit=req.limit,
        chunk_limit=req.chunk_limit,
        reinforce=req.reinforce,
        session_id=req.session_id,
        since=since,
    )
    return QueryResponse(
        query=result.query,
        results=[
            QueryResultItem(
                neo4j_id=r.neo4j_id,
                name=r.name,
                type=r.type,
                distance=r.distance,
                vector_sim=r.vector_sim,
                reinforcement=r.reinforcement,
                edge_confidence=r.edge_confidence,
                score=r.score,
                path_edge_types=r.path_edge_types,
            )
            for r in result.results
        ],
        chunks=[
            QueryChunkItem(
                chunk_neo4j_id=c.chunk_neo4j_id,
                text=c.text,
                doc_id=c.doc_id,
                source_doc=c.source_doc,
                position=c.position,
                score=c.score,
            )
            for c in result.chunks
        ],
        touched_entity_count=len(result.touched_entity_ids),
        touched_edge_count=len(result.touched_edge_ids),
    )
