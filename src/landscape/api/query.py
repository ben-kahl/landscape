from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from landscape.retrieval import query as query_module
from landscape.security import AgentPrincipal

router = APIRouter()


def _normalize_as_of(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


class QueryRequest(BaseModel):
    text: str
    hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    chunk_limit: int = Field(default=3, ge=0, le=20)
    reinforce: bool = True
    session_id: str | None = None
    since_hours: int | None = Field(default=None, ge=1)
    debug: bool = False
    include_historical: bool = False
    as_of: datetime | None = None


class PathNodeModel(BaseModel):
    name: str
    type: str


class PathEdgeModel(BaseModel):
    type: str
    negated: bool = False
    subtype: str | None = None
    memory_fact_id: str | None = None
    quantities: dict[str, object | None] = Field(default_factory=dict)


class EntityPathModel(BaseModel):
    nodes: list[PathNodeModel]
    edges: list[PathEdgeModel]


class QueryResultItem(BaseModel):
    entity_id: str
    name: str
    type: str
    distance: int
    vector_sim: float
    reinforcement: float
    edge_confidence: float
    score: float
    path: EntityPathModel
    retrieval_mode: str
    memory_facts: list[dict[str, object]] = Field(default_factory=list)
    supporting_assertions: list[dict[str, object]] = Field(default_factory=list)


class QueryChunkItem(BaseModel):
    chunk_id: str
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


@router.get("/documents")
async def get_document_endpoint(doc_id: str, auth: AgentPrincipal) -> dict:
    del auth  # principal resolved for authz; not needed in handler body
    from landscape.storage import neo4j_store

    doc = await neo4j_store.get_document_with_chunks(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=404, detail=f"No document with doc_id {doc_id!r}"
        )
    doc["full_text"] = "\n".join(ch["text"] for ch in doc["chunks"])
    return doc


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, auth: AgentPrincipal) -> QueryResponse:
    del auth  # principal resolved for authz; not needed in handler body
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
        debug=req.debug,
        include_historical=req.include_historical,
        as_of=_normalize_as_of(req.as_of),
    )
    return QueryResponse(
        query=result.query,
        results=[
            QueryResultItem(
                entity_id=r.entity_id,
                name=r.name,
                type=r.type,
                distance=r.distance,
                vector_sim=r.vector_sim,
                reinforcement=r.reinforcement,
                edge_confidence=r.edge_confidence,
                score=r.score,
                path=EntityPathModel(
                    nodes=[PathNodeModel(name=n.name, type=n.type) for n in r.path.nodes],
                    edges=[
                        PathEdgeModel(
                            type=e.type,
                            negated=e.negated,
                            subtype=e.subtype,
                            memory_fact_id=e.memory_fact_id,
                            quantities=e.quantities,
                        )
                        for e in r.path.edges
                    ],
                ),
                retrieval_mode=r.retrieval_mode,
                memory_facts=r.memory_facts,
                supporting_assertions=r.supporting_assertions,
            )
            for r in result.results
        ],
        chunks=[
            QueryChunkItem(
                chunk_id=c.chunk_id,
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
