from fastapi import APIRouter
from pydantic import BaseModel, Field

from landscape.retrieval import query as query_module

router = APIRouter()


class QueryRequest(BaseModel):
    text: str
    hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    reinforce: bool = True


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


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResultItem]
    touched_entity_count: int
    touched_edge_count: int


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    result = await query_module.retrieve(
        query_text=req.text,
        hops=req.hops,
        limit=req.limit,
        reinforce=req.reinforce,
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
        touched_entity_count=len(result.touched_entity_ids),
        touched_edge_count=len(result.touched_edge_ids),
    )
