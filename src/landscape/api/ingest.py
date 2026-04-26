from fastapi import APIRouter
from pydantic import BaseModel

from landscape import pipeline
from landscape.security import AgentPrincipal

router = APIRouter()


class IngestRequest(BaseModel):
    text: str
    title: str
    source_type: str = "text"
    session_id: str | None = None
    turn_id: str | None = None
    debug: bool = False


class IngestResponse(BaseModel):
    doc_id: str
    already_existed: bool
    entities_created: int
    entities_reinforced: int
    relations_created: int
    relations_reinforced: int
    relations_superseded: int
    chunks_created: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest, auth: AgentPrincipal) -> IngestResponse:
    del auth  # principal resolved for authz; not needed in handler body
    result = await pipeline.ingest(
        req.text,
        req.title,
        req.source_type,
        session_id=req.session_id,
        turn_id=req.turn_id,
        debug=req.debug,
    )
    return IngestResponse(
        doc_id=result.doc_id,
        already_existed=result.already_existed,
        entities_created=result.entities_created,
        entities_reinforced=result.entities_reinforced,
        relations_created=result.relations_created,
        relations_reinforced=result.relations_reinforced,
        relations_superseded=result.relations_superseded,
        chunks_created=result.chunks_created,
    )
