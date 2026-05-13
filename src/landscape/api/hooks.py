from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from landscape.conversation_ingestion import ConversationTurn, should_auto_ingest_turn
from landscape.hooks.adapters import normalize_role
from landscape.security import AgentPrincipal

router = APIRouter(prefix="/hooks")


class ConversationTurnHookRequest(BaseModel):
    client: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    text: str
    debug: bool = False


class ConversationTurnHookResponse(BaseModel):
    accepted: bool
    scheduled: bool


@router.post("/conversation-turn", response_model=ConversationTurnHookResponse)
async def capture_conversation_turn_hook(
    req: ConversationTurnHookRequest,
    auth: AgentPrincipal,
) -> ConversationTurnHookResponse:
    del auth
    from landscape import mcp_app

    role = normalize_role(req.role)
    text = req.text.strip()
    turn = ConversationTurn(
        session_id=req.session_id,
        turn_id=req.turn_id,
        role=role,
        text=text,
    )
    if not should_auto_ingest_turn(turn, seen_fingerprints=set()):
        return ConversationTurnHookResponse(accepted=False, scheduled=False)

    mcp_app._schedule_auto_ingestion(
        text,
        req.session_id,
        req.turn_id,
        role=role,
        debug=req.debug,
    )
    return ConversationTurnHookResponse(accepted=True, scheduled=True)
