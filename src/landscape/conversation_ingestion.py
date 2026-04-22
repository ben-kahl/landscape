from __future__ import annotations

import hashlib
from dataclasses import dataclass

from landscape.pipeline import IngestResult, ingest


@dataclass(frozen=True)
class ConversationTurn:
    session_id: str
    turn_id: str
    role: str
    text: str


@dataclass(frozen=True)
class ConversationIngestResult:
    doc_id: str
    already_existed: bool
    entities_created: int
    entities_reinforced: int
    relations_created: int
    relations_reinforced: int
    relations_superseded: int
    chunks_created: int


def normalize_turn_text(text: str) -> str:
    return text.strip()


def normalize_turn_role(role: str) -> str:
    normalized = (role or "").strip().lower()
    return normalized or "unknown"


def build_conversation_title(turn: ConversationTurn) -> str:
    role = normalize_turn_role(turn.role)
    return f"conversation:{turn.session_id}:{turn.turn_id}:{role}"


def turn_fingerprint(turn: ConversationTurn) -> str:
    normalized = normalize_turn_text(turn.text)
    role = normalize_turn_role(turn.role)
    raw = f"{turn.session_id}|{turn.turn_id}|{role}|{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


def should_auto_ingest_turn(turn: ConversationTurn, *, seen_fingerprints: set[str]) -> bool:
    normalized = normalize_turn_text(turn.text)
    if not turn.session_id or not turn.turn_id or not normalized:
        return False
    normalized_turn = ConversationTurn(turn.session_id, turn.turn_id, turn.role, normalized)
    if turn_fingerprint(normalized_turn) in seen_fingerprints:
        return False
    return True


async def ingest_conversation_turn(
    turn: ConversationTurn,
    *,
    title: str | None = None,
    source_type: str = "text",
) -> ConversationIngestResult:
    result: IngestResult = await ingest(
        turn.text,
        title or build_conversation_title(turn),
        source_type=source_type,
        session_id=turn.session_id,
        turn_id=turn.turn_id,
    )
    return ConversationIngestResult(
        doc_id=result.doc_id,
        already_existed=result.already_existed,
        entities_created=result.entities_created,
        entities_reinforced=result.entities_reinforced,
        relations_created=result.relations_created,
        relations_reinforced=result.relations_reinforced,
        relations_superseded=result.relations_superseded,
        chunks_created=result.chunks_created,
    )
