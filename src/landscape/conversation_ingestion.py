from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    session_id: str
    turn_id: str
    role: str
    text: str


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
