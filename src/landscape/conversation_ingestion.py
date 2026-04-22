from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    text: str


def normalize_turn_text(text: str) -> str:
    return " ".join(text.split())


def build_conversation_title(turns: list[ConversationTurn]) -> str:
    for turn in turns:
        normalized = normalize_turn_text(turn.text)
        if normalized:
            words = normalized.split(" ")
            return " ".join(words[:8])
    return "Conversation"


def turn_fingerprint(turns: list[ConversationTurn]) -> str:
    parts = []
    for turn in turns:
        normalized = normalize_turn_text(turn.text)
        parts.append(f"{turn.role.strip().lower()}:{normalized}")
    payload = "\n".join(parts).encode("utf-8")
    return sha256(payload).hexdigest()


def should_auto_ingest_turn(turn: ConversationTurn) -> bool:
    return bool(normalize_turn_text(turn.text))
