from __future__ import annotations

import json
from pathlib import Path

from landscape.conversation_ingestion import ConversationTurn

_VALID_ROLES = frozenset({"user", "assistant"})
_SUPPORTED_CLIENTS = frozenset({"claude-code"})


def _coerce_str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_role(role: object) -> str:
    normalized = _coerce_str(role).lower()
    if normalized in {"assistant", "agent", "model"}:
        return "assistant"
    if normalized in {"human", "user"}:
        return "user"
    return normalized or "unknown"


def _text_from_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip())
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        if isinstance(text, str):
            return text.strip()
    return ""


def read_transcript(
    path: Path,
    *,
    session_id: str | None = None,
    client: str = "claude-code",
) -> list[ConversationTurn]:
    """Parse a transcript JSONL into ordered user/assistant ConversationTurns.

    Malformed lines, non-message rows, empty-text turns, and tool-only turns are
    skipped. ``session_id`` overrides the per-row ``sessionId`` when provided.
    """
    if client not in _SUPPORTED_CLIENTS:
        raise ValueError(f"unsupported transcript client: {client!r}")

    turns: list[ConversationTurn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        message = row.get("message") if isinstance(row.get("message"), dict) else row
        role = _normalize_role(message.get("role"))
        if role not in _VALID_ROLES:
            continue
        text = _text_from_content(message.get("content"))
        if not text:
            continue
        sid = session_id or _coerce_str(row.get("sessionId") or row.get("session_id"))
        turn_id = _coerce_str(
            message.get("id") or message.get("uuid") or row.get("uuid")
        ) or f"{role}-{len(turns)}"
        turns.append(
            ConversationTurn(session_id=sid, turn_id=turn_id, role=role, text=text)
        )
    return turns
