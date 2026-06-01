from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HookTurn:
    client: str
    session_id: str
    turn_id: str
    role: str
    text: str


def normalize_role(role: str | None) -> str:
    normalized = (role or "").strip().lower()
    if normalized in {"assistant", "agent", "model"}:
        return "assistant"
    if normalized in {"human", "user"}:
        return "user"
    return normalized or "unknown"


def extract_turns(client: str, payload: dict[str, Any]) -> list[HookTurn]:
    """Extract conversation turns from known agent hook payload shapes."""
    client = (client or "unknown").strip().lower()
    direct = _extract_direct_turn(client, payload)
    if direct is not None:
        return [direct]

    # An explicit prompt (e.g. Claude Code UserPromptSubmit) is the signal for
    # that event and must win over the transcript: those payloads also carry a
    # transcript_path, and reading the transcript would capture the *previous*
    # assistant message instead of the user's just-typed prompt.
    prompt_turn = _extract_prompt_turn(client, payload)
    if prompt_turn is not None:
        return [prompt_turn]

    transcript_turn = _extract_transcript_turn(client, payload)
    if transcript_turn is not None:
        return [transcript_turn]

    event_turn = _extract_opencode_event_turn(client, payload)
    if event_turn is not None:
        return [event_turn]

    assistant_turn = _extract_assistant_turn(client, payload)
    if assistant_turn is not None:
        return [assistant_turn]

    return []


def _extract_direct_turn(client: str, payload: dict[str, Any]) -> HookTurn | None:
    text = _coerce_text(payload.get("text"))
    session_id = _coerce_str(payload.get("session_id") or payload.get("sessionID"))
    if not text or not session_id:
        return None
    role = normalize_role(_coerce_str(payload.get("role")))
    turn_id = _coerce_str(payload.get("turn_id") or payload.get("turnID") or payload.get("id"))
    return HookTurn(
        client=client,
        session_id=session_id,
        turn_id=turn_id or _synthetic_turn_id(client, payload, role, text),
        role=role,
        text=text,
    )


def _extract_prompt_turn(client: str, payload: dict[str, Any]) -> HookTurn | None:
    text = _coerce_text(
        payload.get("prompt")
        or payload.get("user_prompt")
        or payload.get("userPrompt")
        or payload.get("input")
    )
    session_id = _coerce_str(payload.get("session_id") or payload.get("sessionID"))
    if not text or not session_id:
        return None
    role = "user"
    return HookTurn(
        client=client,
        session_id=session_id,
        turn_id=_synthetic_turn_id(client, payload, role, text),
        role=role,
        text=text,
    )


def _extract_assistant_turn(client: str, payload: dict[str, Any]) -> HookTurn | None:
    text = _coerce_text(
        payload.get("response")
        or payload.get("assistant_response")
        or payload.get("assistantResponse")
        or payload.get("message")
    )
    session_id = _coerce_str(payload.get("session_id") or payload.get("sessionID"))
    if not text or not session_id:
        return None
    role = "assistant"
    turn_id = _coerce_str(payload.get("turn_id") or payload.get("message_id") or payload.get("id"))
    return HookTurn(
        client=client,
        session_id=session_id,
        turn_id=turn_id or _synthetic_turn_id(client, payload, role, text),
        role=role,
        text=text,
    )


def _extract_transcript_turn(client: str, payload: dict[str, Any]) -> HookTurn | None:
    transcript_path = _coerce_str(payload.get("transcript_path") or payload.get("transcriptPath"))
    session_id = _coerce_str(payload.get("session_id") or payload.get("sessionID"))
    if not transcript_path or not session_id:
        return None

    path = Path(transcript_path).expanduser()
    if not path.exists():
        return None

    latest: HookTurn | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = row.get("message") if isinstance(row.get("message"), dict) else row
        role = normalize_role(_coerce_str(message.get("role")))
        if role not in {"user", "assistant"}:
            continue
        text = _text_from_content(message.get("content"))
        if not text:
            continue
        turn_id = _coerce_str(message.get("id") or message.get("uuid") or row.get("uuid"))
        latest = HookTurn(
            client=client,
            session_id=session_id,
            turn_id=turn_id or _synthetic_turn_id(client, payload, role, text),
            role=role,
            text=text,
        )
    if latest is None or latest.role != "assistant":
        return None
    return latest


def _extract_opencode_event_turn(client: str, payload: dict[str, Any]) -> HookTurn | None:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    if event.get("type") not in {"message.updated", "message.part.updated"}:
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        properties = event.get("properties")
        message = properties.get("message") if isinstance(properties, dict) else None
    if not isinstance(message, dict):
        return None

    session_id = _coerce_str(
        message.get("session_id")
        or message.get("sessionID")
        or event.get("session_id")
        or event.get("sessionID")
    )
    role = normalize_role(_coerce_str(message.get("role") or event.get("role")))
    text = _text_from_content(message.get("parts") or message.get("content") or message.get("text"))
    if not session_id or role not in {"user", "assistant"} or not text:
        return None

    turn_id = _coerce_str(message.get("id") or message.get("messageID") or event.get("id"))
    return HookTurn(
        client=client,
        session_id=session_id,
        turn_id=turn_id or _synthetic_turn_id(client, payload, role, text),
        role=role,
        text=text,
    )


def _synthetic_turn_id(client: str, payload: dict[str, Any], role: str, text: str) -> str:
    event = _coerce_str(payload.get("hook_event_name") or payload.get("event_name") or "turn")
    session_id = _coerce_str(payload.get("session_id") or payload.get("sessionID"))
    raw = f"{client}|{session_id}|{event}|{role}|{text.strip()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{client}:{event}:{digest}"


def _coerce_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coerce_text(value: Any) -> str:
    return _text_from_content(value).strip()


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
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
            return text
    return ""
