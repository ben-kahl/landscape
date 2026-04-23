from __future__ import annotations

import hashlib
from dataclasses import dataclass

from landscape.pipeline import IngestResult, ingest

_TOOL_NOISE_ROLES = frozenset({"tool", "function"})


@dataclass(frozen=True)
class ConversationTurn:
    session_id: str
    turn_id: str
    role: str
    text: str


@dataclass(frozen=True)
class ConversationIngestResult:
    title: str
    skipped: bool
    reason: str | None
    ingest_result: IngestResult | None

    def _require_ingest_result(self) -> IngestResult:
        if self.ingest_result is None:
            raise AttributeError("skipped conversation ingests do not have a pipeline result")
        return self.ingest_result

    @property
    def doc_id(self) -> str:
        return self._require_ingest_result().doc_id

    @property
    def already_existed(self) -> bool:
        return self._require_ingest_result().already_existed

    @property
    def entities_created(self) -> int:
        return self._require_ingest_result().entities_created

    @property
    def entities_reinforced(self) -> int:
        return self._require_ingest_result().entities_reinforced

    @property
    def relations_created(self) -> int:
        return self._require_ingest_result().relations_created

    @property
    def relations_reinforced(self) -> int:
        return self._require_ingest_result().relations_reinforced

    @property
    def relations_superseded(self) -> int:
        return self._require_ingest_result().relations_superseded

    @property
    def chunks_created(self) -> int:
        return self._require_ingest_result().chunks_created


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


def should_auto_ingest_turn(
    turn: ConversationTurn,
    *,
    seen_fingerprints: set[str] | None = None,
) -> bool:
    normalized = normalize_turn_text(turn.text)
    if not turn.session_id or not turn.turn_id or not normalized:
        return False
    if normalize_turn_role(turn.role) in _TOOL_NOISE_ROLES:
        return False
    if seen_fingerprints is None:
        return True
    normalized_turn = ConversationTurn(turn.session_id, turn.turn_id, turn.role, normalized)
    if turn_fingerprint(normalized_turn) in seen_fingerprints:
        return False
    return True


async def ingest_conversation_turn(
    turn: ConversationTurn,
    *,
    seen_fingerprints: set[str] | None = None,
) -> ConversationIngestResult:
    title = build_conversation_title(turn)
    normalized = normalize_turn_text(turn.text)
    if not turn.session_id or not turn.turn_id or not normalized:
        return ConversationIngestResult(
            title=title,
            skipped=True,
            reason="ineligible",
            ingest_result=None,
        )
    if normalize_turn_role(turn.role) in _TOOL_NOISE_ROLES:
        return ConversationIngestResult(
            title=title,
            skipped=True,
            reason="tool_noise",
            ingest_result=None,
        )

    normalized_turn = ConversationTurn(turn.session_id, turn.turn_id, turn.role, normalized)
    fingerprint = turn_fingerprint(normalized_turn)
    if seen_fingerprints is not None and fingerprint in seen_fingerprints:
        return ConversationIngestResult(
            title=title,
            skipped=True,
            reason="duplicate",
            ingest_result=None,
        )

    result = await ingest(
        turn.text,
        title,
        session_id=turn.session_id,
        turn_id=turn.turn_id,
    )
    if seen_fingerprints is not None:
        seen_fingerprints.add(fingerprint)

    return ConversationIngestResult(
        title=title,
        skipped=False,
        reason=None,
        ingest_result=result,
    )
