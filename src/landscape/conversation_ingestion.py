from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from landscape.observability import create_ingest_log_context
from landscape.pipeline import IngestResult, ingest
from landscape.storage import neo4j_store

if TYPE_CHECKING:
    from landscape.extraction.salience import SalientItem

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
    debug: bool = False,
) -> ConversationIngestResult:
    title = build_conversation_title(turn)
    normalized = normalize_turn_text(turn.text)
    log = create_ingest_log_context(
        title=title,
        source_type="text",
        session_id=turn.session_id or None,
        turn_id=turn.turn_id or None,
        debug=debug,
    )
    log.emit(
        "conversation_ingest_started",
        always=True,
        role=normalize_turn_role(turn.role),
        text_length=len(normalized),
    )
    if not turn.session_id or not turn.turn_id or not normalized:
        log.emit("conversation_ingest_skipped", always=True, reason="ineligible")
        return ConversationIngestResult(
            title=title,
            skipped=True,
            reason="ineligible",
            ingest_result=None,
        )
    if normalize_turn_role(turn.role) in _TOOL_NOISE_ROLES:
        log.emit("conversation_ingest_skipped", always=True, reason="tool_noise")
        return ConversationIngestResult(
            title=title,
            skipped=True,
            reason="tool_noise",
            ingest_result=None,
        )

    normalized_turn = ConversationTurn(turn.session_id, turn.turn_id, turn.role, normalized)
    fingerprint = turn_fingerprint(normalized_turn)
    if seen_fingerprints is not None and fingerprint in seen_fingerprints:
        log.emit("conversation_ingest_skipped", always=True, reason="duplicate")
        return ConversationIngestResult(
            title=title,
            skipped=True,
            reason="duplicate",
            ingest_result=None,
        )

    try:
        result = await ingest(
            turn.text,
            title,
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            debug=debug,
            log_context=log,
        )
    except Exception as exc:
        log.emit(
            "conversation_ingest_failed",
            level=40,
            always=True,
            reason="pipeline_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    if seen_fingerprints is not None:
        seen_fingerprints.add(fingerprint)

    return ConversationIngestResult(
        title=title,
        skipped=False,
        reason=None,
        ingest_result=result,
    )


async def ingest_conversation_window(
    session_id: str,
    salient_items: list[SalientItem],
    *,
    debug: bool = False,
) -> IngestResult | None:
    if not salient_items:
        return None

    ordered_items = sorted(
        enumerate(salient_items),
        key=lambda indexed_item: (
            getattr(indexed_item[1], "turn_index", None) is None,
            getattr(indexed_item[1], "turn_index", None)
            if getattr(indexed_item[1], "turn_index", None) is not None
            else indexed_item[1].turn_id,
            indexed_item[0],
        ),
    )
    salient_items = [item for _, item in ordered_items]

    first = salient_items[0]
    title = f"conversation:{session_id}:window:{first.turn_id}"
    log = create_ingest_log_context(
        title=title,
        source_type="text",
        session_id=session_id,
        turn_id=first.turn_id,
        debug=debug,
    )
    result = await ingest(
        "\n\n".join(item.text for item in salient_items),
        title,
        session_id=session_id,
        turn_id=first.turn_id,
        debug=debug,
        log_context=log,
    )

    for item in salient_items[1:]:
        turn_element_id, _ = await neo4j_store.merge_turn(session_id, item.turn_id)
        await neo4j_store.link_document_to_turn(result.doc_id, turn_element_id)

    return result
