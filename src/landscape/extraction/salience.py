from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from landscape.conversation_ingestion import ConversationTurn, should_auto_ingest_turn
from landscape.extraction import llm_client
from landscape.observability.weave_tracing import traced

SalienceCategory = Literal[
    "identity",
    "preference",
    "decision",
    "fact",
    "relationship",
    "state_change",
]

SALIENCE_CATEGORIES = (
    "identity",
    "preference",
    "decision",
    "fact",
    "relationship",
    "state_change",
)
_SALIENCE_CATEGORY_SET = frozenset(SALIENCE_CATEGORIES)

_SYSTEM_PROMPT = (
    "You decide which conversation turns are worth remembering as long-term memory.\n"
    "You are given NUMBERED turns. Return ONLY the turns that state durable, "
    "future-relevant information: user identity/role, stable preferences, decisions "
    "and commitments, stable facts about people/projects/tools, relationships, or "
    "state changes/corrections.\n"
    "DISCARD: greetings, acknowledgements, small talk, transient task mechanics, "
    "tool chatter, and code or log pastes. Also discard any turn whose main purpose "
    "is to ask a question (including clarifying questions, whoever asks them), purely "
    "hypothetical or speculative ideas ('what if', 'maybe', 'someday', 'we could'), "
    "and short throwaway replies to a clarifying question ('staging', 'the second "
    "one').\n"
    "For each kept turn return its turn_index (the number shown) and a category from: "
    f"{', '.join(SALIENCE_CATEGORIES)}. If nothing is worth remembering, return an "
    "empty list."
)


class SalienceSelectionItem(BaseModel):
    turn_index: int
    category: SalienceCategory


class SalienceSelection(BaseModel):
    selected: list[SalienceSelectionItem]


@dataclass(frozen=True)
class SalientItem:
    turn_id: str
    text: str
    category: str
    turn_index: int | None = None


def _render_turns(turns: list[ConversationTurn]) -> str:
    lines: list[str] = []
    for i, turn in enumerate(turns, start=1):
        lines.append(f"[{i}] ({turn.role}) {turn.text.strip()}")
    return "\n".join(lines)


def _call_salience_model(prompt: str) -> SalienceSelection:
    return llm_client.complete_structured(
        prompt, model_cls=SalienceSelection, schema_name="SalienceSelection"
    )


@traced(name="salience.select")
def select_salient(turns: list[ConversationTurn]) -> list[SalientItem]:
    """Return the memory-worthy turns with provenance intact."""
    eligible = [turn for turn in turns if should_auto_ingest_turn(turn)]
    if not eligible:
        return []

    prompt = f"{_SYSTEM_PROMPT}\n\n{_render_turns(eligible)}"
    selection = _call_salience_model(prompt)

    selected_by_index: dict[int, SalienceCategory] = {}
    seen: set[int] = set()
    for selected in selection.selected:
        idx = selected.turn_index
        if (
            idx < 1
            or idx > len(eligible)
            or idx in seen
            or selected.category not in _SALIENCE_CATEGORY_SET
        ):
            continue
        seen.add(idx)
        selected_by_index[idx] = selected.category

    items: list[SalientItem] = []
    for idx in sorted(selected_by_index):
        turn = eligible[idx - 1]
        items.append(
            SalientItem(
                turn_id=turn.turn_id,
                text=turn.text,
                category=selected_by_index[idx],
                turn_index=idx,
            )
        )
    return items
