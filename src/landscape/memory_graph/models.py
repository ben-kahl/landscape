from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssertionPayload:
    source_kind: str
    source_id: str
    raw_subject_text: str
    raw_relation_text: str
    raw_object_text: str
    confidence: float
    subtype: str | None = None
    family_candidate: str | None = None
    quantity_value: float | str | None = None
    quantity_unit: str | None = None
    quantity_kind: str | None = None
    time_scope: str | None = None
    chunk_refs: list[tuple[str, int | None, int | None]] = field(default_factory=list)
