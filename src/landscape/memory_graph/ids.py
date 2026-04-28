import hashlib
import json

from landscape.memory_graph.families import FamilyConfig


def _stable_digest(parts: list[object]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def alias_id(entity_id: str, alias_text: str) -> str:
    return f"alias:{entity_id}:{_stable_digest([alias_text.strip().lower()])}"


def assertion_id(
    *,
    source_kind: str,
    source_id: str,
    raw_subject_text: str,
    raw_relation_text: str,
    raw_object_text: str,
    subtype: str | None,
    qualifier_payload: dict[str, object],
    chunk_refs: list[tuple[str, int | None, int | None]],
) -> str:
    digest = _stable_digest(
        [
            source_kind,
            source_id,
            raw_subject_text,
            raw_relation_text,
            raw_object_text,
            subtype,
            qualifier_payload,
            chunk_refs,
        ]
    )
    return f"assertion:{source_kind}:{digest}"


def fact_key(family: FamilyConfig, subject_entity_id: str, object_entity_id: str | None, subtype: str | None) -> str:
    parts = [family.family, subject_entity_id]
    if object_entity_id is not None:
        parts.append(object_entity_id)
    if family.identity_uses_subtype:
        parts.append(subtype or "")
    return ":".join(parts)


def slot_key(family: FamilyConfig, subject_entity_id: str, object_entity_id: str | None, subtype: str | None) -> str:
    if family.single_current:
        return f"{family.family}:{subject_entity_id}"
    return fact_key(family, subject_entity_id, object_entity_id, subtype)
