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


def _encode_value_parts(
    *,
    value_text: str | None,
    value_number: float | None,
    value_unit: str | None,
    value_kind: str | None,
    value_time: str | None,
) -> list[str]:
    parts: list[str] = []
    if value_text is not None:
        parts.append(f"value_text={value_text}")
    if value_number is not None:
        parts.append(f"value_number={value_number}")
    if value_unit is not None:
        parts.append(f"value_unit={value_unit}")
    if value_kind is not None:
        parts.append(f"value_kind={value_kind}")
    if value_time is not None:
        parts.append(f"value_time={value_time}")
    return parts


def fact_key(
    family: FamilyConfig,
    subject_entity_id: str,
    object_entity_id: str | None,
    subtype: str | None,
    *,
    value_text: str | None = None,
    value_number: float | None = None,
    value_unit: str | None = None,
    value_kind: str | None = None,
    value_time: str | None = None,
) -> str:
    parts = [family.family, subject_entity_id]
    if object_entity_id is not None:
        parts.append(object_entity_id)
    if subtype is not None:
        parts.append(subtype)
    parts.extend(
        _encode_value_parts(
            value_text=value_text,
            value_number=value_number,
            value_unit=value_unit,
            value_kind=value_kind,
            value_time=value_time,
        )
    )
    return ":".join(parts)


def slot_key(
    family: FamilyConfig,
    subject_entity_id: str,
    object_entity_id: str | None,
    subtype: str | None,
    *,
    value_text: str | None = None,
    value_number: float | None = None,
    value_unit: str | None = None,
    value_kind: str | None = None,
    value_time: str | None = None,
) -> str:
    if family.slot_mode == "subject":
        return f"{family.family}:{subject_entity_id}"
    if family.slot_mode == "object":
        object_part = object_entity_id
        if object_part is None:
            value_parts = _encode_value_parts(
                value_text=value_text,
                value_number=value_number,
                value_unit=value_unit,
                value_kind=value_kind,
                value_time=value_time,
            )
            object_part = "|".join(value_parts) if value_parts else ""
        return ":".join(part for part in (family.family, subject_entity_id, object_part) if part)
    if family.slot_mode == "subtype":
        return ":".join(
            part for part in (family.family, subject_entity_id, subtype or "") if part
        )
    return fact_key(
        family,
        subject_entity_id,
        object_entity_id,
        subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
    )
