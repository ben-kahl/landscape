from dataclasses import dataclass

from landscape.memory_graph.families import FAMILY_REGISTRY
from landscape.memory_graph.ids import fact_key, slot_key
from landscape.memory_graph.models import AssertionPayload


@dataclass(frozen=True)
class NormalizationResult:
    promotable: bool
    family: str | None
    subject_entity_id: str | None
    object_entity_id: str | None
    subtype: str | None
    value_text: str | None
    value_number: float | None
    value_unit: str | None
    value_kind: str | None
    value_time: str | None
    quantity_value: float | str | None
    quantity_unit: str | None
    quantity_kind: str | None
    time_scope: str | None
    fact_key: str | None
    slot_key: str | None
    negated: bool = False


def normalize_assertion(
    payload: AssertionPayload,
    *,
    subject_entity_id: str | None,
    object_entity_id: str | None,
) -> NormalizationResult:
    value_text = payload.value_text
    value_number = payload.value_number
    value_unit = payload.value_unit
    value_kind = payload.value_kind
    value_time = payload.value_time
    quantity_value = payload.quantity_value
    quantity_unit = payload.quantity_unit
    quantity_kind = payload.quantity_kind
    time_scope = payload.time_scope
    family = payload.family_candidate
    if not family or family not in FAMILY_REGISTRY:
        return NormalizationResult(
            False,
            None,
            subject_entity_id,
            object_entity_id,
            payload.subtype,
            value_text,
            value_number,
            value_unit,
            value_kind,
            value_time,
            quantity_value,
            quantity_unit,
            quantity_kind,
            time_scope,
            None,
            None,
        )

    config = FAMILY_REGISTRY[family]
    if subject_entity_id is None:
        return NormalizationResult(
            False,
            None,
            subject_entity_id,
            object_entity_id,
            payload.subtype,
            value_text,
            value_number,
            value_unit,
            value_kind,
            value_time,
            quantity_value,
            quantity_unit,
            quantity_kind,
            time_scope,
            None,
            None,
        )
    if config.object_kind == "entity" and object_entity_id is None:
        return NormalizationResult(
            False,
            None,
            subject_entity_id,
            object_entity_id,
            payload.subtype,
            value_text,
            value_number,
            value_unit,
            value_kind,
            value_time,
            quantity_value,
            quantity_unit,
            quantity_kind,
            time_scope,
            None,
            None,
        )

    if config.object_kind == "value":
        if value_text is None and value_number is None and payload.quantity_value is not None:
            if isinstance(payload.quantity_value, str):
                value_text = payload.quantity_value
            else:
                value_number = float(payload.quantity_value)
        if value_text is None and value_number is None and object_entity_id is None:
            value_text = payload.raw_object_text
        if value_kind is None and quantity_kind is not None:
            value_kind = quantity_kind
        if value_unit is None and quantity_unit is not None:
            value_unit = quantity_unit
        if value_time is None and time_scope is not None:
            value_time = time_scope

    return NormalizationResult(
        promotable=True,
        family=family,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        subtype=payload.subtype,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        value_kind=value_kind,
        value_time=value_time,
        quantity_value=quantity_value,
        quantity_unit=quantity_unit,
        quantity_kind=quantity_kind,
        time_scope=time_scope,
        negated=payload.negated,
        fact_key=fact_key(
            config,
            subject_entity_id,
            object_entity_id,
            payload.subtype,
            negated=payload.negated,
            value_text=value_text,
            value_number=value_number,
            value_unit=value_unit,
            value_kind=value_kind,
            value_time=value_time,
        ),
        slot_key=slot_key(
            config,
            subject_entity_id,
            object_entity_id,
            payload.subtype,
            negated=False,
            value_text=value_text,
            value_number=value_number,
            value_unit=value_unit,
            value_kind=value_kind,
            value_time=value_time,
        ),
    )
