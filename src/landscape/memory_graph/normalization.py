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
    fact_key: str | None
    slot_key: str | None


def normalize_assertion(
    payload: AssertionPayload,
    *,
    subject_entity_id: str | None,
    object_entity_id: str | None,
) -> NormalizationResult:
    family = payload.family_candidate
    if not family or family not in FAMILY_REGISTRY:
        return NormalizationResult(
            False,
            None,
            subject_entity_id,
            object_entity_id,
            payload.subtype,
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
            None,
            None,
        )

    return NormalizationResult(
        promotable=True,
        family=family,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        subtype=payload.subtype,
        fact_key=fact_key(config, subject_entity_id, object_entity_id, payload.subtype),
        slot_key=slot_key(config, subject_entity_id, object_entity_id, payload.subtype),
    )
