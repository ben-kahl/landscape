from dataclasses import dataclass
from typing import Literal

SlotMode = Literal["additive", "subject", "object", "subtype"]
ObjectKind = Literal["entity", "value"]


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    slot_mode: SlotMode
    object_kind: ObjectKind
    traversable: bool


def _e(family: str, slot_mode: SlotMode, traversable: bool = True) -> FamilyConfig:
    return FamilyConfig(family, slot_mode=slot_mode, object_kind="entity", traversable=traversable)


def _v(family: str, slot_mode: SlotMode, traversable: bool = True) -> FamilyConfig:
    return FamilyConfig(family, slot_mode=slot_mode, object_kind="value", traversable=traversable)


FAMILY_REGISTRY: dict[str, FamilyConfig] = {
    "WORKS_FOR":      _e("WORKS_FOR", "subject"),
    "LEADS":          _e("LEADS", "additive"),
    "MEMBER_OF":      _e("MEMBER_OF", "additive"),
    "WORKS_ON":       _e("WORKS_ON", "additive"),
    "MAINTAINS":      _e("MAINTAINS", "additive"),
    "REPORTS_TO":     _e("REPORTS_TO", "subject"),
    "APPROVED":       _e("APPROVED", "additive"),
    "USES":           _e("USES", "additive"),
    "BELONGS_TO":     _e("BELONGS_TO", "subject"),
    "LOCATED_IN":     _e("LOCATED_IN", "additive"),
    "CREATED":        _e("CREATED", "additive"),
    "HAS_TITLE":      _e("HAS_TITLE", "object"),
    "FAMILY_OF":      _e("FAMILY_OF", "additive"),
    "LIVES_IN":       _e("LIVES_IN", "subject"),
    "OWNS":           _e("OWNS", "additive"),
    "DEPENDS_ON":     _e("DEPENDS_ON", "additive"),
    "HAS_PREFERENCE": _v("HAS_PREFERENCE", "subtype"),
    "HAS_ATTRIBUTE":  _v("HAS_ATTRIBUTE", "subtype"),
    "RECOMMENDED":    _v("RECOMMENDED", "additive"),
    "DISCUSSED":      _v("DISCUSSED", "additive"),
    "HAPPENED_ON":    _v("HAPPENED_ON", "subject"),
}
