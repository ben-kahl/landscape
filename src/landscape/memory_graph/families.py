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

    @property
    def single_current(self) -> bool:
        return self.slot_mode == "subject"

    @property
    def identity_uses_subtype(self) -> bool:
        return self.slot_mode == "subtype"


FAMILY_REGISTRY: dict[str, FamilyConfig] = {
    "WORKS_FOR": FamilyConfig("WORKS_FOR", slot_mode="subject", object_kind="entity", traversable=True),
    "LEADS": FamilyConfig("LEADS", slot_mode="additive", object_kind="entity", traversable=True),
    "MEMBER_OF": FamilyConfig("MEMBER_OF", slot_mode="additive", object_kind="entity", traversable=True),
    "WORKS_ON": FamilyConfig("WORKS_ON", slot_mode="additive", object_kind="entity", traversable=True),
    "MAINTAINS": FamilyConfig("MAINTAINS", slot_mode="additive", object_kind="entity", traversable=True),
    "REPORTS_TO": FamilyConfig("REPORTS_TO", slot_mode="subject", object_kind="entity", traversable=True),
    "APPROVED": FamilyConfig("APPROVED", slot_mode="additive", object_kind="entity", traversable=True),
    "USES": FamilyConfig("USES", slot_mode="additive", object_kind="entity", traversable=True),
    "BELONGS_TO": FamilyConfig("BELONGS_TO", slot_mode="subject", object_kind="entity", traversable=True),
    "LOCATED_IN": FamilyConfig("LOCATED_IN", slot_mode="additive", object_kind="entity", traversable=True),
    "CREATED": FamilyConfig("CREATED", slot_mode="additive", object_kind="entity", traversable=True),
    "HAS_TITLE": FamilyConfig("HAS_TITLE", slot_mode="object", object_kind="entity", traversable=True),
    "HAS_PREFERENCE": FamilyConfig("HAS_PREFERENCE", slot_mode="subtype", object_kind="value", traversable=True),
    "HAS_ATTRIBUTE": FamilyConfig("HAS_ATTRIBUTE", slot_mode="subtype", object_kind="value", traversable=True),
    "FAMILY_OF": FamilyConfig("FAMILY_OF", slot_mode="subtype", object_kind="entity", traversable=True),
    "RECOMMENDED": FamilyConfig("RECOMMENDED", slot_mode="additive", object_kind="value", traversable=True),
    "DISCUSSED": FamilyConfig("DISCUSSED", slot_mode="additive", object_kind="value", traversable=True),
    "HAPPENED_ON": FamilyConfig("HAPPENED_ON", slot_mode="subject", object_kind="value", traversable=True),
    "LIVES_IN": FamilyConfig("LIVES_IN", slot_mode="subject", object_kind="entity", traversable=True),
    "OWNS": FamilyConfig("OWNS", slot_mode="additive", object_kind="entity", traversable=True),
    "DEPENDS_ON": FamilyConfig("DEPENDS_ON", slot_mode="additive", object_kind="entity", traversable=True),
}
