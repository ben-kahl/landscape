from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    traversable: bool
    object_kind: str
    single_current: bool
    identity_uses_subtype: bool = False


FAMILY_REGISTRY: dict[str, FamilyConfig] = {
    "WORKS_FOR": FamilyConfig("WORKS_FOR", traversable=True, object_kind="entity", single_current=True),
    "MEMBER_OF": FamilyConfig("MEMBER_OF", traversable=True, object_kind="entity", single_current=False),
    "BELONGS_TO": FamilyConfig("BELONGS_TO", traversable=True, object_kind="entity", single_current=True),
    "REPORTS_TO": FamilyConfig("REPORTS_TO", traversable=True, object_kind="entity", single_current=True),
    "WORKS_ON": FamilyConfig("WORKS_ON", traversable=True, object_kind="entity", single_current=False),
    "MAINTAINS": FamilyConfig("MAINTAINS", traversable=True, object_kind="entity", single_current=False),
    # Transitional redesign families: the registry treats these as first-class,
    # but current extraction normalization still maps raw OWNS -> LEADS and
    # raw DEPENDS_ON -> USES. Task 2+ will consume the family registry directly.
    "OWNS": FamilyConfig("OWNS", traversable=True, object_kind="entity", single_current=False),
    "USES": FamilyConfig("USES", traversable=True, object_kind="entity", single_current=False),
    "DEPENDS_ON": FamilyConfig("DEPENDS_ON", traversable=True, object_kind="entity", single_current=False),
    "CREATED": FamilyConfig("CREATED", traversable=True, object_kind="entity", single_current=False, identity_uses_subtype=True),
    "LEADS": FamilyConfig("LEADS", traversable=True, object_kind="entity", single_current=False),
    "APPROVED": FamilyConfig("APPROVED", traversable=True, object_kind="entity", single_current=False),
    "LOCATED_IN": FamilyConfig("LOCATED_IN", traversable=True, object_kind="entity", single_current=False),
}
