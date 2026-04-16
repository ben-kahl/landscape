from pydantic import BaseModel

# Closed vocabulary of relation types. Any rel_type the LLM produces that's
# not in this set is normalized via RELATION_SYNONYMS before write, or passed
# through unchanged if it's truly novel. This is a minimum-viable fix for
# synonym drift (e.g. WORKS_FOR / EMPLOYED_BY / CURRENTLY_WORKS_AT all
# describing the same semantic relation) — see the "Known limitations"
# section in CLAUDE.md for the broader problem and the path forward.
RELATION_VOCAB: frozenset[str] = frozenset(
    {
        "WORKS_FOR",
        "LEADS",
        "MEMBER_OF",
        "REPORTS_TO",
        "APPROVED",
        "USES",
        "BELONGS_TO",
        "LOCATED_IN",
        "CREATED",
        "RELATED_TO",
    }
)

# Relation types where a subject has at most one live object at a time.
# When a new edge lands with a (subject, rel_type) already present but a
# *different* object, the old edge gets superseded (valid_until set).
#
# Non-functional rels like LEADS, MEMBER_OF, APPROVED, USES, CREATED, LOCATED_IN,
# RELATED_TO are additive: a person can lead multiple teams, a project can use
# multiple technologies, a company can be headquartered in multiple offices.
# Treating those as functional silently marks correct facts stale.
FUNCTIONAL_RELATION_TYPES: frozenset[str] = frozenset(
    {
        "WORKS_FOR",
        "REPORTS_TO",
        "BELONGS_TO",
    }
)

# Synonym → canonical mapping. Only includes *same-direction* synonyms —
# e.g. APPROVED_BY (object approved subject) is the reverse direction and
# would require swapping subject/object, which is deferred.
RELATION_SYNONYMS: dict[str, str] = {
    # WORKS_FOR family
    "EMPLOYED_BY": "WORKS_FOR",
    "EMPLOYED_AT": "WORKS_FOR",
    "WORKS_AT": "WORKS_FOR",
    "WORKED_AT": "WORKS_FOR",
    "WORKED_FOR": "WORKS_FOR",
    "CURRENTLY_WORKS_AT": "WORKS_FOR",
    "CURRENTLY_WORKS_FOR": "WORKS_FOR",
    "HIRED_BY": "WORKS_FOR",
    "AFFILIATED_WITH": "WORKS_FOR",
    # LEADS family
    "MANAGES": "LEADS",
    "HEADS": "LEADS",
    "DIRECTS": "LEADS",
    "OVERSEES": "LEADS",
    "OWNS": "LEADS",
    # MEMBER_OF family
    "PART_OF": "MEMBER_OF",
    "IN_TEAM": "MEMBER_OF",
    "ON_TEAM": "MEMBER_OF",
    "BELONGS_TO_TEAM": "MEMBER_OF",
    # APPROVED family (same direction: subject approved object)
    "SIGNED_OFF_ON": "APPROVED",
    "GREEN_LIT": "APPROVED",
    "AUTHORIZED": "APPROVED",
    "RATIFIED": "APPROVED",
    # USES family
    "DEPENDS_ON": "USES",
    "BUILT_ON": "USES",
    "RUNS_ON": "USES",
    "POWERED_BY": "USES",
    "USES_TECHNOLOGY": "USES",
    # BELONGS_TO family (e.g. project → parent org)
    "OWNED_BY": "BELONGS_TO",
    "SUBSIDIARY_OF": "BELONGS_TO",
    "DIVISION_OF": "BELONGS_TO",
    # LOCATED_IN family
    "BASED_IN": "LOCATED_IN",
    "HEADQUARTERED_IN": "LOCATED_IN",
    "SITUATED_IN": "LOCATED_IN",
    # CREATED family
    "AUTHORED": "CREATED",
    "BUILT": "CREATED",
    "DEVELOPED": "CREATED",
    "WROTE": "CREATED",
    "FOUNDED": "CREATED",
}


def normalize_relation_type(rel_type: str) -> str:
    """Map an LLM-produced rel_type to a canonical form. Unknown types pass
    through unchanged — we'd rather preserve novel semantics than flatten
    everything to RELATED_TO. The cost is that unknown types still don't
    supersede across synonyms; that's acknowledged in the limitations doc."""
    if not rel_type:
        return "RELATED_TO"
    upper = rel_type.strip().upper().replace(" ", "_")
    if upper in RELATION_VOCAB:
        return upper
    return RELATION_SYNONYMS.get(upper, upper)


class ExtractedEntity(BaseModel):
    name: str
    type: str
    confidence: float
    aliases: list[str] = []


class ExtractedRelation(BaseModel):
    subject: str
    object: str
    relation_type: str
    confidence: float


class Extraction(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
