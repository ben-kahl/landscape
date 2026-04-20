import re

from pydantic import BaseModel

# Closed vocabulary of relation types. Any rel_type the LLM produces that's
# not in this set is normalized via RELATION_SYNONYMS before write, or passed
# through unchanged if it's truly novel. This is a minimum-viable fix for
# synonym drift (e.g. WORKS_FOR / EMPLOYED_BY / CURRENTLY_WORKS_AT all
# describing the same semantic relation) — see the "Known limitations"
# section in CLAUDE.md for the broader problem and the path forward.
RELATION_VOCAB: frozenset[str] = frozenset(
    {
        # Original v1
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
        # v2 additions (vocab_expansion spec)
        "HAS_TITLE",
        "HAS_PREFERENCE",
        "HAS_ATTRIBUTE",
        "FAMILY_OF",
        "RECOMMENDED",
        "DISCUSSED",
        "HAPPENED_ON",
        "LIVES_IN",
    }
)

# Per-rel-type functional keys. An edge is "functional" in a given slot when
# its (subject, type, <declared-key-fields>) tuple matches a live edge; if
# that slot already has a different object, the old edge is superseded.
#
# Patterns:
#   ()           — subject-keyed: one object per (subject, type)
#   ("object",)  — object-keyed: one subtype per (subject, type, object)
#   ("subtype",) — subtype-keyed: one object per (subject, type, subtype)
#
# Absent from the dict → additive (both edges coexist).
#
# Null-safety: for any declared key field, if EITHER the new or the old edge
# carries NULL for that field, treat it as a non-match. This keeps object-
# and subtype-keyed rels safe when subtype hasn't been extracted yet (step 3
# of the vocab-expansion rollout) — no silent supersession on missing data.
FUNCTIONAL_KEYS: dict[str, tuple[str, ...]] = {
    # Subject-keyed (at most one live object per subject)
    "WORKS_FOR": (),
    "REPORTS_TO": (),
    "BELONGS_TO": (),
    "LIVES_IN": (),
    "HAPPENED_ON": (),
    # Object-keyed (one subtype per subject+object pair) — superseded once
    # subtype write-through lands (step 3); behaves additively until then.
    "HAS_TITLE": ("object",),
    # Subtype-keyed (one object per subject+subtype pair) — no-op until
    # subtype write-through lands.
    "HAS_PREFERENCE": ("subtype",),
    "HAS_ATTRIBUTE": ("subtype",),
    # Absent (additive): LEADS, MEMBER_OF, APPROVED, USES, CREATED,
    # LOCATED_IN, RELATED_TO, FAMILY_OF, RECOMMENDED, DISCUSSED.
}

# Backwards-compat alias — collapses to the set of subject-keyed rels. Kept
# for one release; callers should migrate to FUNCTIONAL_KEYS. Removal planned
# alongside the step-3 subtype write-through commit.
FUNCTIONAL_RELATION_TYPES: frozenset[str] = frozenset(
    rel for rel, key in FUNCTIONAL_KEYS.items() if key == ()
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
    # HAS_TITLE family (Person → Org, title rides in subtype)
    "TITLE": "HAS_TITLE",
    "ROLE": "HAS_TITLE",
    "POSITION": "HAS_TITLE",
    "WORKS_AS": "HAS_TITLE",
    "HAS_JOB_TITLE": "HAS_TITLE",
    "HAS_ROLE": "HAS_TITLE",
    "APPOINTED_AS": "HAS_TITLE",
    "PROMOTED_TO": "HAS_TITLE",
    # HAS_PREFERENCE family
    "PREFERS": "HAS_PREFERENCE",
    "LIKES_BEST": "HAS_PREFERENCE",
    "FAVORITE": "HAS_PREFERENCE",
    "FAVORITE_OF": "HAS_PREFERENCE",
    # HAS_ATTRIBUTE family (dangerous — generic IS/HAS needs a careful coercion threshold)
    "HAS_TRAIT": "HAS_ATTRIBUTE",
    "ATTRIBUTE_OF": "HAS_ATTRIBUTE",
    # FAMILY_OF family (specifics ride in subtype: daughter, uncle, parent, etc.)
    "RELATED_TO_FAMILY": "FAMILY_OF",
    "KIN_OF": "FAMILY_OF",
    "MARRIED_TO": "FAMILY_OF",
    "SPOUSE_OF": "FAMILY_OF",
    "PARENT_OF": "FAMILY_OF",
    "CHILD_OF": "FAMILY_OF",
    "SIBLING_OF": "FAMILY_OF",
    # LIVES_IN family (subject-keyed; distinct from LOCATED_IN which is additive)
    "RESIDES_IN": "LIVES_IN",
    "LIVES_AT": "LIVES_IN",
    "HOME_IS": "LIVES_IN",
    "CURRENTLY_LIVES_IN": "LIVES_IN",
    # RECOMMENDED / DISCUSSED / HAPPENED_ON — mostly rely on direct match; a
    # few surface variants covered here.
    "SUGGESTED": "RECOMMENDED",
    "TALKED_ABOUT": "DISCUSSED",
    "MENTIONED": "DISCUSSED",
    "OCCURRED_ON": "HAPPENED_ON",
    "TOOK_PLACE_ON": "HAPPENED_ON",
    "DATED": "HAPPENED_ON",
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
    subtype: str | None = None
    quantity_value: float | str | None = None
    quantity_unit: str | None = None
    quantity_kind: str | None = None
    time_scope: str | None = None


_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def normalize_subtype(raw: str | None) -> str | None:
    """snake_case an LLM-supplied subtype. Whitespace/punctuation collapse to
    single underscores; leading/trailing underscores stripped. Returns None
    for empty/None/whitespace-only input so callers can test truthiness."""
    if raw is None:
        return None
    cleaned = _SNAKE_RE.sub("_", raw.strip().lower()).strip("_")
    return cleaned or None


class Extraction(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
