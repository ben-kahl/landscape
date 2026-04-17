# Spec — Entity + Relation Vocabulary Expansion (v2)

Status: draft for review
Target: land before running LongMemEval
Related: CLAUDE.md §"Known Limitations" (relation-type supersession, functional relations)

## 1. Motivation

The current vocabulary (8 entity types, 10 rel types) was tuned for the Helios Robotics corpus — org charts, projects, functional supersession of employment. LongMemEval probes personal memory: preferences, life facts, tasks, temporal reasoning, knowledge updates. Running LongMemEval against today's schema drops most of the interesting facts at extraction time, which will make the benchmark look like a retrieval problem when it's really an extraction-schema problem.

Concrete gap already observed: "Alice works for Atlas Corp as a senior engineer" drops the title entirely — no place to put "senior engineer" in the current schema.

## 2. Core idea — subtypes on relations, not a wider canonical vocab

Entity-type coercion already does the right thing: the canonical `type` is one of 8 primitives, and the raw LLM-produced string is preserved as `subtype` on the node. No information lost, no explosion of canonicals to maintain.

Apply the same pattern to relations. Every `:RELATES_TO` edge carries:

- `type` — one of a small set of primitives (canonical). Determines traversal semantics and supersession behavior.
- `subtype` — snake_case string preserving the LLM's original phrasing or nuance. Informational; does not (by default) participate in supersession.

So instead of 28 canonical rel types, we have ~12 primitives and let nuance ride in `subtype`:

```
(Alice)-[:RELATES_TO {type: "HAS_TITLE",     subtype: "senior_engineer"}]->(Atlas)
(Bob)  -[:RELATES_TO {type: "LIKES",         subtype: "loves"}           ]->(Jazz)
(Carol)-[:RELATES_TO {type: "FAMILY_OF",     subtype: "daughter"}        ]->(Dana)
(Eve)  -[:RELATES_TO {type: "HAS_PREFERENCE",subtype: "favorite_color"}  ]->(Blue)
```

## 3. Supersession is per-rel-type, declared by key

Today's `FUNCTIONAL_RELATION_TYPES` frozenset hides a bug: "functional" isn't a property of the rel type alone — it's a property of *which fields together identify the slot* that a new edge occupies. Three meaningful patterns show up, plus the additive default:

| Pattern | Functional key | Examples |
|---|---|---|
| **Subject-keyed** — one object globally | `(subject, type)` | `WORKS_FOR`, `LIVES_IN`, `MARRIED_TO`, `REPORTS_TO`, `BELONGS_TO` |
| **Object-keyed** — one subtype per (subject, object) pair | `(subject, type, object)` | `HAS_TITLE` @ employer, `HAS_ROLE` @ team/club, `PLAYS_POSITION` @ team, `HAS_ACCOUNT_AT` @ service, `ENROLLED_IN` @ school, `HAS_RELATIONSHIP` @ person |
| **Subtype-keyed** — one object per (subject, subtype) | `(subject, type, subtype)` | `HAS_PREFERENCE(subtype="favorite_color")`, `HAS_ATTRIBUTE(subtype="height")`, `HAS_STATE(subtype="current_mood")` |
| **Additive** — coexistence is correct | no key | `LIKES`, `FAMILY_OF`, `USES`, `LEADS`, `MEMBER_OF`, `DISCUSSED`, `RECOMMENDED`, `APPROVED`, `CREATED` |

The object-keyed pattern is the one we'd previously overlooked — it's common outside employment. Promoting at Atlas supersedes the old Atlas title; moving to Beacon does not supersede the Atlas title (both facts remain meaningfully true at different points in time, and the supersession of `WORKS_FOR Atlas` carries the "no longer at Atlas" story on its own).

Replace the flat frozenset with a per-type key declaration:

```python
FUNCTIONAL_KEYS: dict[str, tuple[str, ...]] = {
    "WORKS_FOR":      (),          # subject is always implicit; () = subject-keyed
    "LIVES_IN":       (),
    "MARRIED_TO":     (),
    "REPORTS_TO":     (),
    "BELONGS_TO":     (),
    "HAPPENED_ON":    (),
    "HAS_TITLE":      ("object",), # one title per (subject, employer)
    "HAS_PREFERENCE": ("subtype",),# one value per (subject, preference_axis)
    "HAS_ATTRIBUTE":  ("subtype",),
    # absent = additive
}
```

`upsert_relation` logic: when a new edge lands, look up any live edge with the same subject + type whose declared-key fields all match the incoming edge. If found and the object differs, supersede the old; if found and the object matches, reinforce.

**Strict generalization of today's behavior.** The three current functional rels (`WORKS_FOR`, `REPORTS_TO`, `BELONGS_TO`) map to `()` — same as today. No existing tests regress.

## 4. Vocabulary additions

### Entity types (+3, total 11)

Add only where a truly new primitive is needed. Everything else rides as an entity's `subtype` or the edge's `subtype`.

| Type | Rationale | Example |
|---|---|---|
| `Role` | A role/title is not a Person, Org, or Concept — it's its own thing, and we'll reference it in `HAS_TITLE` edges | "Senior Engineer", "CTO" |
| `DateTime` | Temporal reasoning needs anchor points as first-class nodes so graph traversal can answer "what happened on X" | "last Friday", "2026-03-05" |
| `Task` | LongMemEval tracks user tasks; Task has distinct lifecycle (assignment, completion) that doesn't fit Event | "buy groceries", "finish Q2 report" |

**Explicitly not adding** — these collapse to subtypes:
- Skill, Goal, Attribute, Preference → Concept with `subtype` set; the semantic is carried by the relation type (`HAS_SKILL`, `HAS_GOAL`, etc.), not the entity type.
- Emotion, Mood → ride as `subtype` on Concept or as the object of `HAS_STATE`.
- File types (Image, Video) → Document with `subtype`.

### Relation types (+7, total 17)

| Type | Key | Purpose |
|---|---|---|
| `HAS_TITLE` | `(object,)` | Person → Role or Org, title in subtype |
| `HAS_PREFERENCE` | `(subtype,)` | Person → anything they like; subtype names the axis ("favorite_color", "dietary") |
| `HAS_ATTRIBUTE` | `(subtype,)` | Person → value; subtype names the attribute ("height", "pronouns") |
| `FAMILY_OF` | additive | Person → Person; subtype = relation ("daughter", "uncle") |
| `RECOMMENDED` | additive | Person → anything, subtype optional ("strongly", "in_passing") |
| `DISCUSSED` | additive | Person → anything; write only at confidence ≥ 0.7 |
| `HAPPENED_ON` | `()` | Event → DateTime |

**Not adding as new types** (handled via subtype on existing rel or skipped for v1):
- `HAS_SKILL`, `HAS_GOAL`, `LIVES_IN` — `LIVES_IN` lands as a distinct rel from `LOCATED_IN` despite surface similarity because `LIVES_IN` is subject-keyed (one current residence) and `LOCATED_IN` is additive (multi-office org). **Add `LIVES_IN` to v1** — it was in the list above under subject-keyed.
- `SCHEDULED_FOR`, `ASSIGNED_TO`, `COMPLETED` — task-lifecycle rels. Add in a v1.5 once Task as an entity type is exercised.

Updated v1 primitive rel list (13 total): current 10 + `HAS_TITLE`, `HAS_PREFERENCE`, `HAS_ATTRIBUTE`, `FAMILY_OF`, `RECOMMENDED`, `DISCUSSED`, `HAPPENED_ON`, `LIVES_IN`.

(Yes, that's 8 additions. The spec's claim of "+7" above was a miscount — I'm holding at 8 including `LIVES_IN`.)

### Synonym map additions (sketch)

Rough sizes:
- `HAS_TITLE`: `TITLE`, `ROLE`, `POSITION`, `WORKS_AS`, `HAS_JOB_TITLE`, `HAS_ROLE`, `APPOINTED_AS`
- `LIVES_IN`: `RESIDES_IN`, `LIVES_AT`, `HOME_IS`, `CURRENTLY_LIVES_IN`
- `FAMILY_OF`: `RELATED_TO_FAMILY`, `KIN_OF`; subtypes carry the specifics (daughter/son/parent/etc.)
- `HAS_PREFERENCE`: `PREFERS`, `LIKES_BEST`, `FAVORITE`
- `HAS_ATTRIBUTE`: `IS`, `HAS` (dangerous — may over-catch; needs careful coercion threshold)

The embedding-based coercion path (`rel_type_coercion.py`) handles the semantic near-misses. Keep `COERCION_THRESHOLD` at 0.55 initially and rerun `bench_extraction.py` to confirm the bigger vocab doesn't degrade small-LLM precision.

## 5. Subtype handling

### Write path
1. Extraction LLM produces `{subject, object, relation_type, subtype?, confidence}`.
2. `normalize_relation_type(rel_type)` maps to canonical primitive (existing logic, synonym map + embedding coercion).
3. `subtype` passes through as `snake_case(strip(lower(raw)))`. No stopword removal; no synonym collapsing at write time. Preserve the LLM's choice verbatim. Empty/None is allowed — additive rels often don't need a subtype.
4. `upsert_relation` applies the `FUNCTIONAL_KEYS` lookup to decide reinforce / supersede / create.

### Retrieval path
1. BFS traversal keys on canonical `type` (same as today) — a query for "who reports to Alice?" still works via the `REPORTS_TO` primitive regardless of subtype.
2. Path display includes the subtype in brackets when present:
   ```
   Alice → HAS_TITLE[senior_engineer] → Atlas
   Bob   → FAMILY_OF[daughter] → Carol
   ```
3. `RetrievedEntity.path_edge_types` stays a list of canonicals for backward compat; add `path_edge_subtypes: list[str | None]` alongside. LangChain retriever displays both.

### Agent write-back API
`add_relation` grows an optional `subtype: str | None = None` kwarg. Backward compatible — callers not passing it get the current behavior.

## 6. Migration

All changes additive. No data migration:
- Existing edges have no `subtype` property → treated as `None` everywhere.
- Existing functional set (3 rels) maps directly to `FUNCTIONAL_KEYS` entries with `()`.
- New rel types and entity types don't affect existing nodes.
- Retrieval path display is purely additive — callers relying on `path_edge_types` alone continue to work.

## 7. Non-goals / deferred

- **Subtype clustering** (e.g. `senior_engineer` vs `senior_software_engineer`). Deferred; we'll see if it matters on real benchmark traces before building embedding-based subtype collapsing.
- **Cross-edge consistency** (e.g. when `WORKS_FOR Atlas` is superseded, should dependent `HAS_TITLE@Atlas` also be marked superseded?). Deferred. Arguably the old title edge remains historically true; adding cross-edge consistency is a correctness-vs-audit-trail tradeoff we should make with data, not by fiat.
- **Reified relations** (Role as a node rather than an edge property): out of scope. Would give us cleaner modeling of "senior eng at Atlas from 2024–2026" but doubles traversal depth.
- **Per-rel confidence thresholds at write time** (e.g. `DISCUSSED` only persists at conf ≥ 0.7): call it out as a config knob but don't ship it in v1 — measure noise first.
- **`HAS_TITLE` between Person and Role (vs Person and Org)**: the spec picks Person→Org with title-in-subtype, because that gives clean object-keyed supersession on promotion. If we later want Role as a first-class queryable node ("who has the CTO title?"), we can add a parallel `HAS_ROLE` edge Person→Role alongside without conflict.

## 8. Rollout plan

1. Land this spec + approve (you). No code.
2. **Schema + supersession change** (1 commit):
   - Add `FUNCTIONAL_KEYS` replacing `FUNCTIONAL_RELATION_TYPES` (keep the old name as an alias for one release).
   - Rewrite `upsert_relation` Case 2 to use the declared key.
   - Add the 8 new canonical rel types + 3 new entity types.
   - Expand synonym map.
   - All existing tests must pass unchanged.
3. **Subtype write-through** (1 commit):
   - `ExtractedRelation.subtype: str | None`.
   - `upsert_relation` accepts + writes subtype.
   - `add_relation` MCP tool grows `subtype` kwarg.
   - snake_case normalization at write time.
4. **Extraction prompt update** (1 commit):
   - New canonical list + 1-line descriptors in the prompt.
   - 2–3 worked examples covering: personal fact with title/family, preference, temporal.
   - Rerun `bench_extraction.py`; update `project_extraction_model_benchmarks.md`. Gate: F1 ≥ prior baseline − 5%.
5. **Retrieval display + LangChain passthrough** (1 commit):
   - `RetrievedEntity.path_edge_subtypes`; path formatting shows subtypes in brackets.
6. **Personal-memory integration test** (1 commit):
   - Small hand-built corpus (3–5 synthetic docs) exercising title, preference, family, temporal.
   - Assert extraction coverage + retrieval over subtype-carrying edges.
7. **LongMemEval single-session-user smoke** (1 commit): run a slice, record numbers.

If step 4 regresses F1 > 5% on the existing benchmark, revert the prompt + vocab expansion and pursue per-category extractors or a larger local LLM instead. The `FUNCTIONAL_KEYS` refactor in step 2 stays regardless — it's an improvement independent of vocab size.

## 9. Open questions

1. **Validation of subtype values.** Reject empty-after-snake_case? Reject subtypes longer than some N? Leaning: no validation beyond snake_case normalization; let weird LLM outputs through and deal with them via retrieval-side filtering if it becomes a problem.
2. **Subtype cardinality explosion.** Index on `(type, subtype)` in Neo4j? Probably not needed until we see query patterns — traversal keys on `type`, subtype is informational.
3. **Should `FUNCTIONAL_KEYS` support composite keys beyond one extra field?** I don't see a v1 use case. Keep the signature `tuple[str, ...]` so we can grow later without a breaking change.
4. **What's our ground truth for step 4's extraction benchmark?** The killer-demo corpus won't exercise the new rel types. Proposal: hand-build a 20-doc personal-memory mini-corpus with known entities/relations before step 4, use it as the benchmark target for the new vocab. Extra work up front but we'd otherwise be flying blind on whether the prompt change helped or hurt.

## 10. Success criteria

- All existing tests green after step 2.
- Extraction benchmark F1 (existing corpus) within 5% of prior baseline after step 4.
- New mini-corpus (step 6) extracts ≥ 80% of planted title/preference/family/temporal facts with correct canonical + subtype.
- LongMemEval single-session-user smoke (step 7) demonstrates subtype recall on at least the title/preference/family question shapes. Absolute scores not gated on this spec — we're unblocking the benchmark, not claiming SoTA.
