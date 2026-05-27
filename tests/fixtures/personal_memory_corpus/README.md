# Personal Memory Corpus

Ground-truth corpus for benchmarking the vocab-expansion proposal in `specs/vocab_expansion.md`. Used to gate the expanded canonical vocabulary + subtype model + `FUNCTIONAL_KEYS` refactor against extraction quality regressions, and to drive LongMemEval-style retrieval tests.

## Contents

- `doc_01..doc_20_*.md` — 20 hand-authored documents. 2–5 sentences each, clearly written but with enough natural phrasing that extraction is non-trivial. Organized across 6 scenarios (employment trajectory, knowledge updates, family, temporal, preferences, multi-hop connectors).
- `ground_truth.json` — per-document expected entities and relations. Each relation records the canonical type, the `subtype` (snake_case nuance), and a list of acceptable synonym strings the extractor may produce. Also includes canonical-vocab metadata under `_meta`.
- `scenarios.json` — multi-document scenarios. Each scenario specifies an ingest sequence, assertions about the final graph state (live vs superseded edges), and retrieval queries with expected top-result entities.

## How to use

### Extraction quality benchmark
Ingest each doc independently; compare extracted (type, subject, object) triples against ground truth. Score precision and recall, with synonym hits counted as correct when the produced type is in `acceptable_synonyms` for that ground-truth relation. Subtype scoring is separate (exact match or cosine similarity — decide at measurement time).

### Supersession correctness
Run the ingest sequences in `scenarios.json` in order. After each scenario, query Neo4j for edges with `valid_until IS NULL` and `valid_until IS NOT NULL` and diff against `final_state.live_relations` / `final_state.superseded_relations`.

### Retrieval quality
Run the retrieval queries in each scenario via `retrieve()`. Score by whether `expect_top_names` all appear in the top-k (k = len(expect_top_names) + 2). Multi-hop queries are marked with `notes` explaining the reasoning path.

### Session scoping
The `session_scoped_retrieval` scenario exercises the Phase 3.1 cross-session filter (#20) with the expanded vocab. Ingest specifies `session_id` and `turn_id` per doc.

## Intentional edge cases

- Direction-ambiguous phrasings ("Her manager is Hiro" — should Alice REPORT_TO Hiro or Hiro MANAGES Alice?) — see the `direction_ambiguity_probe` scenario.
- Historical relations that supersession will mark stale (doc_17's Alice-at-Atlas fact lands after doc_03's job change) — extraction ground truth is per-doc, so the raw fact is still expected to be extracted.
- Entities that stand alone without explicit relations (inference latency as a topic, otter as an animal, etc.) — keeps the Concept type exercised.
- `MARRIED_TO` vs `FAMILY_OF(subtype=spouse)` — ground truth canonicalizes to the latter for v1; synonym map should accept the former.

## What this corpus deliberately does NOT cover

- Image or visual-derived entities (Phase 4).
- Adversarial or ambiguous references that require coreference resolution beyond simple pronouns.
- Cross-language or multi-lingual text.
- Very long documents (all docs are < 100 tokens) — token-budget stress tests live elsewhere.
- Task entities with lifecycle (ASSIGNED_TO, COMPLETED) — deferred to v1.5 after Task as an entity type is exercised.

## Provenance

All names, organizations, locations, events, and works referenced in the docs are invented for this corpus. Any resemblance to real people or companies is coincidental. The real book titles (*The Soul of a New Machine*, *Piranesi*) and real people (Tracy Kidder, Susanna Clarke, Ada Lovelace) are used factually and only to give the preferences scenario realistic content.
