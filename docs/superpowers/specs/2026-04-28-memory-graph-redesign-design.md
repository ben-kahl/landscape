# Landscape Memory Graph Redesign

## Goal

Replace Landscape's current `RELATES_TO`-centric graph with a three-surface
memory model that:

- preserves raw observed claims independently from normalized semantic memory
- applies deterministic, family-specific supersession at the semantic layer
- keeps multi-hop traversal simple enough for agents to use reliably
- preserves a stable enough retrieval contract to A/B benchmark the redesign
  against the current implementation

## Why This Direction

The current model overloads one graph shape to do four jobs at once:

- preserve raw evidence
- represent current semantic memory
- model temporal supersession
- provide the edges used for traversal

That is the root cause behind several recurring problems in the current code:

- supersession depends too heavily on extractor-facing relation labels
- provenance is attached to `RELATES_TO` edges rather than to explicit claims
- unresolved or low-confidence claims are awkward to preserve cleanly
- retrieval walks the same structure that ingestion mutates for semantic updates

The redesign separates those concerns:

- `Assertion` preserves what was observed
- `MemoryFact` represents normalized semantic truth
- `MEMORY_REL` is a rebuildable traversal index over current semantic truth

## Alternatives Considered

### 1. Keep patching the current `RELATES_TO` model

This is the smallest code change, but it keeps evidence, truth, and traversal
coupled. It does not solve the architectural problem; it only moves the bugs
around.

### 2. Move directly from raw ingestion to `MemoryFact` without an `Assertion` layer

This simplifies the graph, but it throws away the key reason to redesign:
preserving claim-level evidence even when normalization or entity resolution is
uncertain.

### 3. Recommended: three-surface model

This adds write-path complexity, but it cleanly separates evidence, semantic
truth, and traversal infrastructure. That is the only option that satisfies the
design decisions agreed during the grilling session.

## Current Code Constraints

The redesign must replace assumptions that currently live in these surfaces:

- `src/landscape/storage/neo4j_store.py`
  - relation persistence
  - alias stub creation
  - graph BFS over `RELATES_TO`
- `src/landscape/pipeline.py`
  - document ingest write path
- `src/landscape/conversation_ingestion.py`
  - turn-based ingest write path
- `src/landscape/writeback.py`
  - explicit agent-authored entities and relations
- `src/landscape/retrieval/query.py`
  - vector seeding and graph expansion
- `src/landscape/cli/graph.py`
  - neighborhood traversal and graph status reporting
- `src/landscape/extraction/schema.py`
  - canonical family vocabulary and functional semantics
- `tests/` plus benchmark fixtures
  - many tests assume `RELATES_TO` is both the truth model and traversal model

The redesign should change the underlying model sharply, but preserve the first
round benchmark corpus, query set, and high-level retrieval contract so A/B
results are interpretable.

## Design Decisions

The following decisions are fixed for this redesign:

- Raw observed claims are a hard requirement and are not replaced by the vector
  store.
- Assertions are persisted even when resolution or normalization is imperfect.
- Assertions always store raw subject and object text.
- `MemoryFact` always anchors to at least one canonical `Entity`.
- `MemoryFact` may be entity-valued or scalar-valued, but the initial promoted
  family set is mostly entity-valued.
- Promotion is synchronous for supported families, with optional later async
  re-normalization.
- Fact identity and supersession are deterministic and family-specific.
- Multiple assertions may support one memory fact.
- `MemoryFact` is versioned; it is not overwritten in place.
- `MEMORY_REL` is derived, rebuildable, and contains current edges only.
- Traversal expands only over `MEMORY_REL`, never over raw assertions.
- Retrieval remains vector-seeded at the `Entity` and `Chunk` level in v1.
- Community detection is derived experimental infrastructure, not a
  correctness dependency.
- Cutover is sharp. There is no long-lived dual-write period.

## In Scope

- new source-of-truth `Assertion` nodes
- new source-of-truth `MemoryFact` nodes
- new derived `MEMORY_REL` traversal edges
- separate `Alias` nodes instead of alias stub `Entity` nodes
- deterministic promotion rules for a closed v1 family catalog
- current-only default traversal with optional historical mode
- fixture, benchmark, and retrieval updates required by the new model

## Out of Scope

- task or workflow lifecycle modeling
- true n-ary facts
- assertion vector embeddings
- community-based retrieval affecting correctness in v1
- long-lived compatibility with the old `RELATES_TO` truth model
- migration of live production memory

## Core Schema

### Node Labels

| Label | Role | Source of truth | Identity contract |
|---|---|---|---|
| `Entity` | Canonical people, orgs, repos, projects, teams, tools, concepts | Yes | Stable application id; must not rely on Neo4j `elementId()` |
| `Alias` | Alternate surface form for a canonical entity | Yes | Stable id derived from canonical entity id plus normalized alias text |
| `Document` | External source document | Yes | Existing stable document id |
| `Chunk` | Source text chunk and vector anchor | Yes | Existing `doc_id:position:content_hash` id |
| `Conversation` | Session container | Yes | Existing `session_id` |
| `Turn` | Conversation turn | Yes | Existing `session_id:turn_id` |
| `Assertion` | One observed claim from a source | Yes | Deterministic source-local fingerprint |
| `MemoryFact` | One normalized semantic fact version | Yes | Unique version id plus stable family-specific identity keys |

`Community` is not part of the v1 source-of-truth schema. If introduced later,
it is derived retrieval infrastructure only.

### Relationship Types

| Relationship | From -> To | Purpose | Source of truth |
|---|---|---|---|
| `ASSERTS` | `Document` or `Turn` -> `Assertion` | Claim provenance anchor | Yes |
| `MENTIONS_CHUNK` | `Assertion` -> `Chunk` | Fine-grained evidence anchor | Yes when available |
| `SUBJECT_ENTITY` | `Assertion` -> `Entity` | Resolved subject entity | Optional |
| `OBJECT_ENTITY` | `Assertion` -> `Entity` | Resolved object entity | Optional |
| `SUPPORTS` | `Assertion` -> `MemoryFact` | Evidence supports semantic fact | Yes |
| `AS_SUBJECT` | `Entity` -> `MemoryFact` | Semantic subject anchor | Yes |
| `AS_OBJECT` | `MemoryFact` -> `Entity` | Semantic object anchor | Optional |
| `SAME_AS` | `Alias` -> `Entity` | Alias to canonical identity link | Yes |
| `MEMORY_REL` | `Entity` -> `Entity` | Derived traversal edge backed by a current `MemoryFact` | Derived |

Existing structural provenance edges such as `PART_OF`, `HAS_TURN`,
`NEXT_TURN`, and `INGESTED_IN` remain valid and do not need to be redesigned in
this slice.

## Identity and Constraint Rules

The redesign requires stable application-level ids for every first-class node
that participates in ingest, retrieval, or cross-store references.

### Required uniqueness

- `Entity.id`
- `Alias.id`
- `Document.id`
- `Chunk.id`
- `Conversation.id`
- `Turn.id`
- `Assertion.id`
- `MemoryFact.id`

### `Assertion` identity

`Assertion` deduplicates by source-local occurrence, not by global semantic
similarity.

The deterministic fingerprint must include:

- source kind and source id
- raw subject text
- raw relation text
- raw object text
- subtype if present
- extracted qualifier payload
- evidence offsets when available

Implications:

- re-ingesting the same source claim is idempotent
- the same claim from two different sources produces two assertions
- a new extraction event from the same source does not create a duplicate node
  unless the source-local claim payload is materially different

### `MemoryFact` identity

`MemoryFact` needs two identity concepts:

- `fact_key`
  - family-specific semantic identity for one normalized fact
- `slot_key`
  - family-specific current-state slot used for supersession

`MemoryFact.id` identifies one versioned fact record.

Examples:

- `WORKS_FOR`
  - `fact_key = WORKS_FOR:{subject_entity_id}:{object_entity_id}`
  - `slot_key = WORKS_FOR:{subject_entity_id}`
- `USES`
  - `fact_key = USES:{subject_entity_id}:{object_entity_id}`
  - `slot_key = USES:{subject_entity_id}:{object_entity_id}`
- scalar family example, deferred from the initial catalog:
  - `HAS_TITLE`
  - `fact_key = HAS_TITLE:{subject_entity_id}:{normalized_value}`
  - `slot_key = HAS_TITLE:{subject_entity_id}`

`fact_key` determines whether a new assertion reinforces an existing semantic
fact. `slot_key` determines whether a new fact supersedes a current one.

## Assertion Contract

Every `Assertion` stores the raw extracted claim payload even when entity
resolution succeeds.

### Required properties

- `id`
- `source_kind`
  - `document` or `turn`
- `raw_subject_text`
- `raw_relation_text`
- `raw_object_text`
- `confidence`
- `status`
  - `active`, `ambiguous`, `low_confidence`, or `retracted`
- `created_at`

### Optional properties

- `family_candidate`
- `subtype`
- `quantity_value`
- `quantity_unit`
- `quantity_kind`
- `time_scope`
- `observed_time_text`
- `chunk_span_start`
- `chunk_span_end`
- `evidence_excerpt`
- `extraction_model`
- `session_id`
- `turn_id`

### Evidence storage rule

Assertions do not copy whole chunks by default.

The preferred provenance payload is:

- source `Document` or `Turn`
- zero, one, or many `MENTIONS_CHUNK` links
- offsets into the chunk or source text when available
- optional short `evidence_excerpt` cache

This keeps assertions self-describing without making Qdrant chunk layout the
source-of-truth provenance contract.

## MemoryFact Contract

`MemoryFact` is the semantic truth layer and is versioned append-and-supersede,
not overwrite-in-place.

### Required properties

- `id`
- `family`
- `fact_key`
- `slot_key`
- `current`
- `support_count`
- `confidence_agg`
- `normalization_policy`
- `created_at`
- `updated_at`

### Optional properties

- `subtype`
- `valid_from`
- `valid_until`
- `value_type`
- `value_text`
- `value_number`
- `value_unit`
- `value_time`

### Structural rules

- `AS_SUBJECT` is required.
- `AS_OBJECT` is required for entity-valued families.
- scalar-valued families store the value on the fact rather than creating fake
  object entities.
- `current=true` means the fact is live for default traversal and ranking.
- superseded facts stay queryable for historical mode and explanation.

## Derived `MEMORY_REL` Contract

`MEMORY_REL` is current-only traversal infrastructure.

### Required properties

- `memory_fact_id`
- `family`
- `current`

### Optional properties

- `subtype`
- `weight`
- `valid_until`

### Rules

- `MEMORY_REL` exists only for traversable current facts.
- `MEMORY_REL` is never authoritative history.
- historical reasoning flows through versioned `MemoryFact`s, not through stale
  traversal edges.
- `MEMORY_REL` must be rebuildable from current `MemoryFact` state.

## Alias Model

Aliases are separate identity records, not stub entities and not a plain
`aliases[]` property.

### Rules

- canonical nodes remain `:Entity`
- alternate names are `:Alias`
- `(:Alias)-[:SAME_AS]->(:Entity)`
- alias provenance may come from assertions or resolver metadata
- retrieval and resolution may index both `Entity.name` and `Alias.name`
- graph traversal walks canonical entities only

This directly replaces the current alias-stub pattern in
`src/landscape/storage/neo4j_store.py`.

## Family Catalog

The family catalog is closed in v1. Claims outside this catalog remain
assertion-only until explicitly supported.

### Core product families

These are the semantic families the redesign is aiming at for agentic coding and
project-development memory:

| Family | Object kind | Traversable | `fact_key` | `slot_key` | Current cardinality |
|---|---|---|---|---|---|
| `WORKS_FOR` | Entity | Yes | subject + object | subject | one current employer per subject |
| `MEMBER_OF` | Entity | Yes | subject + object | subject + object | additive |
| `BELONGS_TO` | Entity | Yes | subject + object | subject | one current parent by default |
| `REPORTS_TO` | Entity | Yes | subject + object | subject | one current manager per subject |
| `WORKS_ON` | Entity | Yes | subject + object | subject + object | additive |
| `MAINTAINS` | Entity | Yes | subject + object | subject + object | additive |
| `OWNS` | Entity | Yes | subject + object | subject + object | additive |
| `USES` | Entity | Yes | subject + object | subject + object | additive |
| `DEPENDS_ON` | Entity | Yes | subject + object | subject + object | additive |
| `CREATED` | Entity | Yes | subject + object + subtype | subject + object + subtype | additive |

### Benchmark-parity carryover families

The first A/B benchmark must stay comparable to the current corpus and query
set. For that reason, the first cut also carries these currently-used families
as promotable instead of forcing them assertion-only:

| Family | Reason retained in first cut | Traversable |
|---|---|---|
| `LEADS` | Existing fixtures and retrieval cases rely on it | Yes |
| `APPROVED` | Existing multi-hop benchmark cases rely on it | Yes |
| `LOCATED_IN` | Existing relation vocabulary and fixtures rely on it | Yes |

These carryover families are still part of the closed v1 catalog. They are
present to preserve A/B comparability, not because they are the final product
center of gravity.

### Explicitly deferred

- `RELATED_TO`
  - remains assertion-only in v1 because it is too vague to support stable
    identity or traversal semantics
- task and workflow families
- scalar families such as `HAS_TITLE` and `HAS_ATTRIBUTE`
  - the schema supports scalar facts, but they are not required in the first
    promoted catalog

## Normalization Contract

Normalization is staged and deterministic.

### 1. LLM extraction

The model extracts raw claim candidates and produces:

- raw subject text
- raw relation text
- raw object text
- subtype
- qualifiers
- confidence

### 2. Deterministic family mapping

Code maps the extracted relation into the closed family catalog or marks the
claim assertion-only.

Inputs may include:

- synonym tables
- current canonical vocabulary
- narrow lexical rules
- limited pattern matching

Regex is allowed only for narrow value parsing, not for semantic identity or
supersession reasoning.

### 3. Deterministic resolution and value parsing

Code resolves canonical subject and object entities when possible and classifies
the object as entity-valued or scalar-valued.

Promotion requirements:

- subject entity must resolve canonically
- object entity must resolve for entity-valued families
- scalar value must validate for scalar-valued families

### 4. Family-specific fact builder

Each family defines:

- required inputs
- object kind
- `fact_key`
- `slot_key`
- whether multiple current facts are allowed
- whether the family materializes `MEMORY_REL`

### 5. Promotion decision

If the claim is outside the closed family catalog, missing required inputs, or
ambiguous after deterministic normalization:

- keep the assertion
- do not create a `MemoryFact`
- do not create a `MEMORY_REL`

Assertion-only is a first-class supported state.

## Promotion and Versioning Rules

### Reinforcement

If a new assertion normalizes to an existing current `MemoryFact` with the same
`fact_key`:

- keep the existing current fact version
- attach the assertion with `SUPPORTS`
- increment `support_count`
- recompute `confidence_agg`
- update timestamps as needed

### Supersession

If a new assertion normalizes to a different `fact_key` within a single-current
slot:

- create a new `MemoryFact` version with `current=true`
- mark the previously current fact in that slot `current=false`
- set the old fact's `valid_until`
- attach the new assertion to the new fact
- update derived `MEMORY_REL` edges so only the new current fact remains
  traversable

### Additive families

If the family allows multiple current facts:

- create a new current `MemoryFact` if no current fact exists for that
  `fact_key`
- do not supersede sibling facts with different `fact_key`s

## Retrieval Contract

The redesign must preserve the first-round retrieval contract closely enough to
benchmark it against the current implementation.

### Query pipeline in v1

1. Embed the user query.
2. Search vectors against `entities` and `chunks`.
3. Promote chunk hits to nearby entities using the existing chunk-to-entity
   bridge.
4. Expand graph candidates over `MEMORY_REL` only.
5. Rank using existing-style signals such as vector similarity, graph distance,
   recency, and reinforcement.
6. Load `MemoryFact`s for semantic explanation.
7. Load supporting `Assertion`s plus source `Document` or `Turn` and chunk
   evidence.

### Important boundaries

- No assertion vector collection in v1.
- No direct traversal over `Assertion`.
- Default traversal uses current facts only.
- Historical facts are included only through an explicit flag or mode.
- Assertion-only claims may surface through chunk/document evidence, but they do
  not create hop-expansion paths.

## Community Detection

Community detection is not a correctness dependency in v1.

### Contract

- derived only
- asynchronous only
- experimental only
- does not affect whether core retrieval is considered correct

If community retrieval is later added, its first role should be seed widening
and reranking rather than graph path expansion.

## Write Path

1. Ingest source text or conversation turn.
2. Create or reuse `Document`, `Chunk`, `Conversation`, and `Turn` records as
   today.
3. Extract raw claims.
4. Create or merge `Assertion` nodes using source-local identity.
5. Resolve entities canonically.
6. Attach optional `SUBJECT_ENTITY` and `OBJECT_ENTITY` links when resolution
   succeeds.
7. Attempt family-specific normalization.
8. Create or update `MemoryFact` versions.
9. Materialize or replace current `MEMORY_REL` edges for traversable families.
10. Write or update entity and chunk embeddings as today.

## Query and API Response Shaping

The redesign changes internal storage, but callers should still receive a
usable hybrid-memory response.

The intended response structure is:

- ranked semantic result
- graph distance and path metadata
- normalized current `MemoryFact`
- supporting `Assertion`s
- source `Document` or `Turn`
- supporting `Chunk`s where available

The API does not need to expose every internal field by default, but it should
be able to explain:

- what semantic fact was used
- what raw claims supported it
- what source text anchored those claims

## Cutover Strategy

This redesign is a sharp cutover.

### Rules

- no long-lived dual-write
- no mixed authoritative truth models
- fixtures and tests are rewritten to the new schema
- old `RELATES_TO`-based write and traversal logic is retired rather than kept
  as a second truth model

### Migration assumptions

- development and benchmark data may be wiped and reseeded
- there is no production customer memory requiring backward-compatible
  migration logic
- tiny shims are acceptable only for test harnesses or temporary tooling during
  the branch, not as a permanent runtime contract

## Benchmark Contract

The first benchmark after the redesign must keep these variables stable:

- same corpus
- same query set
- same top-k and success metrics
- same chunk surfacing expectations
- same multi-hop tasks

The purpose of the first A/B is to evaluate the schema change, not to conflate
schema change with a new retrieval regime.

## Risks and Mitigations

### Risk: write path gets significantly more complex

Mitigation:

- keep the family catalog closed
- keep normalization rule-driven
- make assertion-only a valid terminal state

### Risk: `MEMORY_REL` drifts from `MemoryFact`

Mitigation:

- treat `MEMORY_REL` as derived only
- keep it current-only
- make rebuild from `MemoryFact` part of the design contract

### Risk: benchmark parity families bloat the first cut

Mitigation:

- keep the carryover set explicit and small
- mark them as A/B parity scope, not unbounded permanent expansion

### Risk: history makes traversal noisy

Mitigation:

- default to current-only traversal
- expose history only through explicit mode or explanation

## Acceptance Criteria

The redesign is only considered correctly specified if all of the following are
true:

- `Assertion` is the raw evidence source of truth.
- `MemoryFact` is the normalized semantic source of truth.
- `MEMORY_REL` is clearly derived and current-only.
- assertion persistence does not depend on successful normalization.
- every promoted family defines `fact_key`, `slot_key`, and current-cardinality
  semantics.
- traversal semantics do not depend on raw assertions.
- the spec supports both document and turn provenance.
- aliases are modeled separately from canonical entities.
- the cutover strategy is sharp rather than dual-write.
- the initial benchmark contract remains stable enough for A/B comparison.

## Result

This redesign should replace the current core graph model. It is not just a
cleanup of the existing `RELATES_TO` schema; it is the new architectural
direction for Landscape's memory graph.
