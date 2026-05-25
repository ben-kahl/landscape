# Architecture Doc Draft — Two-Layer Memory Graph for Landscape

  ## Summary

  Landscape should move from a single semantic fact graph to a three-surface memory model:

  - Assertion layer: source-of-truth observed claims from documents and conversations
  - MemoryFact layer: normalized semantic memory objects derived from assertions
  - MEMORY_REL layer: denormalized traversal edges derived from current MemoryFacts for fast multi-hop retrieval

  This preserves raw memory evidence, supports temporal/current-state reasoning, and keeps retrieval simple enough to preserve Landscape’s
  core differentiator: explicit multi-hop traversal.

  ## Design Decisions

  - Observed memory and semantic memory are separate.
    Raw extracted claims must not be collapsed directly into the same structure used for traversal and supersession.
  - Assertion is append-oriented and provenance-rich.
    Assertions preserve what was observed, by whom, from where, and with what confidence, even when normalization is uncertain.
  - MemoryFact is the semantic source of truth.
    Normalized facts such as employment, reporting, preference, family relation, or tool usage live here and carry temporal/current-state
    semantics.
  - MEMORY_REL is retrieval-only infrastructure.
    Traversal should use direct edges derived from MemoryFact so hop counting and BFS stay simple.
  - Supersession applies to MemoryFact, not Assertion.
    Old observations remain preserved; only current semantic memory slots are superseded.
  - Normalization should be conservative.
    If a claim cannot be mapped confidently into a semantic memory family, preserve it as an assertion and omit semantic promotion.
  - Conversation memory is first-class.
    Assertions can originate from Turn as well as Document; the system should not privilege document ingestion over user/agent conversation
    memory.

  ## Concrete Schema Proposal

  ### Node Labels

  | Label | Role | Source of truth | Notes |
  |---|---|---|---|
  | Entity | Canonical people, orgs, projects, tools, locations, concepts, events, tasks, datetimes | Yes | Existing label retained |
  | Document | External source document metadata | Yes | Existing label retained |
  | Chunk | Vectorized text span and provenance anchor | Yes | Existing label retained |
  | Conversation | Session container | Yes | Existing label retained |
  | Turn | Individual user/agent exchange | Yes | Existing label retained |
  | Assertion | One observed claim extracted from a source | Yes | New |
  | MemoryFact | One normalized semantic memory fact | Yes | New |
  | NormalizationRun | Optional ingest/normalization audit record | Optional | Add only if process audit becomes important |

  ### Edge Types

  | Edge | From -> To | Role | Source of truth | Notes |
  |---|---|---|---|---|
  | ASSERTS | Document/Turn -> Assertion | Source provenance | Yes | One source can assert many claims |
  | SUBJECT | Assertion -> Entity | Raw claim subject | Yes | Required |
  | OBJECT | Assertion -> Entity | Raw claim object | Yes | Required |
  | SUPPORTS | Assertion -> MemoryFact | Evidence link | Yes | Many assertions may support one fact |
  | AS_SUBJECT | Entity -> MemoryFact | Semantic fact subject binding | Yes | Required |
  | AS_OBJECT | MemoryFact -> Entity | Semantic fact object binding | Yes | Required |
  | MEMORY_REL | Entity -> Entity | Traversal index edge | Derived | Retrieval-only, points back to MemoryFact |
  | SAME_AS | Entity -> Entity | Canonicalization / alias resolution | Yes | Existing concept retained |

  ## Property Model

  ### Assertion properties

  | Property | Required | Purpose |
  |---|---|---|
  | id | Yes | Stable identifier |
  | raw_relation | Yes | Extractor phrasing or relation string |
  | canonical_candidate | No | Best-effort normalized family before promotion |
  | confidence | Yes | Extraction confidence |
  | subtype | No | Preserved relation nuance |
  | quantity_value | No | Numeric qualifier |
  | quantity_unit | No | Numeric qualifier |
  | quantity_kind | No | Count/duration/price/etc. |
  | time_scope | No | Temporal qualifier from text |
  | asserted_at | Yes | When the assertion was created |
  | observed_time_text | No | Original textual time phrase |
  | source_kind | Yes | document or turn |
  | chunk_refs | No | Bounded source chunk ids for assertion identity/provenance |
  | chunk_ref_count | No | Number of bounded source chunk refs |
  | extraction_model | Yes | Extractor provenance |
  | status | Yes | active, ambiguous, low_confidence, retracted |
  | session_id | No | Conversation scope when source is a turn |
  | turn_id | No | Conversation scope when source is a turn |

  ### Chunk properties

  | Property | Required | Purpose |
  |---|---|---|
  | mentioned_entity_ids | Yes | Canonical entity ids mentioned in this chunk |
  | mentioned_entity_names | Yes | Display names for mentioned entities |

  ### MemoryFact properties

  | Property | Required | Purpose |
  |---|---|---|
  | id | Yes | Stable fact identifier |
  | type | Yes | Canonical semantic relation family |
  | subtype | No | Semantic nuance retained when useful |
  | slot_family | Yes | Family used for supersession policy |
  | slot_key | No | Concrete current-state slot identity |
  | confidence_agg | Yes | Aggregated confidence across supporting assertions |
  | support_count | Yes | Number of supporting assertions |
  | current | Yes | Whether fact is active/current |
  | valid_from | No | Semantic validity start |
  | valid_until | No | Semantic validity end |
  | created_at | Yes | Fact creation timestamp |
  | updated_at | Yes | Fact last normalization update |
  | normalization_policy | Yes | Rule family used to create/update this fact |

  ### MEMORY_REL properties

  | Property | Required | Purpose |
  |---|---|---|
  | memory_fact_id | Yes | Backreference to source MemoryFact |
  | type | Yes | Canonical relation family |
  | subtype | No | Traversal-visible nuance |
  | current | Yes | Traversal filter hook |
  | weight | No | Ranking / traversal weighting |
  | valid_until | No | Optional quick filter for stale edges |

  ## Semantic Relation Families and Supersession Policy

  | Family | Examples | Traversable | Supersession policy |
  |---|---|---|---|
  | Identity / alias | same person, alternate name | Yes | Additive / merge-driven |
  | Employment / affiliation | WORKS_FOR, MEMBER_OF, BELONGS_TO | Yes | Subject-keyed current slot |
  | Role / title | HAS_TITLE | Yes | Object-keyed current slot |
  | Reporting | REPORTS_TO | Yes | Subject-keyed current slot |
  | Residence | LIVES_IN | Yes | Subject-keyed current slot |
  | Location | LOCATED_IN | Yes | Additive |
  | Family / social relation | FAMILY_OF | Yes | Usually additive |
  | Preference | HAS_PREFERENCE | Yes | Subtype-keyed current slot |
  | Attribute / profile fact | HAS_ATTRIBUTE | Yes | Subtype-keyed current slot |
  | Tool / dependency usage | USES | Yes | Additive |
  | Creation / authorship | CREATED | Yes | Additive |
  | Event-time anchor | HAPPENED_ON | Yes | Subject-keyed or event-keyed |
  | Recommendation / mention | RECOMMENDED, DISCUSSED | Usually yes | Additive |
  | Task lifecycle | assignment, completion, due date | Later | Requires dedicated task model |
  | Episodic / weak mention | passing conversational facts | Usually no | Assertion-only unless promoted |

  ## Retrieval and Traversal Design

  - Retrieval should seed from vectors against entities, chunks, and optionally assertions.
  - Multi-hop traversal should walk MEMORY_REL, not the raw Assertion graph and not the fully reified Entity -> MemoryFact -> Entity shape.
  - One semantic hop equals one MEMORY_REL edge.
  - Explanation should dereference memory_fact_id to MemoryFact, then pull supporting Assertions and their source Document/Turn.
  - Traversal should default to current=true facts, with opt-in inclusion of historical facts where the query implies temporal reasoning.
  - Assertions that fail semantic promotion should still be retrievable as evidence/context, but should not expand the graph by default.

  ## Write Path

  1. Ingest source text or turn content.
  2. Extract entities and raw claims.
  3. Create Assertion nodes and attach provenance.
  4. Resolve entities canonically.
  5. Attempt conservative normalization from Assertion into MemoryFact.
  6. Apply supersession rules at the MemoryFact slot level.
  7. Materialize or update the corresponding MEMORY_REL edge for traversable facts.
  8. Write entity and chunk embeddings as today; consider whether assertions need their own vector collection later.

  ## Query Path

  1. Embed query.
  2. Search vectors for seed entities and chunks.
  3. Optionally search assertion text when semantic recall seems sparse.
  4. Expand through MEMORY_REL.
  5. Rank using vector similarity, graph distance, edge confidence, and access-reinforcement with time decay. (Recency, support count, and historical-status signals are tracked on MemoryFact nodes but not yet wired into the scorer — planned backlog.)
  6. For returned paths, load MemoryFact and supporting Assertions for explanation.
  7. Surface both normalized path and raw evidence in API/MCP responses.

  ## Why This Is Better Than Current Landscape

  - Preserves observed claims even when normalization is imperfect.
  - Makes supersession semantically meaningful instead of coupling it to extractor label choice.
  - Separates memory evidence from retrieval infrastructure.
  - Supports conversation-originated memory cleanly.
  - Keeps the retriever simple and performant by traversing direct edges.
  - Makes later improvements to normalization rules non-destructive because assertions remain available.

  ## Risks and Tradeoffs

  - More graph objects and more write complexity.
  - Need consistency guarantees between MemoryFact and MEMORY_REL.
  - Need clear promotion criteria so the semantic layer does not become noisy.
  - Query and API payloads become richer and may need explicit “summary vs evidence” shaping.

  ## Recommended Initial Scope

  - Add Assertion and MemoryFact immediately.
  - Keep Entity, Document, Chunk, Conversation, and Turn unchanged where possible.
  - Materialize MEMORY_REL only for the current semantic families already known to be useful in retrieval.
  - Leave task/commitment modeling out of v1 of this redesign.
  - Preserve existing relation vocabulary initially, but reinterpret it as semantic families rather than raw extractor output.

  ## Acceptance / Review Criteria

  - The doc should make clear that Assertion is the memory source of truth and MEMORY_REL is retrieval-only.
  - Every semantic family listed above must specify whether it is traversable and how it supersedes.
  - The schema must cleanly support both document and turn provenance.
  - The retrieval design must preserve simple multi-hop hop counting.
  - The design must avoid requiring raw assertions to be mutated when current-state memory changes.

  ## Assumptions

  - There is no live production memory to migrate, so correctness and clarity matter more than backward compatibility.
  - Landscape’s primary product goal remains agent memory, not generic document GraphRAG.
  - Multi-hop traversal over a stable semantic layer remains the central differentiator.
  - It is acceptable for some observed claims to remain assertion-only until better normalization exists.
