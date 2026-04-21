# Landscape Architecture

Landscape is a local-first memory system for AI agents. It combines a graph
database, a vector database, and local LLM extraction so agents can retrieve
facts by semantic similarity and by explicit relationships between entities.

The target use case is multi-hop memory: questions where the answer is not in a
single semantically similar chunk, but can be reached by connecting entities
across documents or conversations.

## System Overview

Landscape has four main layers:

| Layer | Responsibility |
|---|---|
| FastAPI service | HTTP ingestion and query endpoints |
| Ingestion pipeline | Chunking, LLM extraction, entity resolution, persistence |
| Storage | Neo4j for graph facts, Qdrant for vector search |
| Agent interfaces | MCP tools, LangChain retriever, CLI |

The ingestion path extracts structured facts from free text and writes them to
both storage systems. The retrieval path starts from a semantic query, finds
candidate entities and chunks in Qdrant, expands from matched entities through
Neo4j, deduplicates results, and ranks them with vector, graph-distance, and
recency signals.

## Data Model

### Neo4j

Primary node labels:

| Label | Purpose |
|---|---|
| `Entity` | Named people, organizations, projects, tools, locations, concepts, and artifacts |
| `Document` | Source document metadata and ingestion provenance |
| `Chunk` | Source text spans with positions and embedding references |
| `Conversation` | Agent/user session container |
| `Turn` | Individual conversation turns used for session-scoped memory |

Primary relationships:

| Relationship | Purpose |
|---|---|
| `RELATES_TO` | Extracted subject-predicate-object fact |
| `EXTRACTED_FROM` | Provenance from facts/entities to source chunks or documents |
| `MENTIONED_IN` | Conversation turn references |
| `INGESTED_IN` | Document-to-turn write provenance |
| `SAME_AS` | Entity-resolution link |

`RELATES_TO` edges carry the important memory properties:

- `type`: canonical relationship type, such as `WORKS_FOR`, `LEADS`, or `USES`
- `subtype`: optional natural-language qualifier for richer semantics
- `confidence`: extraction confidence
- `source_docs`: provenance trail
- `valid_from` and `valid_until`: temporal validity
- `quantity_value`, `quantity_unit`, `quantity_kind`, `time_scope`: numeric and
  temporal qualifiers

Quantified relation fields preserve facts such as:

- "Eric watched 8 hours of Netflix today"
- "Maya owns three bikes"
- "The contract is worth $500"
- "The team meets twice per week"

These stay attached to the graph edge instead of being lost during
subject-predicate-object extraction.

### Qdrant

Landscape uses Qdrant collections for vector lookup:

| Collection | Payload |
|---|---|
| `entities` | Neo4j element ID, entity name, type, source document, timestamp |
| `chunks` | Neo4j chunk ID, document ID, source document, position |

Entity vectors give graph traversal a semantic starting point. Chunk vectors
provide source text context when graph facts alone are too sparse.

## Ingestion Flow

1. Text enters through the API, CLI, MCP `remember` tool, or direct writeback
   helpers.
2. The chunker splits long inputs while preserving source positions.
3. The local LLM extracts entities and relations using a structured schema.
4. Entity and relationship types are normalized to a canonical vocabulary.
5. Entity resolution merges obvious duplicates using type-aware matching.
6. Entities, chunks, documents, and relationships are written to Neo4j.
7. Entity and chunk embeddings are written to Qdrant with Neo4j cross-references.

The extractor is intentionally schema-first. This makes downstream retrieval
more reliable than storing only raw text or model-generated prose summaries.

## Retrieval Flow

1. Embed the user query.
2. Search Qdrant for semantically similar entities and chunks.
3. Expand from matched entities in Neo4j up to the requested hop depth.
4. Include entities connected to high-scoring chunks.
5. Remove duplicate entity hits.
6. Score results with vector similarity, graph distance, recency, and
   reinforcement signals.
7. Return ranked entities, path metadata, edge qualifiers, and source chunks.

The graph expansion is the main differentiator. If a query starts near one node
but the answer lives two or three relationships away, Neo4j can retrieve that
path directly.

## Supersession Model

Landscape supports temporal updates through valid-time properties on relation
edges. When a new fact conflicts with a live functional relationship, the old
edge receives `valid_until` and the new edge becomes current.

Only functional relationship types trigger this behavior by default. For
example, `WORKS_FOR` and `REPORTS_TO` are treated as at-most-one-current-value
per subject, while `USES`, `APPROVED`, and `LOCATED_IN` are additive because a
project can use several tools, a person can approve several items, and an
organization can have several locations.

This avoids a common graph-memory failure mode where a second true fact
incorrectly deletes or hides the first true fact.

## Relationship Vocabulary

Local LLMs are useful extractors, but they can vary relationship labels across
runs. Landscape reduces that drift with a canonical vocabulary and synonym
normalization before persistence.

Examples:

| Raw phrasing | Canonical type |
|---|---|
| `EMPLOYED_BY` | `WORKS_FOR` |
| `MANAGES` | `LEADS` |
| `PART_OF` | `MEMBER_OF` |
| `BUILT_WITH` | `USES` |

Unknown relationship types are preserved rather than dropped. That protects
novel semantics, but it also means unknown near-synonyms may not participate in
supersession until they are normalized or clustered.

## Benchmark Notes

The killer-demo benchmark is a small, controlled corpus designed to isolate the
multi-hop retrieval behavior. It contains seven questions across 1-hop, 2-hop,
and 3-hop cases.

The ChromaDB baseline is evaluated at chunk level, while Landscape is evaluated
at entity level. That means MRR values are not directly comparable across the two
systems. The fair comparison is per-question success: vector-only retrieval
misses the 3-hop chain because no single chunk contains the full answer path,
while hybrid graph retrieval can traverse it.

LongMemEval scripts are included for broader memory-style evaluation, but their
results are more sensitive to local LLM extraction quality, model choice, and
hardware speed.

## Phase 3.5 Exit Criteria

The normative criteria live in [README.md](../README.md#phase-35-exit-criteria). This section keeps the same terminology but does not duplicate the policy text.

### CI Required

- See [README.md](../README.md#ci-required) for the command-level CI Required criteria.

### Local Required

- See [README.md](../README.md#local-required) for the command-level Local Required criteria.

### Exit Condition

- See [README.md](../README.md#exit-condition) for the phase 3.5 exit condition and phase 4 handoff point.

## Known Limitations

### Relation-Type Synonym Drift

The canonical vocabulary handles common synonyms, but novel or ambiguous
relationship labels can still pass through unchanged. These labels remain useful
for retrieval, but they may not trigger supersession or type-specific ranking
logic.

Future work: embed relationship labels and cluster novel types into canonical
groups when similarity is high enough.

### Directional Synonyms

Some relationship synonyms imply a direction change. For example, `APPROVED_BY`
is not just a synonym for `APPROVED`; it also reverses subject and object.
Landscape currently normalizes relationship labels but does not automatically
invert relation direction during ingestion.

Future work: add direction-aware normalization rules for high-confidence cases.

### Entity Resolver Strictness

The resolver requires compatible entity types before merging. This avoids bad
merges, but it can create duplicates when the extractor and agent writeback use
different labels for the same real-world entity.

Future work: improve type compatibility rules and use aliases more aggressively
for person and organization entities.

### Extraction Quality Depends On The Local Model

Landscape is local-first, so extraction quality depends on the Ollama model in
use. Smaller models are fast and cheap, but they can miss implicit relationships,
invent inconsistent labels, or omit qualifiers.

Future work: benchmark extraction profiles across local models and expose
profile-specific recommendations.

## Future Work

Landscape is past the "build the basic stack" stage. The remaining work before
phase 4 is primarily about quality hardening, evaluation clarity, and
operational polish.

- Direction-aware relationship normalization so inverse forms such as
  `APPROVED_BY` can map correctly instead of only by string synonym.
- Stronger entity resolution across type variants and aliases, especially for
  agent-authored write-back that does not match ingestion-time labels exactly.
- Semantic relation clustering / supersession hardening so novel relation labels
  can participate in canonical reasoning and temporal updates.
- Automatic agent-conversation ingestion so useful conversational memory does
  not depend on explicit `add_entity` / `add_relation` calls for every fact.
- Benchmark hardening and reproducibility beyond the current killer-demo and
  LongMemEval smoke harness.
- CI or a formal verification workflow so the current integration surface has a
  stable regression gate.
- Expanded ingestion modes as the next major feature area: richer document
  inputs, drive-platform integrations such as Google Drive, automatic
  conversation capture, and visual/multimodal ingestion through OCR and local
  multimodal models.
