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
Neo4j, deduplicates results, and ranks them with vector, graph-distance, fact
currentness, and recency signals.

## Data Model

### Neo4j

Primary node labels:

| Label | Purpose |
|---|---|
| `Entity` | Canonical people, organizations, projects, tools, locations, concepts, and artifacts |
| `Document` | Source document metadata and ingestion provenance |
| `Chunk` | Source text spans with positions and embedding references |
| `Conversation` | Agent/user session container |
| `Turn` | Individual conversation turns used for session-scoped memory |
| `Assertion` | Raw extracted statement anchored to a document or turn |
| `MemoryFact` | Normalized, queryable fact version used for supersession and retrieval |
| `Alias` | Alternate surface form linked to a canonical entity |

Primary relationships:

| Relationship | Purpose |
|---|---|
| `EXTRACTED_FROM` | Entity-to-document provenance from ingest |
| `MENTIONED_IN` | Entity-to-turn provenance for conversation memory |
| `INGESTED_IN` | Document-to-turn write provenance |
| `HAS_TURN` | Conversation-to-turn containment |
| `ASSERTS` | Document/turn source to raw assertion |
| `MENTIONS_CHUNK` | Assertion-to-chunk provenance |
| `SUBJECT_ENTITY` | Assertion subject binding |
| `OBJECT_ENTITY` | Assertion object binding |
| `SUPPORTS` | Assertion to normalized memory fact |
| `AS_SUBJECT` | Entity to memory fact subject binding |
| `AS_OBJECT` | Memory fact to entity object binding |
| `MEMORY_REL` | Traversable, current fact edge used by retrieval |
| `SAME_AS` | Alias-to-canonical entity resolution link |

The redesigned graph separates raw extraction from queryable memory:

- `Assertion` stores the source-anchored statement exactly as extracted.
- `MemoryFact` stores the normalized family/subtype form used by ranking,
  supersession, and traversal.
- `MEMORY_REL` is the live graph edge the retriever expands across. It carries
  the current fact state plus the normalized family metadata.

This split preserves provenance while avoiding the old one-edge-does-everything
model. A single source can yield multiple assertions, and a single assertion
can support a normalized fact that later gets superseded without losing the
original evidence trail.

### Qdrant

Landscape uses Qdrant collections for vector lookup:

| Collection | Payload |
|---|---|
| `entities` | Neo4j element ID, entity name, type, source document, timestamp |
| `chunks` | Document-scoped chunk ID, document ID, source document, position |

Entity vectors give graph traversal a semantic starting point. Chunk vectors
provide source text context when graph facts alone are too sparse.

Chunk identity is now scoped to the owning document and chunk position rather
than a global `content_hash`. The stable key is:

`chunk_id = f"{doc_id}:{position}:{content_hash}"`

This prevents identical text in different documents from collapsing into one
Neo4j `:Chunk` node or one Qdrant point. It also means older chunk data must be
rebuilt or reingested after this migration, because the identity contract has
changed.

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

Supersession is modeled at the `MemoryFact` layer, not on raw assertions.
`Assertion` records remain immutable source evidence. When a newer extraction
conflicts with an existing current fact, Landscape creates a new versioned
`MemoryFact`, marks the prior version superseded, and only projects the current
version into `MEMORY_REL` for traversal.

This keeps provenance intact while ensuring retrieval only walks live facts.
Non-conflicting facts remain additive, and superseded versions stay available
for audit and temporal reasoning without appearing in the current graph view.

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
- The canonical verification workflow is documented in `README.md` and should
  stay aligned with the current CI-safe regression gate as commands evolve.
- Expanded ingestion modes as the next major feature area: richer document
  inputs, drive-platform integrations such as Google Drive, automatic
  conversation capture, and visual/multimodal ingestion through OCR and local
  multimodal models.
