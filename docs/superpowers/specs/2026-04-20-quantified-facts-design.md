# Quantified Facts Design

## Goal

Preserve numeric and measured personal-memory facts in Landscape's graph so questions like "How many hours did I spend watching documentaries on Netflix last month?" can be answered from structured memory instead of relying only on raw chunk text.

## Problem

Landscape currently extracts relationships as `(subject, relation_type, object)` plus optional `subtype`. That preserves the shape of a fact but drops quantitative qualifiers. For example, "Eric watched 8 hours of Netflix today" can become `Eric -[RELATED_TO or DISCUSSED]-> Netflix`, while `8`, `hours`, and `today` disappear from the graph. LongMemEval contains many questions where the answer is a count, duration, quantity, price, interval, or date-scoped amount, so entity hit rate is not enough.

## Approach

Add quantified qualifiers as properties on existing `:RELATES_TO` edges:

```cypher
(Eric)-[:RELATES_TO {
  type: "DISCUSSED",
  subtype: "watched",
  quantity_kind: "duration",
  quantity_value: 8,
  quantity_unit: "hour",
  time_scope: "today"
}]->(Netflix)
```

This keeps traversal depth unchanged and fits the current extraction, pipeline, and supersession model. Reified `:Fact` nodes are intentionally deferred because most current failures involve one quantity attached to one relation, not multiple independent measurements attached to the same event.

## Data Model

`ExtractedRelation` gains optional fields:

- `quantity_value`: `float | str | None`
- `quantity_unit`: `str | None`
- `quantity_kind`: `str | None`
- `time_scope`: `str | None`

`quantity_kind` uses a small suggested vocabulary but is not a hard enum: `count`, `duration`, `frequency`, `price`, `distance`, `percentage`, `rating`, `measurement`. Unknown values are preserved after light normalization so new facts are not destroyed.

Neo4j `:RELATES_TO` edges store these fields directly. Missing values remain absent or null. Existing graph data needs no migration.

## Extraction

The LLM prompt should explicitly ask for numeric qualifiers on relationships. Worked examples should cover:

- duration: "watched 10 hours of documentaries on Netflix last month"
- count: "owns three bikes"
- count with context: "packed 7 shirts for a 5-day trip"

The prompt must keep the existing closed relation vocabulary. Quantities are not relation types. They are edge qualifiers.

## Write Path

The ingestion pipeline passes quantity fields from `ExtractedRelation` to `neo4j_store.upsert_relation`. Agent write-back can accept the same optional fields later, but the first implementation should keep the MCP tool surface unchanged unless needed by tests. This avoids making agents provide quantity metadata before the extractor proves useful.

`upsert_relation` handles quantities like `subtype`: a new non-null value updates the live exact edge during reinforcement, while null input preserves an existing value. Supersession identity remains governed by `FUNCTIONAL_KEYS`; quantity fields do not participate in edge identity in this first slice.

## Retrieval

Graph expansion should return quantity fields for each path edge. The HTTP and MCP query surfaces should include those qualifiers alongside `path_edge_types` and `path_edge_subtypes`, and LangChain document formatting should render them in human-readable form:

```text
Netflix (Technology) [1 hops via DISCUSSED[watched] {duration=10 hour, scope=last_month}]
```

The LangChain retriever should also surface raw chunks returned by `retrieve()`. Numeric answers often remain most safely answerable from verbatim source text even after graph qualifiers exist.

## Testing

Add focused unit/integration coverage that does not depend on stochastic LLM extraction:

- Pydantic schema accepts quantified relation fields.
- `upsert_relation` writes quantity fields on create.
- `upsert_relation` reinforces quantity fields with non-null-wins behavior.
- Retrieval path expansion carries quantity metadata.
- HTTP query response includes path edge qualifiers.
- LangChain retriever returns chunk documents as well as entity documents.

LLM extraction-quality tests can be added after the prompt settles, but they should not gate the first implementation because local LLM output is non-deterministic.

## Out Of Scope

- Reified `:Fact` nodes.
- Quantity-based Cypher search helpers.
- Numeric comparison queries such as "more than 5 hours".
- Unit conversion and canonicalization beyond light string normalization.
- MCP `add_relation` quantity parameters.
- Official LongMemEval answer judging.

## Success Criteria

- A directly inserted or extracted quantified relation preserves `quantity_value`, `quantity_unit`, `quantity_kind`, and `time_scope` on the graph edge.
- Retrieval result objects can carry those qualifiers from graph expansion to HTTP/LangChain surfaces.
- Raw chunk text remains visible through LangChain retrieval.
- Existing tests continue to pass aside from already-known baseline failures unrelated to this work.
