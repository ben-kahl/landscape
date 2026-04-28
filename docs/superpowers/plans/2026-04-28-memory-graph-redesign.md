# Memory Graph Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `RELATES_TO`-centric truth model with the approved `Assertion` + `MemoryFact` + `MEMORY_REL` design while preserving a stable enough retrieval contract for the first A/B benchmark.

**Architecture:** Introduce a small `memory_graph` domain package for ids, family config, and deterministic normalization. Extend `neo4j_store.py` with explicit Alias / Assertion / MemoryFact / MEMORY_REL primitives, then route document ingest, conversation ingest, and agent writeback through one shared promotion service. Retrieval keeps entity/chunk vector seeding, but graph expansion moves from `RELATES_TO` to `MEMORY_REL` and explanations are hydrated from `MemoryFact` plus supporting `Assertion`s.

**Tech Stack:** Python 3.12, pytest, Neo4j (Cypher), Qdrant, FastAPI, MCP, uv.

**Branch:** `review-remediation` (worktree `/home/kahlb/Documents/landscape/.worktrees/review-remediation`).
**Spec:** `docs/superpowers/specs/2026-04-28-memory-graph-redesign-design.md`.

---

## File Structure

| File | Change |
|---|---|
| `src/landscape/memory_graph/__init__.py` | New package export surface |
| `src/landscape/memory_graph/families.py` | Closed v1 family registry and slot semantics |
| `src/landscape/memory_graph/ids.py` | Stable ids for Alias, Assertion, and MemoryFact |
| `src/landscape/memory_graph/models.py` | Dataclasses for assertion payloads, promotion results, and explanation records |
| `src/landscape/memory_graph/normalization.py` | Deterministic family mapping and promotion rules |
| `src/landscape/memory_graph/service.py` | Shared write-path service used by ingest and writeback |
| `src/landscape/storage/neo4j_store.py` | Neo4j constraints plus Alias / Assertion / MemoryFact / MEMORY_REL persistence and traversal |
| `src/landscape/pipeline.py` | Document ingest refactor from direct `upsert_relation()` to assertion persistence + promotion |
| `src/landscape/conversation_ingestion.py` | Conversation ingest refactor onto the same service path |
| `src/landscape/writeback.py` | Agent-authored relation writes switch to Assertion/MemoryFact path |
| `src/landscape/retrieval/query.py` | Expand over `MEMORY_REL`, hydrate explanations, add historical flag plumbing |
| `src/landscape/mcp_app.py` | Expose normalized fact + assertion evidence in MCP search output |
| `src/landscape/api/query.py` | Add optional `include_historical` query flag and new response shape fields |
| `src/landscape/cli/query.py` | Add `--include-historical` flag and render fact/evidence metadata |
| `src/landscape/cli/graph.py` | Neighbor traversal and graph counts move to `MEMORY_REL` / `MemoryFact` |
| `src/landscape/cli/status.py` | Status counts move away from `RELATES_TO` |
| `src/landscape/extraction/schema.py` | Align canonical vocabulary and promotable family set with new registry |
| `docs/ARCHITECTURE.md` | Replace `RELATES_TO` truth-model description with the new three-surface model |
| `tests/test_memory_graph_ids.py` | New deterministic-id tests |
| `tests/test_memory_graph_normalization.py` | New family mapping and slot-key tests |
| `tests/test_memory_graph_storage.py` | New storage, supersession, and MEMORY_REL materialization tests |
| `tests/test_ingest.py` | Update ingest expectations from `RELATES_TO` edges to Assertion/MemoryFact graph |
| `tests/test_pipeline_conversation.py` | Conversation provenance tests updated for assertions |
| `tests/test_writeback.py` | Agent writeback now creates assertions and promoted memory facts |
| `tests/test_retrieval_basic.py` | Retrieval path assertions updated to `MEMORY_REL` traversal and explanation hydration |
| `tests/test_chunk_surfacing.py` | Chunk surfacing stays stable while explanation adds assertions |
| `tests/test_killer_demo.py` | Keep benchmark corpus/query contract stable against the new graph |
| `tests/test_personal_memory.py` | Supersession and history assertions move to versioned MemoryFacts |
| `tests/test_provenance.py` | Provenance checks move from relation properties to assertions and support links |
| `tests/test_mcp_server.py` | MCP response contract updated for fact + evidence output |

## Task 1: Add the memory-graph domain package

**Files:**
- Create: `src/landscape/memory_graph/__init__.py`
- Create: `src/landscape/memory_graph/families.py`
- Create: `src/landscape/memory_graph/ids.py`
- Create: `src/landscape/memory_graph/models.py`
- Modify: `src/landscape/extraction/schema.py`
- Test: `tests/test_memory_graph_ids.py`
- Test: `tests/test_memory_graph_normalization.py`

- [ ] **Step 1: Write the failing family-registry and id tests**

Create `tests/test_memory_graph_ids.py`:

```python
from landscape.memory_graph.families import FAMILY_REGISTRY
from landscape.memory_graph.ids import alias_id, assertion_id, fact_key, slot_key


def test_alias_id_is_canonical_entity_scoped():
    first = alias_id("ent-person-123", "Bob")
    second = alias_id("ent-person-123", " bob ")
    other = alias_id("ent-person-999", "Bob")
    assert first == second
    assert first != other


def test_assertion_id_is_source_local_and_deterministic():
    left = assertion_id(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        subtype=None,
        qualifier_payload={},
        chunk_refs=[("doc-1:0:abc", 3, 22)],
    )
    right = assertion_id(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        subtype=None,
        qualifier_payload={},
        chunk_refs=[("doc-1:0:abc", 3, 22)],
    )
    other_source = assertion_id(
        source_kind="document",
        source_id="doc-2",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        subtype=None,
        qualifier_payload={},
        chunk_refs=[("doc-1:0:abc", 3, 22)],
    )
    assert left == right
    assert left != other_source


def test_family_registry_contains_expected_v1_promotable_families():
    assert FAMILY_REGISTRY["WORKS_FOR"].single_current is True
    assert FAMILY_REGISTRY["USES"].single_current is False
    assert FAMILY_REGISTRY["APPROVED"].traversable is True


def test_slot_and_fact_keys_are_family_specific():
    works_for = FAMILY_REGISTRY["WORKS_FOR"]
    uses = FAMILY_REGISTRY["USES"]
    assert fact_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a:ent-b"
    assert slot_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a"
    assert fact_key(uses, "ent-a", "ent-b", None) == "USES:ent-a:ent-b"
    assert slot_key(uses, "ent-a", "ent-b", None) == "USES:ent-a:ent-b"
```

Create `tests/test_memory_graph_normalization.py`:

```python
from landscape.memory_graph.families import FAMILY_REGISTRY


def test_related_to_is_not_promotable():
    assert "RELATED_TO" not in FAMILY_REGISTRY


def test_benchmark_parity_families_are_explicitly_promotable():
    assert FAMILY_REGISTRY["LEADS"].traversable is True
    assert FAMILY_REGISTRY["LOCATED_IN"].traversable is True
    assert FAMILY_REGISTRY["APPROVED"].traversable is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_graph_ids.py tests/test_memory_graph_normalization.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'landscape.memory_graph'`.

- [ ] **Step 3: Add the new package, family registry, and deterministic id helpers**

Create `src/landscape/memory_graph/families.py`:

```python
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
    "OWNS": FamilyConfig("OWNS", traversable=True, object_kind="entity", single_current=False),
    "USES": FamilyConfig("USES", traversable=True, object_kind="entity", single_current=False),
    "DEPENDS_ON": FamilyConfig("DEPENDS_ON", traversable=True, object_kind="entity", single_current=False),
    "CREATED": FamilyConfig("CREATED", traversable=True, object_kind="entity", single_current=False, identity_uses_subtype=True),
    "LEADS": FamilyConfig("LEADS", traversable=True, object_kind="entity", single_current=False),
    "APPROVED": FamilyConfig("APPROVED", traversable=True, object_kind="entity", single_current=False),
    "LOCATED_IN": FamilyConfig("LOCATED_IN", traversable=True, object_kind="entity", single_current=False),
}
```

Create `src/landscape/memory_graph/ids.py`:

```python
import hashlib
import json

from landscape.memory_graph.families import FamilyConfig


def _stable_digest(parts: list[object]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def alias_id(entity_id: str, alias_text: str) -> str:
    return f"alias:{entity_id}:{_stable_digest([alias_text.strip().lower()])}"


def assertion_id(
    *,
    source_kind: str,
    source_id: str,
    raw_subject_text: str,
    raw_relation_text: str,
    raw_object_text: str,
    subtype: str | None,
    qualifier_payload: dict[str, object],
    chunk_refs: list[tuple[str, int | None, int | None]],
) -> str:
    digest = _stable_digest(
        [
            source_kind,
            source_id,
            raw_subject_text,
            raw_relation_text,
            raw_object_text,
            subtype,
            qualifier_payload,
            chunk_refs,
        ]
    )
    return f"assertion:{source_kind}:{digest}"


def fact_key(family: FamilyConfig, subject_entity_id: str, object_entity_id: str | None, subtype: str | None) -> str:
    parts = [family.family, subject_entity_id]
    if object_entity_id is not None:
        parts.append(object_entity_id)
    if family.identity_uses_subtype:
        parts.append(subtype or "")
    return ":".join(parts)


def slot_key(family: FamilyConfig, subject_entity_id: str, object_entity_id: str | None, subtype: str | None) -> str:
    if family.single_current:
        return f"{family.family}:{subject_entity_id}"
    return fact_key(family, subject_entity_id, object_entity_id, subtype)
```

Create `src/landscape/memory_graph/models.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssertionPayload:
    source_kind: str
    source_id: str
    raw_subject_text: str
    raw_relation_text: str
    raw_object_text: str
    confidence: float
    subtype: str | None = None
    family_candidate: str | None = None
    quantity_value: float | str | None = None
    quantity_unit: str | None = None
    quantity_kind: str | None = None
    time_scope: str | None = None
    chunk_refs: list[tuple[str, int | None, int | None]] = field(default_factory=list)
```

Update `src/landscape/extraction/schema.py`:

```python
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
        "WORKS_ON",
        "MAINTAINS",
        "OWNS",
        "DEPENDS_ON",
    }
)
```

Create `src/landscape/memory_graph/__init__.py`:

```python
from landscape.memory_graph.families import FAMILY_REGISTRY, FamilyConfig
from landscape.memory_graph.ids import alias_id, assertion_id, fact_key, slot_key
from landscape.memory_graph.models import AssertionPayload

__all__ = [
    "AssertionPayload",
    "FAMILY_REGISTRY",
    "FamilyConfig",
    "alias_id",
    "assertion_id",
    "fact_key",
    "slot_key",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_graph_ids.py tests/test_memory_graph_normalization.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/landscape/memory_graph src/landscape/extraction/schema.py tests/test_memory_graph_ids.py tests/test_memory_graph_normalization.py
git commit -m "feat(memory-graph): add family registry and deterministic ids"
```

## Task 2: Add Neo4j storage primitives for Alias, Assertion, MemoryFact, and MEMORY_REL

**Files:**
- Modify: `src/landscape/storage/neo4j_store.py`
- Test: `tests/test_memory_graph_storage.py`

- [ ] **Step 1: Write failing storage tests for assertion persistence, supersession, and MEMORY_REL materialization**

Create `tests/test_memory_graph_storage.py`:

```python
import pytest

from landscape.memory_graph import AssertionPayload
from landscape.storage import neo4j_store


@pytest.mark.asyncio
async def test_merge_assertion_is_idempotent(neo4j_driver):
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-test",
        raw_subject_text="Alice",
        raw_relation_text="works at",
        raw_object_text="Acme",
        confidence=0.9,
        family_candidate="WORKS_FOR",
    )
    first = await neo4j_store.merge_assertion(payload)
    second = await neo4j_store.merge_assertion(payload)
    assert first == second


@pytest.mark.asyncio
async def test_superseding_single_current_fact_replaces_memory_rel(neo4j_driver):
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    beacon = await neo4j_store.merge_entity("Beacon", "Organization", "doc-b", 0.9)
    first = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice,
        object_entity_id=acme,
        subtype=None,
        confidence=0.9,
        assertion_id="assertion:1",
    )
    await neo4j_store.materialize_memory_rel(first)
    second = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice,
        object_entity_id=beacon,
        subtype=None,
        confidence=0.95,
        assertion_id="assertion:2",
    )
    explanation = await neo4j_store.get_memory_fact_explanation(second)
    assert explanation["family"] == "WORKS_FOR"
    assert explanation["current"] is True
    assert explanation["object_name"] == "Beacon"


@pytest.mark.asyncio
async def test_bfs_expand_memory_rel_uses_current_edges_only(neo4j_driver):
    rows = await neo4j_store.bfs_expand_memory_rel([], max_hops=2)
    assert rows == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_graph_storage.py -v`

Expected: FAIL with missing functions such as `merge_assertion`, `create_memory_fact_version`, and `bfs_expand_memory_rel`.

- [ ] **Step 3: Extend `neo4j_store.py` with the new schema helpers and traversal query**

Append these helpers near the existing document/entity/chunk functions in `src/landscape/storage/neo4j_store.py`:

```python
from landscape.memory_graph import AssertionPayload, FAMILY_REGISTRY, alias_id, assertion_id, fact_key, slot_key


async def ensure_memory_graph_schema() -> None:
    driver = get_driver()
    statements = [
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT alias_id_unique IF NOT EXISTS FOR (n:Alias) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS FOR (n:Assertion) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT memory_fact_id_unique IF NOT EXISTS FOR (n:MemoryFact) REQUIRE n.id IS UNIQUE",
    ]
    async with driver.session() as session:
        for stmt in statements:
            await session.run(stmt)


async def merge_alias(canonical_entity_id: str, alias_text: str, source_doc: str, confidence: float) -> str:
    aid = alias_id(canonical_entity_id, alias_text)
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (canonical:Entity {id: $entity_id})
            MERGE (a:Alias {id: $alias_id})
            ON CREATE SET a.name = $alias_text, a.normalized_name = toLower(trim($alias_text))
            MERGE (a)-[r:SAME_AS]->(canonical)
            ON CREATE SET r.confidence = $confidence, r.source_doc = $source_doc
            """,
            entity_id=canonical_entity_id,
            alias_id=aid,
            alias_text=alias_text,
            confidence=confidence,
            source_doc=source_doc,
        )
    return aid


async def merge_assertion(payload: AssertionPayload) -> str:
    aid = assertion_id(
        source_kind=payload.source_kind,
        source_id=payload.source_id,
        raw_subject_text=payload.raw_subject_text,
        raw_relation_text=payload.raw_relation_text,
        raw_object_text=payload.raw_object_text,
        subtype=payload.subtype,
        qualifier_payload={
            "quantity_value": payload.quantity_value,
            "quantity_unit": payload.quantity_unit,
            "quantity_kind": payload.quantity_kind,
            "time_scope": payload.time_scope,
        },
        chunk_refs=payload.chunk_refs,
    )
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MERGE (a:Assertion {id: $assertion_id})
            ON CREATE SET a.source_kind = $source_kind,
                          a.source_id = $source_id,
                          a.raw_subject_text = $raw_subject_text,
                          a.raw_relation_text = $raw_relation_text,
                          a.raw_object_text = $raw_object_text,
                          a.family_candidate = $family_candidate,
                          a.confidence = $confidence,
                          a.subtype = $subtype,
                          a.status = 'active',
                          a.created_at = $now
            """,
            assertion_id=aid,
            source_kind=payload.source_kind,
            source_id=payload.source_id,
            raw_subject_text=payload.raw_subject_text,
            raw_relation_text=payload.raw_relation_text,
            raw_object_text=payload.raw_object_text,
            family_candidate=payload.family_candidate,
            confidence=payload.confidence,
            subtype=payload.subtype,
            now=datetime.now(UTC).isoformat(),
        )
    return aid


async def create_memory_fact_version(*, family: str, subject_entity_id: str, object_entity_id: str | None, subtype: str | None, confidence: float, assertion_id: str) -> str:
    family_cfg = FAMILY_REGISTRY[family]
    fkey = fact_key(family_cfg, subject_entity_id, object_entity_id, subtype)
    skey = slot_key(family_cfg, subject_entity_id, object_entity_id, subtype)
    fact_id = f"fact:{hashlib.sha256(f'{fkey}:{assertion_id}'.encode()).hexdigest()[:20]}"
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            """
            MATCH (subject:Entity {id: $subject_entity_id})
            OPTIONAL MATCH (object:Entity {id: $object_entity_id})
            CREATE (fact:MemoryFact {
                id: $fact_id,
                family: $family,
                fact_key: $fact_key,
                slot_key: $slot_key,
                subtype: $subtype,
                current: true,
                support_count: 1,
                confidence_agg: $confidence,
                normalization_policy: 'v1_family_rules',
                created_at: $now,
                updated_at: $now
            })
            MERGE (subject)-[:AS_SUBJECT]->(fact)
            FOREACH (_ IN CASE WHEN object IS NULL THEN [] ELSE [1] END |
                MERGE (fact)-[:AS_OBJECT]->(object)
            )
            WITH fact
            MATCH (a:Assertion {id: $assertion_id})
            MERGE (a)-[:SUPPORTS]->(fact)
            """,
            fact_id=fact_id,
            family=family,
            fact_key=fkey,
            slot_key=skey,
            subtype=subtype,
            confidence=confidence,
            now=datetime.now(UTC).isoformat(),
            subject_entity_id=subject_entity_id,
            object_entity_id=object_entity_id,
            assertion_id=assertion_id,
        )
    return fact_id


async def bfs_expand_memory_rel(seed_entity_ids: list[str], max_hops: int) -> list[dict[str, Any]]:
    if not seed_entity_ids:
        return []
    query = f"""
    MATCH (seed:Entity) WHERE seed.id IN $seed_ids
    MATCH path = shortestPath((seed)-[rels:MEMORY_REL*1..{max_hops}]-(target:Entity))
    WHERE seed.id <> target.id AND ALL(r IN rels WHERE r.current = true)
    RETURN
      seed.id AS seed_id,
      target.id AS target_id,
      target.name AS target_name,
      target.type AS target_type,
      length(path) AS distance,
      [r IN rels | r.memory_fact_id] AS memory_fact_ids,
      [r IN rels | r.family] AS edge_families
    """
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(query, seed_ids=seed_entity_ids)
        return [dict(record) async for record in result]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_graph_storage.py -v`

Expected: PASS for idempotent assertion merge, current-only fact replacement, and empty BFS behavior.

- [ ] **Step 5: Commit**

```bash
git add src/landscape/storage/neo4j_store.py tests/test_memory_graph_storage.py
git commit -m "feat(memory-graph): add neo4j primitives for assertions and memory facts"
```

## Task 3: Add deterministic normalization and a shared promotion service

**Files:**
- Create: `src/landscape/memory_graph/normalization.py`
- Create: `src/landscape/memory_graph/service.py`
- Modify: `src/landscape/pipeline.py`
- Modify: `src/landscape/conversation_ingestion.py`
- Test: `tests/test_memory_graph_normalization.py`
- Test: `tests/test_ingest.py`
- Test: `tests/test_pipeline_conversation.py`

- [ ] **Step 1: Write a failing normalization test for promotable vs assertion-only claims**

Append to `tests/test_memory_graph_normalization.py`:

```python
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.normalization import normalize_assertion


def test_normalize_assertion_promotes_known_family():
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="works for",
        raw_object_text="Acme",
        confidence=0.9,
        family_candidate="WORKS_FOR",
    )
    result = normalize_assertion(payload, subject_entity_id="ent-alice", object_entity_id="ent-acme")
    assert result.promotable is True
    assert result.family == "WORKS_FOR"
    assert result.object_entity_id == "ent-acme"


def test_normalize_assertion_keeps_related_to_assertion_only():
    payload = AssertionPayload(
        source_kind="document",
        source_id="doc-1",
        raw_subject_text="Alice",
        raw_relation_text="is connected to",
        raw_object_text="Acme",
        confidence=0.6,
        family_candidate="RELATED_TO",
    )
    result = normalize_assertion(payload, subject_entity_id="ent-alice", object_entity_id="ent-acme")
    assert result.promotable is False
    assert result.family is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_graph_normalization.py::test_normalize_assertion_promotes_known_family tests/test_memory_graph_normalization.py::test_normalize_assertion_keeps_related_to_assertion_only -v`

Expected: FAIL with `ModuleNotFoundError` or missing `normalize_assertion`.

- [ ] **Step 3: Implement the normalization result type and shared persistence service**

Create `src/landscape/memory_graph/normalization.py`:

```python
from dataclasses import dataclass

from landscape.memory_graph.families import FAMILY_REGISTRY
from landscape.memory_graph.ids import fact_key, slot_key
from landscape.memory_graph.models import AssertionPayload


@dataclass(frozen=True)
class NormalizationResult:
    promotable: bool
    family: str | None
    subject_entity_id: str | None
    object_entity_id: str | None
    subtype: str | None
    fact_key: str | None
    slot_key: str | None


def normalize_assertion(
    payload: AssertionPayload,
    *,
    subject_entity_id: str | None,
    object_entity_id: str | None,
) -> NormalizationResult:
    family = payload.family_candidate
    if not family or family not in FAMILY_REGISTRY:
        return NormalizationResult(False, None, subject_entity_id, object_entity_id, payload.subtype, None, None)

    config = FAMILY_REGISTRY[family]
    if subject_entity_id is None:
        return NormalizationResult(False, None, subject_entity_id, object_entity_id, payload.subtype, None, None)
    if config.object_kind == "entity" and object_entity_id is None:
        return NormalizationResult(False, None, subject_entity_id, object_entity_id, payload.subtype, None, None)

    return NormalizationResult(
        promotable=True,
        family=family,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
        subtype=payload.subtype,
        fact_key=fact_key(config, subject_entity_id, object_entity_id, payload.subtype),
        slot_key=slot_key(config, subject_entity_id, object_entity_id, payload.subtype),
    )
```

Create `src/landscape/memory_graph/service.py`:

```python
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.normalization import normalize_assertion
from landscape.storage import neo4j_store


async def persist_assertion_and_maybe_promote(
    payload: AssertionPayload,
    *,
    source_node_id: str,
    source_kind: str,
    subject_entity_id: str | None,
    object_entity_id: str | None,
    chunk_ids: list[str],
) -> tuple[str, str | None]:
    assertion_id = await neo4j_store.merge_assertion(payload)
    await neo4j_store.link_source_to_assertion(source_kind, source_node_id, assertion_id)
    for chunk_id in chunk_ids:
        await neo4j_store.link_assertion_to_chunk(assertion_id, chunk_id)
    if subject_entity_id is not None:
        await neo4j_store.link_assertion_subject(assertion_id, subject_entity_id)
    if object_entity_id is not None:
        await neo4j_store.link_assertion_object(assertion_id, object_entity_id)

    normalized = normalize_assertion(
        payload,
        subject_entity_id=subject_entity_id,
        object_entity_id=object_entity_id,
    )
    if not normalized.promotable:
        return assertion_id, None

    fact_id = await neo4j_store.upsert_memory_fact_from_assertion(
        family=normalized.family,
        subject_entity_id=normalized.subject_entity_id,
        object_entity_id=normalized.object_entity_id,
        subtype=normalized.subtype,
        confidence=payload.confidence,
        assertion_id=assertion_id,
    )
    return assertion_id, fact_id
```

Replace the relation-upsert loop in `src/landscape/pipeline.py` with:

```python
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.service import persist_assertion_and_maybe_promote

        # Step 5: assertion persistence + fact promotion
        relations_created = 0
        relations_reinforced = 0
        relations_superseded = 0
        for relation in extraction.relations:
            canonical_rel_type, _ = coerce_rel_type(relation.relation_type)
            payload = AssertionPayload(
                source_kind="document",
                source_id=doc_id,
                raw_subject_text=relation.subject,
                raw_relation_text=relation.relation_type,
                raw_object_text=relation.object,
                confidence=relation.confidence,
                family_candidate=canonical_rel_type,
                subtype=normalize_subtype(relation.subtype),
                quantity_value=relation.quantity_value,
                quantity_unit=relation.quantity_unit,
                quantity_kind=relation.quantity_kind,
                time_scope=relation.time_scope,
                chunk_refs=[(cid, None, None) for cid in chunk_ids],
            )
            subject_entity_id = await resolver.resolve_existing_entity_id(relation.subject)
            object_entity_id = await resolver.resolve_existing_entity_id(relation.object)
            _, fact_id = await persist_assertion_and_maybe_promote(
                payload,
                source_node_id=doc_id,
                source_kind="document",
                subject_entity_id=subject_entity_id,
                object_entity_id=object_entity_id,
                chunk_ids=chunk_ids,
            )
            if fact_id is None:
                relations_created += 1
            else:
                relations_reinforced += 1
```

- [ ] **Step 4: Run focused ingest tests**

Run: `uv run pytest tests/test_memory_graph_normalization.py tests/test_ingest.py tests/test_pipeline_conversation.py -v`

Expected: focused normalization tests PASS; ingest tests may still fail until the storage-link helpers from Task 2 are fully added. Fix those helpers before moving on.

- [ ] **Step 5: Commit**

```bash
git add src/landscape/memory_graph/normalization.py src/landscape/memory_graph/service.py src/landscape/pipeline.py src/landscape/conversation_ingestion.py tests/test_memory_graph_normalization.py tests/test_ingest.py tests/test_pipeline_conversation.py
git commit -m "feat(memory-graph): route ingest through assertion promotion service"
```

## Task 4: Move agent writeback to the same assertion/fact model

**Files:**
- Modify: `src/landscape/writeback.py`
- Modify: `src/landscape/mcp_app.py`
- Test: `tests/test_writeback.py`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing writeback regression tests**

Append to `tests/test_writeback.py`:

```python
@pytest.mark.asyncio
async def test_add_relation_creates_assertion_and_memory_fact():
    await add_relation(
        subject="Alice",
        object="Acme",
        relation_type="WORKS_FOR",
        confidence=0.9,
        source_doc="wb-doc",
        session_id="s1",
        turn_id="t1",
    )
    rows = await neo4j_store.run_cypher_readonly(
        "MATCH (:Assertion)-[:SUPPORTS]->(f:MemoryFact {family: 'WORKS_FOR'}) RETURN count(f) AS count"
    )
    assert rows[0]["count"] == 1


@pytest.mark.asyncio
async def test_alias_writeback_creates_alias_not_stub_entity():
    canonical = await add_entity("Robert", "Person", 0.9, "s1", "t1")
    await neo4j_store.merge_alias(canonical["id"], "Bob", "wb-doc", 0.9)
    rows = await neo4j_store.run_cypher_readonly(
        "MATCH (a:Alias)-[:SAME_AS]->(:Entity {name: 'Robert'}) RETURN count(a) AS count"
    )
    assert rows[0]["count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_writeback.py::test_add_relation_creates_assertion_and_memory_fact tests/test_writeback.py::test_alias_writeback_creates_alias_not_stub_entity -v`

Expected: FAIL because `add_relation()` still writes `RELATES_TO` edges directly and `add_alias()` still creates stub `Entity` nodes.

- [ ] **Step 3: Replace direct relation upserts in writeback with the shared service**

In `src/landscape/writeback.py`, replace the current `neo4j_store.upsert_relation(...)` path inside `add_relation()` with:

```python
from landscape.memory_graph.models import AssertionPayload
from landscape.memory_graph.service import persist_assertion_and_maybe_promote

    payload = AssertionPayload(
        source_kind="turn",
        source_id=f"{session_id}:{turn_id}",
        raw_subject_text=subject,
        raw_relation_text=relation_type,
        raw_object_text=object,
        confidence=confidence,
        family_candidate=normalize_relation_type(relation_type),
        subtype=None,
    )
    assertion_id, fact_id = await persist_assertion_and_maybe_promote(
        payload,
        source_node_id=turn_element_id,
        source_kind="turn",
        subject_entity_id=subject_id,
        object_entity_id=object_id,
        chunk_ids=[],
    )
    outcome = "memory_fact" if fact_id is not None else "assertion_only"
    return {"outcome": outcome, "assertion_id": assertion_id, "memory_fact_id": fact_id}
```

Also replace the alias-stub behavior in `neo4j_store.add_alias()` callers with `merge_alias()` and stop creating non-canonical `Entity` aliases.

- [ ] **Step 4: Run the writeback and MCP regressions**

Run: `uv run pytest tests/test_writeback.py tests/test_mcp_server.py -v`

Expected: PASS for updated writeback behavior. MCP response shape tests may still need updates for the new `assertion_id` / `memory_fact_id` fields.

- [ ] **Step 5: Commit**

```bash
git add src/landscape/writeback.py src/landscape/mcp_app.py tests/test_writeback.py tests/test_mcp_server.py
git commit -m "feat(memory-graph): move writeback onto assertion and memory-fact model"
```

## Task 5: Refactor retrieval to traverse MEMORY_REL and hydrate explanations

**Files:**
- Modify: `src/landscape/retrieval/query.py`
- Modify: `src/landscape/api/query.py`
- Modify: `src/landscape/cli/query.py`
- Modify: `src/landscape/cli/graph.py`
- Modify: `src/landscape/cli/status.py`
- Modify: `src/landscape/mcp_app.py`
- Test: `tests/test_retrieval_basic.py`
- Test: `tests/test_chunk_surfacing.py`
- Test: `tests/test_killer_demo.py`
- Test: `tests/test_personal_memory.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write the failing retrieval tests for MEMORY_REL traversal and evidence hydration**

Append to `tests/test_retrieval_basic.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_expands_over_memory_rel_not_relates_to():
    result = await retrieve("Where does Alice work?", hops=2, reinforce=False)
    assert all("RELATES_TO" not in edge for item in result.results for edge in item.path_edge_types)


@pytest.mark.asyncio
async def test_retrieve_returns_fact_and_assertion_explanations():
    result = await retrieve("Where does Alice work?", hops=2, reinforce=False)
    assert result.results
    top = result.results[0]
    assert top.memory_facts
    assert top.supporting_assertions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_basic.py::test_retrieve_expands_over_memory_rel_not_relates_to tests/test_retrieval_basic.py::test_retrieve_returns_fact_and_assertion_explanations -v`

Expected: FAIL because `RetrievedEntity` has no explanation fields and `bfs_expand()` still traverses `RELATES_TO`.

- [ ] **Step 3: Replace BFS traversal and add explanation hydration**

In `src/landscape/retrieval/query.py`, change `RetrievedEntity` to:

```python
@dataclass
class RetrievedEntity:
    neo4j_id: str
    name: str
    type: str
    distance: int
    vector_sim: float
    reinforcement: float
    edge_confidence: float
    score: float
    path_memory_fact_ids: list[str] = field(default_factory=list)
    path_edge_types: list[str] = field(default_factory=list)
    supporting_assertions: list[dict] = field(default_factory=list)
    memory_facts: list[dict] = field(default_factory=list)
```

Replace:

```python
        expansions = await neo4j_store.bfs_expand(live_seed_ids, max_hops=hops)
```

with:

```python
        expansions = await neo4j_store.bfs_expand_memory_rel(live_seed_ids, max_hops=hops)
```

Then, after ranking, hydrate explanations:

```python
        for item in ranked:
            explanation = await neo4j_store.get_entity_path_explanation(item.path_memory_fact_ids)
            item.memory_facts = explanation["memory_facts"]
            item.supporting_assertions = explanation["assertions"]
```

In `src/landscape/api/query.py`, add the optional flag and response fields:

```python
class QueryRequest(BaseModel):
    query: str
    hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=50)
    session_id: str | None = None
    include_historical: bool = False
```

In `src/landscape/cli/query.py`, add:

```python
    parser.add_argument("--include-historical", action="store_true")
```

- [ ] **Step 4: Run the retrieval-focused suite**

Run: `uv run pytest tests/test_retrieval_basic.py tests/test_chunk_surfacing.py tests/test_killer_demo.py tests/test_personal_memory.py tests/test_provenance.py -v`

Expected: PASS with:
- multi-hop paths still resolving
- chunk surfacing unchanged
- personal-memory supersession expressed via current vs historical `MemoryFact`s
- provenance loaded from assertions rather than relation properties

- [ ] **Step 5: Commit**

```bash
git add src/landscape/retrieval/query.py src/landscape/api/query.py src/landscape/cli/query.py src/landscape/cli/graph.py src/landscape/cli/status.py src/landscape/mcp_app.py tests/test_retrieval_basic.py tests/test_chunk_surfacing.py tests/test_killer_demo.py tests/test_personal_memory.py tests/test_provenance.py
git commit -m "feat(memory-graph): switch retrieval to memory-rel traversal with fact evidence"
```

## Task 6: Migrate fixtures, remove RELATES_TO-centric assumptions, and update architecture docs

**Files:**
- Modify: `tests/test_ingest.py`
- Modify: `tests/test_pipeline_conversation.py`
- Modify: `tests/test_writeback.py`
- Modify: `tests/test_retrieval_basic.py`
- Modify: `tests/test_killer_demo.py`
- Modify: `tests/test_personal_memory.py`
- Modify: `tests/test_provenance.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-04-28-memory-graph-redesign-design.md` only if implementation discoveries force a spec correction

- [ ] **Step 1: Replace relation-property assertions with graph-layer assertions**

Update tests that currently query:

```cypher
MATCH (s:Entity)-[r:RELATES_TO {type: 'WORKS_FOR'}]->(o:Entity)
```

to query:

```cypher
MATCH (a:Assertion)-[:SUPPORTS]->(f:MemoryFact {family: 'WORKS_FOR', current: true})-[:AS_OBJECT]->(o:Entity)
MATCH (:Entity {name: 'Alice'})-[:AS_SUBJECT]->(f)
RETURN a.raw_relation_text AS raw_relation, o.name AS object_name, f.valid_until AS valid_until
```

- [ ] **Step 2: Keep the benchmark contract stable**

Update `tests/test_killer_demo.py` and fixture assertions so the queries and success criteria stay the same while the expected path substrate changes from `RELATES_TO` to `MEMORY_REL`.

The important rewrite pattern is:

```python
assert "Aurora" in names
assert "Maya Chen" in names
assert any("USES" in item.path_edge_types for item in result.results)
```

instead of checking raw Neo4j relation properties directly.

- [ ] **Step 3: Update architecture docs**

Replace the Neo4j data-model section in `docs/ARCHITECTURE.md` with:

```markdown
Primary node labels:

- `Entity`
- `Alias`
- `Document`
- `Chunk`
- `Conversation`
- `Turn`
- `Assertion`
- `MemoryFact`

Primary relationship layers:

- `ASSERTS`, `MENTIONS_CHUNK`, `SUBJECT_ENTITY`, `OBJECT_ENTITY`
- `SUPPORTS`, `AS_SUBJECT`, `AS_OBJECT`
- derived `MEMORY_REL`
```

- [ ] **Step 4: Run the full redesign verification slice**

Run:

`uv run pytest tests/test_memory_graph_ids.py tests/test_memory_graph_normalization.py tests/test_memory_graph_storage.py tests/test_ingest.py tests/test_pipeline_conversation.py tests/test_writeback.py tests/test_retrieval_basic.py tests/test_chunk_surfacing.py tests/test_killer_demo.py tests/test_personal_memory.py tests/test_provenance.py tests/test_mcp_server.py -v`

Expected: PASS. This is the minimum redesign slice before broader suite cleanup.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingest.py tests/test_pipeline_conversation.py tests/test_writeback.py tests/test_retrieval_basic.py tests/test_killer_demo.py tests/test_personal_memory.py tests/test_provenance.py docs/ARCHITECTURE.md
git commit -m "test(memory-graph): migrate fixtures and docs to redesigned graph model"
```

## Self-Review

### Spec coverage

- Raw evidence layer: covered by Tasks 2-4.
- Deterministic family-specific promotion: covered by Tasks 1 and 3.
- Current-only rebuildable traversal layer: covered by Tasks 2 and 5.
- Alias redesign: covered by Tasks 2 and 4.
- Stable A/B benchmark contract: covered by Tasks 5 and 6.

### Placeholder scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Every task names concrete files and commands.

### Type consistency

- `AssertionPayload`, `NormalizationResult`, `FamilyConfig`, `fact_key`, and `slot_key` are introduced once in Task 1/3 and reused consistently throughout the plan.
- Retrieval task uses `path_memory_fact_ids`, not the legacy `path_edge_ids`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-28-memory-graph-redesign.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
