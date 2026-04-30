import pytest

from landscape.memory_graph import FAMILY_REGISTRY, AssertionPayload, fact_key, slot_key
from landscape.memory_graph.service import persist_assertion_and_maybe_promote
from landscape.storage import neo4j_store


async def _entity_app_id(neo4j_driver, entity_element_id: str) -> str:
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity)
            WHERE elementId(e) = $eid
            RETURN e.id AS entity_id
            """,
            eid=entity_element_id,
        )
        record = await result.single()
        assert record is not None
        return record["entity_id"]


def test_memory_fact_key_modes_follow_family_config():
    works_for = FAMILY_REGISTRY["WORKS_FOR"]
    has_title = FAMILY_REGISTRY["HAS_TITLE"]
    has_pref = FAMILY_REGISTRY["HAS_PREFERENCE"]
    created = FAMILY_REGISTRY["CREATED"]

    assert works_for.slot_mode == "subject"
    assert has_title.slot_mode == "object"
    assert has_pref.slot_mode == "subtype"
    assert created.slot_mode == "additive"

    assert fact_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a:ent-b"
    assert slot_key(works_for, "ent-a", "ent-b", None) == "WORKS_FOR:ent-a"
    assert fact_key(has_title, "ent-a", "ent-b", "senior_engineer") == (
        "HAS_TITLE:ent-a:ent-b:senior_engineer"
    )
    assert slot_key(has_title, "ent-a", "ent-b", "senior_engineer") == "HAS_TITLE:ent-a:ent-b"
    assert fact_key(has_pref, "ent-a", "ent-b", "favorite_color") == (
        "HAS_PREFERENCE:ent-a:ent-b:favorite_color"
    )
    assert slot_key(has_pref, "ent-a", "ent-b", "favorite_color") == "HAS_PREFERENCE:ent-a:favorite_color"
    assert fact_key(created, "ent-a", None, "diagram") == "CREATED:ent-a:diagram"


@pytest.mark.asyncio
async def test_value_backed_family_preserves_value_identity_on_promotion(neo4j_driver):
    doc_id, _ = await neo4j_store.merge_document("hash-happened", "value-backed-test", "text")
    kickoff = await neo4j_store.merge_entity("Kickoff", "EVENT", "value-backed-test", 0.95, doc_id, "test")
    payload = AssertionPayload(
        source_kind="document",
        source_id="value-backed-test",
        raw_subject_text="Kickoff",
        raw_relation_text="happened on",
        raw_object_text="2026-03-05",
        confidence=0.95,
        family_candidate="HAPPENED_ON",
        value_time="2026-03-05",
    )
    promotion = await persist_assertion_and_maybe_promote(
        payload,
        source_node_id=doc_id,
        source_kind="document",
        subject_entity_id=kickoff,
        object_entity_id=None,
        chunk_ids=[],
    )
    assert promotion.fact_id is not None
    assert promotion.outcome == "created"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fact_id})
            RETURN f.fact_key AS fact_key,
                   f.slot_key AS slot_key,
                   f.value_text AS value_text,
                   f.value_time AS value_time,
                   f.time_scope AS time_scope
            """,
            fact_id=promotion.fact_id,
        )
        record = await result.single()
    assert record is not None
    kickoff_id = await _entity_app_id(neo4j_driver, kickoff)
    expected_fact_key = fact_key(
        FAMILY_REGISTRY["HAPPENED_ON"],
        kickoff_id,
        None,
        None,
        value_text="2026-03-05",
        value_time="2026-03-05",
    )
    expected_slot_key = slot_key(
        FAMILY_REGISTRY["HAPPENED_ON"],
        kickoff_id,
        None,
        None,
        value_time="2026-03-05",
    )
    assert record["fact_key"] == expected_fact_key
    assert record["slot_key"] == expected_slot_key
    assert record["value_text"] == "2026-03-05"
    assert record["value_time"] == "2026-03-05"
    assert record["time_scope"] == "2026-03-05"


@pytest.mark.asyncio
async def test_object_keyed_family_supersedes_on_same_slot(neo4j_driver):
    doc1, _ = await neo4j_store.merge_document("hash-title-1", "object-keyed-test-1", "text")
    doc2, _ = await neo4j_store.merge_document("hash-title-2", "object-keyed-test-2", "text")
    alice = await neo4j_store.merge_entity("Alice", "PERSON", "object-keyed-test", 0.95, doc1, "test")
    atlas = await neo4j_store.merge_entity("Atlas", "ORGANIZATION", "object-keyed-test", 0.95, doc1, "test")
    first = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document",
            source_id="object-keyed-test-1",
            raw_subject_text="Alice",
            raw_relation_text="is a senior engineer at",
            raw_object_text="Atlas",
            confidence=0.95,
            family_candidate="HAS_TITLE",
            subtype="senior_engineer",
        ),
        source_node_id=doc1,
        source_kind="document",
        subject_entity_id=alice,
        object_entity_id=atlas,
        chunk_ids=[],
    )
    second = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document",
            source_id="object-keyed-test-2",
            raw_subject_text="Alice",
            raw_relation_text="is a principal engineer at",
            raw_object_text="Atlas",
            confidence=0.96,
            family_candidate="HAS_TITLE",
            subtype="principal_engineer",
        ),
        source_node_id=doc2,
        source_kind="document",
        subject_entity_id=alice,
        object_entity_id=atlas,
        chunk_ids=[],
    )
    assert first.fact_id is not None
    assert second.fact_id is not None
    assert first.outcome == "created"
    assert second.outcome == "superseded"

    alice_id = await _entity_app_id(neo4j_driver, alice)
    atlas_id = await _entity_app_id(neo4j_driver, atlas)
    expected_slot_key = slot_key(FAMILY_REGISTRY["HAS_TITLE"], alice_id, atlas_id, "principal_engineer")
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {family: 'HAS_TITLE', slot_key: $slot_key})
            RETURN count(*) AS total, sum(CASE WHEN f.valid_until IS NULL THEN 1 ELSE 0 END) AS live
            """,
            slot_key=expected_slot_key,
        )
        record = await result.single()
    assert record is not None
    assert record["total"] == 2
    assert record["live"] == 1


@pytest.mark.asyncio
async def test_subtype_keyed_family_supersedes_on_same_slot(neo4j_driver):
    doc1, _ = await neo4j_store.merge_document("hash-pref-1", "subtype-keyed-test-1", "text")
    doc2, _ = await neo4j_store.merge_document("hash-pref-2", "subtype-keyed-test-2", "text")
    alice = await neo4j_store.merge_entity("Alice", "PERSON", "subtype-keyed-test", 0.95, doc1, "test")
    first = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document",
            source_id="subtype-keyed-test-1",
            raw_subject_text="Alice",
            raw_relation_text="prefers",
            raw_object_text="Blue",
            confidence=0.95,
            family_candidate="HAS_PREFERENCE",
            subtype="favorite_color",
        ),
        source_node_id=doc1,
        source_kind="document",
        subject_entity_id=alice,
        object_entity_id=None,
        chunk_ids=[],
    )
    second = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document",
            source_id="subtype-keyed-test-2",
            raw_subject_text="Alice",
            raw_relation_text="prefers",
            raw_object_text="Green",
            confidence=0.96,
            family_candidate="HAS_PREFERENCE",
            subtype="favorite_color",
        ),
        source_node_id=doc2,
        source_kind="document",
        subject_entity_id=alice,
        object_entity_id=None,
        chunk_ids=[],
    )
    assert first.fact_id is not None
    assert second.fact_id is not None
    assert first.outcome == "created"
    assert second.outcome == "superseded"

    alice_id = await _entity_app_id(neo4j_driver, alice)
    expected_slot_key = slot_key(
        FAMILY_REGISTRY["HAS_PREFERENCE"],
        alice_id,
        None,
        "favorite_color",
    )
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {family: 'HAS_PREFERENCE', slot_key: $slot_key})
            RETURN count(*) AS total, sum(CASE WHEN f.valid_until IS NULL THEN 1 ELSE 0 END) AS live
            """,
            slot_key=expected_slot_key,
        )
        record = await result.single()
    assert record is not None
    assert record["total"] == 2
    assert record["live"] == 1


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
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Assertion {id: $assertion_id})
            RETURN count(a) AS count
            """,
            assertion_id=first,
        )
        record = await result.single()
        assert record is not None
        assert record["count"] == 1


@pytest.mark.asyncio
async def test_create_memory_fact_accepts_element_id_and_links_real_assertion(neo4j_driver):
    await neo4j_store.ensure_memory_graph_schema()
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    alice_id = await _entity_app_id(neo4j_driver, alice)
    acme_id = await _entity_app_id(neo4j_driver, acme)
    assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-a",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Acme",
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    first = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice,
        object_entity_id=acme,
        subtype=None,
        confidence=0.9,
        assertion_id=assertion,
    )
    explanation = await neo4j_store.get_memory_fact_explanation(first)
    assert explanation is not None
    assert explanation["subject_entity_id"] == alice_id
    assert explanation["object_entity_id"] == acme_id
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Assertion {id: $assertion_id})-[:SUPPORTS]->(f:MemoryFact {id: $fact_id})
            RETURN count(f) AS count
            """,
            assertion_id=assertion,
            fact_id=first,
        )
        record = await result.single()
        assert record is not None
        assert record["count"] == 1


@pytest.mark.asyncio
async def test_superseding_single_current_fact_replaces_memory_rel(neo4j_driver):
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    beacon = await neo4j_store.merge_entity("Beacon", "Organization", "doc-b", 0.9)
    alice_id = await _entity_app_id(neo4j_driver, alice)
    acme_id = await _entity_app_id(neo4j_driver, acme)
    beacon_id = await _entity_app_id(neo4j_driver, beacon)
    first_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-a",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Acme",
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    second_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-b",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Beacon",
            confidence=0.95,
            family_candidate="WORKS_FOR",
        )
    )
    first = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=acme_id,
        subtype=None,
        confidence=0.9,
        assertion_id=first_assertion,
    )
    await neo4j_store.materialize_memory_rel(first)
    family_cfg = FAMILY_REGISTRY["WORKS_FOR"]
    slot = slot_key(family_cfg, alice_id, acme_id, None)
    second = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=beacon_id,
        subtype=None,
        confidence=0.95,
        assertion_id=second_assertion,
    )
    explanation = await neo4j_store.get_memory_fact_explanation(second)
    assert explanation["family"] == "WORKS_FOR"
    assert explanation["valid_until"] is None
    assert explanation["object_name"] == "Beacon"
    async with neo4j_driver.session() as session:
        old_fact_result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fact_id})
            RETURN f.valid_until AS valid_until,
                   (f.valid_until IS NULL) AS current
            """,
            fact_id=first,
        )
        old_fact = await old_fact_result.single()
        assert old_fact is not None
        assert old_fact["valid_until"] is not None

        old_rel_result = await session.run(
            """
            MATCH (:Entity {id: $subject_id})-[r:MEMORY_REL {memory_fact_id: $fact_id}]->()
            RETURN r.valid_until AS valid_until,
                   (r.valid_until IS NULL) AS current
            """,
            subject_id=alice_id,
            fact_id=first,
        )
        old_rel = await old_rel_result.single()
        assert old_rel is not None
        assert old_rel["valid_until"] is not None

        current_count_result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key})
            WHERE f.valid_until IS NULL
            RETURN count(f) AS count
            """,
            slot_key=slot,
        )
        current_count = await current_count_result.single()
        assert current_count is not None
        assert current_count["count"] == 1


@pytest.mark.asyncio
async def test_superseding_single_current_fact_is_idempotent_on_retry(neo4j_driver):
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    beacon = await neo4j_store.merge_entity("Beacon", "Organization", "doc-b", 0.9)
    alice_id = await _entity_app_id(neo4j_driver, alice)
    acme_id = await _entity_app_id(neo4j_driver, acme)
    beacon_id = await _entity_app_id(neo4j_driver, beacon)
    first_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-a",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Acme",
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    second_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-b",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Beacon",
            confidence=0.95,
            family_candidate="WORKS_FOR",
        )
    )
    first = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=acme_id,
        subtype=None,
        confidence=0.9,
        assertion_id=first_assertion,
    )
    await neo4j_store.materialize_memory_rel(first)
    second = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=beacon_id,
        subtype=None,
        confidence=0.95,
        assertion_id=second_assertion,
    )
    retry = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=beacon_id,
        subtype=None,
        confidence=0.95,
        assertion_id=second_assertion,
    )
    assert retry == second
    explanation = await neo4j_store.get_memory_fact_explanation(retry)
    assert explanation is not None
    assert explanation["valid_until"] is None
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key})
            WHERE f.valid_until IS NULL
            RETURN count(f) AS count
            """,
            slot_key=slot_key(FAMILY_REGISTRY["WORKS_FOR"], alice_id, beacon_id, None),
        )
        record = await result.single()
        assert record is not None
        assert record["count"] == 1


@pytest.mark.asyncio
async def test_bfs_expand_memory_rel_empty_input_returns_empty(neo4j_driver):
    rows = await neo4j_store.bfs_expand_memory_rel([], max_hops=2)
    assert rows == []


@pytest.mark.asyncio
async def test_bfs_expand_memory_rel_validates_max_hops(neo4j_driver):
    with pytest.raises(ValueError):
        await neo4j_store.bfs_expand_memory_rel(["entity:missing"], max_hops=0)


@pytest.mark.asyncio
async def test_bfs_expand_memory_rel_uses_current_edges_only(neo4j_driver):
    alice = await neo4j_store.merge_entity("Alice", "Person", "doc-a", 0.9)
    acme = await neo4j_store.merge_entity("Acme", "Organization", "doc-a", 0.9)
    beacon = await neo4j_store.merge_entity("Beacon", "Organization", "doc-b", 0.9)
    alice_id = await _entity_app_id(neo4j_driver, alice)
    acme_id = await _entity_app_id(neo4j_driver, acme)
    beacon_id = await _entity_app_id(neo4j_driver, beacon)
    current_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-a",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Acme",
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    stale_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="doc-b",
            raw_subject_text="Alice",
            raw_relation_text="works at",
            raw_object_text="Beacon",
            confidence=0.95,
            family_candidate="WORKS_FOR",
        )
    )
    current_fact = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=acme_id,
        subtype=None,
        confidence=0.9,
        assertion_id=current_assertion,
    )
    await neo4j_store.materialize_memory_rel(current_fact)
    current_fact = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=alice_id,
        object_entity_id=beacon_id,
        subtype=None,
        confidence=0.95,
        assertion_id=stale_assertion,
    )
    rows = await neo4j_store.bfs_expand_memory_rel([alice], max_hops=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["seed_id"] == alice_id
    assert row["seed_element_id"] == alice
    assert row["target_id"] == beacon_id
    assert row["target_element_id"] == beacon
    assert row["target_name"] == "Beacon"
    assert row["target_type"] == "Organization"
    assert row["target_access_count"] >= 1
    assert row["target_last_accessed"] is not None
    assert row["distance"] == 1
    assert row["memory_fact_ids"] == [current_fact]
    assert row["path_memory_fact_ids"] == [current_fact]
    assert row["edge_families"] == ["WORKS_FOR"]
    assert len(row["edge_ids"]) == 1
    assert row["edge_confidences"] == [0.95]
    assert row["edge_access_counts"] == [1]
    assert row["edge_last_accessed"][0] is not None
    assert all(row["target_id"] != acme_id for row in rows)
