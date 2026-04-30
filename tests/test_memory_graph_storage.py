import pytest

from landscape.memory_graph import FAMILY_REGISTRY, AssertionPayload, fact_key, slot_key
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
    assert explanation["current"] is True
    assert explanation["object_name"] == "Beacon"
    async with neo4j_driver.session() as session:
        old_fact_result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fact_id})
            RETURN f.current AS current
            """,
            fact_id=first,
        )
        old_fact = await old_fact_result.single()
        assert old_fact is not None
        assert old_fact["current"] is False

        old_rel_result = await session.run(
            """
            MATCH (:Entity {id: $subject_id})-[r:MEMORY_REL {memory_fact_id: $fact_id}]->()
            RETURN r.current AS current
            """,
            subject_id=alice_id,
            fact_id=first,
        )
        old_rel = await old_rel_result.single()
        assert old_rel is not None
        assert old_rel["current"] is False

        current_count_result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key, current: true})
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
    assert explanation["current"] is True
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key, current: true})
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
