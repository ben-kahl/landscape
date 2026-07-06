import pytest

from landscape.memory_graph import FAMILY_REGISTRY, AssertionPayload, fact_key, slot_key
from landscape.memory_graph.service import persist_assertion_and_maybe_promote
from landscape.storage import neo4j_store


async def _entity_app_id(neo4j_driver, entity_id: str) -> str:
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {id: $eid}) RETURN e.id AS entity_id",
            eid=entity_id,
        )
        record = await result.single()
        assert record is not None, f"Entity with id={entity_id!r} not found"
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
    assert slot_key(has_pref, "ent-a", "ent-b", "favorite_color") == (
        "HAS_PREFERENCE:ent-a:favorite_color"
    )
    assert fact_key(created, "ent-a", None, "diagram") == "CREATED:ent-a:diagram"


@pytest.mark.asyncio
async def test_value_backed_family_preserves_value_identity_on_promotion(neo4j_driver):
    doc_id, _, _ = await neo4j_store.merge_document("hash-happened", "value-backed-test", "text")
    kickoff = await neo4j_store.merge_entity(
        "Kickoff", "EVENT", "value-backed-test", 0.95, doc_id, "test"
    )
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
    doc1, _, _ = await neo4j_store.merge_document("hash-title-1", "object-keyed-test-1", "text")
    doc2, _, _ = await neo4j_store.merge_document("hash-title-2", "object-keyed-test-2", "text")
    alice = await neo4j_store.merge_entity(
        "Alice", "PERSON", "object-keyed-test", 0.95, doc1, "test"
    )
    atlas = await neo4j_store.merge_entity(
        "Atlas", "ORGANIZATION", "object-keyed-test", 0.95, doc1, "test"
    )
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
    expected_slot_key = slot_key(
        FAMILY_REGISTRY["HAS_TITLE"], alice_id, atlas_id, "principal_engineer"
    )
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {family: 'HAS_TITLE', slot_key: $slot_key})
            RETURN count(*) AS total, 
            sum(CASE WHEN f.system_until IS NULL THEN 1 ELSE 0 END) AS live
            """,
            slot_key=expected_slot_key,
        )
        record = await result.single()
    assert record is not None
    assert record["total"] == 2
    assert record["live"] == 1


@pytest.mark.asyncio
async def test_subtype_keyed_family_supersedes_on_same_slot(neo4j_driver):
    doc1, _, _ = await neo4j_store.merge_document("hash-pref-1", "subtype-keyed-test-1", "text")
    doc2, _, _ = await neo4j_store.merge_document("hash-pref-2", "subtype-keyed-test-2", "text")
    alice = await neo4j_store.merge_entity(
        "Alice", "PERSON", "subtype-keyed-test", 0.95, doc1, "test"
    )
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
            RETURN count(*) AS total,
            sum(CASE WHEN f.system_until IS NULL THEN 1 ELSE 0 END) AS live
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
    assert explanation["system_until"] is None
    assert explanation["object_name"] == "Beacon"
    async with neo4j_driver.session() as session:
        old_fact_result = await session.run(
            """
            MATCH (f:MemoryFact {id: $fact_id})
            RETURN f.system_until AS system_until,
                   (f.system_until IS NULL) AS current
            """,
            fact_id=first,
        )
        old_fact = await old_fact_result.single()
        assert old_fact is not None
        assert old_fact["system_until"] is not None

        old_rel_result = await session.run(
            """
            MATCH (:Entity {id: $subject_id})-[r:MEMORY_REL {memory_fact_id: $fact_id}]->()
            RETURN r.system_until AS system_until,
                   (r.system_until IS NULL) AS current
            """,
            subject_id=alice_id,
            fact_id=first,
        )
        old_rel = await old_rel_result.single()
        assert old_rel is not None
        assert old_rel["system_until"] is not None

        current_count_result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key})
            WHERE f.system_until IS NULL
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
    assert explanation["system_until"] is None
    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (f:MemoryFact {slot_key: $slot_key})
            WHERE f.system_until IS NULL
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
    assert row["target_id"] == beacon_id
    assert row["target_name"] == "Beacon"
    assert row["target_type"] == "Organization"
    assert row["target_access_count"] >= 1
    assert row["target_last_accessed"] is not None
    assert row["distance"] == 1
    assert row["path_memory_fact_ids"] == [current_fact]
    assert row["path_edge_types"] == ["WORKS_FOR"]
    assert len(row["edge_ids"]) == 1
    assert row["edge_confidences"] == [0.95]
    assert row["edge_access_counts"] == [1]
    assert row["edge_last_accessed"][0] is not None
    assert all(row["target_id"] != acme_id for row in rows)


@pytest.mark.asyncio
async def test_subject_keyed_cross_polarity_supersession(neo4j_driver):
    """Positive WORKS_FOR then negative WORKS_FOR should supersede the positive."""
    doc_id, _, _ = await neo4j_store.merge_document("hash-xpol-1", "xpol-test", "text")
    alice = await neo4j_store.merge_entity(
        "Alice", "Person", "xpol-test", 0.9, doc_id, "test"
    )
    acme = await neo4j_store.merge_entity(
        "Acme", "Organization", "xpol-test", 0.9, doc_id, "test"
    )

    positive_payload = AssertionPayload(
        source_kind="document",
        source_id="xpol-test",
        raw_subject_text="Alice",
        raw_relation_text="works for",
        raw_object_text="Acme",
        confidence=0.9,
        family_candidate="WORKS_FOR",
        negated=False,
    )
    pos_result = await persist_assertion_and_maybe_promote(
        positive_payload,
        source_node_id=doc_id,
        source_kind="document",
        subject_entity_id=alice,
        object_entity_id=acme,
        chunk_ids=[],
    )
    assert pos_result.outcome == "created"

    negative_payload = AssertionPayload(
        source_kind="document",
        source_id="xpol-test",
        raw_subject_text="Alice",
        raw_relation_text="does not work for",
        raw_object_text="Acme",
        confidence=0.9,
        family_candidate="WORKS_FOR",
        negated=True,
    )
    neg_result = await persist_assertion_and_maybe_promote(
        negative_payload,
        source_node_id=doc_id,
        source_kind="document",
        subject_entity_id=alice,
        object_entity_id=acme,
        chunk_ids=[],
    )
    assert neg_result.outcome == "superseded"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {id: $alice_id})-[:AS_SUBJECT]->(f:MemoryFact {family: 'WORKS_FOR'})
            WHERE f.system_until IS NULL
            RETURN f.negated AS negated, f.id AS fact_id
            """,
            alice_id=alice,
        )
        current_facts = [dict(r) async for r in result]

    assert len(current_facts) == 1
    assert current_facts[0]["negated"] is True


@pytest.mark.asyncio
async def test_additive_entity_cross_polarity_supersession(neo4j_driver):
    """Positive USES then negative USES on same subject+object should supersede the positive."""
    doc_id, _, _ = await neo4j_store.merge_document("hash-add-ent", "add-ent-test", "text")
    project = await neo4j_store.merge_entity(
        "Project X", "Project", "add-ent-test", 0.9, doc_id, "test"
    )
    redis = await neo4j_store.merge_entity(
        "Redis", "Technology", "add-ent-test", 0.9, doc_id, "test"
    )
    kafka = await neo4j_store.merge_entity(
        "Kafka", "Technology", "add-ent-test", 0.9, doc_id, "test"
    )

    pos_redis = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document", source_id="add-ent-test",
            raw_subject_text="Project X", raw_relation_text="uses", raw_object_text="Redis",
            confidence=0.9, family_candidate="USES", negated=False,
        ),
        source_node_id=doc_id, source_kind="document",
        subject_entity_id=project, object_entity_id=redis, chunk_ids=[],
    )
    pos_kafka = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document", source_id="add-ent-test",
            raw_subject_text="Project X", raw_relation_text="uses", raw_object_text="Kafka",
            confidence=0.9, family_candidate="USES", negated=False,
        ),
        source_node_id=doc_id, source_kind="document",
        subject_entity_id=project, object_entity_id=kafka, chunk_ids=[],
    )
    assert pos_redis.outcome == "created"
    assert pos_kafka.outcome == "created"

    neg_redis = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document", source_id="add-ent-test",
            raw_subject_text="Project X", raw_relation_text="no longer uses",
            raw_object_text="Redis", confidence=0.9, family_candidate="USES", negated=True,
        ),
        source_node_id=doc_id, source_kind="document",
        subject_entity_id=project, object_entity_id=redis, chunk_ids=[],
    )
    assert neg_redis.outcome == "superseded"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {id: $project_id})-[:AS_SUBJECT]->(f:MemoryFact {family: 'USES'})
            WHERE f.system_until IS NULL
            RETURN f.negated AS negated, f.object_entity_id AS obj
            ORDER BY f.object_entity_id
            """,
            project_id=project,
        )
        current = [dict(r) async for r in result]

    assert len(current) == 2
    redis_fact = next(f for f in current if f["obj"] == redis)
    kafka_fact = next(f for f in current if f["obj"] == kafka)
    assert redis_fact["negated"] is True
    assert kafka_fact["negated"] is False


@pytest.mark.asyncio
async def test_additive_value_cross_polarity_supersession(neo4j_driver):
    """Positive RECOMMENDED then negative RECOMMENDED on same subject+value should supersede."""
    doc_id, _, _ = await neo4j_store.merge_document("hash-add-val", "add-val-test", "text")
    team = await neo4j_store.merge_entity(
        "Platform Team", "Team", "add-val-test", 0.9, doc_id, "test"
    )

    pos = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document", source_id="add-val-test",
            raw_subject_text="Platform Team", raw_relation_text="recommended",
            raw_object_text="Redis", confidence=0.9, family_candidate="RECOMMENDED",
            value_text="Redis", negated=False,
        ),
        source_node_id=doc_id, source_kind="document",
        subject_entity_id=team, object_entity_id=None, chunk_ids=[],
    )
    assert pos.outcome == "created"

    neg = await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document", source_id="add-val-test",
            raw_subject_text="Platform Team", raw_relation_text="no longer recommends",
            raw_object_text="Redis", confidence=0.9, family_candidate="RECOMMENDED",
            value_text="Redis", negated=True,
        ),
        source_node_id=doc_id, source_kind="document",
        subject_entity_id=team, object_entity_id=None, chunk_ids=[],
    )
    assert neg.outcome == "superseded"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {id: $team_id})-[:AS_SUBJECT]->(f:MemoryFact {family: 'RECOMMENDED'})
            WHERE f.system_until IS NULL
            RETURN f.negated AS negated
            """,
            team_id=team,
        )
        current = [dict(r) async for r in result]

    assert len(current) == 1
    assert current[0]["negated"] is True


@pytest.mark.asyncio
async def test_additive_coexistence_unaffected_by_negation(neo4j_driver):
    """Negating USES Redis must not affect the live USES Kafka fact."""
    doc_id, _, _ = await neo4j_store.merge_document("hash-coex", "coex-test", "text")
    project = await neo4j_store.merge_entity(
        "Project Y", "Project", "coex-test", 0.9, doc_id, "test"
    )
    redis = await neo4j_store.merge_entity(
        "Redis", "Technology", "coex-test", 0.9, doc_id, "test"
    )
    kafka = await neo4j_store.merge_entity(
        "Kafka", "Technology", "coex-test", 0.9, doc_id, "test"
    )

    for obj, neg in [(redis, False), (kafka, False), (redis, True)]:
        await persist_assertion_and_maybe_promote(
            AssertionPayload(
                source_kind="document", source_id="coex-test",
                raw_subject_text="Project Y", raw_relation_text="uses",
                raw_object_text="target", confidence=0.9,
                family_candidate="USES", negated=neg,
            ),
            source_node_id=doc_id, source_kind="document",
            subject_entity_id=project, object_entity_id=obj, chunk_ids=[],
        )

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity {id: $project_id})-[:AS_SUBJECT]->(f:MemoryFact {family: 'USES'})
            WHERE f.system_until IS NULL
            RETURN f.negated AS negated, f.object_entity_id AS obj
            """,
            project_id=project,
        )
        current = [dict(r) async for r in result]

    assert len(current) == 2
    kafka_fact = next(f for f in current if f["obj"] == kafka)
    redis_fact = next(f for f in current if f["obj"] == redis)
    assert kafka_fact["negated"] is False
    assert redis_fact["negated"] is True


@pytest.mark.asyncio
async def test_negated_fact_surfaces_in_retrieval_output(neo4j_driver):
    """Negated MemoryFact must appear with negated=True in get_current_fact_details_for_entities."""
    from landscape.storage.neo4j_memory import get_current_fact_details_for_entities

    doc_id, _, _ = await neo4j_store.merge_document("hash-surf", "surf-test", "text")
    alice = await neo4j_store.merge_entity(
        "Alice", "Person", "surf-test", 0.9, doc_id, "test"
    )
    acme = await neo4j_store.merge_entity(
        "Acme", "Organization", "surf-test", 0.9, doc_id, "test"
    )

    await persist_assertion_and_maybe_promote(
        AssertionPayload(
            source_kind="document", source_id="surf-test",
            raw_subject_text="Alice", raw_relation_text="does not work for",
            raw_object_text="Acme", confidence=0.9, family_candidate="WORKS_FOR",
            negated=True,
        ),
        source_node_id=doc_id, source_kind="document",
        subject_entity_id=alice, object_entity_id=acme, chunk_ids=[],
    )

    facts, _ = await get_current_fact_details_for_entities([alice])
    assert len(facts) == 1
    assert facts[0]["negated"] is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memory_fact_defaults_effective_from_to_ingested_at(neo4j_driver):
    """When no extracted effective_from is supplied, storage floors
    effective_from to ingested_at so as_of queries can always anchor."""
    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store

    subj = "BiTempAlice"
    obj = "BiTempAcme"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, obj],
        )
    subject_id = await neo4j_store.merge_entity(subj, "PERSON", "bitemp-test", 0.9)
    object_id = await neo4j_store.merge_entity(obj, "ORGANIZATION", "bitemp-test", 0.9)

    assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="bitemp-default-test",
            raw_subject_text=subj,
            raw_relation_text="works for",
            raw_object_text=obj,
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    fact_id = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=subject_id,
        object_entity_id=object_id,
        subtype=None,
        confidence=0.9,
        assertion_id=assertion,
    )
    await neo4j_store.materialize_memory_rel(fact_id)

    async with neo4j_driver.session() as session:
        record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.ingested_at AS ingested_at, "
                "       f.effective_from AS effective_from, "
                "       f.effective_until AS effective_until, "
                "       f.system_until AS system_until",
                fid=fact_id,
            )
        ).single()
        edge = await (
            await session.run(
                "MATCH ()-[r:MEMORY_REL {memory_fact_id: $fid}]->() "
                "RETURN r.ingested_at AS ingested_at, "
                "       r.effective_from AS effective_from, "
                "       r.effective_until AS effective_until, "
                "       r.system_until AS system_until",
                fid=fact_id,
            )
        ).single()

    assert record is not None
    assert record["ingested_at"] is not None
    assert record["effective_from"] == record["ingested_at"], (
        "effective_from must default to ingested_at when no extracted value"
    )
    assert record["effective_until"] is None
    assert record["system_until"] is None

    assert edge is not None
    assert edge["ingested_at"] == record["ingested_at"]
    assert edge["effective_from"] == record["effective_from"]
    assert edge["effective_until"] is None
    assert edge["system_until"] is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_memory_fact_stores_explicit_effective_range(neo4j_driver):
    """When effective_from/effective_until are supplied, they are stored
    verbatim on both MemoryFact node and MEMORY_REL edge."""
    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store

    subj = "BiTempRangeBob"
    obj = "BiTempRangeZylos"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, obj],
        )
    subject_id = await neo4j_store.merge_entity(subj, "PERSON", "bitemp-range", 0.9)
    object_id = await neo4j_store.merge_entity(obj, "ORGANIZATION", "bitemp-range", 0.9)

    assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="bitemp-range-test",
            raw_subject_text=subj,
            raw_relation_text="worked for",
            raw_object_text=obj,
            confidence=0.9,
            family_candidate="WORKS_FOR",
        )
    )
    fact_id = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=subject_id,
        object_entity_id=object_id,
        subtype=None,
        confidence=0.9,
        assertion_id=assertion,
        effective_from="2020-03-01T00:00:00+00:00",
        effective_until="2023-09-30T23:59:59+00:00",
    )
    await neo4j_store.materialize_memory_rel(fact_id)

    async with neo4j_driver.session() as session:
        record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.effective_from AS effective_from, "
                "       f.effective_until AS effective_until",
                fid=fact_id,
            )
        ).single()
        edge = await (
            await session.run(
                "MATCH ()-[r:MEMORY_REL {memory_fact_id: $fid}]->() "
                "RETURN r.effective_from AS effective_from, "
                "       r.effective_until AS effective_until",
                fid=fact_id,
            )
        ).single()

    assert record["effective_from"] == "2020-03-01T00:00:00+00:00"
    assert record["effective_until"] == "2023-09-30T23:59:59+00:00"
    assert edge["effective_from"] == record["effective_from"]
    assert edge["effective_until"] == record["effective_until"]


@pytest.mark.unit
def test_assertion_payload_accepts_effective_fields():
    from landscape.memory_graph import AssertionPayload

    payload = AssertionPayload(
        source_kind="document",
        source_id="x",
        raw_subject_text="Alice",
        raw_relation_text="worked for",
        raw_object_text="Acme",
        confidence=0.9,
        effective_from="2020-01-01",
        effective_until="2023-12-31",
    )
    assert payload.effective_from == "2020-01-01"
    assert payload.effective_until == "2023-12-31"

    default = AssertionPayload(
        source_kind="document",
        source_id="x",
        raw_subject_text="Alice",
        raw_relation_text="works for",
        raw_object_text="Zylos",
        confidence=0.9,
    )
    assert default.effective_from is None
    assert default.effective_until is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_supersession_shares_now_with_new_fact_creation(neo4j_driver):
    """Within one ingestion pass, the closed old fact's system_until and
    the new fact's ingested_at must share the exact same timestamp value.
    Today both call datetime.now() independently and diverge by microseconds."""
    from datetime import UTC, datetime

    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store

    subj = "NowAlice"
    old_obj = "NowAcme"
    new_obj = "NowZylos"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, old_obj, new_obj],
        )
    subject_id = await neo4j_store.merge_entity(subj, "PERSON", "now-test", 0.9)
    old_object_id = await neo4j_store.merge_entity(old_obj, "ORGANIZATION", "now-test", 0.9)
    new_object_id = await neo4j_store.merge_entity(new_obj, "ORGANIZATION", "now-test", 0.9)

    shared_now = datetime.now(UTC).isoformat()

    old_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="now-old",
            raw_subject_text=subj,
            raw_relation_text="works for",
            raw_object_text=old_obj,
            confidence=0.9,
            family_candidate="WORKS_FOR",
        ),
        now=shared_now,
    )
    old_fact = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR",
        subject_entity_id=subject_id,
        object_entity_id=old_object_id,
        subtype=None,
        confidence=0.9,
        assertion_id=old_assertion,
        now=shared_now,
    )
    await neo4j_store.materialize_memory_rel(old_fact, now=shared_now)

    new_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document",
            source_id="now-new",
            raw_subject_text=subj,
            raw_relation_text="works for",
            raw_object_text=new_obj,
            confidence=0.95,
            family_candidate="WORKS_FOR",
        ),
        now=shared_now,
    )
    new_fact = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR",
        subject_entity_id=subject_id,
        object_entity_id=new_object_id,
        subtype=None,
        confidence=0.95,
        assertion_id=new_assertion,
        now=shared_now,
    )

    async with neo4j_driver.session() as session:
        old_record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.system_until AS system_until",
                fid=old_fact,
            )
        ).single()
        new_record = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.ingested_at AS ingested_at",
                fid=new_fact,
            )
        ).single()

    assert old_record["system_until"] == shared_now
    assert new_record["ingested_at"] == shared_now


@pytest.mark.asyncio
@pytest.mark.integration
async def test_supersession_closes_effective_until_with_new_fact_effective_from(
    neo4j_driver,
):
    """When a new fact has an extracted effective_from, the superseded old
    fact's effective_until must close to that timestamp, not to ingested_at."""
    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store

    subj = "EffSupAlice"
    old_obj = "EffSupAcme"
    new_obj = "EffSupZylos"
    new_effective_from = "2025-03-15T00:00:00+00:00"

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, old_obj, new_obj],
        )
    subject_id = await neo4j_store.merge_entity(subj, "PERSON", "eff-sup", 0.9)
    old_object_id = await neo4j_store.merge_entity(old_obj, "ORGANIZATION", "eff-sup", 0.9)
    new_object_id = await neo4j_store.merge_entity(new_obj, "ORGANIZATION", "eff-sup", 0.9)

    old_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document", source_id="eff-sup-old",
            raw_subject_text=subj, raw_relation_text="works for", raw_object_text=old_obj,
            confidence=0.9, family_candidate="WORKS_FOR",
        )
    )
    old_fact = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR", subject_entity_id=subject_id,
        object_entity_id=old_object_id, subtype=None,
        confidence=0.9, assertion_id=old_assertion,
    )
    await neo4j_store.materialize_memory_rel(old_fact)

    new_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document", source_id="eff-sup-new",
            raw_subject_text=subj, raw_relation_text="works for", raw_object_text=new_obj,
            confidence=0.95, family_candidate="WORKS_FOR",
            effective_from=new_effective_from,
        )
    )
    await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR", subject_entity_id=subject_id,
        object_entity_id=new_object_id, subtype=None,
        confidence=0.95, assertion_id=new_assertion,
        effective_from=new_effective_from,
    )

    async with neo4j_driver.session() as session:
        rec = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.effective_until AS effective_until, "
                "       f.system_until AS system_until",
                fid=old_fact,
            )
        ).single()
    assert rec["effective_until"] == new_effective_from
    assert rec["system_until"] is not None
    assert rec["system_until"] != new_effective_from  # ingestion-time, not the event time


@pytest.mark.asyncio
@pytest.mark.integration
async def test_supersession_falls_back_to_now_when_no_effective_from(neo4j_driver):
    from datetime import UTC, datetime

    from landscape.memory_graph import AssertionPayload
    from landscape.storage import neo4j_store

    subj = "EffFbAlice"
    old_obj = "EffFbAcme"
    new_obj = "EffFbZylos"
    shared_now = datetime.now(UTC).isoformat()

    await neo4j_store.ensure_memory_graph_schema()
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
            names=[subj, old_obj, new_obj],
        )
    subject_id = await neo4j_store.merge_entity(subj, "PERSON", "eff-fb", 0.9)
    old_object_id = await neo4j_store.merge_entity(old_obj, "ORGANIZATION", "eff-fb", 0.9)
    new_object_id = await neo4j_store.merge_entity(new_obj, "ORGANIZATION", "eff-fb", 0.9)

    old_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document", source_id="eff-fb-old",
            raw_subject_text=subj, raw_relation_text="works for", raw_object_text=old_obj,
            confidence=0.9, family_candidate="WORKS_FOR",
        ),
        now=shared_now,
    )
    old_fact = await neo4j_store.create_memory_fact_version(
        family="WORKS_FOR", subject_entity_id=subject_id,
        object_entity_id=old_object_id, subtype=None,
        confidence=0.9, assertion_id=old_assertion, now=shared_now,
    )
    await neo4j_store.materialize_memory_rel(old_fact, now=shared_now)

    new_assertion = await neo4j_store.merge_assertion(
        AssertionPayload(
            source_kind="document", source_id="eff-fb-new",
            raw_subject_text=subj, raw_relation_text="works for", raw_object_text=new_obj,
            confidence=0.95, family_candidate="WORKS_FOR",
        ),
        now=shared_now,
    )
    new_fact = await neo4j_store.supersede_single_current_fact(
        family="WORKS_FOR", subject_entity_id=subject_id,
        object_entity_id=new_object_id, subtype=None,
        confidence=0.95, assertion_id=new_assertion,
        now=shared_now,
    )

    async with neo4j_driver.session() as session:
        old_rec = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.effective_until AS effective_until",
                fid=old_fact,
            )
        ).single()
        new_rec = await (
            await session.run(
                "MATCH (f:MemoryFact {id: $fid}) "
                "RETURN f.ingested_at AS ingested_at",
                fid=new_fact,
            )
        ).single()

    assert old_rec["effective_until"] == shared_now
    assert old_rec["effective_until"] == new_rec["ingested_at"]
