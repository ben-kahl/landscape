"""Supersession and temporal chain integration tests."""
import pytest


async def _clear_doc(neo4j_driver, title: str) -> None:
    async with neo4j_driver.session() as session:
        await session.run("MATCH (d:Document {title: $title}) DETACH DELETE d", title=title)


async def _clear_relation(neo4j_driver, subject: str, rel_type: str) -> None:
    """Delete all RELATES_TO edges of a given type from the subject."""
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (s:Entity {name: $s})-[r:RELATES_TO {type: $t}]->() DELETE r",
            s=subject,
            t=rel_type,
        )


@pytest.mark.asyncio
async def test_reinforcement_appends_source_docs(http_client, neo4j_driver):
    """Same relation from two docs → one edge with both source_docs entries."""
    title1 = "supersession-test-reinforce-1"
    title2 = "supersession-test-reinforce-2"
    for t in (title1, title2):
        await _clear_doc(neo4j_driver, t)

    # Ensure the entities exist first via a forced direct merge
    from landscape.storage import neo4j_store

    doc_id1, _ = await neo4j_store.merge_document("hash-reinf-1", title1, "text")
    doc_id2, _ = await neo4j_store.merge_document("hash-reinf-2", title2, "text")
    await neo4j_store.merge_entity("Alice", "PERSON", title1, 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("Project Atlas", "PROJECT", title1, 0.9, doc_id1, "test")

    # Clear any stale LEADS edges
    await _clear_relation(neo4j_driver, "Alice", "LEADS")

    # First relation
    outcome1, _ = await neo4j_store.upsert_relation(
        "Alice", "Project Atlas", "LEADS", 0.8, title1
    )
    assert outcome1 == "created"

    # Second ingest same relation from different doc
    outcome2, _ = await neo4j_store.upsert_relation(
        "Alice", "Project Atlas", "LEADS", 0.9, title2
    )
    assert outcome2 == "reinforced"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO {type: 'LEADS'}]->
                  (o:Entity {name: 'Project Atlas'})
            WHERE r.valid_until IS NULL
            RETURN r.source_docs AS source_docs, r.confidence AS conf
            """
        )
        record = await result.single()

    assert record is not None
    source_docs = record["source_docs"] or []
    assert title1 in source_docs
    assert title2 in source_docs
    # Confidence is max of both
    assert abs(record["conf"] - 0.9) < 0.01


@pytest.mark.asyncio
async def test_supersession_marks_valid_until(http_client, neo4j_driver):
    """Alice WORKS_FOR Acme (doc1) → Alice WORKS_FOR Zylos (doc2) → old edge gets valid_until."""
    title1 = "supersession-test-sup-1"
    title2 = "supersession-test-sup-2"
    for t in (title1, title2):
        await _clear_doc(neo4j_driver, t)

    from landscape.storage import neo4j_store

    doc_id1, _ = await neo4j_store.merge_document("hash-sup-1", title1, "text")
    doc_id2, _ = await neo4j_store.merge_document("hash-sup-2", title2, "text")
    await neo4j_store.merge_entity("Alice", "PERSON", title1, 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("Acme Corp", "ORGANIZATION", title1, 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("Zylos", "ORGANIZATION", title2, 0.9, doc_id2, "test")

    # Clear stale WORKS_FOR edges
    await _clear_relation(neo4j_driver, "Alice", "WORKS_FOR")

    outcome1, _ = await neo4j_store.upsert_relation("Alice", "Acme Corp", "WORKS_FOR", 0.9, title1)
    assert outcome1 == "created"

    outcome2, _ = await neo4j_store.upsert_relation("Alice", "Zylos", "WORKS_FOR", 0.9, title2)
    assert outcome2 == "superseded"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO {type: 'WORKS_FOR'}]->(o:Entity)
            RETURN o.name AS target, r.valid_until AS valid_until, r.valid_from AS valid_from
            ORDER BY r.valid_from
            """
        )
        records = await result.data()

    assert len(records) == 2, f"Expected 2 WORKS_FOR edges, got {len(records)}: {records}"
    acme_edge = next((r for r in records if r["target"] == "Acme Corp"), None)
    zylos_edge = next((r for r in records if r["target"] == "Zylos"), None)

    assert acme_edge is not None
    assert zylos_edge is not None
    assert acme_edge["valid_until"] is not None, "Old edge must have valid_until set"
    assert zylos_edge["valid_until"] is None, "New edge must have valid_until = null"


@pytest.mark.asyncio
async def test_additive_different_rel_type(http_client, neo4j_driver):
    """Alice LEADS Atlas + Alice OWNS Atlas → both edges coexist, neither superseded."""
    title = "supersession-test-additive"
    await _clear_doc(neo4j_driver, title)

    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-add-1", title, "text")
    await neo4j_store.merge_entity("Alice", "PERSON", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Project Atlas", "PROJECT", title, 0.9, doc_id, "test")

    await _clear_relation(neo4j_driver, "Alice", "LEADS")
    await _clear_relation(neo4j_driver, "Alice", "OWNS")

    outcome1, _ = await neo4j_store.upsert_relation("Alice", "Project Atlas", "LEADS", 0.9, title)
    outcome2, _ = await neo4j_store.upsert_relation("Alice", "Project Atlas", "OWNS", 0.9, title)

    assert outcome1 == "created"
    assert outcome2 == "created"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO]->(o:Entity {name: 'Project Atlas'})
            WHERE r.valid_until IS NULL AND r.type IN ['LEADS', 'OWNS']
            RETURN count(r) AS cnt
            """
        )
        record = await result.single()

    assert record["cnt"] == 2, f"Both edges must coexist, got {record['cnt']}"


@pytest.mark.asyncio
async def test_object_keyed_subtype_supersedes(http_client, neo4j_driver):
    """HAS_TITLE is object-keyed — a new subtype on the same (s, o) supersedes.

    Alice HAS_TITLE Atlas[senior_engineer] → Alice HAS_TITLE Atlas[principal_engineer]
    The old edge should be marked superseded; the new edge live.
    """
    title1 = "supersession-test-has-title-1"
    title2 = "supersession-test-has-title-2"
    for t in (title1, title2):
        await _clear_doc(neo4j_driver, t)

    from landscape.storage import neo4j_store

    doc_id1, _ = await neo4j_store.merge_document("hash-ht-1", title1, "text")
    doc_id2, _ = await neo4j_store.merge_document("hash-ht-2", title2, "text")
    await neo4j_store.merge_entity("Alice", "Person", title1, 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("Atlas Corp", "Organization", title1, 0.9, doc_id1, "test")

    await _clear_relation(neo4j_driver, "Alice", "HAS_TITLE")

    o1, _ = await neo4j_store.upsert_relation(
        "Alice", "Atlas Corp", "HAS_TITLE", 0.9, title1, subtype="senior_engineer"
    )
    o2, _ = await neo4j_store.upsert_relation(
        "Alice", "Atlas Corp", "HAS_TITLE", 0.9, title2, subtype="principal_engineer"
    )
    assert o1 == "created"
    assert o2 == "superseded"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO {type: 'HAS_TITLE'}]->
                  (o:Entity {name: 'Atlas Corp'})
            RETURN r.subtype AS subtype, r.valid_until AS valid_until
            ORDER BY r.valid_from
            """
        )
        records = await result.data()

    assert len(records) == 2
    sr = next(r for r in records if r["subtype"] == "senior_engineer")
    pr = next(r for r in records if r["subtype"] == "principal_engineer")
    assert sr["valid_until"] is not None, "old title must be superseded"
    assert pr["valid_until"] is None, "new title must be live"


@pytest.mark.asyncio
async def test_object_keyed_different_object_is_additive(http_client, neo4j_driver):
    """HAS_TITLE at different orgs coexists — Alice held titles at both.

    Past Atlas title remains a live fact alongside current Beacon title,
    because the story of "no longer at Atlas" lives on WORKS_FOR, not HAS_TITLE.
    """
    title = "supersession-test-has-title-multi-org"
    await _clear_doc(neo4j_driver, title)

    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-htm", title, "text")
    await neo4j_store.merge_entity("Alice", "Person", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Atlas Corp", "Organization", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Beacon Corp", "Organization", title, 0.9, doc_id, "test")

    await _clear_relation(neo4j_driver, "Alice", "HAS_TITLE")

    o1, _ = await neo4j_store.upsert_relation(
        "Alice", "Atlas Corp", "HAS_TITLE", 0.9, title, subtype="senior_engineer"
    )
    o2, _ = await neo4j_store.upsert_relation(
        "Alice", "Beacon Corp", "HAS_TITLE", 0.9, title, subtype="principal_engineer"
    )
    assert o1 == "created"
    assert o2 == "created"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO {type: 'HAS_TITLE'}]->(o)
            WHERE r.valid_until IS NULL
            RETURN count(r) AS cnt
            """
        )
        record = await result.single()
    assert record["cnt"] == 2


@pytest.mark.asyncio
async def test_subtype_keyed_pref_supersedes_on_same_axis(http_client, neo4j_driver):
    """HAS_PREFERENCE is subtype-keyed. Changing favorite_color supersedes."""
    title = "supersession-test-pref-axis"
    await _clear_doc(neo4j_driver, title)

    from landscape.storage import neo4j_store

    doc_id, _ = await neo4j_store.merge_document("hash-pref-1", title, "text")
    await neo4j_store.merge_entity("Alice", "Person", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Blue", "Concept", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Green", "Concept", title, 0.9, doc_id, "test")
    await neo4j_store.merge_entity("Sushi", "Concept", title, 0.9, doc_id, "test")

    await _clear_relation(neo4j_driver, "Alice", "HAS_PREFERENCE")

    # Same axis, different object → supersede
    o1, _ = await neo4j_store.upsert_relation(
        "Alice", "Blue", "HAS_PREFERENCE", 0.9, title, subtype="favorite_color"
    )
    o2, _ = await neo4j_store.upsert_relation(
        "Alice", "Green", "HAS_PREFERENCE", 0.9, title, subtype="favorite_color"
    )
    # Different axis → additive, coexists
    o3, _ = await neo4j_store.upsert_relation(
        "Alice", "Sushi", "HAS_PREFERENCE", 0.9, title, subtype="favorite_food"
    )
    assert o1 == "created"
    assert o2 == "superseded"
    assert o3 == "created"

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Alice'})-[r:RELATES_TO {type: 'HAS_PREFERENCE'}]->(o)
            WHERE r.valid_until IS NULL
            RETURN o.name AS target, r.subtype AS subtype
            ORDER BY target
            """
        )
        records = await result.data()
    targets = {(r["target"], r["subtype"]) for r in records}
    assert targets == {("Green", "favorite_color"), ("Sushi", "favorite_food")}


@pytest.mark.asyncio
async def test_current_facts_query(http_client, neo4j_driver):
    """Query with valid_until IS NULL returns only current state.

    Uses REPORTS_TO — a functional relation — because MEMBER_OF is
    non-functional (a person can be a member of multiple teams) and no
    longer triggers Case 2 supersession."""
    title = "supersession-test-current"
    await _clear_doc(neo4j_driver, title)

    from landscape.storage import neo4j_store

    doc_id1, _ = await neo4j_store.merge_document("hash-curr-1", title, "text")
    doc_id2, _ = await neo4j_store.merge_document("hash-curr-2", title + "-2", "text")
    await neo4j_store.merge_entity("Bob", "PERSON", title, 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("MgrA", "PERSON", title, 0.9, doc_id1, "test")
    await neo4j_store.merge_entity("MgrB", "PERSON", title + "-2", 0.9, doc_id2, "test")

    await _clear_relation(neo4j_driver, "Bob", "REPORTS_TO")

    await neo4j_store.upsert_relation("Bob", "MgrA", "REPORTS_TO", 0.9, title)
    await neo4j_store.upsert_relation("Bob", "MgrB", "REPORTS_TO", 0.9, title + "-2")

    async with neo4j_driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Entity {name: 'Bob'})-[r:RELATES_TO {type: 'REPORTS_TO'}]->(o:Entity)
            WHERE r.valid_until IS NULL
            RETURN o.name AS current_manager
            """
        )
        records = await result.data()

    assert len(records) == 1, f"Expected 1 current fact, got {len(records)}: {records}"
    assert records[0]["current_manager"] == "MgrB"
