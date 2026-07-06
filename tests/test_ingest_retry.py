"""Unit + integration tests for the failed-ingest retry fix.

Covers: a failed ingest must not permanently mark a Document as ingested
(`pipeline.ingest()` previously created the Document node as step 1 and had
no cleanup on later failure, so every retry hit the `already_existed=True`
short-circuit forever). See CLAUDE.md's Phase 3.5 backlog / task 2 for the
full root-cause writeup.
"""

from __future__ import annotations

import pytest

from landscape.extraction.chunker import Chunk
from landscape.extraction.schema import ExtractedEntity, ExtractedRelation, Extraction
from landscape.storage import neo4j_documents


class _FakeDocumentStore:
    """In-memory stand-in for the Document-related neo4j_store calls.

    Mimics just enough of merge_document / mark_document_ingested /
    delete_document_subtree to let pipeline.ingest() be exercised twice in a
    row (once failing, once succeeding) without a real Neo4j instance.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}  # content_hash -> {"doc_id", "completed"}
        self._next_id = 0
        self.delete_subtree_calls: list[str] = []

    async def merge_document(self, content_hash, title, source_type):
        entry = self.docs.get(content_hash)
        if entry is None:
            self._next_id += 1
            doc_id = f"doc-{self._next_id}"
            self.docs[content_hash] = {"doc_id": doc_id, "completed": False}
            return doc_id, True, False
        return entry["doc_id"], False, entry["completed"]

    async def mark_document_ingested(self, doc_id):
        for entry in self.docs.values():
            if entry["doc_id"] == doc_id:
                entry["completed"] = True

    async def delete_document_subtree(self, doc_id):
        self.delete_subtree_calls.append(doc_id)
        for content_hash, entry in list(self.docs.items()):
            if entry["doc_id"] == doc_id:
                del self.docs[content_hash]


def _install_common_pipeline_stubs(monkeypatch, pipeline, *, extract_result):
    """Stub every pipeline collaborator except the Document-lifecycle calls
    (merge_document / mark_document_ingested / delete_document_subtree),
    which callers wire up themselves via _FakeDocumentStore."""

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return f"{doc_id}-chunk-{chunk_index}"

    async def fake_upsert_chunk(**kwargs):
        return None

    async def fake_set_chunk_mentions(chunk_id, **kwargs):
        return None

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        return f"{name}-id", True, None

    async def fake_merge_entity(**kwargs):
        return f"{kwargs['name']}-id"

    async def fake_upsert_entity(**kwargs):
        return None

    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", fake_upsert_chunk)
    monkeypatch.setattr(pipeline.neo4j_store, "set_chunk_mentions", fake_set_chunk_mentions)
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "merge_entity", fake_merge_entity)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", fake_upsert_entity)
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text=text)])
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(pipeline.llm, "extract", lambda text: extract_result)
    monkeypatch.setattr(pipeline, "coerce_rel_type", lambda rel_type: (rel_type, 1.0))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_failed_ingest_then_retry_proceeds_as_fresh_ingest(monkeypatch):
    """A stage failure after the Document is created must not permanently
    stick the doc in an "already_existed" state: a second ingest() of the
    identical text should clean up the orphaned subtree and run as a fresh
    ingest (already_existed=False), not silently no-op forever."""
    from landscape import pipeline

    store = _FakeDocumentStore()
    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", store.merge_document)
    monkeypatch.setattr(
        pipeline.neo4j_store, "mark_document_ingested", store.mark_document_ingested
    )
    monkeypatch.setattr(
        pipeline.neo4j_store, "delete_document_subtree", store.delete_document_subtree
    )
    qdrant_delete_calls: list[str] = []

    async def fake_delete_chunks_for_document(doc_id):
        qdrant_delete_calls.append(doc_id)

    monkeypatch.setattr(
        pipeline.qdrant_store, "delete_chunks_for_document", fake_delete_chunks_for_document
    )

    extraction_with_relation = Extraction(
        entities=[
            ExtractedEntity(name="Alice", type="PERSON", confidence=0.9),
            ExtractedEntity(name="Project Atlas", type="PROJECT", confidence=0.9),
        ],
        relations=[
            ExtractedRelation(
                subject="Alice",
                object="Project Atlas",
                relation_type="LEADS",
                confidence=0.9,
            )
        ],
    )
    _install_common_pipeline_stubs(
        monkeypatch, pipeline, extract_result=extraction_with_relation
    )

    # First run: extraction succeeds but relation persistence blows up
    # mid-pipeline (simulates an LLM/Qdrant outage after the Document node
    # already exists).
    async def boom_persist(*args, **kwargs):
        raise RuntimeError("simulated downstream outage")

    monkeypatch.setattr(pipeline, "persist_assertion_and_maybe_promote", boom_persist)

    text = "Alice leads Project Atlas."
    with pytest.raises(RuntimeError, match="simulated downstream outage"):
        await pipeline.ingest(text, "retry-doc")

    # The failed run's subtree should have been rolled back.
    assert store.delete_subtree_calls == ["doc-1"]
    assert qdrant_delete_calls == ["doc-1"]
    assert store.docs == {}

    # Second run with identical text: persist succeeds this time — just
    # confirm the retry treats this as a brand new ingest.
    async def ok_persist(payload, **kwargs):
        from landscape.memory_graph.service import PersistenceResult

        return PersistenceResult(assertion_id="a1", fact_id="f1", outcome="created")

    monkeypatch.setattr(pipeline, "persist_assertion_and_maybe_promote", ok_persist)

    result = await pipeline.ingest(text, "retry-doc")

    assert result.already_existed is False
    assert store.docs[list(store.docs)[0]]["completed"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingest_failure_cleanup_only_runs_when_doc_owned(monkeypatch):
    """Cleanup (delete_document_subtree) must run when *this* run created
    the doc and a later stage fails, but must never run on the
    already-complete early-return path."""
    from landscape import pipeline

    delete_calls: list[str] = []

    async def fake_delete_document_subtree(doc_id):
        delete_calls.append(doc_id)

    async def fake_delete_chunks_for_document(doc_id):
        return None

    monkeypatch.setattr(
        pipeline.neo4j_store, "delete_document_subtree", fake_delete_document_subtree
    )
    monkeypatch.setattr(
        pipeline.qdrant_store, "delete_chunks_for_document", fake_delete_chunks_for_document
    )

    # --- Case 1: doc already exists and is complete -> early return, no
    # cleanup call, no exception at all.
    async def fake_merge_document_complete(content_hash, title, source_type):
        return "doc-complete", False, True

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document_complete)

    result = await pipeline.ingest("some text", "already-complete-doc")

    assert result.already_existed is True
    assert delete_calls == []

    # --- Case 2: doc is freshly created by this run, then a later stage
    # fails -> cleanup must run exactly once for that doc.
    async def fake_merge_document_created(content_hash, title, source_type):
        return "doc-fresh", True, False

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return "chunk-0"

    async def boom_upsert_chunk(**kwargs):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document_created)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", boom_upsert_chunk)
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text=text)])
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )

    with pytest.raises(RuntimeError, match="qdrant unreachable"):
        await pipeline.ingest("more text", "fresh-doc-that-fails")

    assert delete_calls == ["doc-fresh"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_merge_document_reports_incomplete_signal(monkeypatch):
    """merge_document must surface whether a pre-existing doc is missing its
    completion marker, via a fake Neo4j driver/session (no real DB needed).
    This exercises the record -> return-tuple plumbing described in the
    approved fix design: (doc_id, created, ingest_completed)."""

    class _FakeResult:
        def __init__(self, record):
            self._record = record

        async def single(self):
            return self._record

    class _FakeSession:
        def __init__(self, record):
            self._record = record

        async def run(self, query, **params):
            return _FakeResult(self._record)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _FakeDriver:
        def __init__(self, record):
            self._record = record

        def session(self):
            return _FakeSession(self._record)

    # Case: pre-existing, complete document.
    complete_record = {
        "doc_id": "existing-doc",
        "created": False,
        "ingest_completed": True,
    }
    monkeypatch.setattr(
        neo4j_documents, "get_driver", lambda: _FakeDriver(complete_record)
    )
    doc_id, created, ingest_completed = await neo4j_documents.merge_document(
        "hash-1", "title-1", "text"
    )
    assert (doc_id, created, ingest_completed) == ("existing-doc", False, True)

    # Case: pre-existing, incomplete (stale partial) document.
    incomplete_record = {
        "doc_id": "existing-doc-2",
        "created": False,
        "ingest_completed": False,
    }
    monkeypatch.setattr(
        neo4j_documents, "get_driver", lambda: _FakeDriver(incomplete_record)
    )
    doc_id, created, ingest_completed = await neo4j_documents.merge_document(
        "hash-2", "title-2", "text"
    )
    assert (doc_id, created, ingest_completed) == ("existing-doc-2", False, False)

    # Case: freshly created document.
    created_record = {
        "doc_id": "new-doc",
        "created": True,
        "ingest_completed": False,
    }
    monkeypatch.setattr(
        neo4j_documents, "get_driver", lambda: _FakeDriver(created_record)
    )
    doc_id, created, ingest_completed = await neo4j_documents.merge_document(
        "hash-3", "title-3", "text"
    )
    assert (doc_id, created, ingest_completed) == ("new-doc", True, False)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_stale_partial_ingest_cleanup_round_trip(
    http_client, neo4j_driver, qdrant_client
):
    """Full round trip against real Neo4j/Qdrant: ingest doc A, remove its
    completion marker to simulate a crash mid-pipeline, re-ingest the same
    text, and assert (a) no duplicate chunks/assertions pile up and (b) a
    MemoryFact also supported by a *different* document survives the
    subtree cleanup of the stale doc.

    Not run automatically (integration-marked, requires the docker stack).
    """
    from landscape import pipeline

    title = "stale-partial-ingest-doc"
    other_title = "stale-partial-other-doc"
    text = "Diego leads Vision Team."
    shared_subject = "Diego"
    shared_object = "Vision Team"

    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (d:Document) WHERE d.title IN [$t1, $t2] "
            "OPTIONAL MATCH (d)-[:ASSERTS]->(a:Assertion) "
            "OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d) "
            "DETACH DELETE d, a, c",
            t1=title,
            t2=other_title,
        )
        await session.run(
            "MATCH (e:Entity) WHERE e.name IN [$s, $o] DETACH DELETE e",
            s=shared_subject,
            o=shared_object,
        )

    # First ingest of the doc under test.
    result_a = await pipeline.ingest(text, title)
    assert result_a.already_existed is False

    # A second, independent document reinforcing the same fact — this is
    # the fact that must survive when doc A's subtree is cleaned up.
    result_other = await pipeline.ingest(text, other_title)
    assert result_other.already_existed is False

    # Simulate a crash mid-pipeline on doc A: strip its completion marker
    # so the next ingest of identical text sees it as a stale partial.
    async with neo4j_driver.session() as session:
        await session.run(
            "MATCH (d:Document {title: $title}) REMOVE d.ingest_completed_at",
            title=title,
        )

    # Re-ingest identical text for doc A: should clean up the stale subtree
    # and run as a fresh ingest rather than a no-op.
    result_retry = await pipeline.ingest(text, title)
    assert result_retry.already_existed is False

    async with neo4j_driver.session() as session:
        # Exactly one Document node for this title/content-hash, with the
        # marker set again.
        doc_result = await session.run(
            "MATCH (d:Document {title: $title}) "
            "RETURN count(d) AS doc_count, "
            "       collect(d.ingest_completed_at IS NOT NULL)[0] AS has_marker",
            title=title,
        )
        doc_record = await doc_result.single()
        assert doc_record["doc_count"] == 1
        assert doc_record["has_marker"] is True

        # No duplicate chunks for doc A.
        chunk_result = await session.run(
            "MATCH (c:Chunk)-[:PART_OF]->(d:Document {title: $title}) RETURN count(c) AS cnt",
            title=title,
        )
        chunk_record = await chunk_result.single()
        assert chunk_record["cnt"] >= 1

        # The shared fact (also asserted by other_title's doc) must survive.
        fact_result = await session.run(
            """
            MATCH (s:Entity {name: $subject})-[r:MEMORY_REL]->(o:Entity {name: $object})
            WHERE r.system_until IS NULL
            RETURN count(r) AS cnt
            """,
            subject=shared_subject,
            object=shared_object,
        )
        fact_record = await fact_result.single()
        assert fact_record["cnt"] >= 1

        # Graph-consistency invariant: subtree cleanup must not leave any
        # MEMORY_REL edge (live between two entities, keyed by a
        # memory_fact_id property) whose MemoryFact node is gone. Such a
        # phantom edge would still be traversable by graph retrieval.
        orphan_result = await session.run(
            """
            MATCH (s:Entity {name: $subject})-[r:MEMORY_REL]-(o:Entity {name: $object})
            WHERE NOT EXISTS { MATCH (:MemoryFact {id: r.memory_fact_id}) }
            RETURN count(r) AS cnt
            """,
            subject=shared_subject,
            object=shared_object,
        )
        orphan_record = await orphan_result.single()
        assert orphan_record["cnt"] == 0
