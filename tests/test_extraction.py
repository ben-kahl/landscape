"""Unit tests for LLM extraction wiring of effective_from / effective_until fields.

These tests stub pipeline.llm.extract so no LLM connection is required.
They verify that:
  1. effective_from / effective_until on ExtractedRelation are parsed via
     _parse_iso_temporal and forwarded as ISO-8601 UTC strings on the resulting
     AssertionPayload.
  2. Backwards ranges (effective_until < effective_from) drop both endpoints and
     emit a warning log.
"""

import pytest

from landscape.extraction.schema import ExtractedEntity, ExtractedRelation, Extraction
from landscape.memory_graph.service import PersistenceResult

# ---------------------------------------------------------------------------
# Shared pipeline stub helpers
# ---------------------------------------------------------------------------

def _make_pipeline_stubs(monkeypatch, extraction: Extraction) -> dict:
    """Patch all pipeline I/O so ingest() runs fully in-process.

    Returns a dict that will hold the captured AssertionPayload after ingest
    completes.
    """
    from landscape import pipeline
    from landscape.extraction.chunker import Chunk

    captured: dict = {}

    async def fake_merge_document(content_hash, title, source_type):
        return "doc-stub", True, False

    async def fake_create_chunk(doc_id, chunk_index, text, content_hash):
        return f"chunk-{chunk_index}"

    async def fake_upsert_chunk(**kwargs):
        return None

    async def fake_resolve_entity(name, entity_type, vector, source_doc):
        return f"{name}-id", True, 0.0

    async def fake_merge_entity(**kwargs):
        return f"{kwargs['name']}-id"

    async def fake_upsert_entity(**kwargs):
        return None

    async def fake_set_chunk_mentions(chunk_id, **kwargs):
        return None

    async def fake_mark_document_ingested(doc_id):
        return None

    async def fake_persist(
        payload,
        *,
        source_node_id,
        source_kind,
        subject_entity_id,
        object_entity_id,
        chunk_ids,
        now=None,
    ):
        captured["payload"] = payload
        return PersistenceResult(
            assertion_id="assertion-stub",
            fact_id="fact-stub",
            outcome="created",
        )

    monkeypatch.setattr(pipeline.neo4j_store, "merge_document", fake_merge_document)
    monkeypatch.setattr(pipeline.neo4j_store, "create_chunk", fake_create_chunk)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_chunk", fake_upsert_chunk)
    monkeypatch.setattr(pipeline.resolver, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "merge_entity", fake_merge_entity)
    monkeypatch.setattr(pipeline.qdrant_store, "upsert_entity", fake_upsert_entity)
    monkeypatch.setattr(pipeline.neo4j_store, "set_chunk_mentions", fake_set_chunk_mentions)
    monkeypatch.setattr(
        pipeline.neo4j_store, "mark_document_ingested", fake_mark_document_ingested
    )
    monkeypatch.setattr(pipeline, "persist_assertion_and_maybe_promote", fake_persist)
    monkeypatch.setattr(pipeline, "coerce_rel_type", lambda rel_type: (rel_type, 1.0))
    # Stub chunking so the real HF tokenizer (chunker._get_splitter) never loads —
    # keeps these unit tests hermetic and offline-safe.
    monkeypatch.setattr(pipeline, "chunk_text", lambda text: [Chunk(index=0, text=text)])
    monkeypatch.setattr(
        pipeline.encoder,
        "embed_documents",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(pipeline.llm, "extract", lambda text: extraction)

    return captured


# ---------------------------------------------------------------------------
# Test 1: effective fields propagate through pipeline → AssertionPayload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.unit
async def test_extractor_propagates_effective_fields_from_llm_output(monkeypatch):
    """Stub the LLM to return a relation with effective_from / effective_until;
    assert the resulting AssertionPayload carries fully-normalised ISO-8601
    UTC timestamps produced by _parse_iso_temporal."""
    from landscape import pipeline

    extraction = Extraction(
        entities=[
            ExtractedEntity(name="Alice", type="PERSON", confidence=0.95),
            ExtractedEntity(name="Acme", type="ORGANIZATION", confidence=0.95),
        ],
        relations=[
            ExtractedRelation(
                subject="Alice",
                object="Acme",
                relation_type="WORKS_FOR",
                confidence=0.9,
                effective_from="2020-03",
                effective_until="2023-09",
            )
        ],
    )

    captured = _make_pipeline_stubs(monkeypatch, extraction)

    await pipeline.ingest(
        "Alice worked at Acme from March 2020 to September 2023.",
        "test-effective-propagation",
    )

    assert "payload" in captured, "persist_assertion_and_maybe_promote was never called"
    p = captured["payload"]
    assert p.effective_from == "2020-03-01T00:00:00+00:00", (
        f"effective_from mismatch: {p.effective_from!r}"
    )
    assert p.effective_until == "2023-09-30T23:59:59+00:00", (
        f"effective_until mismatch: {p.effective_until!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: backwards range → both fields drop, warning logged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.unit
async def test_extractor_drops_backwards_effective_range(monkeypatch, caplog):
    """Stub the LLM to return effective_from > effective_until (backwards range);
    assert both are None on the resulting AssertionPayload and a warning is
    logged containing 'backwards effective range'."""
    import logging

    from landscape import pipeline

    extraction = Extraction(
        entities=[
            ExtractedEntity(name="Alice", type="PERSON", confidence=0.95),
            ExtractedEntity(name="Acme", type="ORGANIZATION", confidence=0.95),
        ],
        relations=[
            ExtractedRelation(
                subject="Alice",
                object="Acme",
                relation_type="WORKS_FOR",
                confidence=0.9,
                effective_from="2025-01-01",
                effective_until="2020-01-01",
            )
        ],
    )

    captured = _make_pipeline_stubs(monkeypatch, extraction)

    with caplog.at_level(logging.WARNING, logger="landscape.pipeline"):
        await pipeline.ingest("nonsense backwards range text", "test-backwards-range")

    assert "payload" in captured, "persist_assertion_and_maybe_promote was never called"
    p = captured["payload"]
    assert p.effective_from is None, f"expected effective_from=None, got {p.effective_from!r}"
    assert p.effective_until is None, f"expected effective_until=None, got {p.effective_until!r}"
    assert "backwards effective range" in caplog.text.lower(), (
        f"expected warning not found in log. caplog.text={caplog.text!r}"
    )
