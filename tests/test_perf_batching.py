"""Tests for the batching + parallelization perf work (spec: specs/perf_batching_and_parallelization.md)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from landscape.config import EMBEDDING_MODEL_DIMS, Settings


def test_embedding_model_dims_map_contains_known_models():
    assert EMBEDDING_MODEL_DIMS["nomic-ai/nomic-embed-text-v1.5"] == 768
    assert EMBEDDING_MODEL_DIMS["sentence-transformers/all-MiniLM-L6-v2"] == 384


def test_settings_embedding_dims_resolves_from_model_name():
    s = Settings(embedding_model="nomic-ai/nomic-embed-text-v1.5")
    assert s.embedding_dims == 768
    s2 = Settings(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert s2.embedding_dims == 384


def test_settings_embedding_dims_raises_for_unknown_model():
    s = Settings(embedding_model="made-up/not-a-real-model")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        _ = s.embedding_dims


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,expected_dims",
    [
        ("nomic-ai/nomic-embed-text-v1.5", 768),
        ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ],
)
async def test_init_collection_uses_configured_dims(model_name, expected_dims):
    from landscape.storage import qdrant_store

    fake_client = MagicMock()
    fake_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    fake_client.create_collection = AsyncMock()

    with patch.object(qdrant_store, "get_client", return_value=fake_client), \
         patch("landscape.storage.qdrant_store.settings") as mock_settings:
        mock_settings.embedding_dims = expected_dims

        await qdrant_store.init_collection()

    call = fake_client.create_collection.call_args
    assert call.kwargs["vectors_config"].size == expected_dims


@pytest.mark.asyncio
async def test_retrieve_runs_both_filter_queries_in_parallel():
    """When session_id AND since are both set, the four Neo4j filter calls
    (entities-in-conv, entities-since, chunks-in-conv, chunks-since) must be
    gathered, not awaited serially."""
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from qdrant_client.models import ScoredPoint
    from landscape.retrieval import query as query_mod

    call_log: list[tuple[str, float]] = []
    start_time = asyncio.get_event_loop().time()

    async def slow_return(label, value):
        # 50ms per call — if serial, total wait is 200ms; if parallel, 50ms.
        await asyncio.sleep(0.05)
        call_log.append((label, asyncio.get_event_loop().time() - start_time))
        return value

    async def side_ent_conv(*a, **kw):
        return await slow_return("ent_conv", ["e1"])

    async def side_ent_since(*a, **kw):
        return await slow_return("ent_since", ["e1"])

    async def side_chunk_conv(*a, **kw):
        return await slow_return("chunk_conv", [])

    async def side_chunk_since(*a, **kw):
        return await slow_return("chunk_since", [])

    # Build a fake ScoredPoint so seed_sims is non-empty and the early
    # `if not seed_sims` return is bypassed.
    fake_hit = MagicMock(spec=ScoredPoint)
    fake_hit.score = 0.9
    fake_hit.payload = {"neo4j_node_id": "e1"}

    # _hydrate_entities and bfs_expand must be patched to prevent real Neo4j
    # connections while allowing the function to reach the filter block.
    fake_entity_row = {
        "eid": "e1",
        "name": "FakeEntity",
        "type": "Person",
        "access_count": 0,
        "last_accessed": None,
    }

    with patch.object(
        query_mod.qdrant_store, "search_entities_any_type",
        AsyncMock(return_value=[fake_hit]),
    ), patch.object(
        query_mod.qdrant_store, "search_chunks",
        AsyncMock(return_value=[]),
    ), patch.object(
        query_mod.encoder, "embed_query", return_value=[0.0] * 4,
    ), patch.object(
        query_mod.neo4j_store, "get_entities_from_chunks",
        AsyncMock(return_value=[]),
    ), patch(
        "landscape.retrieval.query._hydrate_entities",
        AsyncMock(return_value=[fake_entity_row]),
    ), patch.object(
        query_mod.neo4j_store, "bfs_expand",
        AsyncMock(return_value=[]),
    ), patch.object(
        query_mod.neo4j_store, "get_entities_in_conversation",
        AsyncMock(side_effect=side_ent_conv),
    ), patch.object(
        query_mod.neo4j_store, "get_entities_since",
        AsyncMock(side_effect=side_ent_since),
    ), patch.object(
        query_mod.neo4j_store, "get_chunks_in_conversation",
        AsyncMock(side_effect=side_chunk_conv),
    ), patch.object(
        query_mod.neo4j_store, "get_chunks_since",
        AsyncMock(side_effect=side_chunk_since),
    ):
        await query_mod.retrieve(
            "q",
            session_id="s1",
            since=datetime(2026, 1, 1, tzinfo=UTC),
            reinforce=False,
        )

    timestamps = [t for _, t in call_log]
    assert len(timestamps) == 4, f"expected 4 filter calls, got {len(timestamps)}: {call_log}"
    assert max(timestamps) - min(timestamps) < 0.03, (
        f"filter calls not parallelized — spread: {max(timestamps) - min(timestamps):.3f}s"
    )


@pytest.mark.asyncio
async def test_retrieve_runs_seed_searches_in_parallel():
    """search_entities_any_type and search_chunks query two different Qdrant
    collections and are independent — must be gathered."""
    from landscape.retrieval import query as query_mod

    start = asyncio.get_event_loop().time()
    times: dict[str, float] = {}

    async def slow_entities(*a, **kw):
        await asyncio.sleep(0.05)
        times["entities"] = asyncio.get_event_loop().time() - start
        return []

    async def slow_chunks(*a, **kw):
        await asyncio.sleep(0.05)
        times["chunks"] = asyncio.get_event_loop().time() - start
        return []

    with patch.object(query_mod.qdrant_store, "search_entities_any_type", AsyncMock(side_effect=slow_entities)), \
         patch.object(query_mod.qdrant_store, "search_chunks", AsyncMock(side_effect=slow_chunks)), \
         patch.object(query_mod.encoder, "embed_query", return_value=[0.0] * 4), \
         patch.object(query_mod.neo4j_store, "get_entities_from_chunks", AsyncMock(return_value=[])):
        await query_mod.retrieve("q", reinforce=False)

    assert abs(times["entities"] - times["chunks"]) < 0.03, (
        f"seed searches not parallel: entities={times['entities']:.3f}, "
        f"chunks={times['chunks']:.3f}"
    )


@pytest.mark.asyncio
async def test_retrieve_runs_reinforcement_writes_in_parallel():
    from landscape.retrieval import query as query_mod

    start = asyncio.get_event_loop().time()
    times: dict[str, float] = {}

    async def slow_ents(*a, **kw):
        await asyncio.sleep(0.05)
        times["ents"] = asyncio.get_event_loop().time() - start

    async def slow_rels(*a, **kw):
        await asyncio.sleep(0.05)
        times["rels"] = asyncio.get_event_loop().time() - start

    # Seed the retrieval with a single entity hit so reinforce runs.
    fake_hit = MagicMock()
    fake_hit.payload = {"neo4j_node_id": "e1"}
    fake_hit.score = 0.9

    with patch.object(query_mod.qdrant_store, "search_entities_any_type", AsyncMock(return_value=[fake_hit])), \
         patch.object(query_mod.qdrant_store, "search_chunks", AsyncMock(return_value=[])), \
         patch.object(query_mod.encoder, "embed_query", return_value=[0.0] * 4), \
         patch.object(query_mod.neo4j_store, "get_entities_from_chunks", AsyncMock(return_value=[])), \
         patch.object(query_mod.neo4j_store, "bfs_expand", AsyncMock(return_value=[])), \
         patch.object(query_mod, "_hydrate_entities", AsyncMock(return_value=[
             {"eid": "e1", "name": "E1", "type": "T", "access_count": 0, "last_accessed": None}
         ])), \
         patch.object(query_mod.neo4j_store, "touch_entities", AsyncMock(side_effect=slow_ents)), \
         patch.object(query_mod.neo4j_store, "touch_relations", AsyncMock(side_effect=slow_rels)):
        await query_mod.retrieve("q", reinforce=True)

    assert abs(times["ents"] - times["rels"]) < 0.03, (
        f"reinforce writes not parallel: ents={times['ents']:.3f}, rels={times['rels']:.3f}"
    )


@pytest.mark.asyncio
async def test_ingest_batch_encodes_chunks_once():
    """A 10-chunk document should trigger exactly one embed_documents call,
    not 10 per-chunk encode() calls."""
    from landscape import pipeline

    fake_chunks = [MagicMock(text=f"chunk {i}", index=i) for i in range(10)]
    with patch.object(pipeline, "chunk_text", return_value=fake_chunks), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.neo4j_store, "create_chunk",
                      AsyncMock(side_effect=[f"ch{i}" for i in range(10)])), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4 for _ in range(10)]) as batch_encode, \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4) as per_call, \
         patch.object(pipeline.qdrant_store, "upsert_chunk", AsyncMock()) as upsert, \
         patch.object(pipeline.llm, "extract",
                      return_value=MagicMock(entities=[], relations=[])):
        await pipeline.ingest(text="x" * 500, title="doc")

    assert batch_encode.call_count == 1
    assert len(batch_encode.call_args.args[0]) == 10
    # The per-chunk encode path should NOT be used for chunks anymore.
    chunk_encode_texts = {c.text for c in fake_chunks}
    per_call_texts = {c.args[0] for c in per_call.call_args_list if c.args}
    assert not chunk_encode_texts & per_call_texts, (
        "chunks are still going through per-call encoder.encode"
    )
    assert upsert.call_count == 10


@pytest.mark.asyncio
async def test_ingest_upserts_chunks_in_parallel():
    from landscape import pipeline

    start = asyncio.get_event_loop().time()
    finish_times: list[float] = []

    async def slow_upsert(*a, **kw):
        await asyncio.sleep(0.05)
        finish_times.append(asyncio.get_event_loop().time() - start)

    fake_chunks = [MagicMock(text=f"chunk {i}", index=i) for i in range(5)]
    with patch.object(pipeline, "chunk_text", return_value=fake_chunks), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.neo4j_store, "create_chunk",
                      AsyncMock(side_effect=[f"ch{i}" for i in range(5)])), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4 for _ in range(5)]), \
         patch.object(pipeline.qdrant_store, "upsert_chunk", AsyncMock(side_effect=slow_upsert)), \
         patch.object(pipeline.llm, "extract",
                      return_value=MagicMock(entities=[], relations=[])):
        await pipeline.ingest(text="x" * 500, title="doc")

    # Serial: 5 * 0.05 = 0.25s. Parallel: all within 0.06s of each other.
    assert max(finish_times) - min(finish_times) < 0.03


@pytest.mark.asyncio
async def test_ingest_batch_encodes_entities_once():
    from landscape import pipeline

    # Build entity mocks with distinct names so none dedupe.
    fake_entities = [MagicMock() for _ in range(5)]
    for i, e in enumerate(fake_entities):
        e.name = f"E{i}"
        e.type = "Person"
        e.confidence = 0.9

    extraction = MagicMock(entities=fake_entities, relations=[])

    with patch.object(pipeline, "chunk_text", return_value=[]), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4 for _ in range(5)]) as batch_encode, \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4), \
         patch.object(pipeline.resolver, "resolve_entity",
                      AsyncMock(return_value=(None, True, None))), \
         patch.object(pipeline.neo4j_store, "merge_entity",
                      AsyncMock(side_effect=[f"ent{i}" for i in range(5)])), \
         patch.object(pipeline.qdrant_store, "upsert_entity", AsyncMock()), \
         patch.object(pipeline.llm, "extract", return_value=extraction):
        await pipeline.ingest(text="some doc", title="t")

    # Entity encode should be batched into one call that encodes all 5 names.
    entity_batch_calls = [c for c in batch_encode.call_args_list if len(c.args[0]) == 5]
    assert entity_batch_calls, (
        f"expected one batch of 5 entity texts, got batches: "
        f"{[len(c.args[0]) for c in batch_encode.call_args_list]}"
    )


@pytest.mark.asyncio
async def test_ingest_dedupes_identical_entity_mentions_before_resolving():
    """Two extracted entities with the same (name, type) in one doc should
    resolve once, not twice — the serial loop used to rely on the first
    resolution creating a canonical node that the second would match."""
    from landscape import pipeline

    e1 = MagicMock()
    e1.name = "Alice"
    e1.type = "Person"
    e1.confidence = 0.9
    e2 = MagicMock()
    e2.name = "alice"  # case difference — must dedupe
    e2.type = "Person"
    e2.confidence = 0.9
    extraction = MagicMock(entities=[e1, e2], relations=[])

    resolve_mock = AsyncMock(return_value=(None, True, None))

    with patch.object(pipeline, "chunk_text", return_value=[]), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4]), \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4), \
         patch.object(pipeline.resolver, "resolve_entity", resolve_mock), \
         patch.object(pipeline.neo4j_store, "merge_entity",
                      AsyncMock(return_value="ent1")), \
         patch.object(pipeline.neo4j_store, "link_entity_to_doc", AsyncMock()), \
         patch.object(pipeline.qdrant_store, "upsert_entity", AsyncMock()), \
         patch.object(pipeline.llm, "extract", return_value=extraction):
        await pipeline.ingest(text="some doc", title="t")

    assert resolve_mock.call_count == 1, (
        f"expected one resolve call for deduped (Alice/alice), got {resolve_mock.call_count}"
    )


@pytest.mark.asyncio
async def test_ingest_dedupes_whitespace_variants_of_same_entity():
    """Leading/trailing whitespace in extracted names should not create
    duplicate resolve calls for the same underlying entity."""
    from landscape import pipeline

    e1 = MagicMock()
    e1.name = "Alice "  # trailing space
    e1.type = "Person"
    e1.confidence = 0.9
    e2 = MagicMock()
    e2.name = " Alice"  # leading space
    e2.type = "Person"
    e2.confidence = 0.9
    extraction = MagicMock(entities=[e1, e2], relations=[])

    resolve_mock = AsyncMock(return_value=(None, True, None))

    with patch.object(pipeline, "chunk_text", return_value=[]), \
         patch.object(pipeline.neo4j_store, "merge_document",
                      AsyncMock(return_value=("doc1", True))), \
         patch.object(pipeline.encoder, "embed_documents",
                      return_value=[[0.0] * 4]), \
         patch.object(pipeline.encoder, "encode", return_value=[0.0] * 4), \
         patch.object(pipeline.resolver, "resolve_entity", resolve_mock), \
         patch.object(pipeline.neo4j_store, "merge_entity",
                      AsyncMock(return_value="ent1")), \
         patch.object(pipeline.neo4j_store, "link_entity_to_doc", AsyncMock()), \
         patch.object(pipeline.qdrant_store, "upsert_entity", AsyncMock()), \
         patch.object(pipeline.llm, "extract", return_value=extraction):
        await pipeline.ingest(text="some doc", title="t")

    assert resolve_mock.call_count == 1, (
        f"expected one resolve call for whitespace-variant names, got {resolve_mock.call_count}"
    )
    # The canonical stored name should be trimmed (no leading/trailing whitespace).
    resolve_args = resolve_mock.call_args
    stored_name = resolve_args.kwargs["name"]
    assert stored_name == stored_name.strip(), (
        f"stored name still has whitespace: {stored_name!r}"
    )
