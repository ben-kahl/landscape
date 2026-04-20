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
