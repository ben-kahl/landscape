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
