import types

import pytest

from landscape.embeddings import encoder


@pytest.mark.unit
def test_load_model_disables_remote_code_by_default(monkeypatch):
    captured = {}

    def fake_embeddings(*, model_name, model_kwargs, encode_kwargs):
        captured["model_name"] = model_name
        captured["model_kwargs"] = model_kwargs
        captured["encode_kwargs"] = encode_kwargs
        return object()

    monkeypatch.delenv("ALLOW_REMOTE_MODEL_CODE", raising=False)
    monkeypatch.setattr(
        encoder,
        "settings",
        types.SimpleNamespace(
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            hf_token=None,
        ),
    )
    monkeypatch.setattr(encoder, "HuggingFaceEmbeddings", fake_embeddings)

    encoder._embeddings = None
    encoder.load_model()

    assert captured["model_name"] == "nomic-ai/nomic-embed-text-v1.5"
    assert captured["model_kwargs"]["trust_remote_code"] is False
    assert captured["encode_kwargs"] == {"normalize_embeddings": True}


@pytest.mark.unit
def test_load_model_allows_allowlisted_remote_code_when_opted_in(monkeypatch):
    captured = {}

    def fake_embeddings(*, model_name, model_kwargs, encode_kwargs):
        captured["model_name"] = model_name
        captured["model_kwargs"] = model_kwargs
        captured["encode_kwargs"] = encode_kwargs
        return object()

    monkeypatch.setenv("ALLOW_REMOTE_MODEL_CODE", "true")
    monkeypatch.setattr(
        encoder,
        "settings",
        types.SimpleNamespace(
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            hf_token="hf-test-token",
        ),
    )
    monkeypatch.setattr(encoder, "HuggingFaceEmbeddings", fake_embeddings)

    encoder._embeddings = None
    encoder.load_model()

    assert captured["model_kwargs"]["trust_remote_code"] is True
    assert captured["model_kwargs"]["token"] == "hf-test-token"


@pytest.mark.unit
def test_load_model_does_not_trust_non_allowlisted_models(monkeypatch):
    captured = {}

    def fake_embeddings(*, model_name, model_kwargs, encode_kwargs):
        captured["model_name"] = model_name
        captured["model_kwargs"] = model_kwargs
        captured["encode_kwargs"] = encode_kwargs
        return object()

    monkeypatch.setenv("ALLOW_REMOTE_MODEL_CODE", "true")
    monkeypatch.setattr(
        encoder,
        "settings",
        types.SimpleNamespace(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            hf_token=None,
        ),
    )
    monkeypatch.setattr(encoder, "HuggingFaceEmbeddings", fake_embeddings)

    encoder._embeddings = None
    encoder.load_model()

    assert captured["model_kwargs"]["trust_remote_code"] is False
