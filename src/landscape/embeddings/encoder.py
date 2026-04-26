import os

from langchain_huggingface import HuggingFaceEmbeddings

from landscape.config import settings

_embeddings: HuggingFaceEmbeddings | None = None
TRUST_REMOTE_CODE_MODELS = frozenset({"nomic-ai/nomic-embed-text-v1.5"})


def _allow_remote_model_code() -> bool:
    return os.getenv("ALLOW_REMOTE_MODEL_CODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _trust_remote_code_for(model_name: str) -> bool:
    return _allow_remote_model_code() and model_name in TRUST_REMOTE_CODE_MODELS


def load_model() -> None:
    global _embeddings
    model_kwargs: dict = {
        "trust_remote_code": _trust_remote_code_for(settings.embedding_model)
    }
    if settings.hf_token:
        model_kwargs["token"] = settings.hf_token
    _embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    if _embeddings is None:
        raise RuntimeError("Encoder not loaded — call load_model() during app lifespan")
    return _embeddings


def embed_query(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def encode(text: str) -> list[float]:
    """Legacy alias used by the existing ingest pipeline."""
    return embed_query(text)
