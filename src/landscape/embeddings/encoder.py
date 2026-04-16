from langchain_huggingface import HuggingFaceEmbeddings

from landscape.config import settings

_embeddings: HuggingFaceEmbeddings | None = None


def load_model() -> None:
    global _embeddings
    model_kwargs: dict = {"trust_remote_code": True}
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
