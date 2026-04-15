from sentence_transformers import SentenceTransformer

from landscape.config import settings

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    kwargs: dict = {"trust_remote_code": True}
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    _model = SentenceTransformer(settings.embedding_model, **kwargs)


def encode(text: str) -> list[float]:
    if _model is None:
        raise RuntimeError("Encoder not loaded — call load_model() during app lifespan")
    return _model.encode(text).tolist()
