import math
from dataclasses import dataclass

from pydantic_settings import BaseSettings


@dataclass(frozen=True)
class LLMProfile:
    """A named bundle of LLM settings. Switched via LANDSCAPE_LLM_PROFILE.

    To add a new profile:
      1. Append an entry to LLM_PROFILES below (any ollama-compatible tag).
      2. `docker compose --profile gpu-nvidia exec ollama-nvidia ollama pull <tag>`
      3. Set `LANDSCAPE_LLM_PROFILE=<name>` in .env or the shell.

    `ollama_tag` selects the model and `thinking` is forwarded to Ollama's
    chat API for thinking-capable models."""

    ollama_tag: str
    temperature: float = 0.0
    num_ctx: int = 8192
    thinking: bool = False
    notes: str = ""


# Seeded with the current default only. Add new profiles here — the user's
# own additions, model bakeoffs, etc. — instead of editing `llm_model` directly.
LLM_PROFILES: dict[str, LLMProfile] = {
    "llama31_8b": LLMProfile(
        ollama_tag="llama3.1:8b",
        thinking=False,
        notes="Phase 2 baseline. Known-good for the killer-demo corpus.",
    ),
    "qwen25_7b_nothink": LLMProfile(
           ollama_tag="qwen2.5:7b",
           thinking=False,
           notes="Qwen 2.5 7B with thinking disabled",
    ),
}


# Embedding model → output dimension. Extend this when switching/adding models.
# Source: published model card of each encoder.
EMBEDDING_MODEL_DIMS: dict[str, int] = {
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "landscape-dev"
    qdrant_url: str = "http://qdrant:6333"
    ollama_url: str = "http://ollama:11434"
    llm_profile: str = "llama31_8b"
    # Escape hatch: set LANDSCAPE_LLM_MODEL to override the profile's ollama_tag
    # without touching LLM_PROFILES. Useful for one-off A/B tests.
    llm_model: str | None = None
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    hf_token: str | None = None

    scoring_alpha: float = 1.0  # vector similarity
    scoring_beta: float = 0.8  # graph proximity
    scoring_gamma: float = 0.2  # reinforcement (multiplicative, was 0.4 additive)
    scoring_delta: float = 0.3  # edge confidence
    # decay_lambda chosen so exp(-lambda * 7 days) == 0.5 — a true 7-day half-life
    decay_lambda: float = math.log(2) / (7 * 86400)
    reinforcement_cap: float = 2.0

    mcp_issuer_url: str = "http://127.0.0.1:8000"
    auth_update_last_used_interval_seconds: int = 300
    # Local SQLite file backing the api-client / bearer-secret tables. Default
    # is XDG-ish under the user's home so a fresh checkout doesn't write into
    # the repo. Override via AUTH_DB_PATH (e.g. docker-compose pins this to
    # /var/lib/landscape/auth.db on a named volume); tests point it at a
    # tmp_path-backed file for isolation.
    auth_db_path: str = "~/.config/landscape/auth.db"

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    def model_post_init(self, _context: object) -> None:
        if self.llm_profile not in LLM_PROFILES:
            available = ", ".join(sorted(LLM_PROFILES))
            raise ValueError(
                f"Unknown LLM profile {self.llm_profile!r}. "
                f"Known profiles: {available}. Add a new entry to "
                f"LLM_PROFILES in src/landscape/config.py."
            )
        if self.llm_model is None:
            # Resolve the profile's ollama_tag into llm_model so every caller
            # (pipeline.py, extraction/llm.py) keeps reading a single field.
            self.llm_model = LLM_PROFILES[self.llm_profile].ollama_tag

    @property
    def embedding_dims(self) -> int:
        try:
            return EMBEDDING_MODEL_DIMS[self.embedding_model]
        except KeyError as exc:
            known = ", ".join(sorted(EMBEDDING_MODEL_DIMS))
            raise ValueError(
                f"Unknown embedding model {self.embedding_model!r}. "
                f"Add it to EMBEDDING_MODEL_DIMS in src/landscape/config.py. "
                f"Known: {known}."
            ) from exc


settings = Settings()
