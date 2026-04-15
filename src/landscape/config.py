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

    `ollama_tag` is the only load-bearing field today; temperature and num_ctx
    are reserved for when extraction/llm.py starts honoring them."""

    ollama_tag: str
    temperature: float = 0.0
    num_ctx: int = 8192
    notes: str = ""


# Seeded with the current default only. Add new profiles here — the user's
# own additions, model bakeoffs, etc. — instead of editing `llm_model` directly.
LLM_PROFILES: dict[str, LLMProfile] = {
    "llama31_8b": LLMProfile(
        ollama_tag="llama3.1:8b",
        notes="Phase 2 baseline. Known-good for the killer-demo corpus.",
    ),
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
    scoring_gamma: float = 0.4  # reinforcement
    scoring_delta: float = 0.3  # edge confidence
    # decay_lambda chosen so exp(-lambda * 7 days) == 0.5 — a true 7-day half-life
    decay_lambda: float = math.log(2) / (7 * 86400)
    reinforcement_cap: float = 2.0

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


settings = Settings()
