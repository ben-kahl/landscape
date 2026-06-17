import math
from dataclasses import dataclass

from pydantic_settings import BaseSettings


@dataclass(frozen=True)
class LLMProfile:
    """A named provider/model preset selected via the LLM_PROFILE env var.

    `base_url` + `model` point the OpenAI SDK at either a local llama-server
    (OpenAI-compatible) or a cloud endpoint. `api_key_env` is the NAME of the
    env var holding the key (never the secret itself) so the registry can be
    safely enumerated by a future model-switcher UI. `ctx_size` is the declared
    context budget; for local llama-server it must match the server's
    --ctx-size launch flag (wired via LLAMA_CTX_SIZE in docker-compose).
    """

    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    no_think: bool = False
    ctx_size: int = 32768
    notes: str = ""


LLM_PROFILES: dict[str, LLMProfile] = {
    "local_qwen35": LLMProfile(
        base_url="http://llama-server:8080/v1",
        model="Qwen3.5-9B-Q4_K_M",
        api_key_env=None,
        no_think=True,
        ctx_size=16384,
        notes="Default. Qwen 3.5 9B Q4_K_M via llama-server.",
    ),
    "local_llama31": LLMProfile(
        base_url="http://llama-server:8080/v1",
        model="llama-3.1-8b",
        api_key_env=None,
        ctx_size=16384,
        notes="A/B incumbent (prior benchmark baseline).",
    ),
    "openai_gpt5": LLMProfile(
        base_url="https://api.openai.com/v1",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        ctx_size=16384,
        notes="Cloud preset. Any OpenAI-compatible endpoint works.",
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
    llm_profile: str = "local_qwen35"
    # Overrides the active profile's base_url at connection time. Host-run CLI
    # and scripts set this to http://localhost:8080/v1 since the profile's
    # docker-internal alias (llama-server:8080) is unreachable from the host.
    llm_base_url: str | None = None
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    hf_token: str | None = None

    scoring_alpha: float = 1.0  # vector similarity
    scoring_beta: float = 0.8  # graph proximity
    scoring_gamma: float = 0.2  # reinforcement (multiplicative, was 0.4 additive)
    scoring_delta: float = 0.3  # edge confidence
    scoring_sim_decay: float = 0.6  # per-hop decay applied to inherited vector sim
    # decay_lambda chosen so exp(-lambda * 7 days) == 0.5 — a true 7-day half-life
    decay_lambda: float = math.log(2) / (7 * 86400)
    reinforcement_cap: float = 2.0
    # Minimum cosine similarity for a Qdrant-derived seed to anchor graph
    # expansion. Direct substring/alias seeds (assigned 1.0) are never gated.
    seed_sim_floor: float = 0.3

    # Weave (W&B) tracing for the LLM extraction pipeline. Unset = disabled.
    # Set to e.g. "landscape-extraction" or "<entity>/landscape-extraction".
    weave_project: str | None = None

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
