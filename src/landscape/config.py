import math

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "landscape-dev"
    qdrant_url: str = "http://qdrant:6333"
    ollama_url: str = "http://ollama:11434"
    llm_model: str = "llama3.1:8b"
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


settings = Settings()
