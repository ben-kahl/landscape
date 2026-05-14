from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

LIVE_NEO4J_URI = "bolt://localhost:7687"
LIVE_QDRANT_URL = "http://localhost:6333"

DEFAULT_TEST_NEO4J_URI = "bolt://localhost:17687"
DEFAULT_TEST_QDRANT_URL = "http://localhost:16333"
DEFAULT_TEST_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class StackConfig:
    neo4j_uri: str
    neo4j_auth: tuple[str, str]
    qdrant_url: str
    ollama_url: str


def resolve_test_stack_config(env: Mapping[str, str]) -> StackConfig:
    return StackConfig(
        neo4j_uri=env.get("LANDSCAPE_TEST_NEO4J_URI", DEFAULT_TEST_NEO4J_URI),
        neo4j_auth=(
            env.get("LANDSCAPE_TEST_NEO4J_USER", "neo4j"),
            env.get("LANDSCAPE_TEST_NEO4J_PASSWORD", "landscape-test"),
        ),
        qdrant_url=env.get("LANDSCAPE_TEST_QDRANT_URL", DEFAULT_TEST_QDRANT_URL),
        ollama_url=env.get("LANDSCAPE_TEST_OLLAMA_URL", DEFAULT_TEST_OLLAMA_URL),
    )


def assert_safe_to_wipe(config: StackConfig, env: Mapping[str, str]) -> None:
    if env.get("LANDSCAPE_ALLOW_LIVE_TEST_WIPE") == "1":
        return

    if config.neo4j_uri == LIVE_NEO4J_URI or config.qdrant_url == LIVE_QDRANT_URL:
        raise RuntimeError(
            "Refusing to wipe the default live Landscape stack during tests. "
            "Start the isolated test stack with `docker compose -f docker-compose.test.yml up -d`, "
            "or set LANDSCAPE_TEST_NEO4J_URI / LANDSCAPE_TEST_QDRANT_URL to "
            "dedicated test services. "
            "If you really intend to wipe the live default ports, set "
            "LANDSCAPE_ALLOW_LIVE_TEST_WIPE=1."
        )
