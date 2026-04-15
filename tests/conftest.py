import os

# The docker stack already holds most of the GPU (Ollama + app-gpu's encoder);
# force the host test process onto CPU torch to avoid CUDA OOM.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from neo4j import AsyncGraphDatabase  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

# Fixtures assume docker-compose stack is running locally on default ports.
# Env vars must be set before landscape.config is imported anywhere.

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "landscape-dev")
QDRANT_URL = "http://localhost:6333"

os.environ.setdefault("NEO4J_URI", NEO4J_URI)
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", QDRANT_URL)
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")


@pytest_asyncio.fixture(autouse=True)
async def _reset_app_singletons():
    # ASGITransport does not trigger FastAPI lifespan events, so module-level
    # Neo4j/Qdrant clients would leak across event loops between tests.
    from landscape.storage import neo4j_store, qdrant_store

    yield
    await neo4j_store.close_driver()
    await qdrant_store.close_client()


@pytest_asyncio.fixture
async def neo4j_driver():
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    yield driver
    await driver.close()


@pytest_asyncio.fixture
async def qdrant_client():
    client = AsyncQdrantClient(url=QDRANT_URL)
    yield client
    await client.close()


@pytest_asyncio.fixture
async def http_client():
    from landscape.embeddings import encoder
    from landscape.main import app
    from landscape.storage import qdrant_store

    encoder.load_model()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
