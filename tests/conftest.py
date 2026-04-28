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
async def _isolated_test(request):
    """Per-test isolation + singleton reset.

    **Before each test (setup):** wipe Neo4j + drop Qdrant collections so
    the test starts with an empty stack. Resolver non-determinism across
    tests was silently merging entities with the same name (e.g. 'Alice',
    'Project Aurora'), which let one test reinforce an entity that another
    test later observed as pre-warmed. Per-test wipe kills that.

    **After each test (teardown):** close the module-level Neo4j/Qdrant
    client singletons. ASGITransport doesn't trigger FastAPI lifespan
    events, so those singletons would otherwise leak across event loops.

    **Opt-out:** tests marked ``unit`` or ``smoke`` skip stack-backed
    isolation entirely so GitHub-safe tests can run without Neo4j/Qdrant.
    Tests marked @pytest.mark.retrieval skip the pre-test wipe. The
    killer-demo corpus fixture is expensive (~40s of LLM extraction over
    7 docs) and amortizes across the 4 retrieval tests via its own
    module-level `_INGESTED` flag. It wipes once on first use and shares
    state across the marker's tests."""
    if request.node.get_closest_marker("unit") or request.node.get_closest_marker("smoke"):
        yield
        return

    from landscape.storage import neo4j_store, qdrant_store

    if not request.node.get_closest_marker("retrieval"):
        # Fresh ad-hoc clients so the wipe doesn't entangle with the
        # module-level singletons that teardown will close.
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        try:
            async with driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
        finally:
            await driver.close()

        qclient = AsyncQdrantClient(url=QDRANT_URL)
        try:
            existing = await qclient.get_collections()
            names = {c.name for c in existing.collections}
            for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
                if coll in names:
                    await qclient.delete_collection(coll)
        finally:
            await qclient.close()

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
async def http_client(request, monkeypatch):
    from landscape.main import app
    from landscape.auth import AuthContext
    from landscape.security import resolve_request_auth

    # The shared client targets API plumbing, not auth. Override the FastAPI
    # dependency so tests that aren't specifically testing auth don't need to
    # run the full PKCE flow. Tests that exercise auth use their own fixtures
    # in test_api_security.py and test_oauth_flow.py.
    _test_auth = AuthContext(
        client_id="test-client",
        client_name="test",
        token_id="test-token",
        scopes=frozenset({"agent", "graph_query"}),
    )
    app.dependency_overrides[resolve_request_auth] = lambda: _test_auth

    is_ci_safe = request.node.get_closest_marker("unit") or request.node.get_closest_marker("smoke")
    try:
        if is_ci_safe:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client
        else:
            async with app.router.lifespan_context(app):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    yield client
    finally:
        app.dependency_overrides.pop(resolve_request_auth, None)
