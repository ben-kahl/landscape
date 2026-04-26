"""Auth regressions for the FastAPI surface.

Covers the four states the security model needs to distinguish:

* loopback caller, no credentials, bypass enabled -> allowed (agent scope only)
* non-loopback caller, no credentials -> 401
* any caller, valid bearer token for an active client -> allowed
* any caller, malformed/tampered bearer -> 401

The tests are unit-scoped (no Neo4j / Qdrant / Ollama). We monkeypatch the
pipeline / retrieval entry points so the route handlers run without the
backing stack, and we point ``settings.auth_db_path`` at ``tmp_path`` so
the bearer store is isolated per-test.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


@pytest_asyncio.fixture
async def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    monkeypatch.setattr(settings, "allow_unauthenticated_loopback", True)
    await auth_store.ensure_schema()
    yield db_path


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stub pipeline/retrieval so route handlers don't need a real backend."""
    from landscape.pipeline import IngestResult
    from landscape.retrieval.query import RetrievalResult

    async def fake_ingest(
        text, title, source_type="text", session_id=None, turn_id=None, debug=False
    ):
        return IngestResult(
            doc_id="doc-1",
            already_existed=False,
            entities_created=0,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=0,
        )

    async def fake_retrieve(
        query_text,
        hops=2,
        limit=10,
        chunk_limit=3,
        weights=None,
        reinforce=True,
        session_id=None,
        since=None,
        debug=False,
        log_context=None,
    ):
        return RetrievalResult(
            query=query_text,
            results=[],
            touched_entity_ids=[],
            touched_edge_ids=[],
            chunks=[],
        )

    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)
    monkeypatch.setattr("landscape.retrieval.query.retrieve", fake_retrieve)


@pytest_asyncio.fixture
async def loopback_client(auth_db, fake_pipeline):
    from landscape.main import app

    transport = ASGITransport(app=app, client=("127.0.0.1", 5000))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def remote_client(auth_db, fake_pipeline):
    from landscape.main import app

    # Use a public-internet address so the loopback check fails.
    transport = ASGITransport(app=app, client=("203.0.113.5", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


_QUERY_PAYLOAD = {"text": "who leads atlas?"}
_INGEST_PAYLOAD = {"text": "Alice joined Acme.", "title": "note"}


async def test_loopback_bypass_allows_query(loopback_client):
    response = await loopback_client.post("/query", json=_QUERY_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == _QUERY_PAYLOAD["text"]


async def test_loopback_bypass_allows_ingest(loopback_client):
    response = await loopback_client.post("/ingest", json=_INGEST_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == "doc-1"


async def test_remote_query_without_bearer_returns_401(remote_client):
    response = await remote_client.post("/query", json=_QUERY_PAYLOAD)
    assert response.status_code == 401


async def test_remote_ingest_without_bearer_returns_401(remote_client):
    response = await remote_client.post("/ingest", json=_INGEST_PAYLOAD)
    assert response.status_code == 401


async def test_valid_bearer_token_allows_query(remote_client):
    created = await auth_store.create_api_client(name="api-test", scopes=["agent"])
    response = await remote_client.post(
        "/query",
        json=_QUERY_PAYLOAD,
        headers={"Authorization": f"Bearer {created.bearer_token}"},
    )
    assert response.status_code == 200


async def test_tampered_bearer_returns_401(remote_client):
    created = await auth_store.create_api_client(name="tamper-test", scopes=["agent"])
    head, _, tail = created.bearer_token.rpartition("_")
    tampered = f"{head}_{'x' * len(tail)}"
    response = await remote_client.post(
        "/query",
        json=_QUERY_PAYLOAD,
        headers={"Authorization": f"Bearer {tampered}"},
    )
    assert response.status_code == 401


async def test_unknown_bearer_returns_401(remote_client):
    response = await remote_client.post(
        "/query",
        json=_QUERY_PAYLOAD,
        headers={"Authorization": "Bearer lsk_does-not-exist_AAAAAAAAAA"},
    )
    assert response.status_code == 401


async def test_loopback_bypass_disabled_requires_bearer(monkeypatch, auth_db, fake_pipeline):
    monkeypatch.setattr(settings, "allow_unauthenticated_loopback", False)
    from landscape.main import app

    transport = ASGITransport(app=app, client=("127.0.0.1", 5000))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json=_QUERY_PAYLOAD)
    assert response.status_code == 401


async def test_healthz_remains_public(remote_client):
    response = await remote_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_client_lacking_agent_scope_is_forbidden(remote_client):
    created = await auth_store.create_api_client(name="no-scope", scopes=["other"])
    response = await remote_client.post(
        "/query",
        json=_QUERY_PAYLOAD,
        headers={"Authorization": f"Bearer {created.bearer_token}"},
    )
    assert response.status_code == 403
