"""Auth regressions for the FastAPI surface.

Covers the states the OAuth security model needs to distinguish:
* No bearer → 401
* Valid OAuth-issued bearer → 200
* Revoked/unknown bearer → 401
* Missing required scope → 403
* /healthz always public → 200
* Old lsk_* format → 401
"""
from __future__ import annotations

import secrets
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


@pytest_asyncio.fixture
async def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    monkeypatch.setattr(settings, "mcp_issuer_url", "http://127.0.0.1:8000")
    await auth_store.ensure_schema()
    yield db_path


async def _store_live_token(client_id: str = "test-client", scopes: list[str] | None = None) -> str:
    """Register a client and store a live access token. Returns the token string."""
    scopes = scopes or ["agent"]
    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id=client_id,
            client_name="Test Client",
            redirect_uris=[AnyUrl("http://localhost/cb")],
            scope=" ".join(scopes),
        )
    )
    token = secrets.token_urlsafe(32)
    await auth_store.store_oauth_token(
        token_id=secrets.token_urlsafe(16),
        client_id=client_id,
        client_name="Test Client",
        access_token=token,
        refresh_token=None,
        scopes=scopes,
        expires_at=None,
    )
    return token


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stub pipeline/retrieval so route handlers don't need a real backend."""
    from landscape.pipeline import IngestResult

    async def fake_ingest(text, title, source_type="text", 
                          session_id=None, turn_id=None, debug=False):
        return IngestResult(
            doc_id="doc-1", already_existed=False,
            entities_created=0, entities_reinforced=0,
            relations_created=0, relations_reinforced=0,
            relations_superseded=0, chunks_created=0,
        )

    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)


@pytest_asyncio.fixture
async def api_client(auth_db, fake_pipeline):
    from landscape.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_no_bearer_returns_401(api_client):
    resp = await api_client.post("/ingest", json={"text": "hi", "title": "t"})
    assert resp.status_code == 401


async def test_unknown_bearer_returns_401(api_client):
    resp = await api_client.post(
        "/ingest",
        json={"text": "hi", "title": "t"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


async def test_valid_bearer_allows_ingest(auth_db, api_client):
    token = await _store_live_token()
    resp = await api_client.post(
        "/ingest",
        json={"text": "hello world", "title": "test doc"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200


async def test_revoked_token_returns_401(auth_db, api_client):
    token = await _store_live_token(client_id="revoke-client")
    row = await auth_store.load_oauth_token_by_access(token)
    await auth_store.revoke_oauth_token_by_id(row["token_id"])

    resp = await api_client.post(
        "/ingest",
        json={"text": "hi", "title": "t"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


async def test_old_lsk_bearer_returns_401(api_client):
    resp = await api_client.post(
        "/ingest",
        json={"text": "hi", "title": "t"},
        headers={"Authorization": "Bearer lsk_fakeid_fakematerial"},
    )
    assert resp.status_code == 401


async def test_healthz_remains_public(api_client):
    resp = await api_client.get("/healthz")
    assert resp.status_code == 200


async def test_client_lacking_agent_scope_returns_403(auth_db, api_client):
    token = await _store_live_token(client_id="noscope-client", scopes=["graph_query"])
    resp = await api_client.post(
        "/ingest",
        json={"text": "hi", "title": "t"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
