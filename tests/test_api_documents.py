"""GET /documents endpoint tests (unit; storage monkeypatched).

Auth fixture pattern duplicated from test_api_security.py — test modules
can't import from each other in this repo."""

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


async def _store_live_token() -> str:
    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="test-client",
            client_name="Test Client",
            redirect_uris=[AnyUrl("http://localhost/cb")],
            scope="agent",
        )
    )
    token = secrets.token_urlsafe(32)
    await auth_store.store_oauth_token(
        token_id=secrets.token_urlsafe(16),
        client_id="test-client",
        client_name="Test Client",
        access_token=token,
        refresh_token=None,
        scopes=["agent"],
        expires_at=None,
    )
    return token


@pytest.fixture
def fake_document_store(monkeypatch):
    from landscape.storage import neo4j_store

    async def fake_get(doc_id: str):
        if doc_id != "4:abc:12":
            return None
        return {
            "doc_id": "4:abc:12",
            "title": "Ticket 119987",
            "source_type": "text",
            "ingested_at": "2026-07-12T00:00:00+00:00",
            "chunks": [
                {"chunk_id": "d:0:h1", "position": 0, "text": "first chunk"},
                {"chunk_id": "d:1:h1", "position": 1, "text": "second chunk"},
            ],
            "sessions": ["sess-1"],
        }

    monkeypatch.setattr(neo4j_store, "get_document_with_chunks", fake_get)


@pytest.mark.asyncio
async def test_get_document_endpoint_returns_full_text(auth_db, fake_document_store):
    from landscape.main import app

    token = await _store_live_token()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/documents",
            params={"doc_id": "4:abc:12"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Ticket 119987"
    assert body["full_text"] == "first chunk\nsecond chunk"


@pytest.mark.asyncio
async def test_get_document_endpoint_404_when_missing(auth_db, fake_document_store):
    from landscape.main import app

    token = await _store_live_token()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/documents",
            params={"doc_id": "4:nope:0"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 404
