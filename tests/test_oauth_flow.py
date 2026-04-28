"""Integration test: full OAuth 2.1 + PKCE round-trip.

Exercises the complete flow that Claude Code / Codex will execute
against the live server. Uses httpx.AsyncClient with ASGITransport
so no real TCP port is needed.

Marked unit (not integration) because these tests only need the auth DB;
no Neo4j/Qdrant/Ollama required.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


def _pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge_S256)."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


@pytest_asyncio.fixture
async def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    monkeypatch.setattr(settings, "mcp_issuer_url", "http://127.0.0.1:8000")
    await auth_store.ensure_schema()
    yield db_path


@pytest_asyncio.fixture
async def client(auth_db):
    from landscape.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
        follow_redirects=False,
    ) as c:
        yield c


async def _full_pkce_flow(client: AsyncClient) -> dict:
    """Execute register → authorize → token exchange. Returns token data + metadata."""
    verifier, challenge = _pkce()
    redirect_uri = "http://localhost:8080/callback"

    # 1. Register client (token_endpoint_auth_method="none" → public PKCE client,
    #    no client_secret required on token exchange)
    reg_resp = await client.post(
        "/register",
        json={
            "client_name": "test-integration-client",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": "agent",
            "token_endpoint_auth_method": "none",
        },
    )
    assert reg_resp.status_code in (200, 201), f"register failed: {reg_resp.text}"
    client_id = reg_resp.json()["client_id"]

    # 2. Authorization (auto-approved, returns redirect with code)
    state = secrets.token_urlsafe(8)
    auth_resp = await client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": "agent",
        },
    )
    assert auth_resp.status_code == 302, f"authorize failed: {auth_resp.text}"
    location = auth_resp.headers["location"]
    qs = parse_qs(urlparse(location).query)
    code = qs["code"][0]
    assert qs["state"][0] == state

    # 3. Token exchange
    token_resp = await client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )
    assert token_resp.status_code == 200, f"token exchange failed: {token_resp.text}"
    token_data = token_resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"].lower() == "bearer"
    assert "refresh_token" in token_data
    return token_data | {"client_id": client_id, "verifier": verifier}


async def test_discovery_returns_rfc8414_metadata(client):
    resp = await client.get("/.well-known/oauth-authorization-server")
    assert resp.status_code == 200
    data = resp.json()
    assert "authorization_endpoint" in data
    assert "token_endpoint" in data
    assert "registration_endpoint" in data


async def test_full_pkce_flow_issues_access_token(client, auth_db):
    token_data = await _full_pkce_flow(client)
    assert token_data["access_token"]
    # Verify the token is in the DB
    row = await auth_store.load_oauth_token_by_access(token_data["access_token"])
    assert row is not None
    assert "agent" in row["scopes"]


async def test_auth_code_replay_is_rejected(client, auth_db):
    """The same auth code cannot be exchanged twice."""
    verifier, challenge = _pkce()
    redirect_uri = "http://localhost:8080/callback"

    reg_resp = await client.post(
        "/register",
        json={
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": "agent",
            "token_endpoint_auth_method": "none",
        },
    )
    client_id = reg_resp.json()["client_id"]

    auth_resp = await client.get(
        "/authorize",
        params={
            "response_type": "code", "client_id": client_id,
            "redirect_uri": redirect_uri, "code_challenge": challenge,
            "code_challenge_method": "S256", "scope": "agent",
        },
    )
    code = parse_qs(urlparse(auth_resp.headers["location"]).query)["code"][0]

    # First exchange — succeeds
    r1 = await client.post("/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": redirect_uri, "client_id": client_id,
        "code_verifier": verifier,
    })
    assert r1.status_code == 200

    # Second exchange — must fail
    r2 = await client.post("/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": redirect_uri, "client_id": client_id,
        "code_verifier": verifier,
    })
    assert r2.status_code == 400


async def test_wrong_code_verifier_fails(client, auth_db):
    verifier, challenge = _pkce()
    wrong_verifier = secrets.token_urlsafe(32)
    redirect_uri = "http://localhost:8080/callback"

    reg_resp = await client.post(
        "/register",
        json={
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": "agent",
            "token_endpoint_auth_method": "none",
        },
    )
    client_id = reg_resp.json()["client_id"]

    auth_resp = await client.get(
        "/authorize",
        params={
            "response_type": "code", "client_id": client_id,
            "redirect_uri": redirect_uri, "code_challenge": challenge,
            "code_challenge_method": "S256", "scope": "agent",
        },
    )
    code = parse_qs(urlparse(auth_resp.headers["location"]).query)["code"][0]

    resp = await client.post("/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": redirect_uri, "client_id": client_id,
        "code_verifier": wrong_verifier,
    })
    assert resp.status_code == 400


async def test_revoke_invalidates_token(client, auth_db):
    token_data = await _full_pkce_flow(client)
    access_token = token_data["access_token"]
    client_id = token_data["client_id"]

    # The SDK's RevocationRequest model declares client_secret as required
    # (str | None with no default). Public PKCE clients have no secret, so
    # pass an empty string so Pydantic validation passes; the SDK's
    # ClientAuthenticator skips the secret check for token_endpoint_auth_method="none".
    revoke_resp = await client.post(
        "/revoke",
        data={"token": access_token, "client_id": client_id, "client_secret": ""},
    )
    assert revoke_resp.status_code == 200

    row = await auth_store.load_oauth_token_by_access(access_token)
    assert row is None


async def test_refresh_token_issues_new_access_token(client, auth_db):
    token_data = await _full_pkce_flow(client)
    old_access = token_data["access_token"]
    refresh_token = token_data["refresh_token"]
    client_id = token_data["client_id"]

    refresh_resp = await client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
    )
    assert refresh_resp.status_code == 200
    new_data = refresh_resp.json()
    assert new_data["access_token"] != old_access
    assert await auth_store.load_oauth_token_by_access(old_access) is None
