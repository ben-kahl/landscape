"""Unit tests for LandscapeOAuthProvider.

Uses a real SQLite file (tmp_path). No FastMCP, no network.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


def _make_client(client_id: str = "test-client") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        client_name=f"Test {client_id}",
        redirect_uris=[AnyUrl("http://localhost:8080/callback")],
        scope="agent graph_query",
    )


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _make_params(challenge: str, state: str | None = None):
    """Build AuthorizationParams with the fields the installed SDK version supports."""
    from mcp.server.auth.provider import AuthorizationParams
    fields = AuthorizationParams.model_fields.keys()
    kwargs = dict(
        scopes=["agent"],
        code_challenge=challenge,
        redirect_uri=AnyUrl("http://localhost:8080/callback"),
    )
    if "state" in fields:
        kwargs["state"] = state
    if "redirect_uri_provided_explicitly" in fields:
        kwargs["redirect_uri_provided_explicitly"] = True
    return AuthorizationParams(**kwargs)


@pytest_asyncio.fixture
async def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "oauth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    await auth_store.ensure_schema()
    yield db_path


@pytest_asyncio.fixture
async def provider(auth_db):
    from landscape.storage.oauth_provider import LandscapeOAuthProvider
    return LandscapeOAuthProvider()


@pytest_asyncio.fixture
async def registered_client(provider, auth_db):
    client = _make_client()
    await provider.register_client(client)
    return client


async def test_get_client_returns_none_for_unknown(provider):
    result = await provider.get_client("no-such-id")
    assert result is None


async def test_register_and_get_client_round_trips(provider):
    client = _make_client("round-trip-client")
    await provider.register_client(client)
    loaded = await provider.get_client("round-trip-client")
    assert loaded is not None
    assert loaded.client_id == "round-trip-client"
    assert loaded.client_name == "Test round-trip-client"


async def test_authorize_returns_redirect_url_with_code(provider, registered_client):
    _, challenge = _pkce_pair()
    params = _make_params(challenge, state="my-state")
    redirect_url = await provider.authorize(registered_client, params)
    assert "http://localhost:8080/callback" in redirect_url
    assert "code=" in redirect_url


async def test_load_authorization_code_returns_code_object(provider, registered_client):
    from urllib.parse import parse_qs, urlparse
    _, challenge = _pkce_pair()
    params = _make_params(challenge)
    redirect_url = await provider.authorize(registered_client, params)
    code = parse_qs(urlparse(redirect_url).query)["code"][0]

    auth_code = await provider.load_authorization_code(registered_client, code)
    assert auth_code is not None
    assert auth_code.code_challenge == challenge
    assert auth_code.client_id == registered_client.client_id


async def test_load_authorization_code_returns_none_for_unknown(provider, registered_client):
    result = await provider.load_authorization_code(registered_client, "no-such-code")
    assert result is None


async def test_exchange_authorization_code_returns_oauth_token(provider, registered_client):
    from urllib.parse import parse_qs, urlparse
    _, challenge = _pkce_pair()
    params = _make_params(challenge)
    redirect_url = await provider.authorize(registered_client, params)
    code_str = parse_qs(urlparse(redirect_url).query)["code"][0]
    auth_code = await provider.load_authorization_code(registered_client, code_str)

    token = await provider.exchange_authorization_code(registered_client, auth_code)
    assert token.access_token
    assert token.token_type == "Bearer"
    assert token.refresh_token is not None


async def test_exchange_code_marks_it_used(provider, registered_client):
    from urllib.parse import parse_qs, urlparse
    _, challenge = _pkce_pair()
    params = _make_params(challenge)
    redirect_url = await provider.authorize(registered_client, params)
    code_str = parse_qs(urlparse(redirect_url).query)["code"][0]
    auth_code = await provider.load_authorization_code(registered_client, code_str)
    await provider.exchange_authorization_code(registered_client, auth_code)

    second_load = await provider.load_authorization_code(registered_client, code_str)
    assert second_load is None


async def test_load_access_token_returns_token_for_valid(provider, registered_client):
    from urllib.parse import parse_qs, urlparse
    _, challenge = _pkce_pair()
    params = _make_params(challenge)
    redirect_url = await provider.authorize(registered_client, params)
    code_str = parse_qs(urlparse(redirect_url).query)["code"][0]
    auth_code = await provider.load_authorization_code(registered_client, code_str)
    issued = await provider.exchange_authorization_code(registered_client, auth_code)

    loaded = await provider.load_access_token(issued.access_token)
    assert loaded is not None
    assert loaded.client_id == registered_client.client_id
    assert "agent" in loaded.scopes


async def test_load_access_token_returns_none_for_unknown(provider):
    result = await provider.load_access_token("not-a-real-token")
    assert result is None


async def test_revoke_token_makes_access_token_invalid(provider, registered_client):
    from urllib.parse import parse_qs, urlparse
    _, challenge = _pkce_pair()
    params = _make_params(challenge)
    redirect_url = await provider.authorize(registered_client, params)
    code_str = parse_qs(urlparse(redirect_url).query)["code"][0]
    auth_code = await provider.load_authorization_code(registered_client, code_str)
    issued = await provider.exchange_authorization_code(registered_client, auth_code)
    access_token_obj = await provider.load_access_token(issued.access_token)

    await provider.revoke_token(access_token_obj)
    result = await provider.load_access_token(issued.access_token)
    assert result is None


async def test_exchange_refresh_token_issues_new_access_token(provider, registered_client):
    from urllib.parse import parse_qs, urlparse
    _, challenge = _pkce_pair()
    params = _make_params(challenge)
    redirect_url = await provider.authorize(registered_client, params)
    code_str = parse_qs(urlparse(redirect_url).query)["code"][0]
    auth_code = await provider.load_authorization_code(registered_client, code_str)
    first_token = await provider.exchange_authorization_code(registered_client, auth_code)
    refresh_token_obj = await provider.load_refresh_token(registered_client, first_token.refresh_token)

    new_token = await provider.exchange_refresh_token(registered_client, refresh_token_obj, ["agent"])
    assert new_token.access_token != first_token.access_token
    old = await provider.load_access_token(first_token.access_token)
    assert old is None
