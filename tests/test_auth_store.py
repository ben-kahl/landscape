"""Tests for the OAuth-era auth store.

Hits a real SQLite file (per-test tmp_path). Marked unit so they run
without docker-compose (no Neo4j/Qdrant needed).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import pytest_asyncio

from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


@pytest_asyncio.fixture
async def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    await auth_store.ensure_schema()
    await auth_store.ensure_schema()  # idempotency check
    yield db_path


# ── Schema ──────────────────────────────────────────────────────────────────

async def test_ensure_schema_creates_all_tables(auth_db: Path):
    import aiosqlite
    async with aiosqlite.connect(auth_db) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in await cursor.fetchall()}
    assert {"api_clients", "authorization_codes", "oauth_tokens"} <= tables


async def test_ensure_schema_upgrades_legacy_api_clients_table(tmp_path: Path, monkeypatch):
    import aiosqlite
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    db_path = tmp_path / "legacy-auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE api_clients (
                client_id    TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                description  TEXT,
                scopes       TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'active',
                created_at   TEXT NOT NULL,
                last_used_at TEXT
            )
            """
        )
        await db.execute(
            """
            INSERT INTO api_clients
                (client_id, name, description, scopes, status, created_at, last_used_at)
            VALUES (?, ?, NULL, ?, 'active', ?, NULL)
            """,
            ("legacy-client", "Legacy Client", '["agent"]', "2026-01-01T00:00:00+00:00"),
        )
        await db.commit()

    await auth_store.ensure_schema()

    client_info = OAuthClientInformationFull(
        client_id="test-client-id",
        client_name="test-client",
        redirect_uris=[AnyUrl("http://localhost:8080/callback")],
        scope="agent",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    await auth_store.store_oauth_client(client_info)
    loaded = await auth_store.get_oauth_client("test-client-id")

    assert loaded is not None
    assert loaded.client_id == "test-client-id"

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("PRAGMA table_info(api_clients)")
        columns = {row[1] for row in await cursor.fetchall()}
    assert {"redirect_uris", "client_metadata"} <= columns


# ── OAuth client store ───────────────────────────────────────────────────────

async def test_store_and_get_oauth_client_round_trips(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    client_info = OAuthClientInformationFull(
        client_id="test-client-id",
        client_name="test-client",
        redirect_uris=[AnyUrl("http://localhost:8080/callback")],
        scope="agent",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    await auth_store.store_oauth_client(client_info)
    loaded = await auth_store.get_oauth_client("test-client-id")
    assert loaded is not None
    assert loaded.client_id == "test-client-id"
    assert loaded.client_name == "test-client"
    assert loaded.scope == "agent"


async def test_get_oauth_client_returns_none_for_unknown(auth_db: Path):
    result = await auth_store.get_oauth_client("does-not-exist")
    assert result is None


async def test_get_oauth_client_returns_none_for_disabled(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    client_info = OAuthClientInformationFull(
        client_id="disabled-client",
        redirect_uris=[AnyUrl("http://localhost/cb")],
        scope="agent",
    )
    await auth_store.store_oauth_client(client_info)
    await auth_store.disable_client("disabled-client")
    result = await auth_store.get_oauth_client("disabled-client")
    assert result is None


async def test_list_api_clients_returns_registered_clients(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", client_name="Client One",
            redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    clients = await auth_store.list_api_clients()
    assert any(c["client_id"] == "c1" for c in clients)
    assert any(c["name"] == "Client One" for c in clients)


# ── Authorization code store ─────────────────────────────────────────────────

async def test_store_and_load_authorization_code(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    expires_at = time.time() + 600
    await auth_store.store_authorization_code(
        code="testcode123",
        client_id="c1",
        redirect_uri="http://localhost/cb",
        redirect_uri_provided_explicitly=True,
        scopes=["agent"],
        code_challenge="abc123challenge",
        expires_at=expires_at,
    )
    row = await auth_store.load_authorization_code_record("testcode123")
    assert row is not None
    assert row["client_id"] == "c1"
    assert row["code_challenge"] == "abc123challenge"
    assert row["used_at"] is None


async def test_load_authorization_code_returns_none_for_unknown(auth_db: Path):
    result = await auth_store.load_authorization_code_record("no-such-code")
    assert result is None


async def test_mark_code_used_makes_it_unretrievable(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    await auth_store.store_authorization_code(
        code="useme", client_id="c1", redirect_uri="http://localhost/cb",
        redirect_uri_provided_explicitly=True, scopes=["agent"],
        code_challenge="ch", expires_at=time.time() + 600,
    )
    await auth_store.mark_code_used("useme")
    result = await auth_store.load_authorization_code_record("useme")
    assert result is None


# ── OAuth token store ────────────────────────────────────────────────────────

async def test_store_and_load_oauth_token_by_access(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", client_name="MyClient",
            redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    await auth_store.store_oauth_token(
        token_id="tid1", client_id="c1", client_name="MyClient",
        access_token="access_abc", refresh_token="refresh_xyz",
        scopes=["agent"], expires_at=None,
    )
    row = await auth_store.load_oauth_token_by_access("access_abc")
    assert row is not None
    assert row["token_id"] == "tid1"
    assert row["client_id"] == "c1"
    assert row["client_name"] == "MyClient"
    assert row["scopes"] == ["agent"]


async def test_load_oauth_token_by_access_returns_none_for_unknown(auth_db: Path):
    result = await auth_store.load_oauth_token_by_access("no-such-token")
    assert result is None


async def test_load_oauth_token_by_refresh_round_trips(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    await auth_store.store_oauth_token(
        token_id="tid2", client_id="c1", client_name="c1",
        access_token="acc2", refresh_token="ref2",
        scopes=["agent"], expires_at=None,
    )
    row = await auth_store.load_oauth_token_by_refresh("ref2")
    assert row is not None
    assert row["token_id"] == "tid2"


async def test_revoke_oauth_token_by_id_stamps_revoked_at(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    await auth_store.store_oauth_token(
        token_id="tid3", client_id="c1", client_name="c1",
        access_token="acc3", refresh_token=None,
        scopes=["agent"], expires_at=None,
    )
    await auth_store.revoke_oauth_token_by_id("tid3")
    row = await auth_store.load_oauth_token_by_access("acc3")
    assert row is None  # revoked tokens are invisible to load


async def test_load_oauth_token_returns_none_for_expired(auth_db: Path):
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id="c1", redirect_uris=[AnyUrl("http://localhost/cb")], scope="agent",
        )
    )
    await auth_store.store_oauth_token(
        token_id="tid4", client_id="c1", client_name="c1",
        access_token="acc4", refresh_token=None,
        scopes=["agent"], expires_at=time.time() - 1,  # already expired
    )
    row = await auth_store.load_oauth_token_by_access("acc4")
    assert row is None
