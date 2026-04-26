"""Tests for ``landscape.storage.auth_store``.

These hit a real SQLite file (per-test ``tmp_path`` fixture). The goal is
to prove that records land in the expected shape, that secrets are
stored as Argon2 hashes only, and that the verification path actually
rejects revoked / disabled / expired / malformed credentials.

The auth store is independent of the Neo4j/Qdrant stack -- these tests
are marked ``unit`` so they skip the stack-wipe in ``conftest`` and run
without requiring docker-compose to be up.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from landscape.config import settings
from landscape.storage import auth_store
from landscape.storage.auth_store import (
    authenticate_bearer_token,
    create_api_client,
    disable_client,
    ensure_schema,
    get_client_secret_by_id,
    revoke_client_secret,
)

pytestmark = pytest.mark.unit


@pytest_asyncio.fixture
async def auth_db(tmp_path, monkeypatch):
    """Point the auth store at a tmp-isolated SQLite file and bootstrap it.

    Each test gets its own DB file, so there's no shared state between
    tests. We monkeypatch ``settings.auth_db_path`` directly because
    ``_db_path()`` reads from ``settings`` on every call (no caching), so
    the monkeypatch propagates into every store function.
    """
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    await ensure_schema()
    # Idempotency check: running ensure_schema twice should be a no-op.
    await ensure_schema()
    yield db_path


async def test_create_api_client_stores_argon2_hash_only(auth_db: Path):
    created = await create_api_client(name="claude-desktop", scopes=["agent"])

    assert created.bearer_token.startswith("lsk_")
    assert created.secret_id in created.bearer_token
    assert created.secret_prefix == created.bearer_token[:18]
    assert created.scopes == ("agent",)

    stored = await get_client_secret_by_id(created.secret_id)
    assert stored is not None
    assert stored["secret_hash"] != created.bearer_token
    assert stored["secret_hash"].startswith("$argon2")
    assert stored["status"] == "active"
    assert stored["revoked_at"] is None

    # Belt-and-braces: the plaintext token must not be stored anywhere
    # in the SQLite file. Read every row of both tables and scan strings.
    async with aiosqlite.connect(auth_db) as db:
        for table in ("api_clients", "client_secrets"):
            cursor = await db.execute(f"SELECT * FROM {table}")
            rows = await cursor.fetchall()
            await cursor.close()
            for row in rows:
                for value in row:
                    if isinstance(value, str):
                        assert created.bearer_token not in value


async def test_authenticate_bearer_token_accepts_valid_token(auth_db: Path):
    created = await create_api_client(name="bench-runner", scopes=["agent", "ingest"])
    ctx = await authenticate_bearer_token(created.bearer_token)
    assert ctx is not None
    assert ctx.client_id == created.client_id
    assert ctx.client_name == "bench-runner"
    assert ctx.secret_id == created.secret_id
    assert ctx.scopes == frozenset({"agent", "ingest"})
    assert ctx.is_loopback_bypass is False


async def test_authenticate_bearer_token_rejects_tampered_material(auth_db: Path):
    created = await create_api_client(name="tamper-target", scopes=["agent"])
    # Flip the trailing material; the secret_id still resolves but Argon2
    # verification must fail.
    head, _, tail = created.bearer_token.rpartition("_")
    tampered = f"{head}_{'x' * len(tail)}"
    assert tampered != created.bearer_token
    assert await authenticate_bearer_token(tampered) is None


async def test_authenticate_bearer_token_rejects_unknown_secret_id(auth_db: Path):
    assert await authenticate_bearer_token("lsk_does-not-exist_AAAAAAAAAAAA") is None
    assert await authenticate_bearer_token("not-a-token") is None
    assert await authenticate_bearer_token("lsk__missingid") is None


async def test_authenticate_bearer_token_rejects_revoked_secret(auth_db: Path):
    created = await create_api_client(name="revoked-runner", scopes=["agent"])
    await revoke_client_secret(created.secret_id)

    assert await authenticate_bearer_token(created.bearer_token) is None

    # Idempotent: a second revoke should not flip the timestamp.
    stored = await get_client_secret_by_id(created.secret_id)
    assert stored is not None
    first_revoked_at = stored["revoked_at"]
    assert first_revoked_at is not None

    await revoke_client_secret(created.secret_id)
    stored_again = await get_client_secret_by_id(created.secret_id)
    assert stored_again is not None
    assert stored_again["revoked_at"] == first_revoked_at


async def test_authenticate_bearer_token_rejects_disabled_client(auth_db: Path):
    created = await create_api_client(name="disabled-runner", scopes=["agent"])
    await disable_client(created.client_id)

    assert await authenticate_bearer_token(created.bearer_token) is None


async def test_authenticate_bearer_token_rejects_expired_secret(auth_db: Path):
    created = await create_api_client(name="expired-runner", scopes=["agent"])

    # Force-expire the secret directly so we don't have to wait for clock drift.
    async with aiosqlite.connect(auth_db) as db:
        await db.execute(
            "UPDATE client_secrets SET expires_at = '2000-01-01T00:00:00+00:00' "
            "WHERE secret_id = ?",
            (created.secret_id,),
        )
        await db.commit()

    assert await authenticate_bearer_token(created.bearer_token) is None


async def test_get_client_secret_by_id_returns_none_for_unknown(auth_db: Path):
    assert await get_client_secret_by_id("nope") is None


async def test_authenticate_updates_last_used(auth_db: Path, monkeypatch):
    # Make every successful auth touch last_used_at so the test isn't racy
    # against the default 300s throttle window.
    monkeypatch.setattr(settings, "auth_update_last_used_interval_seconds", 0)

    created = await create_api_client(name="touch-runner", scopes=["agent"])
    stored_before = await get_client_secret_by_id(created.secret_id)
    assert stored_before is not None
    assert stored_before["last_used_at"] is None

    ctx = await authenticate_bearer_token(created.bearer_token)
    assert ctx is not None

    stored_after = await get_client_secret_by_id(created.secret_id)
    assert stored_after is not None
    assert stored_after["last_used_at"] is not None


async def test_ensure_schema_enforces_unique_client_id(auth_db: Path):
    """The api_clients PRIMARY KEY must reject duplicate client_ids."""
    async with aiosqlite.connect(auth_db) as db:
        await db.execute(
            "INSERT INTO api_clients (client_id, name, scopes, status, created_at) "
            "VALUES ('dup-test', 'a', '[]', 'active', '2026-01-01T00:00:00+00:00')"
        )
        await db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(
                "INSERT INTO api_clients (client_id, name, scopes, status, created_at) "
                "VALUES ('dup-test', 'b', '[]', 'active', '2026-01-01T00:00:00+00:00')"
            )
            await db.commit()


# Sanity check that the module exposes the expected dataclass shape so
# Task 4 / Task 5 can rely on it.
def test_created_client_secret_is_frozen():
    record = auth_store.CreatedClientSecret(
        client_id="c",
        client_name="n",
        secret_id="s",
        bearer_token="lsk_s_x",
        secret_prefix="lsk_s_x",
        scopes=("agent",),
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(Exception):
        record.client_id = "other"  # type: ignore[misc]
