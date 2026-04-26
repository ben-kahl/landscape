"""Persistence layer for API clients and their hashed bearer secrets.

Storage shape (SQLite, local file at ``settings.auth_db_path``)::

    api_clients(client_id PK, name, description, scopes JSON,
                status, created_at, last_used_at)
    client_secrets(secret_id PK, client_id FK -> api_clients.client_id,
                   secret_hash, secret_prefix,
                   created_at, expires_at, revoked_at)

Why SQLite (not Neo4j) for auth state? Auth records are small, relational, and
need strict uniqueness/integrity. Neo4j is the right tool for traversing
multi-hop memory; it is the wrong tool for "is this bearer token valid?". A
local SQLite file keeps auth orthogonal to the graph stack: tests don't need
Neo4j running, and the auth file is trivially backed up / wiped independently
of the memory graph.

* ``client_id`` and ``secret_id`` are UUIDs minted at creation time. The
  client-facing bearer token is ``lsk_<secret_id>_<material>`` -- the
  secret_id is intentionally derivable from the token so we can do an
  indexed lookup without scanning every hash.
* ``secret_hash`` is the Argon2id hash of the full bearer token; the
  plaintext is never persisted. Verification goes through
  ``password_hash.verify`` -- never a string compare.
* A client may rotate through multiple ClientSecret rows. The currently
  live secret is whichever row has ``revoked_at IS NULL`` and (if set) an
  ``expires_at`` in the future.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from landscape.auth import (
    AuthContext,
    mint_client_secret,
    parse_bearer_token,
    password_hash,
)
from landscape.config import settings


@dataclass(frozen=True)
class CreatedClientSecret:
    """Returned by ``create_api_client`` (Task 5 will reuse the shape for ``rotate_client_secret``).

    The plaintext ``bearer_token`` is the only chance the caller has to
    capture the credential; the store keeps just the Argon2 hash.
    """

    client_id: str
    client_name: str
    secret_id: str
    bearer_token: str
    secret_prefix: str
    scopes: tuple[str, ...]
    created_at: str


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS api_clients (
        client_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        scopes TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        last_used_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS client_secrets (
        secret_id TEXT PRIMARY KEY,
        client_id TEXT NOT NULL REFERENCES api_clients(client_id),
        secret_hash TEXT NOT NULL,
        secret_prefix TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT,
        revoked_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_api_clients_status ON api_clients(status)",
    "CREATE INDEX IF NOT EXISTS idx_client_secrets_client_id ON client_secrets(client_id)",
    "CREATE INDEX IF NOT EXISTS idx_client_secrets_revoked_at ON client_secrets(revoked_at)",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _db_path() -> Path:
    """Resolve the configured auth DB path, expanding ``~`` etc.

    Centralized so tests can override ``settings.auth_db_path`` and have
    every call pick up the new value without restarting the process.
    """
    return Path(settings.auth_db_path).expanduser()


async def _connect() -> aiosqlite.Connection:
    """Open a fresh aiosqlite connection with the PRAGMAs we want.

    WAL gives us cheap concurrent readers (relevant once Task 4 wires
    bearer auth into every request). ``foreign_keys=ON`` enforces the
    ``client_secrets.client_id`` FK -- SQLite ignores it by default.
    ``busy_timeout=5000`` lets writers wait up to 5s for the WAL writer
    rather than failing immediately under contention.
    """
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=5000")
    return db


async def ensure_schema() -> None:
    """Apply every DDL statement against the auth DB.

    Idempotent: every CREATE is ``IF NOT EXISTS``-guarded. Called once at
    FastAPI startup (see ``landscape.main._startup_storage``) and from
    test fixtures so each test sees a fully-initialized DB.
    """
    db = await _connect()
    try:
        for statement in SCHEMA_STATEMENTS:
            await db.execute(statement)
        await db.commit()
    finally:
        await db.close()


async def create_api_client(
    *,
    name: str,
    scopes: list[str],
    description: str | None = None,
) -> CreatedClientSecret:
    """Insert a new api_clients row + a single live client_secrets row.

    UUID4 collisions are vanishingly rare; the PRIMARY KEY constraints
    surface them as ``IntegrityError`` rather than silently overwriting.
    """
    client_id = str(uuid4())
    secret_id = str(uuid4())
    bearer_token, secret_prefix, secret_hash = mint_client_secret(secret_id)
    now = _now_iso()
    scopes_list = list(scopes)
    db = await _connect()
    try:
        await db.execute(
            """
            INSERT INTO api_clients
                (client_id, name, description, scopes, status, created_at, last_used_at)
            VALUES (?, ?, ?, ?, 'active', ?, NULL)
            """,
            (client_id, name, description, json.dumps(scopes_list), now),
        )
        await db.execute(
            """
            INSERT INTO client_secrets
                (secret_id, client_id, secret_hash, secret_prefix,
                 created_at, expires_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, NULL, NULL)
            """,
            (secret_id, client_id, secret_hash, secret_prefix, now),
        )
        await db.commit()
    finally:
        await db.close()
    return CreatedClientSecret(
        client_id=client_id,
        client_name=name,
        secret_id=secret_id,
        bearer_token=bearer_token,
        secret_prefix=secret_prefix,
        scopes=tuple(scopes_list),
        created_at=now,
    )


async def get_client_secret_by_id(secret_id: str) -> dict[str, Any] | None:
    """Look up a stored client_secrets row joined to its api_clients parent.

    Returns ``None`` when the secret_id is unknown. The returned dict
    intentionally exposes ``secret_hash`` so callers can run
    ``password_hash.verify`` themselves (handy in tests). For the
    request hot path, prefer ``authenticate_bearer_token``.
    """
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT
                c.client_id        AS client_id,
                c.name             AS client_name,
                c.scopes           AS scopes,
                c.status           AS status,
                c.last_used_at     AS last_used_at,
                s.secret_id        AS secret_id,
                s.secret_hash      AS secret_hash,
                s.secret_prefix    AS secret_prefix,
                s.created_at       AS created_at,
                s.expires_at       AS expires_at,
                s.revoked_at       AS revoked_at
            FROM client_secrets s
            JOIN api_clients c ON c.client_id = s.client_id
            WHERE s.secret_id = ?
            """,
            (secret_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        columns = [
            "client_id",
            "client_name",
            "scopes",
            "status",
            "last_used_at",
            "secret_id",
            "secret_hash",
            "secret_prefix",
            "created_at",
            "expires_at",
            "revoked_at",
        ]
        record = dict(zip(columns, row, strict=True))
    finally:
        await db.close()
    record["scopes"] = json.loads(record["scopes"]) if record["scopes"] else []
    return record


async def revoke_client_secret(secret_id: str) -> None:
    """Stamp ``revoked_at`` on a client_secrets row. Idempotent.

    A revoked secret can never be reactivated -- rotation means inserting
    a new client_secrets row. ``authenticate_bearer_token`` rejects any
    secret with ``revoked_at`` set. ``COALESCE`` preserves the original
    revoke timestamp on a re-revoke.
    """
    db = await _connect()
    try:
        await db.execute(
            """
            UPDATE client_secrets
            SET revoked_at = COALESCE(revoked_at, ?)
            WHERE secret_id = ?
            """,
            (_now_iso(), secret_id),
        )
        await db.commit()
    finally:
        await db.close()


async def disable_client(client_id: str) -> None:
    """Flip an api_clients row to ``status='disabled'``. Idempotent.

    Disabled clients fail authentication even if their secret is still
    technically valid. Task 5 will expose this via the CLI.
    """
    db = await _connect()
    try:
        await db.execute(
            """
            UPDATE api_clients
            SET status = 'disabled'
            WHERE client_id = ?
            """,
            (client_id,),
        )
        await db.commit()
    finally:
        await db.close()


async def _touch_last_used(client_id: str, now_iso: str) -> None:
    """Update ``api_clients.last_used_at`` only when the configured interval has lapsed.

    Writing on every authenticated request would generate pointless write
    traffic. The ``last_used_at`` field is for debugging / rotation
    hygiene, not for billing -- "fresh enough" is fine. ISO-8601 strings
    are lexicographically sortable, so we can compare them with string
    literal ``<`` and avoid datetime parsing in SQL.
    """
    interval_seconds = settings.auth_update_last_used_interval_seconds
    db = await _connect()
    try:
        cursor = await db.execute(
            "SELECT last_used_at FROM api_clients WHERE client_id = ?",
            (client_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return
        last_used_at = row[0]
        if last_used_at is not None:
            try:
                last_dt = datetime.fromisoformat(last_used_at)
                now_dt = datetime.fromisoformat(now_iso)
                if (now_dt - last_dt).total_seconds() < interval_seconds:
                    return
            except ValueError:
                # Corrupt timestamp -- overwrite it rather than silently skip.
                pass
        await db.execute(
            "UPDATE api_clients SET last_used_at = ? WHERE client_id = ?",
            (now_iso, client_id),
        )
        await db.commit()
    finally:
        await db.close()


async def authenticate_bearer_token(bearer_token: str) -> AuthContext | None:
    """Verify a bearer token and return the authenticated context.

    Returns ``None`` for any failure mode -- bad shape, unknown secret_id,
    Argon2 mismatch, revoked secret, expired secret, or disabled parent
    client. Callers should not branch on the failure reason; that
    information is reserved for audit logs added in Task 4.
    """
    parsed = parse_bearer_token(bearer_token)
    if parsed is None:
        return None
    secret_id, _material = parsed

    record = await get_client_secret_by_id(secret_id)
    if record is None:
        return None
    if record.get("revoked_at") is not None:
        return None
    if record.get("status") != "active":
        return None
    expires_at = record.get("expires_at")
    if expires_at is not None and expires_at <= _now_iso():
        return None
    if not password_hash.verify(bearer_token, record["secret_hash"]):
        return None

    await _touch_last_used(record["client_id"], _now_iso())

    scopes = record.get("scopes") or []
    return AuthContext(
        client_id=record["client_id"],
        client_name=record["client_name"],
        secret_id=record["secret_id"],
        scopes=frozenset(scopes),
        is_loopback_bypass=False,
    )
