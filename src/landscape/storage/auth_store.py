"""Persistence layer for OAuth clients, authorization codes, and tokens.

Storage shape (SQLite, local file at ``settings.auth_db_path``)::

    api_clients(client_id PK, name, description, scopes JSON,
                status, created_at, last_used_at,
                redirect_uris JSON, client_metadata JSON)
    authorization_codes(code PK, client_id FK, redirect_uri,
                        redirect_uri_provided_explicitly INTEGER,
                        scopes JSON, code_challenge,
                        expires_at REAL, used_at REAL)
    oauth_tokens(token_id PK, client_id FK, client_name,
                 access_token UNIQUE, refresh_token UNIQUE,
                 scopes JSON, expires_at REAL, revoked_at REAL)

Why SQLite (not Neo4j) for auth state? Auth records are small, relational,
and need strict uniqueness/integrity. Neo4j is the right tool for traversing
multi-hop memory; it is the wrong tool for "is this bearer token valid?".
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from mcp.shared.auth import OAuthClientInformationFull

from landscape.config import settings

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS api_clients (
        client_id       TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        description     TEXT,
        scopes          TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'active',
        created_at      TEXT NOT NULL,
        last_used_at    TEXT,
        redirect_uris   TEXT,
        client_metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS authorization_codes (
        code                          TEXT PRIMARY KEY,
        client_id                     TEXT NOT NULL REFERENCES api_clients(client_id),
        redirect_uri                  TEXT NOT NULL,
        redirect_uri_provided_explicitly INTEGER NOT NULL DEFAULT 1,
        scopes                        TEXT NOT NULL,
        code_challenge                TEXT NOT NULL,
        expires_at                    REAL NOT NULL,
        used_at                       REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS oauth_tokens (
        token_id       TEXT PRIMARY KEY,
        client_id      TEXT NOT NULL REFERENCES api_clients(client_id),
        client_name    TEXT NOT NULL,
        access_token   TEXT UNIQUE NOT NULL,
        refresh_token  TEXT UNIQUE,
        scopes         TEXT NOT NULL,
        expires_at     REAL,
        revoked_at     REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_api_clients_status ON api_clients(status)",
    "CREATE INDEX IF NOT EXISTS idx_auth_codes_client ON authorization_codes(client_id)",
    "CREATE INDEX IF NOT EXISTS idx_oauth_tokens_access  ON oauth_tokens(access_token)",
    "CREATE INDEX IF NOT EXISTS idx_oauth_tokens_refresh ON oauth_tokens(refresh_token)",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _db_path() -> Path:
    return Path(settings.auth_db_path).expanduser()


async def _connect() -> aiosqlite.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=5000")
    return db


async def ensure_schema() -> None:
    """Apply all DDL statements. Idempotent (CREATE IF NOT EXISTS)."""
    db = await _connect()
    try:
        for statement in SCHEMA_STATEMENTS:
            await db.execute(statement)
        await db.commit()
    finally:
        await db.close()


# ── OAuth client CRUD ────────────────────────────────────────────────────────

async def store_oauth_client(client_info: OAuthClientInformationFull) -> None:
    """Persist a client registered via dynamic registration."""
    client_id = client_info.client_id or str(uuid4())
    name = client_info.client_name or client_id[:16]
    scopes_list = client_info.scope.split() if client_info.scope else ["agent"]
    redirect_uris = json.dumps(
        [str(u) for u in (client_info.redirect_uris or [])]
    )
    metadata = client_info.model_dump_json()
    now = _now_iso()
    db = await _connect()
    try:
        await db.execute(
            """
            INSERT INTO api_clients
                (client_id, name, description, scopes, status,
                 created_at, last_used_at, redirect_uris, client_metadata)
            VALUES (?, ?, NULL, ?, 'active', ?, NULL, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
                name = excluded.name,
                scopes = excluded.scopes,
                redirect_uris = excluded.redirect_uris,
                client_metadata = excluded.client_metadata
            """,
            (client_id, name, json.dumps(scopes_list), now, redirect_uris, metadata),
        )
        await db.commit()
    finally:
        await db.close()


async def get_oauth_client(client_id: str) -> OAuthClientInformationFull | None:
    """Load a registered client. Returns None if unknown or disabled."""
    db = await _connect()
    try:
        cursor = await db.execute(
            "SELECT status, client_metadata FROM api_clients WHERE client_id = ?",
            (client_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    finally:
        await db.close()
    if row is None:
        return None
    status, metadata_json = row
    if status == "disabled":
        return None
    if metadata_json is None:
        return None
    return OAuthClientInformationFull.model_validate_json(metadata_json)


async def list_api_clients() -> list[dict[str, Any]]:
    """Return all api_clients rows for CLI display."""
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT client_id, name, description, scopes, status, created_at, last_used_at
            FROM api_clients
            ORDER BY created_at ASC
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
    finally:
        await db.close()
    columns = ["client_id", "name", "description", "scopes", "status", "created_at", "last_used_at"]
    out: list[dict[str, Any]] = []
    for row in rows:
        record = dict(zip(columns, row, strict=True))
        record["scopes"] = json.loads(record["scopes"]) if record["scopes"] else []
        out.append(record)
    return out


async def disable_client(client_id: str) -> None:
    """Set status='disabled'. Idempotent."""
    db = await _connect()
    try:
        await db.execute(
            "UPDATE api_clients SET status = 'disabled' WHERE client_id = ?",
            (client_id,),
        )
        await db.commit()
    finally:
        await db.close()


async def enable_client(client_id: str) -> None:
    """Set status='active'. Idempotent."""
    db = await _connect()
    try:
        await db.execute(
            "UPDATE api_clients SET status = 'active' WHERE client_id = ?",
            (client_id,),
        )
        await db.commit()
    finally:
        await db.close()


async def _touch_last_used(client_id: str) -> None:
    """Update last_used_at at most once per configured interval."""
    interval = settings.auth_update_last_used_interval_seconds
    now_iso = _now_iso()
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
        last = row[0]
        if last is not None:
            try:
                last_dt = datetime.fromisoformat(last)
                now_dt = datetime.fromisoformat(now_iso)
                if (now_dt - last_dt).total_seconds() < interval:
                    return
            except ValueError:
                pass
        await db.execute(
            "UPDATE api_clients SET last_used_at = ? WHERE client_id = ?",
            (now_iso, client_id),
        )
        await db.commit()
    finally:
        await db.close()


# ── Authorization code store ─────────────────────────────────────────────────

async def store_authorization_code(
    *,
    code: str,
    client_id: str,
    redirect_uri: str,
    redirect_uri_provided_explicitly: bool,
    scopes: list[str],
    code_challenge: str,
    expires_at: float,
) -> None:
    db = await _connect()
    try:
        await db.execute(
            """
            INSERT INTO authorization_codes
                (code, client_id, redirect_uri, redirect_uri_provided_explicitly,
                 scopes, code_challenge, expires_at, used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                code, client_id, redirect_uri,
                1 if redirect_uri_provided_explicitly else 0,
                json.dumps(scopes), code_challenge, expires_at,
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def load_authorization_code_record(code: str) -> dict[str, Any] | None:
    """Return the auth code row if unused and not expired, else None."""
    now = time.time()
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT code, client_id, redirect_uri, redirect_uri_provided_explicitly,
                   scopes, code_challenge, expires_at
            FROM authorization_codes
            WHERE code = ? AND used_at IS NULL AND expires_at > ?
            """,
            (code, now),
        )
        row = await cursor.fetchone()
        await cursor.close()
    finally:
        await db.close()
    if row is None:
        return None
    columns = [
        "code", "client_id", "redirect_uri", "redirect_uri_provided_explicitly",
        "scopes", "code_challenge", "expires_at",
    ]
    record = dict(zip(columns, row, strict=True))
    record["scopes"] = json.loads(record["scopes"])
    record["redirect_uri_provided_explicitly"] = bool(record["redirect_uri_provided_explicitly"])
    # Add used_at so callers can check it; always None for live codes
    record["used_at"] = None
    return record


async def mark_code_used(code: str) -> bool:
    """Stamp used_at only if not already used. Returns True on first use, False if already used."""
    db = await _connect()
    try:
        cursor = await db.execute(
            "UPDATE authorization_codes SET used_at = ? WHERE code = ? AND used_at IS NULL",
            (time.time(), code),
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# ── OAuth token store ────────────────────────────────────────────────────────

async def store_oauth_token(
    *,
    token_id: str,
    client_id: str,
    client_name: str,
    access_token: str,
    refresh_token: str | None,
    scopes: list[str],
    expires_at: float | None,
) -> None:
    db = await _connect()
    try:
        await db.execute(
            """
            INSERT INTO oauth_tokens
                (token_id, client_id, client_name, access_token, refresh_token,
                 scopes, expires_at, revoked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (token_id, client_id, client_name, access_token, refresh_token,
             json.dumps(scopes), expires_at),
        )
        await db.commit()
    finally:
        await db.close()


async def load_oauth_token_by_access(access_token: str) -> dict[str, Any] | None:
    """Return token row if live (not revoked, not expired), else None."""
    now = time.time()
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT token_id, client_id, client_name, access_token, refresh_token,
                   scopes, expires_at
            FROM oauth_tokens
            WHERE access_token = ?
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (access_token, now),
        )
        row = await cursor.fetchone()
        await cursor.close()
    finally:
        await db.close()
    if row is None:
        return None
    columns = ["token_id", "client_id", "client_name", "access_token",
               "refresh_token", "scopes", "expires_at"]
    record = dict(zip(columns, row, strict=True))
    record["scopes"] = json.loads(record["scopes"])
    await _touch_last_used(record["client_id"])
    return record


async def load_oauth_token_by_refresh(refresh_token: str) -> dict[str, Any] | None:
    """Return token row matching the refresh token if live, else None."""
    now = time.time()
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT token_id, client_id, client_name, access_token, refresh_token,
                   scopes, expires_at
            FROM oauth_tokens
            WHERE refresh_token = ?
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (refresh_token, now),
        )
        row = await cursor.fetchone()
        await cursor.close()
    finally:
        await db.close()
    if row is None:
        return None
    columns = ["token_id", "client_id", "client_name", "access_token",
               "refresh_token", "scopes", "expires_at"]
    record = dict(zip(columns, row, strict=True))
    record["scopes"] = json.loads(record["scopes"])
    return record


async def revoke_oauth_token_by_id(token_id: str) -> None:
    """Stamp revoked_at. Idempotent. Revokes both access and refresh tokens."""
    db = await _connect()
    try:
        await db.execute(
            "UPDATE oauth_tokens SET revoked_at = COALESCE(revoked_at, ?) WHERE token_id = ?",
            (time.time(), token_id),
        )
        await db.commit()
    finally:
        await db.close()
