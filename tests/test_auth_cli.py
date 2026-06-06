"""CLI auth command tests (OAuth era).

Only list-clients, disable-client, enable-client remain.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


def _run_cli(*args: str) -> int:
    from landscape.cli.main import _build_parser
    parser = _build_parser()
    parsed = parser.parse_args(list(args))
    return asyncio.run(parsed.func(parsed))


@pytest_asyncio.fixture
async def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    await auth_store.ensure_schema()
    yield db_path


async def _register(client_id: str) -> None:
    await auth_store.store_oauth_client(
        OAuthClientInformationFull(
            client_id=client_id,
            client_name=f"Client {client_id}",
            redirect_uris=[AnyUrl("http://localhost/cb")],
            scope="agent",
        )
    )


def test_list_clients_empty(auth_db, capsys):
    rc = _run_cli("auth", "list-clients")
    assert rc == 0
    out = capsys.readouterr().out
    assert "no registered clients" in out


async def test_list_clients_shows_registered(auth_db, capsys):
    await _register("cli-test-client")
    rc = await asyncio.to_thread(_run_cli, "auth", "list-clients")
    assert rc == 0
    out = capsys.readouterr().out
    assert "cli-test-client" in out


async def test_disable_then_enable_round_trips(auth_db):
    await _register("toggle-client")
    rc = await asyncio.to_thread(_run_cli, "auth", "disable-client", "--client-id", "toggle-client")
    assert rc == 0
    assert await auth_store.get_oauth_client("toggle-client") is None

    rc = await asyncio.to_thread(_run_cli, "auth", "enable-client", "--client-id", "toggle-client")
    assert rc == 0
    assert await auth_store.get_oauth_client("toggle-client") is not None
