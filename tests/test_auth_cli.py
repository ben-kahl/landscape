"""Tests for ``landscape auth ...`` CLI commands.

These run the argparse entry point (``landscape.cli.main``) directly --
no subprocess, no network -- and assert against captured stdout. Each
test gets a tmp-isolated SQLite auth DB via the same monkeypatch pattern
as ``tests/test_auth_store.py``.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from landscape import cli
from landscape.config import settings
from landscape.storage import auth_store

pytestmark = pytest.mark.unit


async def _run_cli(argv: list[str]) -> int:
    """Run the sync CLI entry point in a worker thread.

    ``cli.main`` calls ``asyncio.run`` for coroutine handlers; that fails when
    invoked from an already-running event loop (i.e. inside async tests). A
    worker thread gives the CLI its own loop and matches real CLI behavior.
    """
    return await asyncio.to_thread(cli.main, argv)


@pytest.fixture
def auth_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "auth.db"
    monkeypatch.setattr(settings, "auth_db_path", str(db_path))
    return db_path


_BEARER_RE = re.compile(r"bearer_token:\s+(\S+)")
_SECRET_ID_RE = re.compile(r"secret_id:\s+(\S+)")
_CLIENT_ID_RE = re.compile(r"client_id:\s+(\S+)")


def _extract_bearer(out: str) -> str:
    m = _BEARER_RE.search(out)
    assert m is not None, f"no bearer_token in stdout: {out!r}"
    return m.group(1)


def _extract_first(pattern: re.Pattern[str], out: str) -> str:
    m = pattern.search(out)
    assert m is not None, f"no match for {pattern.pattern!r} in stdout: {out!r}"
    return m.group(1)


def test_auth_help_lists_subcommands(auth_db, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["auth", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for cmd in (
        "create-client",
        "list-clients",
        "list-secrets",
        "create-secret",
        "rotate-secret",
        "revoke-secret",
        "disable-client",
        "enable-client",
    ):
        assert cmd in output


def test_create_client_prints_bearer_once(auth_db, capsys):
    code = cli.main(
        ["auth", "create-client", "--name", "claude", "--scope", "agent"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "bearer_token:" in out
    assert "will not be shown again" in out
    bearer = _extract_bearer(out)
    assert bearer.startswith("lsk_")


async def test_create_client_token_authenticates(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "alpha", "--scope", "agent"])
    bearer = _extract_bearer(capsys.readouterr().out)
    ctx = await auth_store.authenticate_bearer_token(bearer)
    assert ctx is not None
    assert ctx.client_name == "alpha"
    assert ctx.scopes == frozenset({"agent"})


async def test_rotate_secret_keeps_old_secret_until_revoked(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "bench", "--scope", "agent"])
    first_out = capsys.readouterr().out
    first_bearer = _extract_bearer(first_out)
    first_secret_id = _extract_first(_SECRET_ID_RE, first_out)
    client_id = _extract_first(_CLIENT_ID_RE, first_out)

    code = await _run_cli(["auth", "rotate-secret", "--client-id", client_id])
    assert code == 0
    second_out = capsys.readouterr().out
    second_bearer = _extract_bearer(second_out)
    second_secret_id = _extract_first(_SECRET_ID_RE, second_out)

    assert first_secret_id != second_secret_id
    assert first_bearer != second_bearer

    # Both secrets must still authenticate -- rotation does not auto-revoke.
    assert await auth_store.authenticate_bearer_token(first_bearer) is not None
    assert await auth_store.authenticate_bearer_token(second_bearer) is not None


async def test_revoke_secret_kills_only_that_secret(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "rev", "--scope", "agent"])
    first_out = capsys.readouterr().out
    first_bearer = _extract_bearer(first_out)
    first_secret_id = _extract_first(_SECRET_ID_RE, first_out)
    client_id = _extract_first(_CLIENT_ID_RE, first_out)

    await _run_cli(["auth", "create-secret", "--client-id", client_id])
    second_out = capsys.readouterr().out
    second_bearer = _extract_bearer(second_out)

    code = await _run_cli(["auth", "revoke-secret", "--secret-id", first_secret_id])
    assert code == 0

    assert await auth_store.authenticate_bearer_token(first_bearer) is None
    assert await auth_store.authenticate_bearer_token(second_bearer) is not None


async def test_disable_then_enable_client_round_trips(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "tgl", "--scope", "agent"])
    out = capsys.readouterr().out
    bearer = _extract_bearer(out)
    client_id = _extract_first(_CLIENT_ID_RE, out)

    assert await auth_store.authenticate_bearer_token(bearer) is not None

    await _run_cli(["auth", "disable-client", "--client-id", client_id])
    assert await auth_store.authenticate_bearer_token(bearer) is None

    await _run_cli(["auth", "enable-client", "--client-id", client_id])
    assert await auth_store.authenticate_bearer_token(bearer) is not None


async def test_list_clients_never_exposes_secret_hash(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "ls-c", "--scope", "agent"])
    capsys.readouterr()  # discard creation output
    code = await _run_cli(["auth", "list-clients"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ls-c" in out
    assert "secret_hash" not in out
    assert "$argon2" not in out
    assert "bearer_token" not in out


async def test_list_secrets_never_exposes_secret_hash(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "ls-s", "--scope", "agent"])
    create_out = capsys.readouterr().out
    client_id = _extract_first(_CLIENT_ID_RE, create_out)

    code = await _run_cli(["auth", "list-secrets", "--client-id", client_id])
    assert code == 0
    out = capsys.readouterr().out
    assert "secret_id:" in out
    assert "secret_hash" not in out
    assert "$argon2" not in out
    assert "bearer_token" not in out


async def test_create_secret_on_unknown_client_errors(auth_db, capsys):
    code = await _run_cli(["auth", "create-secret", "--client-id", "does-not-exist"])
    # `landscape.cli.main.main` traps exceptions and returns 1.
    assert code == 1
    err = capsys.readouterr().err
    assert "Unknown client_id" in err or "Error:" in err


async def test_rotate_secret_with_zero_revoke_after_revokes_old(auth_db, capsys):
    await _run_cli(["auth", "create-client", "--name", "rot0", "--scope", "agent"])
    first_out = capsys.readouterr().out
    first_bearer = _extract_bearer(first_out)
    client_id = _extract_first(_CLIENT_ID_RE, first_out)

    code = await _run_cli(
        [
            "auth",
            "rotate-secret",
            "--client-id",
            client_id,
            "--revoke-after",
            "0",
        ]
    )
    assert code == 0
    second_bearer = _extract_bearer(capsys.readouterr().out)

    assert await auth_store.authenticate_bearer_token(first_bearer) is None
    assert await auth_store.authenticate_bearer_token(second_bearer) is not None
