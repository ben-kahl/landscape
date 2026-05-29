from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def test_mcp_app_import_does_not_load_pipeline():
    code = textwrap.dedent(
        """
        import sys
        import landscape.mcp_app
        raise SystemExit(1 if "landscape.pipeline" in sys.modules else 0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_schedule_auto_ingestion_appends_to_buffer(monkeypatch):
    from landscape import mcp_app

    added: list[tuple[str, str, str]] = []

    async def fake_add_turn(turn):
        added.append((turn.session_id, turn.turn_id, turn.text))
        return True

    monkeypatch.setattr(mcp_app._buffer_manager, "add_turn", fake_add_turn)

    task = mcp_app._schedule_auto_ingestion("I use Neo4j.", "s1", "t1", role="user")
    await task

    assert added == [("s1", "t1", "I use Neo4j.")]


@pytest.mark.asyncio
async def test_auto_ingestion_debug_reaches_window_ingest(monkeypatch):
    from landscape import mcp_app

    captured: list[bool] = []

    def fake_select_salient(window):
        return []

    async def fake_ingest_conversation_window(session_id, salient, *, debug=False):
        captured.append(debug)

    monkeypatch.setattr(
        "landscape.extraction.salience.select_salient",
        fake_select_salient,
    )
    monkeypatch.setattr(
        "landscape.conversation_ingestion.ingest_conversation_window",
        fake_ingest_conversation_window,
    )

    await mcp_app._auto_ingest_turn("I use Neo4j.", "s-debug", "t1", debug=True)
    await mcp_app.flush_conversation_session("s-debug")

    assert captured == [True]
