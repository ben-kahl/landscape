from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


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
