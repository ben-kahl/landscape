from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_session_end_schedules_background_flush(monkeypatch):
    """The endpoint must NOT await the flush inline: a flush runs salience +
    extraction (multi-second) and would be cancelled by the client hook's short
    POST timeout, restoring the buffer and ingesting nothing. It schedules a
    background flush and returns immediately."""
    import landscape.mcp_app as mcp_app
    from landscape.api import hooks

    scheduled: list[str] = []

    def fake_schedule(session_id: str):
        scheduled.append(session_id)

    monkeypatch.setattr(mcp_app, "schedule_conversation_flush", fake_schedule)

    req = hooks.SessionEndHookRequest(client="claude-code", session_id="s1")
    resp = await hooks.session_end_hook(req, auth=None)

    assert resp.flushed is True
    assert scheduled == ["s1"]


async def test_schedule_conversation_flush_runs_flush_in_background(monkeypatch):
    """The scheduler wrapper actually runs the underlying flush off the caller's
    stack so it survives the request returning / the client disconnecting."""
    import landscape.mcp_app as mcp_app

    flushed: list[str] = []

    async def fake_flush(session_id: str) -> None:
        flushed.append(session_id)

    monkeypatch.setattr(mcp_app, "flush_conversation_session", fake_flush)

    task = mcp_app.schedule_conversation_flush("s1")
    await task

    assert flushed == ["s1"]
