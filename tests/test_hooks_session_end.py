from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_session_end_triggers_flush(monkeypatch):
    from landscape.api import hooks

    flushed: list[str] = []

    async def fake_flush(session_id: str) -> None:
        flushed.append(session_id)

    import landscape.mcp_app as mcp_app

    monkeypatch.setattr(mcp_app, "flush_conversation_session", fake_flush)

    req = hooks.SessionEndHookRequest(client="claude-code", session_id="s1")
    resp = await hooks.session_end_hook(req, auth=None)

    assert resp.flushed is True
    assert flushed == ["s1"]
