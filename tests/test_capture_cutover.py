from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_debug_flag_survives_later_rejected_turn(monkeypatch):
    """A debug turn buffered first must keep the session's debug flag even when a
    later turn in the same session is rejected by the buffer (regression: the
    rejected turn used to wipe the session-scoped flag, losing debug ingest)."""
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
    mcp_app._DEBUG_CAPTURE_SESSIONS.discard("s-mixed")

    # t1 is buffered with debug=True -> session flagged.
    await mcp_app._auto_ingest_turn("durable fact", "s-mixed", "t1", debug=True)
    # t2 (non-debug, empty) is rejected by the real buffer; the rejection must
    # not clear the debug flag set by t1, which is still pending.
    await mcp_app._auto_ingest_turn("   ", "s-mixed", "t2", debug=False)

    await mcp_app.flush_conversation_session("s-mixed")

    assert captured == [True]


@pytest.mark.asyncio
async def test_flush_window_skips_turns_written_by_explicit_memory(monkeypatch):
    from landscape import mcp_app
    from landscape.conversation_ingestion import ConversationTurn

    captured: list[list[str]] = []

    def fake_select_salient(window):
        captured.append([turn.turn_id for turn in window])
        return []

    async def fake_ingest_conversation_window(session_id, salient, *, debug=False):
        return None

    monkeypatch.setattr(
        "landscape.extraction.salience.select_salient",
        fake_select_salient,
    )
    monkeypatch.setattr(
        "landscape.conversation_ingestion.ingest_conversation_window",
        fake_ingest_conversation_window,
    )
    monkeypatch.setattr(mcp_app, "_EXPLICIT_MEMORY_TURN_KEYS", {("s1", "t1")})

    await mcp_app._flush_window(
        "s1",
        [
            ConversationTurn("s1", "t1", "user", "explicitly remembered"),
            ConversationTurn("s1", "t2", "user", "auto captured"),
        ],
    )

    assert captured == [["t2"]]


@pytest.mark.asyncio
async def test_flush_window_rechecks_explicit_memory_after_salience(monkeypatch):
    from landscape import mcp_app
    from landscape.conversation_ingestion import ConversationTurn
    from landscape.extraction.salience import SalientItem

    captured_salient: list[list[str]] = []
    mcp_app._EXPLICIT_MEMORY_TURN_KEYS.clear()
    mcp_app._EXPLICIT_MEMORY_IN_FLIGHT_TURN_KEYS.clear()

    def fake_select_salient(window):
        mcp_app._EXPLICIT_MEMORY_IN_FLIGHT_TURN_KEYS.add(("s1", "t1"))
        return [
            SalientItem(turn_id="t1", text="explicitly remembered", category="fact"),
            SalientItem(turn_id="t2", text="auto captured", category="fact"),
        ]

    async def fake_ingest_conversation_window(session_id, salient, *, debug=False):
        captured_salient.append([item.turn_id for item in salient])

    monkeypatch.setattr(
        "landscape.extraction.salience.select_salient",
        fake_select_salient,
    )
    monkeypatch.setattr(
        "landscape.conversation_ingestion.ingest_conversation_window",
        fake_ingest_conversation_window,
    )

    await mcp_app._flush_window(
        "s1",
        [
            ConversationTurn("s1", "t1", "user", "explicitly remembered"),
            ConversationTurn("s1", "t2", "user", "auto captured"),
        ],
    )

    assert captured_salient == [["t2"]]


@pytest.mark.asyncio
async def test_flush_window_clears_debug_state_when_ingest_fails(monkeypatch):
    from landscape import mcp_app

    def fake_select_salient(window):
        return []

    async def fake_ingest_conversation_window(session_id, salient, *, debug=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "landscape.extraction.salience.select_salient",
        fake_select_salient,
    )
    monkeypatch.setattr(
        "landscape.conversation_ingestion.ingest_conversation_window",
        fake_ingest_conversation_window,
    )
    mcp_app._DEBUG_CAPTURE_SESSIONS.add("s-fail")

    with pytest.raises(RuntimeError, match="boom"):
        await mcp_app._flush_window("s-fail", [])

    assert "s-fail" not in mcp_app._DEBUG_CAPTURE_SESSIONS


@pytest.mark.asyncio
async def test_remember_reserves_turn_before_ingest_completes(monkeypatch):
    from landscape import mcp_app
    from landscape.pipeline import IngestResult

    entered = asyncio.Event()
    release = asyncio.Event()
    mcp_app._EXPLICIT_MEMORY_TURN_KEYS.clear()
    mcp_app._EXPLICIT_MEMORY_IN_FLIGHT_TURN_KEYS.clear()

    async def fake_ingest(*args, **kwargs):
        entered.set()
        await release.wait()
        return IngestResult(
            doc_id="doc-1",
            already_existed=False,
            entities_created=1,
            entities_reinforced=0,
            relations_created=0,
            relations_reinforced=0,
            relations_superseded=0,
            chunks_created=1,
        )

    monkeypatch.setattr(mcp_app, "require_current_scope", lambda scope: None)
    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)

    task = asyncio.create_task(
        mcp_app.remember("Alice joined Acme.", "alice", "s1", "t1")
    )
    try:
        await entered.wait()
        assert mcp_app._is_explicit_memory_turn("s1", "t1") is True
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_remember_clears_in_flight_reservation_on_failure(monkeypatch):
    from landscape import mcp_app

    mcp_app._EXPLICIT_MEMORY_TURN_KEYS.clear()
    mcp_app._EXPLICIT_MEMORY_IN_FLIGHT_TURN_KEYS.clear()

    async def fake_ingest(*args, **kwargs):
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(mcp_app, "require_current_scope", lambda scope: None)
    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)

    with pytest.raises(RuntimeError, match="ingest failed"):
        await mcp_app.remember("Alice joined Acme.", "alice", "s1", "t1")

    assert mcp_app._is_explicit_memory_turn("s1", "t1") is False


@pytest.mark.asyncio
async def test_auto_ingest_clears_debug_state_when_buffer_append_raises(monkeypatch):
    from landscape import mcp_app

    async def fake_add_turn(turn):
        raise RuntimeError("append failed")

    monkeypatch.setattr(mcp_app._buffer_manager, "add_turn", fake_add_turn)

    with pytest.raises(RuntimeError, match="append failed"):
        await mcp_app._auto_ingest_turn("I use Neo4j.", "s-append", "t1", debug=True)

    assert "s-append" not in mcp_app._DEBUG_CAPTURE_SESSIONS


@pytest.mark.asyncio
async def test_remember_clears_in_flight_reservation_on_cancellation(monkeypatch):
    from landscape import mcp_app

    mcp_app._EXPLICIT_MEMORY_TURN_KEYS.clear()
    mcp_app._EXPLICIT_MEMORY_IN_FLIGHT_TURN_KEYS.clear()

    async def fake_ingest(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(mcp_app, "require_current_scope", lambda scope: None)
    monkeypatch.setattr("landscape.pipeline.ingest", fake_ingest)

    with pytest.raises(asyncio.CancelledError):
        await mcp_app.remember("Alice joined Acme.", "alice", "s1", "t1")

    assert mcp_app._is_explicit_memory_turn("s1", "t1") is False
