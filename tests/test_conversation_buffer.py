import asyncio

import pytest

from landscape.conversation_ingestion import ConversationTurn

pytestmark = pytest.mark.unit


def _turn(tid, text, role="user"):
    return ConversationTurn(session_id="s1", turn_id=tid, role=role, text=text)


def test_append_dedups_and_filters_noise():
    from landscape.conversation_buffer import SessionBuffer

    buf = SessionBuffer(session_id="s1")
    assert buf.append(_turn("t1", "I work at Acme.")) is True
    assert buf.append(_turn("t1", "I work at Acme.")) is False  # duplicate fingerprint
    assert buf.append(_turn("t2", "   ")) is False  # empty
    assert buf.append(_turn("t3", "tool blob", role="tool")) is False  # tool-noise
    assert [t.turn_id for t in buf.pending] == ["t1"]


def test_size_trigger_fires_at_threshold():
    from landscape.conversation_buffer import SessionBuffer

    buf = SessionBuffer(session_id="s1")
    for i in range(3):
        buf.append(_turn(f"t{i}", f"durable fact number {i}"))
    assert buf.should_flush_on_size(max_turns=3) is True
    assert buf.should_flush_on_size(max_turns=4) is False


def test_take_window_includes_tail_overlap_and_marks_flushed():
    from landscape.conversation_buffer import SessionBuffer

    buf = SessionBuffer(session_id="s1")
    for i in range(4):
        buf.append(_turn(f"t{i}", f"fact {i}"))

    first = buf.take_window(overlap_turns=1)
    assert [t.turn_id for t in first] == ["t0", "t1", "t2", "t3"]
    assert buf.pending == []  # all consumed

    # New turns arrive; the next window is prefixed with the last flushed turn.
    buf.append(_turn("t4", "fact 4"))
    second = buf.take_window(overlap_turns=1)
    assert [t.turn_id for t in second] == ["t3", "t4"]  # t3 = overlap, t4 = new


def test_take_window_empty_when_no_pending():
    from landscape.conversation_buffer import SessionBuffer

    buf = SessionBuffer(session_id="s1")
    assert buf.take_window(overlap_turns=2) == []


@pytest.mark.asyncio
async def test_manager_size_trigger_invokes_flush_fn():
    from landscape.conversation_buffer import ConversationBufferManager

    flushed: list[tuple[str, list[str]]] = []

    async def flush_fn(session_id, window):
        flushed.append((session_id, [t.turn_id for t in window]))

    mgr = ConversationBufferManager(flush_fn, max_turns=2, idle_seconds=999, overlap_turns=0)
    await mgr.add_turn(_turn("t0", "fact zero"))
    assert flushed == []
    await mgr.add_turn(_turn("t1", "fact one"))
    assert flushed == [("s1", ["t0", "t1"])]


@pytest.mark.asyncio
async def test_manager_idle_trigger_completes_flush_fn():
    from landscape.conversation_buffer import ConversationBufferManager

    flushed: list[tuple[str, list[str]]] = []
    flushed_event = asyncio.Event()

    async def flush_fn(session_id, window):
        await asyncio.sleep(0)
        flushed.append((session_id, [t.turn_id for t in window]))
        flushed_event.set()

    mgr = ConversationBufferManager(flush_fn, max_turns=99, idle_seconds=0.01, overlap_turns=0)
    await mgr.add_turn(_turn("t0", "fact zero"))

    await asyncio.wait_for(flushed_event.wait(), timeout=0.5)
    assert flushed == [("s1", ["t0"])]


@pytest.mark.asyncio
async def test_manager_failed_flush_preserves_pending_for_retry():
    from landscape.conversation_buffer import ConversationBufferManager

    attempts: list[tuple[str, list[str]]] = []

    async def flush_fn(session_id, window):
        attempts.append((session_id, [t.turn_id for t in window]))
        if len(attempts) == 1:
            raise RuntimeError("temporary flush failure")

    mgr = ConversationBufferManager(flush_fn, max_turns=2, idle_seconds=999, overlap_turns=1)
    await mgr.add_turn(_turn("t0", "fact zero"))
    await mgr.add_turn(_turn("t1", "fact one"))

    assert attempts == [("s1", ["t0", "t1"])]

    await mgr.flush_session("s1")
    assert attempts == [("s1", ["t0", "t1"]), ("s1", ["t0", "t1"])]


@pytest.mark.asyncio
async def test_manager_cancelled_flush_restores_pending_and_reraises():
    from landscape.conversation_buffer import ConversationBufferManager

    attempts: list[tuple[str, list[str]]] = []

    async def flush_fn(session_id, window):
        attempts.append((session_id, [t.turn_id for t in window]))
        if len(attempts) == 1:
            raise asyncio.CancelledError

    mgr = ConversationBufferManager(flush_fn, max_turns=2, idle_seconds=999, overlap_turns=1)
    await mgr.add_turn(_turn("t0", "fact zero"))
    with pytest.raises(asyncio.CancelledError):
        await mgr.add_turn(_turn("t1", "fact one"))

    assert attempts == [("s1", ["t0", "t1"])]

    await mgr.flush_session("s1")
    assert attempts == [("s1", ["t0", "t1"]), ("s1", ["t0", "t1"])]
