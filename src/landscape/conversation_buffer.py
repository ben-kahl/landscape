from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from landscape.conversation_ingestion import (
    ConversationTurn,
    should_auto_ingest_turn,
    turn_fingerprint,
)

logger = logging.getLogger(__name__)


class SessionBuffer:
    """Per-session pending turns plus pure flush decisions."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.pending: list[ConversationTurn] = []
        self._seen: set[str] = set()
        self._last_flushed_tail: list[ConversationTurn] = []
        self._flush_snapshot: tuple[list[ConversationTurn], list[ConversationTurn]] | None = None

    def append(self, turn: ConversationTurn) -> bool:
        if not should_auto_ingest_turn(turn, seen_fingerprints=self._seen):
            return False
        normalized = ConversationTurn(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            role=turn.role,
            text=turn.text.strip(),
        )
        self._seen.add(turn_fingerprint(normalized))
        self.pending.append(normalized)
        return True

    def should_flush_on_size(self, *, max_turns: int) -> bool:
        return len(self.pending) >= max_turns

    def has_pending(self) -> bool:
        return bool(self.pending)

    def take_window(self, *, overlap_turns: int) -> list[ConversationTurn]:
        if not self.pending:
            return []
        self._flush_snapshot = (self.pending.copy(), self._last_flushed_tail.copy())
        window = self._last_flushed_tail + self.pending
        overlap = max(0, overlap_turns)
        self._last_flushed_tail = self.pending[-overlap:] if overlap else []
        self.pending = []
        return window

    def commit_window(self) -> None:
        self._flush_snapshot = None

    def restore_window(self) -> None:
        if self._flush_snapshot is None:
            return
        pending, last_flushed_tail = self._flush_snapshot
        self.pending = pending + self.pending
        self._last_flushed_tail = last_flushed_tail
        self._flush_snapshot = None


FlushFn = Callable[[str, list[ConversationTurn]], Awaitable[None]]


class ConversationBufferManager:
    """Owns session buffers, session locks, idle timers, and flush callbacks."""

    def __init__(
        self,
        flush_fn: FlushFn,
        *,
        max_turns: int,
        idle_seconds: float,
        overlap_turns: int,
    ) -> None:
        self._flush_fn = flush_fn
        self._max_turns = max_turns
        self._idle_seconds = idle_seconds
        self._overlap_turns = overlap_turns
        self._buffers: dict[str, SessionBuffer] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._idle_tasks: dict[str, asyncio.Task] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _buffer(self, session_id: str) -> SessionBuffer:
        return self._buffers.setdefault(session_id, SessionBuffer(session_id))

    async def add_turn(self, turn: ConversationTurn) -> bool:
        async with self._lock(turn.session_id):
            accepted = self._buffer(turn.session_id).append(turn)
            if not accepted:
                return False
            if self._buffer(turn.session_id).should_flush_on_size(max_turns=self._max_turns):
                await self._flush_locked(turn.session_id)
            else:
                self._arm_idle_timer(turn.session_id)
            return True

    async def flush_session(self, session_id: str) -> None:
        async with self._lock(session_id):
            await self._flush_locked(session_id)

    async def _flush_locked(self, session_id: str) -> None:
        self._cancel_idle_timer(session_id)
        buffer = self._buffer(session_id)
        window = buffer.take_window(overlap_turns=self._overlap_turns)
        if not window:
            return
        try:
            await self._flush_fn(session_id, window)
        except Exception:
            buffer.restore_window()
            logger.exception("conversation buffer flush failed for session %s", session_id)
        else:
            buffer.commit_window()

    def _arm_idle_timer(self, session_id: str) -> None:
        self._cancel_idle_timer(session_id)
        self._idle_tasks[session_id] = asyncio.create_task(self._idle_flush(session_id))

    def _cancel_idle_timer(self, session_id: str) -> None:
        task = self._idle_tasks.pop(session_id, None)
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _idle_flush(self, session_id: str) -> None:
        try:
            await asyncio.sleep(self._idle_seconds)
        except asyncio.CancelledError:
            return
        await self.flush_session(session_id)
