from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

LOGGER_NAME = "landscape.retrieval"
DEFAULT_LOG_DIR = Path("logs") / "retrieval"
_RETRIEVAL_FILE_HANDLER: logging.FileHandler | None = None
_RETRIEVAL_LOG_PATH: Path | None = None
_PROCESS_LOG_BASENAME = (
    f"retrieval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-pid{os.getpid()}.jsonl"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _compact_fields(fields: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in fields.items() if value is not None}


def _default_process_log_path(log_dir: Path) -> Path:
    return log_dir / _PROCESS_LOG_BASENAME


def ensure_retrieval_log_sink(
    log_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    global _RETRIEVAL_FILE_HANDLER, _RETRIEVAL_LOG_PATH

    if log_dir is None and not force and _RETRIEVAL_LOG_PATH is not None:
        return _RETRIEVAL_LOG_PATH

    resolved_dir = (log_dir or DEFAULT_LOG_DIR).resolve()
    log_path = _default_process_log_path(resolved_dir)

    if (
        not force
        and _RETRIEVAL_FILE_HANDLER is not None
        and _RETRIEVAL_LOG_PATH == log_path
    ):
        return log_path

    logger = logging.getLogger(LOGGER_NAME)
    if _RETRIEVAL_FILE_HANDLER is not None:
        logger.removeHandler(_RETRIEVAL_FILE_HANDLER)
        _RETRIEVAL_FILE_HANDLER.close()

    resolved_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    _RETRIEVAL_FILE_HANDLER = handler
    _RETRIEVAL_LOG_PATH = log_path
    return log_path


def ensure_query_cli_logging() -> None:
    ensure_retrieval_log_sink()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class RetrievalLogContext:
    query_text: str
    hops: int
    limit: int
    chunk_limit: int
    reinforce: bool
    session_id: str | None = None
    since: datetime | None = None
    debug: bool = False
    retrieval_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=perf_counter)
    current_stage: str = "retrieval_started"

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(LOGGER_NAME)

    def set_stage(self, stage: str) -> float:
        self.current_stage = stage
        return perf_counter()

    def emit(
        self,
        event: str,
        *,
        level: int = logging.INFO,
        always: bool = False,
        **fields: object,
    ) -> None:
        if not always and not self.debug:
            return

        payload = _compact_fields(
            {
                "timestamp": _now_iso(),
                "event": event,
                "retrieval_id": self.retrieval_id,
                "query_text": self.query_text,
                "hops": self.hops,
                "limit": self.limit,
                "chunk_limit": self.chunk_limit,
                "reinforce": self.reinforce,
                "session_id": self.session_id,
                "since": self.since.isoformat() if self.since is not None else None,
                "debug": self.debug,
                **fields,
            }
        )
        self.logger.log(level, json.dumps(payload, sort_keys=True, default=str))

    def emit_started(self) -> None:
        self.current_stage = "retrieval_started"
        self.emit(
            "retrieval_started",
            always=True,
            query_length=len(self.query_text),
        )

    def emit_completed(self, **fields: object) -> None:
        self.current_stage = "retrieval_completed"
        self.emit(
            "retrieval_completed",
            always=True,
            duration_ms=_elapsed_ms(self.started_at),
            **fields,
        )

    def emit_failed(self, error: Exception) -> None:
        self.emit(
            "retrieval_failed",
            level=logging.ERROR,
            always=True,
            failed_stage=self.current_stage,
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=_elapsed_ms(self.started_at),
        )


def create_retrieval_log_context(
    *,
    query_text: str,
    hops: int,
    limit: int,
    chunk_limit: int,
    reinforce: bool,
    session_id: str | None = None,
    since: datetime | None = None,
    debug: bool = False,
    retrieval_id: str | None = None,
) -> RetrievalLogContext:
    ensure_retrieval_log_sink()
    return RetrievalLogContext(
        query_text=query_text,
        hops=hops,
        limit=limit,
        chunk_limit=chunk_limit,
        reinforce=reinforce,
        session_id=session_id,
        since=since,
        debug=debug,
        retrieval_id=retrieval_id or uuid.uuid4().hex,
    )
