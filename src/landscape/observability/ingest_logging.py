from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

LOGGER_NAME = "landscape.ingest"
DEFAULT_LOG_DIR = Path("logs") / "ingest"
_INGEST_FILE_HANDLER: logging.FileHandler | None = None
_INGEST_LOG_PATH: Path | None = None
_PROCESS_LOG_BASENAME = (
    f"ingest-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-pid{os.getpid()}.jsonl"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _compact_fields(fields: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in fields.items() if value is not None}


def _default_process_log_path(log_dir: Path) -> Path:
    return log_dir / _PROCESS_LOG_BASENAME


def ensure_ingest_log_sink(
    log_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    global _INGEST_FILE_HANDLER, _INGEST_LOG_PATH

    if log_dir is None and not force and _INGEST_LOG_PATH is not None:
        return _INGEST_LOG_PATH

    resolved_dir = (log_dir or DEFAULT_LOG_DIR).resolve()
    log_path = _default_process_log_path(resolved_dir)

    if (
        not force
        and _INGEST_FILE_HANDLER is not None
        and _INGEST_LOG_PATH == log_path
    ):
        return log_path

    logger = logging.getLogger(LOGGER_NAME)
    if _INGEST_FILE_HANDLER is not None:
        logger.removeHandler(_INGEST_FILE_HANDLER)
        _INGEST_FILE_HANDLER.close()

    resolved_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    _INGEST_FILE_HANDLER = handler
    _INGEST_LOG_PATH = log_path
    return log_path


def ensure_cli_logging() -> None:
    ensure_ingest_log_sink()
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class IngestLogContext:
    title: str
    source_type: str
    session_id: str | None = None
    turn_id: str | None = None
    debug: bool = False
    ingest_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=perf_counter)
    current_stage: str = "ingest_started"

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
                "ingest_id": self.ingest_id,
                "title": self.title,
                "source_type": self.source_type,
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "debug": self.debug,
                **fields,
            }
        )
        self.logger.log(level, json.dumps(payload, sort_keys=True, default=str))

    def emit_started(self, *, content_hash: str, text_length: int) -> None:
        self.current_stage = "ingest_started"
        self.emit(
            "ingest_started",
            always=True,
            content_hash=content_hash,
            text_length=text_length,
        )

    def emit_completed(self, **fields: object) -> None:
        self.current_stage = "ingest_completed"
        self.emit(
            "ingest_completed",
            always=True,
            duration_ms=_elapsed_ms(self.started_at),
            **fields,
        )

    def emit_failed(self, error: Exception) -> None:
        self.emit(
            "ingest_failed",
            level=logging.ERROR,
            always=True,
            failed_stage=self.current_stage,
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=_elapsed_ms(self.started_at),
        )


def create_ingest_log_context(
    *,
    title: str,
    source_type: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    debug: bool = False,
    ingest_id: str | None = None,
) -> IngestLogContext:
    ensure_ingest_log_sink()
    return IngestLogContext(
        title=title,
        source_type=source_type,
        session_id=session_id,
        turn_id=turn_id,
        debug=debug,
        ingest_id=ingest_id or uuid.uuid4().hex,
    )
