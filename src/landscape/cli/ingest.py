from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from landscape.cli.runtime import close_runtime
from landscape.observability import IngestLogContext, ensure_cli_logging


def _get_runtime():
    from landscape import pipeline
    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store

    return pipeline, encoder, neo4j_store, qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest one file (markdown, text, PDF, DOCX, PPTX, XLSX, HTML, CSV, ...)",
        description=(
            "Ingest one file through the local pipeline. The file format is "
            "inferred from its extension and converted to markdown via "
            "markitdown when needed (PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, "
            "XML, EPUB, RTF). Markdown and plain text are passed through "
            "unchanged. Unknown extensions are read as utf-8 text."
        ),
    )
    ingest_parser.add_argument("path", help="Path to the input file")
    ingest_parser.add_argument("--title", help="Document title", default=None)
    ingest_parser.add_argument(
        "--source-type",
        help=(
            "Override the source type recorded with the document. "
            "Default: inferred from the file extension."
        ),
        default=None,
    )
    ingest_parser.add_argument("--session-id", default=None)
    ingest_parser.add_argument("--turn-id", default=None)
    ingest_parser.add_argument("--debug", action="store_true")
    progress_group = ingest_parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        action="store_true",
        default=None,
        help="Force a Rich progress bar even when stderr is not a TTY.",
    )
    progress_group.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable the ingest progress bar.",
    )
    ingest_parser.set_defaults(func=handle_ingest)

    dir_parser = subparsers.add_parser(
        "ingest-dir",
        help="Ingest files from a directory",
        description=(
            "Ingest files from a directory in sorted order. By default, every "
            "file whose extension has a known converter (markdown, text, PDF, "
            "DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPUB, RTF) is picked up; "
            "unknown extensions are skipped with a log line. Pass --glob to "
            "restrict to a single glob pattern instead."
        ),
    )
    dir_parser.add_argument("path", help="Directory to ingest")
    dir_parser.add_argument(
        "--glob",
        default=None,
        help=(
            "Restrict ingestion to files matching this glob (e.g. '*.md'). "
            "Default: walk all files and dispatch by extension."
        ),
    )
    dir_parser.add_argument(
        "--source-type",
        default=None,
        help=(
            "Override the source type for every ingested file. "
            "Default: inferred per-file from the extension."
        ),
    )
    dir_parser.add_argument("--session-id", default=None)
    dir_parser.add_argument("--debug", action="store_true")
    dir_parser.add_argument("--stop-on-error", action="store_true")
    dir_parser.set_defaults(func=handle_ingest_dir)


def _validate_provenance(
    parser: argparse.ArgumentParser,
    session_id: str | None,
    turn_id: str | None,
) -> None:
    if (session_id is None) != (turn_id is None):
        parser.error("session-id and turn-id must be provided together")
    if session_id is not None and (not session_id.strip() or not turn_id.strip()):
        parser.error("session-id and turn-id must be non-empty")


def _format_summary(result) -> str:
    return "\n".join(
        [
            f"doc_id: {result.doc_id}",
            f"already_existed: {result.already_existed}",
            f"entities: created={result.entities_created} reinforced={result.entities_reinforced}",
            (
                "relations: "
                f"created={result.relations_created} "
                f"reinforced={result.relations_reinforced} "
                f"superseded={result.relations_superseded}"
            ),
            f"chunks_created: {result.chunks_created}",
        ]
    )


@dataclass
class CliIngestProgress(IngestLogContext):
    def __post_init__(self) -> None:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=False,
        )
        self._stage_task = self._progress.add_task("Preparing ingest", total=None)
        self._chunk_task: int | None = None

    def __enter__(self) -> "CliIngestProgress":
        self._progress.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._progress.stop()

    def emit(
        self,
        event: str,
        *,
        level: int = 20,
        always: bool = False,
        **fields: object,
    ) -> None:
        self._update_progress(event, fields)
        super().emit(event, level=level, always=always, **fields)

    def _update_progress(self, event: str, fields: dict[str, object]) -> None:
        if event == "ingest_started":
            self._progress.update(self._stage_task, description="Starting ingest")
            return
        if event == "chunking_completed":
            chunk_count = int(fields.get("chunk_count") or 0)
            self._progress.update(self._stage_task, description="Chunking complete")
            if chunk_count:
                self._chunk_task = self._progress.add_task(
                    "Extracting chunks",
                    total=chunk_count,
                )
            return
        if event == "chunk_extraction_completed" and self._chunk_task is not None:
            chunk_number = int(fields.get("chunk_number") or 0)
            chunk_count = int(fields.get("chunk_count") or 0)
            self._progress.update(
                self._chunk_task,
                completed=chunk_number,
                description=f"Extracting chunks {chunk_number}/{chunk_count}",
            )
            return
        if event == "entity_writes_completed":
            self._progress.update(self._stage_task, description="Entities written")
            return
        if event == "relation_upserts_completed":
            self._progress.update(self._stage_task, description="Relations written")
            return
        if event == "ingest_completed":
            self._progress.update(self._stage_task, description="Ingest complete")
            return
        if event == "ingest_failed":
            self._progress.update(self._stage_task, description="Ingest failed")


def _progress_enabled(progress_flag: bool | None) -> bool:
    if progress_flag is not None:
        return progress_flag
    return sys.stderr.isatty()


async def _ingest_text(
    text: str,
    title: str,
    source_type: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    debug: bool = False,
    log_context: IngestLogContext | None = None,
):
    pipeline, encoder, neo4j_store, qdrant_store = _get_runtime()
    try:
        ensure_cli_logging()
        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()
        return await pipeline.ingest(
            text,
            title,
            source_type,
            session_id=session_id,
            turn_id=turn_id,
            debug=debug,
            log_context=log_context,
        )
    finally:
        await close_runtime(neo4j_store, qdrant_store)


async def handle_ingest(args: argparse.Namespace) -> int:
    from landscape.ingestion.converters import (
        ConverterError,
        convert_to_markdown,
    )

    parser = argparse.ArgumentParser(prog="landscape ingest")
    _validate_provenance(parser, args.session_id, args.turn_id)
    path = Path(args.path)

    # Convert once, up front, so we can surface a clean error before spinning
    # up the encoder / Neo4j / Qdrant connections — and so we know the final
    # source_type before constructing the progress bar.
    try:
        converted = convert_to_markdown(path)
    except ConverterError as exc:
        parser.error(str(exc))

    title = args.title or converted.title_hint or path.stem
    source_type = args.source_type or converted.source_type

    if _progress_enabled(args.progress):
        with CliIngestProgress(
            title=title,
            source_type=source_type,
            session_id=args.session_id,
            turn_id=args.turn_id,
            debug=args.debug,
        ) as progress:
            result = await _ingest_text(
                text=converted.text,
                title=title,
                source_type=source_type,
                session_id=args.session_id,
                turn_id=args.turn_id,
                debug=args.debug,
                log_context=progress,
            )
    else:
        result = await _ingest_text(
            text=converted.text,
            title=title,
            source_type=source_type,
            session_id=args.session_id,
            turn_id=args.turn_id,
            debug=args.debug,
        )
    print(_format_summary(result))
    return 0


async def handle_ingest_dir(args: argparse.Namespace) -> int:
    from landscape.ingestion.converters import (
        ConverterError,
        convert_to_markdown,
        is_supported_extension,
    )

    parser = argparse.ArgumentParser(prog="landscape ingest-dir")
    root = Path(args.path)
    if not root.exists():
        parser.error(f"path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"path is not a directory: {root}")
    if args.session_id is not None and not args.session_id.strip():
        parser.error("session-id must be non-empty")

    if args.glob is not None:
        # Strict-pattern mode: caller knows exactly which files they want.
        candidates = sorted(p for p in root.glob(args.glob) if p.is_file())
        if not candidates:
            print(f"No files matched {args.glob!r} under {root}")
            return 0
        paths = candidates
        skipped: list[Path] = []
    else:
        # Default: walk everything, dispatch by extension. Files with no
        # known converter are skipped with a log line so the user can see
        # what was passed over.
        all_files = sorted(p for p in root.iterdir() if p.is_file())
        paths = [p for p in all_files if is_supported_extension(p)]
        skipped = [p for p in all_files if not is_supported_extension(p)]
        if not paths and not skipped:
            print(f"No files found under {root}")
            return 0
        for path in skipped:
            print(f"skip: {path.name} (no converter for {path.suffix or '<none>'})")
        if not paths:
            print(f"No supported files found under {root}")
            return 0

    failures = 0
    for index, path in enumerate(paths, start=1):
        try:
            converted = convert_to_markdown(path)
            title = converted.title_hint or path.stem
            source_type = args.source_type or converted.source_type
            turn_id = f"t{index}" if args.session_id is not None else None
            result = await _ingest_text(
                text=converted.text,
                title=title,
                source_type=source_type,
                session_id=args.session_id,
                turn_id=turn_id,
                debug=args.debug,
            )
            print(f"[{index}/{len(paths)}] {path.name}")
            print(_format_summary(result))
        except ConverterError as exc:
            failures += 1
            print(f"[{index}/{len(paths)}] {path.name}: CONVERT_ERROR {exc}")
            if args.stop_on_error:
                return 1
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(paths)}] {path.name}: ERROR {exc}")
            if args.stop_on_error:
                return 1
    return 1 if failures else 0
