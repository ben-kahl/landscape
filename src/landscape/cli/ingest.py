from __future__ import annotations

import argparse
from pathlib import Path

from landscape.cli.runtime import close_runtime


def _get_runtime():
    from landscape import pipeline
    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store

    return pipeline, encoder, neo4j_store, qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest one UTF-8 text or Markdown file",
        description="Ingest one UTF-8 text or Markdown file through the local pipeline.",
    )
    ingest_parser.add_argument("path", help="Path to the input file")
    ingest_parser.add_argument("--title", help="Document title", default=None)
    ingest_parser.add_argument(
        "--source-type",
        help="Source type recorded with the document",
        default="text",
    )
    ingest_parser.add_argument("--session-id", default=None)
    ingest_parser.add_argument("--turn-id", default=None)
    ingest_parser.set_defaults(func=handle_ingest)

    dir_parser = subparsers.add_parser(
        "ingest-dir",
        help="Ingest files from a directory",
        description="Ingest matching UTF-8 files from a directory in sorted order.",
    )
    dir_parser.add_argument("path", help="Directory to ingest")
    dir_parser.add_argument("--glob", default="*.md", help="Glob pattern, default: *.md")
    dir_parser.add_argument("--source-type", default="text")
    dir_parser.add_argument("--session-id", default=None)
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


def _read_file(parser: argparse.ArgumentParser, path: Path) -> str:
    if not path.exists():
        parser.error(f"path does not exist: {path}")
    if not path.is_file():
        parser.error(f"path is not a file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        parser.error(str(exc))


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


async def _ingest_text(
    text: str,
    title: str,
    source_type: str,
    session_id: str | None = None,
    turn_id: str | None = None,
):
    pipeline, encoder, neo4j_store, qdrant_store = _get_runtime()
    try:
        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()
        return await pipeline.ingest(
            text,
            title,
            source_type,
            session_id=session_id,
            turn_id=turn_id,
        )
    finally:
        await close_runtime(neo4j_store, qdrant_store)


async def handle_ingest(args: argparse.Namespace) -> int:
    parser = argparse.ArgumentParser(prog="landscape ingest")
    _validate_provenance(parser, args.session_id, args.turn_id)
    path = Path(args.path)
    text = _read_file(parser, path)
    result = await _ingest_text(
        text=text,
        title=args.title or path.stem,
        source_type=args.source_type,
        session_id=args.session_id,
        turn_id=args.turn_id,
    )
    print(_format_summary(result))
    return 0


async def handle_ingest_dir(args: argparse.Namespace) -> int:
    parser = argparse.ArgumentParser(prog="landscape ingest-dir")
    root = Path(args.path)
    if not root.exists():
        parser.error(f"path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"path is not a directory: {root}")
    if args.session_id is not None and not args.session_id.strip():
        parser.error("session-id must be non-empty")

    paths = sorted(p for p in root.glob(args.glob) if p.is_file())
    if not paths:
        print(f"No files matched {args.glob!r} under {root}")
        return 0

    failures = 0
    for index, path in enumerate(paths, start=1):
        try:
            text = path.read_text(encoding="utf-8")
            turn_id = f"t{index}" if args.session_id is not None else None
            result = await _ingest_text(
                text=text,
                title=path.stem,
                source_type=args.source_type,
                session_id=args.session_id,
                turn_id=turn_id,
            )
            print(f"[{index}/{len(paths)}] {path.name}")
            print(_format_summary(result))
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(paths)}] {path.name}: ERROR {exc}")
            if args.stop_on_error:
                return 1
    return 1 if failures else 0
