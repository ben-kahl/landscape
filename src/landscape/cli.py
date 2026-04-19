from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from landscape import pipeline
from landscape.embeddings import encoder
from landscape.storage import neo4j_store, qdrant_store


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="landscape")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a UTF-8 text file")
    ingest_parser.add_argument("path", help="Path to the input file")
    ingest_parser.add_argument("--title", help="Document title", default=None)
    ingest_parser.add_argument(
        "--source-type",
        help="Source type recorded with the document",
        default="text",
    )
    ingest_parser.add_argument("--session-id", default=None)
    ingest_parser.add_argument("--turn-id", default=None)

    return parser


def _validate_ingest_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, str]:
    path = Path(args.path)
    if not path.exists():
        parser.error(f"path does not exist: {path}")
    if not path.is_file():
        parser.error(f"path is not a file: {path}")
    session_id = args.session_id
    turn_id = args.turn_id
    if (session_id is None) != (turn_id is None):
        parser.error("session-id and turn-id must be provided together")
    if session_id is not None and (not session_id.strip() or not turn_id.strip()):
        parser.error("session-id and turn-id must be non-empty")

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        parser.error(str(exc))

    title = args.title or path.stem
    return text, title, session_id, turn_id


async def _run_ingest(
    text: str,
    title: str,
    source_type: str,
    session_id: str | None,
    turn_id: str | None,
):
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
        await _close_runtime()


async def _close_runtime() -> None:
    try:
        await neo4j_store.close_driver()
    except Exception as exc:
        print(f"Warning: neo4j close failed: {exc}", file=sys.stderr)

    try:
        await qdrant_store.close_client()
    except Exception as exc:
        print(f"Warning: qdrant close failed: {exc}", file=sys.stderr)


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "ingest":
        parser.error(f"unknown command: {args.command}")

    text, title, session_id, turn_id = _validate_ingest_args(parser, args)

    try:
        result = asyncio.run(
            _run_ingest(
                text=text,
                title=title,
                source_type=args.source_type,
                session_id=session_id,
                turn_id=turn_id,
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(_format_summary(result))
    return 0
