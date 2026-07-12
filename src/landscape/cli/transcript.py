from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from landscape.cli.runtime import close_runtime
from landscape.observability import ensure_cli_logging


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ingest-transcript",
        help="Ingest a completed conversation transcript (end-of-session capture)",
        description=(
            "Read a Claude Code transcript JSONL, select salient turns, and ingest "
            "them into memory. Pass a path argument for manual/backfill use, or "
            "omit it to read the hook event JSON (with transcript_path) from stdin."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Transcript JSONL path. If omitted, hook JSON is read from stdin.",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--client",
        default="claude-code",
        help="Transcript client format. Currently only 'claude-code' is supported.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(func=handle_ingest_transcript)


def _resolve_input(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.path:
        return Path(args.path).expanduser(), args.session_id
    raw = sys.stdin.read()
    payload = json.loads(raw) if raw.strip() else {}
    transcript_path = payload.get("transcript_path") or payload.get("transcriptPath")
    if not transcript_path:
        raise ValueError(
            "no transcript path: pass a path argument or pipe hook JSON "
            "containing transcript_path on stdin"
        )
    session_id = args.session_id or payload.get("session_id") or payload.get("sessionID")
    return Path(transcript_path).expanduser(), session_id


async def handle_ingest_transcript(args: argparse.Namespace) -> int:
    from landscape.conversation_ingestion import ingest_conversation_window
    from landscape.embeddings import encoder
    from landscape.extraction.salience import select_salient
    from landscape.ingestion.transcript import read_transcript
    from landscape.storage import neo4j_store, qdrant_store

    ensure_cli_logging()
    path, session_id = _resolve_input(args)
    if not path.exists():
        print(f"Error: transcript not found: {path}", file=sys.stderr)
        return 1

    turns = read_transcript(path, session_id=session_id, client=args.client)
    if not turns:
        print("No eligible turns in transcript; nothing to ingest.")
        return 0

    resolved_session = session_id or turns[0].session_id
    if not resolved_session:
        print(
            "Error: no session id (pass --session-id or ensure the transcript "
            "rows carry sessionId)",
            file=sys.stderr,
        )
        return 1

    try:
        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()
        await neo4j_store.backfill_ingest_completed_marker()
        salient = select_salient(turns)
        if not salient:
            print("No salient turns; nothing to ingest.")
            return 0
        await ingest_conversation_window(resolved_session, salient, debug=args.debug)
    finally:
        await close_runtime(neo4j_store, qdrant_store)

    print(f"Ingested {len(salient)} salient turn(s) from {path.name}.")
    return 0
