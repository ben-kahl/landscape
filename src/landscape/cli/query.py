from __future__ import annotations

import argparse

from landscape.cli.runtime import close_runtime
from landscape.observability import ensure_query_cli_logging
from landscape.retrieval.render import render_fact, render_path


def _parse_as_of(value: str) -> str:
    from datetime import UTC, datetime
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--as-of must be an ISO-8601 timestamp; got {value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _get_runtime():
    from landscape.embeddings import encoder
    from landscape.retrieval.query import retrieve
    from landscape.storage import neo4j_store, qdrant_store

    return encoder, retrieve, neo4j_store, qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "query",
        help="Run hybrid graph and vector retrieval",
        description="Run hybrid graph + vector retrieval against local Landscape memory.",
    )
    parser.add_argument("text", help="Natural-language query")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-reinforce", action="store_true")
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=None,
        help="ISO-8601 timestamp; return facts true in the world at this moment.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(func=handle_query)


async def handle_query(args: argparse.Namespace) -> int:
    encoder, retrieve, neo4j_store, qdrant_store = _get_runtime()
    try:
        ensure_query_cli_logging()
        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()
        result = await retrieve(
            args.text,
            hops=args.hops,
            limit=args.limit,
            reinforce=not args.no_reinforce,
            debug=args.debug,
            include_historical=args.include_historical,
            as_of=args.as_of,
        )
        if not result.results:
            print("No results.")
            return 0
        seen_fact_ids: set[str] = set()
        for index, item in enumerate(result.results, start=1):
            neg_flag = (
                " [NEGATED]"
                if any(edge.negated for edge in item.path.edges)
                else ""
            )
            print(
                f"{index}. {item.name} [{item.type}]{neg_flag} "
                f"score={item.score:.4f} distance={item.distance} via={item.retrieval_mode}"
            )
            print(f"   path: {render_path(item.path)}")
            for fact in item.memory_facts:
                fid = str(fact.get("memory_fact_id") or "")
                if fid in seen_fact_ids:
                    continue
                seen_fact_ids.add(fid)
                conf = float(fact.get("confidence_agg") or 0.0)
                print(f"     fact: {render_fact(fact)}  (c={conf:.2f})")
        if result.chunks:
            print()
            print("Relevant chunks:")
            for chunk in result.chunks:
                preview = chunk.text[:120].replace("\n", " ")
                print(f"  [{chunk.source_doc}] {preview}")
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)
