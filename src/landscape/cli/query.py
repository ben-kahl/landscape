from __future__ import annotations

import argparse

from landscape.cli.runtime import close_runtime


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
    parser.set_defaults(func=handle_query)


async def handle_query(args: argparse.Namespace) -> int:
    encoder, retrieve, neo4j_store, qdrant_store = _get_runtime()
    try:
        encoder.load_model()
        await qdrant_store.init_collection()
        await qdrant_store.init_chunks_collection()
        result = await retrieve(
            args.text,
            hops=args.hops,
            limit=args.limit,
            reinforce=not args.no_reinforce,
        )
        if not result.results:
            print("No results.")
            return 0
        for index, item in enumerate(result.results, start=1):
            path = " -> ".join(item.path_edge_types) if item.path_edge_types else "(seed)"
            print(
                f"{index}. {item.name} [{item.type}] "
                f"score={item.score:.4f} distance={item.distance} path={path}"
            )
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)
