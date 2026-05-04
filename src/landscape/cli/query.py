from __future__ import annotations

import argparse

from landscape.cli.runtime import close_runtime
from landscape.observability import ensure_query_cli_logging


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
        )
        if not result.results:
            print("No results.")
            return 0
        for index, item in enumerate(result.results, start=1):
            if item.path_edge_types:
                edges = item.path_edge_types
                negated = item.path_edge_negated or [False] * len(edges)
                path_parts: list[str] = []
                for edge, neg in zip(edges, negated):
                    label = f"NOT {edge}" if neg else edge
                    path_parts.append(f"-[{label}]->")
                path_parts.append(f"{item.name} [{item.type}]")
                path_str = f"(seed) " + " ".join(path_parts)
            else:
                path_str = f"(seed) {item.name} [{item.type}]"
            neg_flag = " [NEGATED]" if any(item.path_edge_negated) else ""
            print(
                f"{index}. {item.name} [{item.type}]{neg_flag} "
                f"score={item.score:.4f} distance={item.distance}"
            )
            print(f"   path: {path_str}")
        if result.chunks:
            print()
            print("Relevant chunks:")
            for chunk in result.chunks:
                preview = chunk.text[:120].replace("\n", " ")
                print(f"  [{chunk.source_doc}] {preview}")
        return 0
    finally:
        await close_runtime(neo4j_store, qdrant_store)
