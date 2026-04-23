from __future__ import annotations

import argparse
from importlib import resources

from landscape.cli.runtime import close_runtime
from landscape.cli.wipe import wipe_state
from landscape.observability import ensure_cli_logging


def _get_runtime():
    from landscape.embeddings import encoder
    from landscape.pipeline import ingest
    from landscape.storage import neo4j_store, qdrant_store

    return encoder, ingest, neo4j_store, qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("seed", help="Load demo datasets")
    seed_subparsers = parser.add_subparsers(dest="seed_name", required=True)
    killer = seed_subparsers.add_parser("killer-demo", help="Seed the killer-demo corpus")
    killer.add_argument("--confirm", action="store_true", help="Required because seed wipes first")
    killer.set_defaults(func=handle_killer_demo)


async def handle_killer_demo(args: argparse.Namespace) -> int:
    if not args.confirm:
        print("Refusing to seed killer-demo without --confirm because seeding wipes first")
        return 2

    ensure_cli_logging()

    corpus_dir = resources.files("landscape.demo_corpora.killer_demo_corpus")
    session_id = "seed-killer-demo"
    docs = sorted(path for path in corpus_dir.iterdir() if path.name.endswith(".md"))
    if not docs:
        raise RuntimeError(f"No .md files found in {corpus_dir}")

    print(f"Seeding Landscape from {corpus_dir}")
    print(f"Session id: {session_id}")
    print("Step 1/3  Wiping Neo4j + Qdrant state...")
    await wipe_state()

    encoder, ingest, neo4j_store, qdrant_store = _get_runtime()

    print("Step 2/3  Initializing encoder + Qdrant collections...")
    encoder.load_model()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

    print(f"Step 3/3  Ingesting {len(docs)} docs...")
    try:
        for index, path in enumerate(docs, start=1):
            result = await ingest(
                path.read_text(encoding="utf-8"),
                f"killer-demo:{path.stem}",
                session_id=session_id,
                turn_id=f"t{index}",
            )
            print(
                f"  [{index}/{len(docs)}] {path.name} "
                f"entities={result.entities_created} "
                f"relations={result.relations_created} "
                f"superseded={result.relations_superseded}"
            )
    finally:
        await close_runtime(neo4j_store, qdrant_store)
    return 0
