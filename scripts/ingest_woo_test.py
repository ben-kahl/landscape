"""Ingest woo.txt into Landscape's isolated test databases.

Run from the repo root:
    uv run python scripts/ingest_woo_test.py

The script defaults to the test stack, not the live stack:
    Neo4j:  bolt://localhost:17687
    Qdrant: http://localhost:16333
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_NEO4J_URI = "bolt://localhost:7687"
LIVE_QDRANT_URL = "http://localhost:6333"
TEST_NEO4J_URI = "bolt://localhost:17687"
TEST_QDRANT_URL = "http://localhost:16333"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest woo.txt into the isolated Landscape test databases."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=REPO_ROOT / "woo.txt",
        help="Text file to ingest. Defaults to repo-root woo.txt.",
    )
    parser.add_argument(
        "--title",
        default="woo-test:woo.txt",
        help="Document title to store in Landscape.",
    )
    parser.add_argument("--session-id", default="woo-test")
    parser.add_argument("--turn-id", default="t1")
    parser.add_argument("--source-type", default="text")
    parser.add_argument("--neo4j-uri", default=TEST_NEO4J_URI)
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="landscape-test")
    parser.add_argument("--qdrant-url", default=TEST_QDRANT_URL)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Clear the target test Neo4j and Qdrant stores before ingesting.",
    )
    parser.add_argument(
        "--allow-live-target",
        action="store_true",
        help="Allow targeting the live default ports. Use only when intentional.",
    )
    return parser.parse_args(argv)


def assert_safe_target(args: argparse.Namespace) -> None:
    if args.allow_live_target:
        return
    if args.neo4j_uri == LIVE_NEO4J_URI or args.qdrant_url == LIVE_QDRANT_URL:
        raise SystemExit(
            "Refusing to target the live Landscape stack. Use the test stack "
            "defaults or pass --allow-live-target if this is intentional."
        )


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["NEO4J_URI"] = args.neo4j_uri
    os.environ["NEO4J_USER"] = args.neo4j_user
    os.environ["NEO4J_PASSWORD"] = args.neo4j_password
    os.environ["QDRANT_URL"] = args.qdrant_url
    os.environ["OLLAMA_URL"] = args.ollama_url
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


async def wipe_state() -> None:
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient

    from landscape.storage import qdrant_store

    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()

    qclient = AsyncQdrantClient(url=os.environ["QDRANT_URL"])
    try:
        existing = await qclient.get_collections()
        names = {collection.name for collection in existing.collections}
        for collection in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
            if collection in names:
                await qclient.delete_collection(collection)
    finally:
        await qclient.close()


async def summarize() -> dict[str, int]:
    from landscape.storage import neo4j_store

    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            CALL () { MATCH (d:Document) RETURN count(d) AS documents }
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
            CALL () {
              MATCH ()-[r:MEMORY_REL]->()
              WHERE r.valid_until IS NULL
              RETURN count(r) AS live_memory_rels
            }
            CALL () { MATCH ()-[r:MENTIONS_CHUNK]->() RETURN count(r) AS mentions_chunk }
            CALL () {
              MATCH (c:Chunk)
              WHERE size(coalesce(c.mentioned_entity_ids, [])) > 0
              RETURN count(c) AS chunks_with_mentions
            }
            RETURN documents, entities, chunks, live_memory_rels,
                   mentions_chunk, chunks_with_mentions
            """
        )
        record = await result.single()
    return dict(record)


async def ingest_woo(args: argparse.Namespace) -> None:
    from landscape.embeddings import encoder
    from landscape.pipeline import ingest
    from landscape.storage import neo4j_store, qdrant_store

    if not args.path.exists():
        raise SystemExit(f"Input file not found: {args.path}")

    text = args.path.read_text(encoding="utf-8")

    print("Targeting test databases")
    print(f"  Neo4j:  {args.neo4j_uri}")
    print(f"  Qdrant: {args.qdrant_url}")
    print(f"  File:   {args.path}")
    print(f"  Title:  {args.title}")
    print()

    if args.wipe:
        print("Wiping target Neo4j and Qdrant stores...")
        await wipe_state()

    print("Initializing model and stores...")
    encoder.load_model()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()
    await neo4j_store.ensure_memory_graph_schema()

    print("Ingesting woo.txt...")
    result = await ingest(
        text=text,
        title=args.title,
        source_type=args.source_type,
        session_id=args.session_id,
        turn_id=args.turn_id,
    )

    counts = await summarize()
    print()
    print("Ingest result")
    print(f"  already_existed: {result.already_existed}")
    print(f"  chunks_created: {result.chunks_created}")
    print(f"  entities: created={result.entities_created} reinforced={result.entities_reinforced}")
    print(
        "  relations: "
        f"created={result.relations_created} "
        f"reinforced={result.relations_reinforced} "
        f"superseded={result.relations_superseded}"
    )
    print()
    print("Target graph counts")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    await neo4j_store.close_driver()
    await qdrant_store.close_client()


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assert_safe_target(args)
    configure_environment(args)
    await ingest_woo(args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
