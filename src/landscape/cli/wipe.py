from __future__ import annotations

import argparse

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from landscape.config import settings
from landscape.storage import qdrant_store


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("wipe", help="Clear local Landscape state")
    parser.add_argument("--confirm", action="store_true", help="Required to wipe state")
    parser.set_defaults(func=handle_wipe)


async def wipe_state() -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()

    qclient = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        existing = await qclient.get_collections()
        names = {c.name for c in existing.collections}
        for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
            if coll in names:
                await qclient.delete_collection(coll)
    finally:
        await qclient.close()


async def handle_wipe(args: argparse.Namespace) -> int:
    if not args.confirm:
        print("Refusing to wipe without --confirm")
        return 2

    await wipe_state()
    print("Wiped Neo4j graph and Qdrant Landscape collections.")
    return 0
