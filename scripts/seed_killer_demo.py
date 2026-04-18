"""Seed Landscape with the Helios Robotics killer-demo corpus.

Standalone script (no pytest) intended for demo setup — e.g. before pointing
OpenAI Codex or another MCP client at Landscape as a memory backend. Wipes
Neo4j + Qdrant first, then ingests the 7-doc corpus sequentially. Each doc is
ingested under a deterministic session_id/turn_id so the resulting graph has
Conversation/Turn provenance that's queryable via conversation_history.

Run:
    uv run python scripts/seed_killer_demo.py

After it completes you'll have:
  - ~40 entities, ~60 relations (varies with LLM extraction)
  - A single :Conversation {id: "seed:killer-demo"} with 7 :Turn nodes
  - All entities linked to their source Turn via :MENTIONED_IN
  - All docs linked via :INGESTED_IN

Takes ~40s on a laptop GPU with Llama 3.1 8B via Ollama.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "killer_demo_corpus"
SESSION_ID = "seed-killer-demo"
TITLE_PREFIX = "killer-demo:"


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
        names = {c.name for c in existing.collections}
        for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
            if coll in names:
                await qclient.delete_collection(coll)
    finally:
        await qclient.close()


async def main() -> None:
    from landscape.embeddings import encoder
    from landscape.pipeline import ingest
    from landscape.storage import neo4j_store, qdrant_store

    print(f"Seeding Landscape from {CORPUS_DIR}")
    print(f"Session id: {SESSION_ID}")
    print()

    print("Step 1/3  Wiping Neo4j + Qdrant state...")
    await wipe_state()

    print("Step 2/3  Initializing encoder + Qdrant collections...")
    encoder.load_model()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

    docs = sorted(CORPUS_DIR.glob("*.md"))
    assert docs, f"No .md files found in {CORPUS_DIR}"

    print(f"Step 3/3  Ingesting {len(docs)} docs under session {SESSION_ID!r}...")
    for idx, path in enumerate(docs, start=1):
        turn_id = f"t{idx}"
        title = f"{TITLE_PREFIX}{path.stem}"
        text = path.read_text()
        result = await ingest(
            text, title, session_id=SESSION_ID, turn_id=turn_id
        )
        print(
            f"  [{idx}/{len(docs)}] {path.name}  "
            f"entities={result.entities_created} "
            f"relations={result.relations_created} "
            f"superseded={result.relations_superseded}"
        )

    # Final summary via a quick count
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (e:Entity) WITH count(e) AS ents
            MATCH ()-[r:RELATES_TO]->() WHERE r.valid_until IS NULL
            RETURN ents, count(r) AS live_rels
            """
        )
        record = await result.single()
    print()
    print(f"Done. entities={record['ents']} live_relations={record['live_rels']}")
    print(f"Ready. Point an MCP client at `uv run landscape-mcp` to query.")

    await neo4j_store.close_driver()
    await qdrant_store.close_client()


if __name__ == "__main__":
    asyncio.run(main())
