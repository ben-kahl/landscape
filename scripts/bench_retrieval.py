#!/usr/bin/env python3
"""Benchmark: hybrid vs vector-only vs graph-only retrieval on the
Helios Robotics killer-demo corpus.

Reports precision@k and MRR for each mode on a hand-labeled set of
1-, 2-, and 3-hop queries where the expected answers are known.

Usage:
    # Requires the Docker stack running (Neo4j, Qdrant, Ollama)
    uv run python scripts/bench_retrieval.py

    # Control the ingest (skip if data is already loaded):
    uv run python scripts/bench_retrieval.py --skip-ingest
"""
import argparse
import asyncio

# -- Bootstrap env before any landscape imports. --------------------------
import os
import pathlib
import time
from dataclasses import dataclass

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from landscape.embeddings import encoder  # noqa: E402
from landscape.retrieval.query import retrieve  # noqa: E402
from landscape.storage import neo4j_store, qdrant_store  # noqa: E402

CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "../tests/fixtures/killer_demo_corpus"
TITLE_PREFIX = "bench:"


@dataclass
class Query:
    text: str
    expected: list[str]  # substrings — any result name containing one counts as a hit
    hops_needed: int  # for labeling only


QUERIES = [
    Query("Who leads the Vision Team?", ["Diego"], 1),
    Query("What does Sentinel use for computer vision?", ["PyTorch"], 1),
    Query("Who approved the database for Project Aurora?", ["Maya"], 2),
    Query("What team does the person who approved Aurora's database lead?", 
          ["Platform Team", "Platform"], 3),
    Query("What database does Project Aurora use?", ["PostgreSQL"], 1),
    Query("Who created Project Aurora?", ["Priya"], 1),
    Query("What technology does Project Beacon use?", ["Kafka"], 1),
]


def _hit(names: set[str], expected: list[str]) -> bool:
    return any(
        any(sub.lower() in n.lower() for sub in expected)
        for n in names
    )


def _reciprocal_rank(ranked_names: list[str], expected: list[str]) -> float:
    for i, name in enumerate(ranked_names, start=1):
        if any(sub.lower() in name.lower() for sub in expected):
            return 1.0 / i
    return 0.0


# -- Retrieval modes -------------------------------------------------------

async def hybrid_retrieve(query: str, limit: int) -> list[dict]:
    result = await retrieve(query, hops=3, limit=limit, reinforce=False)
    return [{"name": e.name, "score": e.score} for e in result.results]


async def vector_only_retrieve(query: str, limit: int) -> list[dict]:
    """Qdrant entity search + chunk→entity walk, skip graph BFS."""
    qvec = encoder.embed_query(query)

    entity_hits = await qdrant_store.search_entities_any_type(qvec, limit=limit)
    chunk_hits = await qdrant_store.search_chunks(qvec, limit=limit)

    seen: dict[str, float] = {}
    for h in entity_hits:
        p = h.payload or {}
        nid = p.get("neo4j_node_id")
        if nid:
            seen[nid] = max(seen.get(nid, 0.0), float(h.score))

    chunk_ids = [
        h.payload["chunk_neo4j_id"]
        for h in chunk_hits
        if h.payload and h.payload.get("chunk_neo4j_id")
    ]
    chunk_ents = await neo4j_store.get_entities_from_chunks(chunk_ids)
    for ent in chunk_ents:
        seen.setdefault(ent["eid"], 0.0)

    if not seen:
        return []

    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE elementId(e) IN $ids AND e.canonical = true "
            "RETURN elementId(e) AS eid, e.name AS name",
            ids=list(seen.keys()),
        )
        rows = [dict(r) async for r in result]

    out = [{"name": row["name"], "score": seen.get(row["eid"], 0.0)} for row in rows]
    return sorted(out, key=lambda x: x["score"], reverse=True)[:limit]


async def graph_only_retrieve(query: str, limit: int) -> list[dict]:
    """Seed by exact entity-name search in Neo4j, then BFS-expand."""
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) "
            "AND e.canonical = true "
            "RETURN elementId(e) AS eid, e.name AS name LIMIT 5",
            q=query.split()[-1],  # crude last-word heuristic
        )
        seeds = [dict(r) async for r in result]

    if not seeds:
        return []

    seed_ids = [s["eid"] for s in seeds]
    expansions = await neo4j_store.bfs_expand(seed_ids, max_hops=3)

    seen: dict[str, str] = {s["eid"]: s["name"] for s in seeds}
    for row in expansions:
        if row["target_id"] not in seen:
            seen[row["target_id"]] = row["target_name"]

    return [{"name": name, "score": 0.0} for name in seen.values()][:limit]


# -- Ingest -----------------------------------------------------------------

async def ingest_corpus():
    from landscape.pipeline import ingest

    print("Ingesting killer-demo corpus...", flush=True)
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await qdrant_store.close_client()
    client = qdrant_store.get_client()
    existing = await client.get_collections()
    names = {c.name for c in existing.collections}
    for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
        if coll in names:
            await client.delete_collection(coll)
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

    for path in sorted(CORPUS_DIR.glob("*.md")):
        title = f"{TITLE_PREFIX}{path.stem}"
        print(f"  {title}...", flush=True)
        await ingest(path.read_text(), title)
    print("Ingest complete.\n", flush=True)


# -- Main -------------------------------------------------------------------

async def run(skip_ingest: bool):
    encoder.load_model()
    if not skip_ingest:
        await ingest_corpus()

    modes = {
        "hybrid": hybrid_retrieve,
        "vector-only": vector_only_retrieve,
        "graph-only": graph_only_retrieve,
    }

    limit = 15
    results: dict[str, list[dict]] = {m: [] for m in modes}

    for mode_name, mode_fn in modes.items():
        print(f"Running {mode_name}...", flush=True)
        for q in QUERIES:
            t0 = time.perf_counter()
            items = await mode_fn(q.text, limit)
            elapsed = time.perf_counter() - t0
            names = {r["name"] for r in items}
            ranked = [r["name"] for r in items]
            hit = _hit(names, q.expected)
            rr = _reciprocal_rank(ranked, q.expected)
            results[mode_name].append({
                "query": q.text[:60],
                "hops": q.hops_needed,
                "hit": hit,
                "rr": rr,
                "elapsed": elapsed,
            })
    print()

    # -- Table output -------------------------------------------------------
    header = f"{'Mode':<14} {'P@k':>5} {'MRR':>6} {'Avg ms':>8}"
    print(header)
    print("-" * len(header))
    for mode_name, rows in results.items():
        pk = sum(1 for r in rows if r["hit"]) / len(rows)
        mrr = sum(r["rr"] for r in rows) / len(rows)
        avg_ms = sum(r["elapsed"] for r in rows) / len(rows) * 1000
        print(f"{mode_name:<14} {pk:>5.1%} {mrr:>6.3f} {avg_ms:>7.0f}ms")

    print()
    print("Per-query breakdown:")
    print(f"{'Query':<62} {'Hops':>4} ", end="")
    for m in modes:
        print(f" {m:<14}", end="")
    print()
    for i, q in enumerate(QUERIES):
        print(f"{q.text[:60]:<62} {q.hops_needed:>4} ", end="")
        for m in modes:
            r = results[m][i]
            mark = "hit" if r["hit"] else "MISS"
            print(f" {mark:<14}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Retrieval benchmark")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.skip_ingest))


if __name__ == "__main__":
    main()
