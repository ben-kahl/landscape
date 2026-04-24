#!/usr/bin/env python3
"""ChromaDB baseline benchmark — Helios Robotics killer-demo corpus.

Ingests the same 7 markdown files used by bench_retrieval.py into a fresh
ChromaDB PersistentClient collection and runs the SAME query set, producing
precision@k and MRR numbers for direct comparison with Landscape hybrid retrieval.

Apples-to-apples design:
  - Same embedding model (nomic-ai/nomic-embed-text-v1.5, 768-dim)
  - Same chunker (landscape.extraction.chunker)
  - Same query set and expected-answer definitions
  - Same metric functions (P@1, P@5, MRR)

Usage:
    uv run python scripts/bench_chromadb.py
    uv run python scripts/bench_chromadb.py --results-json /tmp/chroma_results.json
"""

import argparse
import json

# Bootstrap env before landscape imports so Settings picks up defaults.
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is required because chromadb
# bundles opentelemetry-exporter-otlp-proto-grpc which uses pre-v4 protobuf
# descriptors that are incompatible with protobuf >= 4.x (present in this env).
# The pure-Python implementation is slower but correct.
import os
import pathlib
import sys
import tempfile
import uuid

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# Force CPU for embedding to avoid VRAM contention with other processes.
# bench_retrieval.py does the same; keeps both benchmarks on equal footing.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

try:
    import chromadb
except ImportError as e:
    print(
        "chromadb is not installed. Install with: uv pip install -e .[bench]",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from landscape.embeddings import encoder
from landscape.extraction.chunker import chunk_text

CORPUS_DIR = (
    pathlib.Path(__file__).resolve().parent / "../tests/fixtures/killer_demo_corpus"
)
COLLECTION_NAME = "killer_demo"

# ---------------------------------------------------------------------------
# Query set — identical to bench_retrieval.py
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "text": "Who leads the Vision Team?",
        "expected": ["Diego"],
        "hops": 1,
    },
    {
        "text": "What does Sentinel use for computer vision?",
        "expected": ["PyTorch"],
        "hops": 1,
    },
    {
        "text": "Who approved the database for Project Aurora?",
        "expected": ["Maya"],
        "hops": 2,
    },
    {
        "text": "What team does the person who approved Aurora's database lead?",
        "expected": ["Platform Team", "Platform"],
        "hops": 3,
    },
    {
        "text": "What database does Project Aurora use?",
        "expected": ["PostgreSQL"],
        "hops": 1,
    },
    {
        "text": "Who created Project Aurora?",
        "expected": ["Priya"],
        "hops": 1,
    },
    {
        "text": "What technology does Project Beacon use?",
        "expected": ["Kafka"],
        "hops": 1,
    },
]

# ---------------------------------------------------------------------------
# Metric helpers (same semantics as bench_retrieval.py)
# ---------------------------------------------------------------------------


def _chunk_hits_expected(chunk_text_: str, expected: list[str]) -> bool:
    """Return True if the chunk contains any expected substring (case-insensitive)."""
    lower = chunk_text_.lower()
    return any(sub.lower() in lower for sub in expected)


def precision_at_k(ranked_docs: list[str], expected: list[str], k: int) -> float:
    top_k = ranked_docs[:k]
    hits = sum(1 for doc in top_k if _chunk_hits_expected(doc, expected))
    return hits / k if k > 0 else 0.0


def mrr(ranked_docs: list[str], expected: list[str]) -> float:
    for i, doc in enumerate(ranked_docs, start=1):
        if _chunk_hits_expected(doc, expected):
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed a list of chunk texts using Landscape's encoder."""
    return encoder.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return encoder.embed_query(text)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def build_collection(
    chroma_path: str,
    corpus_dir: pathlib.Path = CORPUS_DIR,
) -> chromadb.Collection:
    """Load corpus, chunk, embed, and insert into a fresh ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete any leftover collection (e.g. from a previous aborted run at same path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # We supply embeddings manually — no built-in embedding function
        metadata={"hnsw:space": "cosine"},
    )

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for path in sorted(corpus_dir.glob("*.md")):
        doc_text = path.read_text()
        chunks = chunk_text(doc_text)
        for chunk in chunks:
            chunk_id = f"{path.stem}_chunk_{chunk.index}_{uuid.uuid4().hex[:8]}"
            all_texts.append(chunk.text)
            all_ids.append(chunk_id)
            all_metadatas.append({"source": path.stem, "chunk_index": chunk.index})

    print(f"  Embedding {len(all_texts)} chunks...", flush=True)
    all_embeddings = embed_chunks(all_texts)

    # ChromaDB.add() wants list[list[float]] — already that type
    collection.add(
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
        ids=all_ids,
    )
    print(f"  Inserted {len(all_texts)} chunks into ChromaDB.", flush=True)
    return collection


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    collection: chromadb.Collection,
    n_results: int = 10,
) -> list[dict]:
    """Run all queries; return per-query result dicts."""
    rows = []
    for q in QUERIES:
        qvec = embed_query(q["text"])
        results = collection.query(
            query_embeddings=[qvec],
            n_results=n_results,
            include=["documents"],
        )
        ranked_docs: list[str] = results["documents"][0]  # list for single query

        p1 = precision_at_k(ranked_docs, q["expected"], k=1)
        p5 = precision_at_k(ranked_docs, q["expected"], k=5)
        rr = mrr(ranked_docs, q["expected"])

        rows.append(
            {
                "query": q["text"],
                "hops": q["hops"],
                "p1": p1,
                "p5": p5,
                "mrr": rr,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def _band_stats(rows: list[dict], hops: int) -> dict:
    band = [r for r in rows if r["hops"] == hops]
    if not band:
        return {"p1": 0.0, "p5": 0.0, "mrr": 0.0, "n": 0}
    n = len(band)
    return {
        "p1": sum(r["p1"] for r in band) / n,
        "p5": sum(r["p5"] for r in band) / n,
        "mrr": sum(r["mrr"] for r in band) / n,
        "n": n,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print("## ChromaDB Baseline Results")
    print()
    print("| Question band | ChromaDB P@1 | ChromaDB P@5 | ChromaDB MRR |")
    print("|---|---|---|---|")
    for hops_label, hop_n in [("1-hop", 1), ("2-hop", 2), ("3-hop", 3)]:
        s = _band_stats(rows, hop_n)
        print(
            f"| {hops_label} (n={s['n']}) "
            f"| {s['p1']:.0%} "
            f"| {s['p5']:.0%} "
            f"| {s['mrr']:.3f} |"
        )
    print()

    # Overall
    all_n = len(rows)
    p1_all = sum(r["p1"] for r in rows) / all_n
    p5_all = sum(r["p5"] for r in rows) / all_n
    mrr_all = sum(r["mrr"] for r in rows) / all_n
    print(f"| **Overall** (n={all_n}) | **{p1_all:.0%}** | **{p5_all:.0%}** | **{mrr_all:.3f}** |")
    print()

    print("Per-query breakdown:")
    print(f"{'Query':<62} {'Hops':>4} {'P@1':>5} {'P@5':>5} {'MRR':>6}")
    print("-" * 85)
    for r in rows:
        print(
            f"{r['query'][:60]:<62} {r['hops']:>4} "
            f"{r['p1']:>5.0%} {r['p5']:>5.0%} {r['mrr']:>6.3f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ChromaDB baseline benchmark")
    parser.add_argument(
        "--results-json",
        metavar="PATH",
        default=None,
        help="Write results as JSON to this path (default: next to this script)",
    )
    parser.add_argument(
        "--chroma-path",
        metavar="DIR",
        default=None,
        help="Directory for ChromaDB persistent storage (default: auto tempdir)",
    )
    args = parser.parse_args()

    print("Loading embedding model...", flush=True)
    encoder.load_model()

    with tempfile.TemporaryDirectory(prefix="landscape-bench-chromadb-") as tmp:
        chroma_path = args.chroma_path if args.chroma_path else tmp

        print(f"Building ChromaDB collection at {chroma_path}...", flush=True)
        collection = build_collection(chroma_path)

        print("Running queries...", flush=True)
        rows = run_benchmark(collection)

    print_table(rows)

    json_path = args.results_json or (
        pathlib.Path(__file__).resolve().parent / "bench_chromadb_results.json"
    )
    pathlib.Path(json_path).write_text(json.dumps(rows, indent=2))
    print(f"Results written to {json_path}")


if __name__ == "__main__":
    main()
