"""Smoke tests for the ChromaDB baseline benchmark.

These tests are deliberately minimal and fast (< 30 s). They verify:
  1. chromadb is importable.
  2. embed_chunks() returns 768-dim vectors (nomic-embed-text-v1.5).
  3. A 3-doc in-memory ChromaDB collection ranks the best-match doc first.
  4. The bench script's build_collection + run_benchmark helpers run end-to-end
     on a 2-doc subset of the corpus without crashing.

The full corpus benchmark (bench_chromadb.py) is NOT run here — it takes
several minutes and is invoked manually / in CI as a separate step.
"""

# chromadb bundles opentelemetry-exporter-otlp-proto-grpc which uses the old
# protobuf descriptor API (pre-v4).  Set the pure-Python implementation before
# the first chromadb import to avoid the "Descriptors cannot be created
# directly" TypeError on protobuf >= 4.x.
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import importlib.util
import pathlib
import sys
import tempfile

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external,
    pytest.mark.xfail(
        importlib.util.find_spec("chromadb") is None,
        reason="chromadb is an optional bench dependency; install with `uv run --extra bench ...`",
        strict=False,
    ),
]

# Add scripts/ to sys.path so we can import bench_chromadb directly
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# 1. Import guard
# ---------------------------------------------------------------------------


def test_chromadb_importable():
    import chromadb  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Embedding dim
# ---------------------------------------------------------------------------


def test_embed_chunks_dim():
    from landscape.embeddings import encoder

    encoder.load_model()
    from bench_chromadb import embed_chunks  # noqa: E402

    vecs = embed_chunks(["Hello, world!"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 768, f"Expected 768-dim, got {len(vecs[0])}"


# ---------------------------------------------------------------------------
# 3. In-memory collection ranking smoke test
# ---------------------------------------------------------------------------


def test_in_memory_collection_ranking():
    """Insert 3 docs; the most specific match should rank first."""
    import chromadb
    from landscape.embeddings import encoder

    encoder.load_model()
    from bench_chromadb import embed_chunks, embed_query

    docs = [
        "Project Aurora uses PostgreSQL as its primary database.",
        "The weather in Paris is usually mild in spring.",
        "Kafka is used by Project Beacon for streaming pipelines.",
    ]
    ids = ["doc_aurora", "doc_weather", "doc_beacon"]

    client = chromadb.Client()
    collection = client.create_collection(
        name="smoke_test",
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = embed_chunks(docs)
    collection.add(embeddings=embeddings, documents=docs, ids=ids)

    qvec = embed_query("What database does Aurora use?")
    results = collection.query(query_embeddings=[qvec], n_results=3, include=["documents"])
    top_doc = results["documents"][0][0]

    assert "PostgreSQL" in top_doc or "Aurora" in top_doc, (
        f"Expected Aurora/PostgreSQL doc to rank first, got: {top_doc!r}"
    )


# ---------------------------------------------------------------------------
# 4. End-to-end harness on 2-doc subset
# ---------------------------------------------------------------------------


def test_bench_harness_two_doc_subset(tmp_path: pathlib.Path):
    """Run build_collection + run_benchmark on a 2-doc subset; just no crash."""
    from landscape.embeddings import encoder

    encoder.load_model()
    from bench_chromadb import QUERIES, build_collection, run_benchmark

    # Write a mini corpus (first two fixture docs) to tmp_path
    fixture_dir = (
        pathlib.Path(__file__).resolve().parent
        / "fixtures"
        / "killer_demo_corpus"
    )
    for mdfile in sorted(fixture_dir.glob("*.md"))[:2]:
        (tmp_path / mdfile.name).write_text(mdfile.read_text())

    with tempfile.TemporaryDirectory(prefix="landscape-test-chromadb-") as chroma_dir:
        collection = build_collection(chroma_dir, corpus_dir=tmp_path)
        rows = run_benchmark(collection, n_results=5)

    # Basic shape assertions — not correctness (only 2 docs ingested)
    assert len(rows) == len(QUERIES)
    for row in rows:
        assert "p1" in row
        assert "p5" in row
        assert "mrr" in row
        assert 0.0 <= row["p1"] <= 1.0
        assert 0.0 <= row["p5"] <= 1.0
        assert 0.0 <= row["mrr"] <= 1.0
