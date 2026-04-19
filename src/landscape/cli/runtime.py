from __future__ import annotations

import os
import sys


def _apply_cli_process_defaults() -> None:
    """Prefer host-reachable defaults for local console commands."""
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


async def close_runtime(neo4j_store, qdrant_store) -> None:
    try:
        await neo4j_store.close_driver()
    except Exception as exc:
        print(f"Warning: neo4j close failed: {exc}", file=sys.stderr)

    try:
        await qdrant_store.close_client()
    except Exception as exc:
        print(f"Warning: qdrant close failed: {exc}", file=sys.stderr)
