#!/usr/bin/env python3
"""A/B benchmark for Landscape retrieval quality and supersession correctness.

Sections:
  A. LongMemEval retrieval quality (hit@1, hit@5, hit@10) — optional
  B. Killer-demo multi-hop assertions
  C. Synthetic supersession scenarios (deterministic, no LLM)

Token accounting uses tiktoken cl100k_base on serialized retrieve() output,
matching what TokenCounterMiddleware counts on the live API.

Usage:
    uv run python scripts/bench_ab.py --branch main
    uv run python scripts/bench_ab.py --branch memory-graph-redesign \\
        --longmemeval-data tests/longmemeval_s_cleaned.json \\
        --n-questions 20
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import sys
import time
from datetime import UTC, datetime

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import httpx  # noqa: E402
import tiktoken  # noqa: E402

from landscape import pipeline  # noqa: E402
from landscape.embeddings import encoder  # noqa: E402
from landscape.middleware.token_counter import get_usage, reset_counters  # noqa: E402
from landscape.retrieval.query import retrieve  # noqa: E402
from landscape.storage import neo4j_store, qdrant_store  # noqa: E402

_ENCODING = tiktoken.get_encoding("cl100k_base")
_CORPUS_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "killer_demo_corpus"
)

QUESTION_TYPE = "single-session-user"


def _count_tokens(obj) -> int:
    text = json.dumps(obj, default=str)
    return len(_ENCODING.encode(text))


def _safe_id(raw: str) -> str:
    return raw.replace(":", "-").replace("/", "-").replace(" ", "_")


async def _wipe_stack() -> None:
    driver = neo4j_store.get_driver()
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    client = qdrant_store.get_client()
    existing = await client.get_collections()
    names = {c.name for c in existing.collections}
    for coll in (qdrant_store.COLLECTION, qdrant_store.CHUNKS_COLLECTION):
        if coll in names:
            await client.delete_collection(coll)
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()


def _format_session(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Section A: LongMemEval
# ---------------------------------------------------------------------------

async def _run_longmemeval(data_path: pathlib.Path, n_questions: int) -> dict:
    print(f"\n=== Section A: LongMemEval ({n_questions} questions) ===")
    with data_path.open() as f:
        all_questions = json.load(f)

    questions = [q for q in all_questions if q.get("question_type") == QUESTION_TYPE]
    questions = questions[:n_questions]

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    ingest_times: list[float] = []
    query_times: list[float] = []
    response_tokens: list[int] = []
    ollama_start = get_usage()["ollama"].copy()

    encoder.load_model()

    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {q['question'][:60]}...")
        await _wipe_stack()

        gold_entity = q.get("answer", "")
        sessions = q.get("haystack_sessions", [])

        t0 = time.perf_counter()
        for sess in sessions:
            text = _format_session(sess.get("turns", []))
            if text.strip():
                sid = _safe_id(sess.get("session_id", f"sess-{i}"))
                await pipeline.ingest(text, title=f"lme:{sid}")
        ingest_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        result = await retrieve(q["question"], hops=2, limit=10, reinforce=False)
        query_times.append(time.perf_counter() - t0)

        names = [e.name.lower() for e in result.results]
        gold_lower = gold_entity.lower()
        token_count = _count_tokens(dataclasses.asdict(result))
        response_tokens.append(token_count)

        if any(gold_lower in n or n in gold_lower for n in names[:1]):
            hit_at_1 += 1
        if any(gold_lower in n or n in gold_lower for n in names[:5]):
            hit_at_5 += 1
        if any(gold_lower in n or n in gold_lower for n in names[:10]):
            hit_at_10 += 1

    n = len(questions)
    ollama_end = get_usage()["ollama"]
    ollama_delta_prompt = ollama_end["total_prompt_tokens"] - ollama_start["total_prompt_tokens"]
    ollama_delta_completion = (
        ollama_end["total_completion_tokens"] - ollama_start["total_completion_tokens"]
    )
    ollama_total = ollama_delta_prompt + ollama_delta_completion

    section = {
        "n_questions": n,
        "hit_at_1": round(hit_at_1 / n, 4) if n else 0.0,
        "hit_at_5": round(hit_at_5 / n, 4) if n else 0.0,
        "hit_at_10": round(hit_at_10 / n, 4) if n else 0.0,
        "avg_ingest_s": round(sum(ingest_times) / n, 3) if n else 0.0,
        "avg_query_s": round(sum(query_times) / n, 3) if n else 0.0,
        "ollama_extraction_tokens": {
            "total": ollama_total,
            "avg": round(ollama_total / n, 1) if n else 0.0,
        },
        "avg_response_tokens": round(sum(response_tokens) / n, 1) if n else 0.0,
    }
    print(
        f"  hit@1={section['hit_at_1']:.2f}  hit@5={section['hit_at_5']:.2f}"
        f"  hit@10={section['hit_at_10']:.2f}"
    )
    return section


# ---------------------------------------------------------------------------
# Section B: Killer-demo multi-hop
# ---------------------------------------------------------------------------

_KILLER_DEMO_QUERIES = [
    {
        "query": "Who leads the Vision Team?",
        "expected": ["diego"],
        "hops": 2,
        "limit": 10,
    },
    {
        "query": "What does Project Sentinel use for computer vision?",
        "expected": ["pytorch"],
        "hops": 2,
        "limit": 10,
    },
    {
        "query": "Who approved the database for Project Aurora?",
        "expected": ["maya"],
        "hops": 3,
        "limit": 15,
    },
    {
        "query": "What team does the person who approved Aurora's database lead?",
        "expected": ["platform"],
        "hops": 3,
        "limit": 20,
    },
]


async def _run_killer_demo() -> dict:
    print("\n=== Section B: Killer-demo multi-hop ===")
    await _wipe_stack()
    encoder.load_model()

    for path in sorted(_CORPUS_DIR.glob("*.md")):
        title = f"killer-demo:{path.stem}"
        await pipeline.ingest(path.read_text(), title=title)

    per_query = []
    response_tokens: list[int] = []

    for item in _KILLER_DEMO_QUERIES:
        result = await retrieve(
            item["query"],
            hops=item["hops"],
            limit=item["limit"],
            reinforce=False,
        )
        names = [e.name.lower() for e in result.results]
        passed = any(
            any(exp in n or n in exp for n in names)
            for exp in item["expected"]
        )
        token_count = _count_tokens(dataclasses.asdict(result))
        response_tokens.append(token_count)
        per_query.append({
            "query": item["query"],
            "passed": passed,
            "response_tokens": token_count,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {item['query'][:60]}")

    passed_count = sum(1 for pq in per_query if pq["passed"])
    total = len(per_query)
    section = {
        "assertions_passed": passed_count,
        "assertions_total": total,
        "pass_rate": round(passed_count / total, 4) if total else 0.0,
        "avg_response_tokens": (
            round(sum(response_tokens) / total, 1) if total else 0.0
        ),
        "per_query": per_query,
    }
    print(f"  pass_rate={section['pass_rate']:.2f} ({passed_count}/{total})")
    return section


# ---------------------------------------------------------------------------
# Section C: Supersession scenarios
# ---------------------------------------------------------------------------

async def _run_supersession() -> dict:
    print("\n=== Section C: Supersession scenarios ===")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "supersession_scenarios",
        pathlib.Path(__file__).resolve().parent / "fixtures" / "supersession_scenarios.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_all_scenarios = mod.run_all_scenarios

    results = await run_all_scenarios()
    per_scenario = [
        {"id": r.id, "name": r.name, "passed": r.passed, "error": r.error}
        for r in results
    ]
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = f" — {r.error}" if r.error else ""
        print(f"  [{status}] {r.name}{detail}")

    section = {
        "scenarios_passed": passed_count,
        "scenarios_total": total,
        "pass_rate": round(passed_count / total, 4) if total else 0.0,
        "per_scenario": per_scenario,
    }
    print(f"  pass_rate={section['pass_rate']:.2f} ({passed_count}/{total})")
    return section


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> None:
    print(f"Landscape A/B bench — branch: {args.branch}")

    # Verify stack is reachable
    try:
        r = httpx.get(f"{args.api_url}/healthz", timeout=5.0)
        if r.status_code != 200:
            print(f"ERROR: /healthz returned {r.status_code}", file=sys.stderr)
            sys.exit(1)
    except httpx.ConnectError:
        print(
            f"ERROR: Cannot reach {args.api_url}/healthz — is the stack running?",
            file=sys.stderr,
        )
        sys.exit(1)

    reset_counters()

    longmemeval: dict | None = None
    if args.longmemeval_data:
        data_path = pathlib.Path(args.longmemeval_data)
        if not data_path.exists():
            print(f"ERROR: --longmemeval-data {data_path} does not exist", file=sys.stderr)
            sys.exit(1)
        longmemeval = await _run_longmemeval(data_path, args.n_questions)

    killer_demo = await _run_killer_demo()
    supersession = await _run_supersession()

    final_usage = get_usage()
    response_tokens_total = sum(
        ep["total_response_tokens"] for ep in final_usage["endpoints"].values()
    )
    ollama_tokens_total = (
        final_usage["ollama"]["total_prompt_tokens"]
        + final_usage["ollama"]["total_completion_tokens"]
    )

    output = {
        "branch": args.branch,
        "timestamp": datetime.now(UTC).isoformat(),
        "api_url": args.api_url,
        "killer_demo": killer_demo,
        "supersession": supersession,
        "token_totals": {
            "response_tokens_total": response_tokens_total,
            "ollama_tokens_total": ollama_tokens_total,
        },
    }
    if longmemeval is not None:
        output["longmemeval"] = longmemeval

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Landscape A/B benchmark")
    p.add_argument("--branch", required=True, help="Branch label (used in output filename)")
    p.add_argument("--longmemeval-data", default=None, help="Path to LongMemEval JSON")
    p.add_argument("--api-url", default="http://localhost:8000", help="Landscape API base URL")
    p.add_argument("--n-questions", type=int, default=10, help="LongMemEval questions to run")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: bench_ab_{branch}.json)",
    )
    args = p.parse_args()
    if args.output is None:
        args.output = f"bench_ab_{args.branch}.json"
    return args


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
