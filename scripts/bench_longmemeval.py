#!/usr/bin/env python3
"""Smoke benchmark: Landscape hybrid retrieval on a LongMemEval slice.

LongMemEval (Wu et al.) probes personal-memory recall: given a haystack of
conversation sessions, answer a question whose evidence lives in a known
subset of those sessions. This script runs a small slice and reports a
retrieval-coverage proxy metric — "did top-k retrieval surface at least one
entity from an evidence session" — which is what step 7 of the
vocab_expansion spec asks for.

We are NOT running the official LLM-judge scorer. This smoke measures
whether the expanded vocab + subtype layer routes evidence sessions to
top-k, which is the precondition for any downstream answer-quality gain.

Usage:
    # Requires docker stack up (Neo4j, Qdrant, Ollama).
    # LongMemEval data is NOT in-repo. Point at a local JSON dump:
    uv run python scripts/bench_longmemeval.py \\
        --data /path/to/longmemeval_s.json \\
        --n-questions 10 \\
        --max-sessions 6

    # Download LongMemEval from the authors' release:
    #   https://github.com/xiaowu0162/LongMemEval
    # Pick longmemeval_s.json (small haystack) for this smoke.

The script ingests each question's haystack sessions (capped at
--max-sessions, always including the gold evidence sessions first), runs
hybrid retrieval for the question text, and records top-k hit status.
Results land at scripts/bench_longmemeval_results.json.
"""
import argparse
import asyncio
import json
import os
import pathlib
import random
import sys
import time

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from landscape import pipeline  # noqa: E402
from landscape.embeddings import encoder  # noqa: E402
from landscape.retrieval.query import retrieve  # noqa: E402
from landscape.storage import neo4j_store, qdrant_store  # noqa: E402

QUESTION_TYPE = "single-session-user"
RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "bench_longmemeval_results.json"


def _safe_id(raw: str) -> str:
    """Session/turn IDs can't contain ':' per neo4j_store._validate_id_segment."""
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


def _format_session(session_turns: list[dict]) -> str:
    """Render a list of {role, content} turns as a readable transcript."""
    lines = []
    for turn in session_turns:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        if content:
            lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def _format_docs_for_prompt(docs: list) -> str:
    """Render retrieved Documents (entities + chunks) as a plain-text context block."""
    return "\n\n".join(d.page_content for d in docs)


def _parse_judge_response(raw: str) -> dict:
    """Parse a Bedrock judge response string into {judgment, reason}.

    Handles plain JSON and JSON wrapped in markdown code fences.
    Returns {"judgment": "incorrect", "reason": "parse error: ..."} on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between the first pair of fences
        inner = parts[1] if len(parts) > 1 else ""
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()
    try:
        parsed = json.loads(text)
        if "judgment" not in parsed:
            raise ValueError("missing 'judgment' key")
        return parsed
    except Exception as exc:
        return {"judgment": "incorrect", "reason": f"parse error: {exc!r} raw={raw!r}"}


def _bedrock_invoke(client, model_id: str, prompt: str) -> str:
    """Send a single-turn prompt to Bedrock and return the response text."""
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0},
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def _generate_answer(client, model_id: str, docs: list, question: str) -> str:
    """Call Bedrock to answer question from retrieved document context."""
    context = _format_docs_for_prompt(docs)
    prompt = (
        "You are answering a question about a person's memories and experiences.\n"
        "Use only the context below. If the context does not contain enough information "
        'to answer, respond with exactly: "I don\'t have that information."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely in one or two sentences."
    )
    return _bedrock_invoke(client, model_id, prompt)


def _judge_answer(client, model_id: str, question: str, gold: str, generated: str) -> dict:
    """Call Bedrock to judge whether generated matches gold. Returns {judgment, reason}."""
    prompt = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Generated answer: {generated}\n\n"
        "Does the generated answer correctly answer the question given the gold answer?\n"
        'Respond with a JSON object only: {"judgment": "correct"|"incorrect"|"abstained", "reason": "..."}\n\n'
        "Rules:\n"
        '- "correct": the generated answer conveys the same information as the gold answer, '
        "allowing for paraphrase.\n"
        '- "abstained": the generated answer says it does not have the information AND the '
        "gold answer also indicates the information was not available.\n"
        '- "incorrect": any other case.'
    )
    raw = _bedrock_invoke(client, model_id, prompt)
    return _parse_judge_response(raw)


def _pick_sessions(q: dict, max_sessions: int) -> list[int]:
    """Return indices into haystack_sessions: gold evidence first, then
    distractors, capped at max_sessions."""
    session_ids = q.get("haystack_session_ids") or []
    sessions = q.get("haystack_sessions") or []
    gold = set(q.get("answer_session_ids") or [])
    gold_idx = [i for i, sid in enumerate(session_ids) if sid in gold]
    distractor_idx = [i for i in range(len(sessions)) if i not in set(gold_idx)]
    random.shuffle(distractor_idx)
    picked = gold_idx + distractor_idx
    return picked[:max_sessions]


async def _run_question(q: dict, max_sessions: int, k: int) -> dict:
    """Wipe → ingest sampled sessions (tagged by Landscape session_id) →
    query question → check top-k coverage of evidence sessions."""
    qid = q["question_id"]
    question = q["question"]
    answer = q.get("answer")
    session_ids = q.get("haystack_session_ids") or []
    sessions = q.get("haystack_sessions") or []
    gold_source_ids = set(q.get("answer_session_ids") or [])

    picked_idx = _pick_sessions(q, max_sessions)

    await _wipe_stack()

    gold_landscape_sids: list[str] = []
    ingest_t0 = time.time()
    for i in picked_idx:
        source_sid = session_ids[i] if i < len(session_ids) else f"s{i}"
        landscape_sid = _safe_id(f"lme-{qid}-{source_sid}")
        text = _format_session(sessions[i])
        if not text:
            continue
        try:
            await pipeline.ingest(
                text=text,
                title=f"longmemeval:{qid}:{source_sid}",
                source_type="text",
                session_id=landscape_sid,
                turn_id="t0",
            )
        except Exception as exc:  # extraction-layer failures shouldn't abort the smoke
            print(f"  [warn] ingest failed for {source_sid}: {exc!r}", file=sys.stderr)
            continue
        if source_sid in gold_source_ids:
            gold_landscape_sids.append(landscape_sid)
    ingest_s = time.time() - ingest_t0

    gold_entity_ids: set[str] = set()
    for sid in gold_landscape_sids:
        gold_entity_ids.update(await neo4j_store.get_entities_in_conversation(sid))

    query_t0 = time.time()
    result = await retrieve(
        query_text=question,
        hops=2,
        limit=k,
        reinforce=False,
    )
    query_s = time.time() - query_t0

    top_names = [r.name for r in result.results]
    top_ids = [r.neo4j_id for r in result.results]
    hit = any(eid in gold_entity_ids for eid in top_ids)

    return {
        "question_id": qid,
        "question": question,
        "answer": answer,
        "n_sessions_ingested": len(picked_idx),
        "n_gold_sessions": len(gold_landscape_sids),
        "n_gold_entities": len(gold_entity_ids),
        "hit_at_k": hit,
        "top_names": top_names,
        "ingest_s": round(ingest_s, 2),
        "query_s": round(query_s, 3),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to longmemeval_*.json")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--max-sessions", type=int, default=6,
                    help="Haystack sessions to ingest per question (gold first, then distractors)")
    ap.add_argument("--k", type=int, default=10, help="top-k for retrieval hit check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(RESULTS_PATH))
    args = ap.parse_args()

    random.seed(args.seed)

    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"error: {data_path} does not exist", file=sys.stderr)
        return 2

    raw = json.loads(data_path.read_text())
    # LongMemEval release JSON is a list of question dicts.
    all_qs = raw if isinstance(raw, list) else raw.get("questions", [])
    slice_qs = [q for q in all_qs if q.get("question_type") == QUESTION_TYPE]
    if not slice_qs:
        print(f"error: no questions of type {QUESTION_TYPE!r} found", file=sys.stderr)
        return 2

    random.shuffle(slice_qs)
    sampled = slice_qs[: args.n_questions]
    print(f"LongMemEval smoke: {len(sampled)} question(s) "
          f"(slice={QUESTION_TYPE}, max_sessions={args.max_sessions}, k={args.k})")

    encoder.load_model()

    results: list[dict] = []
    for i, q in enumerate(sampled, start=1):
        print(f"[{i}/{len(sampled)}] {q['question_id']}: {q['question'][:80]}")
        row = await _run_question(q, max_sessions=args.max_sessions, k=args.k)
        print(f"    hit@{args.k}={row['hit_at_k']}  ingest={row['ingest_s']}s  query={row['query_s']}s")
        results.append(row)

    n = len(results)
    hits = sum(1 for r in results if r["hit_at_k"])
    summary = {
        "slice": QUESTION_TYPE,
        "n_questions": n,
        "max_sessions_per_q": args.max_sessions,
        "k": args.k,
        "hit_at_k_rate": round(hits / n, 3) if n else None,
        "avg_ingest_s": round(sum(r["ingest_s"] for r in results) / n, 2) if n else None,
        "avg_query_s": round(sum(r["query_s"] for r in results) / n, 3) if n else None,
    }

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print()
    print(f"hit@{args.k} = {hits}/{n} = {summary['hit_at_k_rate']}")
    print(f"avg ingest = {summary['avg_ingest_s']}s   avg query = {summary['avg_query_s']}s")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
