#!/usr/bin/env python3
"""LongMemEval benchmark for Landscape hybrid retrieval with LLM judging.

LongMemEval (Wu et al.) probes personal-memory recall: given a haystack of
conversation sessions, answer a question whose evidence lives in a known
subset of those sessions.

Pipeline per question:
  1. Ingest haystack sessions (gold evidence first, then distractors).
  2. Retrieve via LandscapeRetriever — entities with edge quantities rendered
     inline, plus raw text chunks.
  3. Call AWS Bedrock to generate an answer from the retrieved context.
  4. Call AWS Bedrock to judge the generated answer against the gold answer.

Metrics reported:
  hit_at_k_rate      — retrieval coverage proxy (was the gold entity in top-k?)
  judge_correct_rate — answer quality (correct + abstained) / total

Usage:
    # Requires docker stack up (Neo4j, Qdrant, Ollama) and AWS credentials.
    uv run python scripts/bench_longmemeval.py \\
        --data tests/longmemeval_s_cleaned.json \\
        --n-questions 10 \\
        --max-sessions 4

    # Skip Bedrock judging for a quick retrieval-only smoke run:
    uv run python scripts/bench_longmemeval.py \\
        --data tests/longmemeval_s_cleaned.json \\
        --n-questions 10 \\
        --skip-judge

    # Override judge model:
    uv run python scripts/bench_longmemeval.py \\
        --data tests/longmemeval_s_cleaned.json \\
        --judge-model anthropic.claude-3-5-sonnet-20241022-v2:0

AWS credentials required (unless --skip-judge):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    BEDROCK_JUDGE_MODEL_ID (optional override, default claude-3-5-haiku)

    Enable the model in the AWS Bedrock console before first use.
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
from landscape.retrieval.langchain_retriever import LandscapeRetriever  # noqa: E402
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

    Handles plain JSON, JSON wrapped in markdown code fences, and Bedrock
    responses that prepend prose before the fence.
    Returns {"judgment": "incorrect", "reason": "parse error: ..."} on failure.
    """
    text = raw.strip()
    fence_start = text.find("```")
    if fence_start != -1:
        text = text[fence_start:]
        parts = text.split("```")
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


def _bedrock_invoke(client, model_id: str, prompt: str, max_tokens: int = 512) -> str:
    """Send a single-turn prompt to Bedrock and return the response text."""
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
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
    raw = _bedrock_invoke(client, model_id, prompt, max_tokens=256)
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


async def _run_question(
    q: dict,
    max_sessions: int,
    k: int,
    bedrock_client,
    judge_model: str,
    skip_judge: bool,
) -> dict:
    """Wipe → ingest sampled sessions → retrieve via LandscapeRetriever →
    optionally generate answer and judge against gold."""
    qid = q["question_id"]
    question = q["question"]
    answer = q.get("answer")
    session_ids = q.get("haystack_session_ids") or []
    sessions = q.get("haystack_sessions") or []
    gold_source_ids = set(q.get("answer_session_ids") or [])

    picked_idx = _pick_sessions(q, max_sessions)

    await _wipe_stack()

    gold_landscape_sids: list[str] = []
    ingested_count = 0
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
        except Exception as exc:
            print(f"  [warn] ingest failed for {source_sid}: {exc!r}", file=sys.stderr)
            continue
        ingested_count += 1
        if source_sid in gold_source_ids:
            gold_landscape_sids.append(landscape_sid)
    ingest_s = time.time() - ingest_t0

    gold_entity_ids: set[str] = set()
    for sid in gold_landscape_sids:
        gold_entity_ids.update(await neo4j_store.get_entities_in_conversation(sid))

    retriever = LandscapeRetriever(hops=2, limit=k, reinforce=False)
    query_t0 = time.time()
    docs = await retriever.ainvoke(question)
    query_s = time.time() - query_t0

    top_names = [
        d.metadata.get("name", d.page_content[:40])
        for d in docs
        if d.metadata.get("kind") == "entity"
    ]
    top_ids = [
        d.metadata.get("neo4j_id")
        for d in docs
        if d.metadata.get("kind") == "entity"
    ]
    hit = any(eid in gold_entity_ids for eid in top_ids if eid)

    row: dict = {
        "question_id": qid,
        "question": question,
        "answer": answer,
        "n_sessions_ingested": ingested_count,
        "n_gold_sessions": len(gold_landscape_sids),
        "n_gold_entities": len(gold_entity_ids),
        "hit_at_k": hit,
        "top_names": top_names,
        "ingest_s": round(ingest_s, 2),
        "query_s": round(query_s, 3),
    }

    if not skip_judge and bedrock_client is not None:
        try:
            generated = _generate_answer(bedrock_client, judge_model, docs, question)
            judgment = _judge_answer(bedrock_client, judge_model, question, answer or "", generated)
            row["generated_answer"] = generated
            row["judgment"] = judgment.get("judgment", "incorrect")
            row["judgment_reason"] = judgment.get("reason", "")
        except Exception as exc:
            print(f"  [warn] Bedrock call failed for {qid}: {exc!r}", file=sys.stderr)
            row["generated_answer"] = None
            row["judgment"] = "error"
            row["judgment_reason"] = str(exc)

    return row


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to longmemeval_*.json")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--max-sessions", type=int, default=6,
                    help="Haystack sessions to ingest per question (gold first, then distractors)")
    ap.add_argument("--k", type=int, default=10, help="top-k for retrieval hit check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(RESULTS_PATH))
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("BEDROCK_JUDGE_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"),
        help="Bedrock model ID for answer generation and judging",
    )
    ap.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region for Bedrock",
    )
    ap.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Bedrock generation and judging; run retrieval only",
    )
    args = ap.parse_args()

    bedrock_client = None
    if not args.skip_judge:
        try:
            import boto3
            bedrock_client = boto3.client("bedrock-runtime", region_name=args.aws_region)
        except ImportError:
            print(
                "warning: boto3 not installed. Install with: uv sync --extra bench\n"
                "Falling back to --skip-judge mode.",
                file=sys.stderr,
            )
            args.skip_judge = True

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
        row = await _run_question(
            q,
            max_sessions=args.max_sessions,
            k=args.k,
            bedrock_client=bedrock_client,
            judge_model=args.judge_model,
            skip_judge=args.skip_judge,
        )
        judgment_str = f"  judgment={row['judgment']}" if "judgment" in row else ""
        print(f"    hit@{args.k}={row['hit_at_k']}{judgment_str}  ingest={row['ingest_s']}s  query={row['query_s']}s")
        results.append(row)

    n = len(results)
    hits = sum(1 for r in results if r["hit_at_k"])
    judged = [r for r in results if "judgment" in r]
    n_correct = sum(1 for r in judged if r["judgment"] == "correct")
    n_abstained = sum(1 for r in judged if r["judgment"] == "abstained")
    n_incorrect = sum(1 for r in judged if r["judgment"] == "incorrect")

    summary: dict = {
        "slice": QUESTION_TYPE,
        "n_questions": n,
        "max_sessions_per_q": args.max_sessions,
        "k": args.k,
        "hit_at_k_rate": round(hits / n, 3) if n else None,
        "avg_ingest_s": round(sum(r["ingest_s"] for r in results) / n, 2) if n else None,
        "avg_query_s": round(sum(r["query_s"] for r in results) / n, 3) if n else None,
    }
    if judged:
        summary["judge_correct_rate"] = round((n_correct + n_abstained) / len(judged), 3)
        summary["judge_correct_count"] = n_correct
        summary["judge_abstained_count"] = n_abstained
        summary["judge_incorrect_count"] = n_incorrect
        summary["judge_model"] = args.judge_model

    output_path = pathlib.Path(args.output)
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print()
    print(f"hit@{args.k} = {hits}/{n} = {summary['hit_at_k_rate']}")
    if judged:
        print(
            f"judge: correct={n_correct}  abstained={n_abstained}  incorrect={n_incorrect}  "
            f"rate={summary['judge_correct_rate']}  model={args.judge_model}"
        )
    print(f"avg ingest = {summary['avg_ingest_s']}s   avg query = {summary['avg_query_s']}s")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
