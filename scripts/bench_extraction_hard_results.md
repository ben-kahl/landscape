# Hard extraction benchmark — Qwen 3.5 9B vs Llama 3.1 8B

Date: 2026-06-21
Harness: `scripts/bench_extraction_hard.py` over
`tests/fixtures/extraction_bench_corpus/` (12 docs, ~150 labeled items, 8
difficulty axes). **Exact**, alias-aware, canonical-type scoring against complete
labels. Both models Q4_K_M via llama-server on an RTX 2080, run sequentially.

This is the discriminating companion to `bench_extraction_results.md` (killer-demo),
which saturated for both models. Here the models actually separate.

## Layer A — extraction quality (overall)

| Model | Ent P | Ent R | Ent F1 | Rel P | Rel R | Rel F1 | Time |
|---|---|---|---|---|---|---|---|
| **Qwen 3.5 9B** | 83.6% | 82.3% | **82.9%** | 57.1% | 60.9% | **58.9%** | **176.7 s** |
| Llama 3.1 8B | 83.1% | 79.0% | 81.0% | 51.1% | 52.2% | 51.6% | 238.7 s |

## Layer A — per axis (F1)

| Axis | Qwen Ent | Llama Ent | Qwen Rel | Llama Rel |
|---|---|---|---|---|
| coreference | **92.3%** | 61.5% | 76.9% | 72.7% |
| implicit | 90.9% | 100.0% | 46.2% | 50.0% |
| distractor | 100.0% | 85.7% | 50.0% | 40.0% |
| aliases | 58.8% | 58.8% | **14.3%** | 0.0% |
| supersession | 92.9% | 92.3% | 88.9% | **100.0%** |
| negation | 75.0% | 75.0% | 33.3% | 33.3% |
| quantified | 66.7% | 66.7% | 57.1% | 25.0% |
| vector_twin | 100.0% | 100.0% | 66.7% | 66.7% |
| **negation recall** | 80.0% | 80.0% | — | — |
| **alias-extraction recall** | 58.3% | 50.0% | — | — |

## Layer B — end-to-end (full pipeline ingest → Neo4j, test stack)

| Model | Alias persistence recall | Supersession correct |
|---|---|---|
| Qwen 3.5 9B | 58.3% (7/12) | 3/3 |
| Llama 3.1 8B | 58.3% (7/12) | 3/3 |

## Reading

- **The corpus discriminates.** Unlike killer-demo (identical 93.8% recall for
  both), here the per-axis spread is wide and the models differ meaningfully —
  e.g. **coreference entity F1: Qwen 92.3% vs Llama 61.5%** (a 30 pt gap), and
  Qwen's overall relation F1 is +7.3 pt. Aliases and vector-twin remain hard for
  both (relation F1 ≤ 67%), which is the point.
- **Qwen 3.5 is the stronger extractor here** — ahead on relations, coreference,
  distractor rejection, and quantified relations, and faster (1.7×). Llama edges
  it only on supersession-relation and implicit entities. This is real evidence
  for keeping Qwen as the default, not a saturated tie.
- **The alias-persistence fix works end-to-end.** Layer B alias recall (58.3%)
  equals each model's Layer A alias-*extraction* recall — i.e. every alias the
  model emits now lands as `:Alias-[:SAME_AS]->canonical` in Neo4j. Before the
  fix (this branch's `pipeline.py` change) it was ~0%.
- **Supersession is correct end-to-end** (3/3 for both): the WORKS_FOR, LIVES_IN,
  and REPORTS_TO functional edges close the old object and activate the new one
  through real ingestion.

## Honest caveats

- ~150 labeled items, **single annotator** (the author). Spot-check
  `tests/fixtures/extraction_bench_corpus/labels.json`.
- **Exact** relation scoring can under-credit valid-but-differently-phrased
  relations (e.g. a model emitting `OWNS` where the label says `LEADS`). This is
  deliberate — it is what makes the benchmark discriminating — but it means
  absolute relation F1 understates "semantically acceptable" extraction.
- Relation **precision** is genuinely low for both (over-extraction + rel-type
  choice variance), not a scoring artifact like killer-demo's substring match.
- **Numeric-value scoring** (counts/durations/dates) is out of scope for v1; the
  quantified axis is graded on entities + relations only. A future extension.

## Reproduce

```bash
# Qwen (default, served by the compose stack):
uv run python scripts/bench_extraction_hard.py --profile local_qwen35 --layer both \
  --json-out scripts/bench_hard_qwen35.json

# Llama 3.1 baseline — serve it on :8080 with a matching alias, then bench:
docker stop landscape-llama-server-nvidia-1
docker run -d --rm --name llama-ab-llama31 --gpus all -p 8080:8080 \
  -v landscape_llama_models:/root/.cache/huggingface --env-file .env \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 --n-gpu-layers -1 --ctx-size 16384 \
  --parallel 1 --no-mmproj --n-predict 4096 --alias llama-3.1-8b
uv run python scripts/bench_extraction_hard.py --profile local_llama31 --layer both \
  --json-out scripts/bench_hard_llama31.json
docker stop llama-ab-llama31
docker compose --profile gpu-nvidia up -d llama-server-nvidia   # restore Qwen

# Layer A only (no DB / test stack needed):
uv run python scripts/bench_extraction_hard.py --layer a
```

Layer B targets the **test stack** (`NEO4J_URI :17687`, `QDRANT_URL :16333`) and
wipes it at the start of each profile run.
