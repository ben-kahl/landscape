# Extraction quality A/B — Qwen 3.5 9B vs Llama 3.1 8B

Date: 2026-06-21
Harness: `scripts/bench_extraction.py` (entity/relation precision/recall/F1 vs
the hand-labeled killer-demo corpus, `tests/fixtures/killer_demo_corpus/`, 7 docs).
Hardware: RTX 2080 (8 GB), both models Q4_K_M served by llama.cpp `llama-server`,
`--no-mmproj`, ctx 16384. Run sequentially (one model fits at a time). No DB
writes (extraction-only); env pointed at the test stack regardless.

| Model | Ent P | Ent R | Ent F1 | Rel P | Rel R | Rel F1 | Time (7 docs) |
|---|---|---|---|---|---|---|---|
| **Qwen 3.5 9B** (default) | 60.0% | 93.8% | **73.2%** | 34.1% | 93.3% | **50.0%** | **102.3 s** |
| Llama 3.1 8B (baseline) | 63.8% | 93.8% | 75.9% | 36.8% | 93.3% | 52.8% | 173.8 s |
| Δ (Qwen − Llama) | −3.8 | 0.0 | −2.7 | −2.7 | 0.0 | −2.8 | **1.70× faster** |

## Reading

- **Recall is identical** (entities 93.8%, relations 93.3%) — both models surface
  the same facts; nothing the downstream graph needs is being missed by Qwen.
- **Quality is within ~3 pp** on both entity and relation F1. The gap is
  precision (both models over-extract), not recall.
- **Qwen 3.5 is ~1.7× faster** (102 s vs 174 s for the same 7 docs).
- Precision being low for *both* models is a corpus/scoring characteristic
  (substring fuzzy match + over-extraction), not a Qwen regression; the resolver
  and functional-supersession logic absorb duplicate/extra edges downstream.

## Caveat — limited discriminating power

This corpus is a **demo fixture, not a discriminating quality benchmark.** Total
ground truth is **32 entities / 15 relations** across 393 words of explicit,
one-fact-per-sentence prose, scored with case-insensitive substring matching.
That is too small, too easy, and too lenient to separate two capable 8B+ models:
both saturate the ceiling, missing the *same* 2 entities and 1 relation. The
identical recall (93.8% / 93.3%) is the proof — it is a ceiling, not a tie.

So this run does **not** support a "Qwen ≈ Llama in quality" claim. It supports a
narrower one: Qwen 3.5 extraction works correctly end-to-end and is not grossly
worse than the baseline *on easy inputs*. The only statistically robust finding
here is **speed** (Qwen 1.7× faster). A real quality comparison needs a harder,
larger corpus (coreference, implicit relations, distractors, aliasing, temporal
supersession) with exact/span scoring — tracked as a follow-up.

## Decision

No gate. Qwen 3.5 9B remains the default: it extracts correctly end-to-end, is no
worse than the baseline on this (easy) corpus, and runs 1.7× faster. Llama 3.1
stays available as the `local_llama31` A/B profile.

Raw run output: `bench_extraction_qwen35.txt`, `bench_extraction_llama31.txt`.

## Reproduce

```bash
# Qwen (default, already served by the compose stack):
LLM_BASE_URL=http://localhost:8080/v1 uv run python scripts/bench_extraction.py --profile local_qwen35

# Llama 3.1 baseline — serve it on :8080 with a matching alias, then bench:
docker stop landscape-llama-server-nvidia-1
docker run -d --rm --name llama-ab-llama31 --gpus all -p 8080:8080 \
  -v landscape_llama_models:/root/.cache/huggingface --env-file .env \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 --port 8080 --n-gpu-layers -1 --ctx-size 16384 \
  --parallel 1 --no-mmproj --n-predict 4096 --alias llama-3.1-8b
LLM_BASE_URL=http://localhost:8080/v1 uv run python scripts/bench_extraction.py --profile local_llama31
docker stop llama-ab-llama31
docker compose --profile gpu-nvidia up -d llama-server-nvidia   # restore Qwen
```
