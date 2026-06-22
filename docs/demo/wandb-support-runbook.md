# Landscape × W&B Support — demo runbook

**What this shows:** Landscape gives an agent (Claude) persistent, queryable
memory of real W&B support knowledge. Point Claude at Landscape's MCP server and
it answers support questions over real closed GitHub issues + the real CHANGELOG,
with provenance — and where answering needs more than one source, the graph
connects them. Multi-hop is a natural payoff of real knowledge at scale, not a
staged trick.

**Why it matters for support:** support answers routinely span sources — a user's
symptom + version, a known issue, and which release fixed it. Vector-only search
finds the most *similar* chunk; Landscape can *traverse* from a symptom to a
related issue to a CHANGELOG fix. It also tracks **fixed-in-version** facts from
the real changelog, so "is this fixed yet?" has a real answer.

**W&B-native:** Landscape's own extraction/retrieval pipeline is instrumented with
**Weave** — you can show the traces live (we dogfood W&B's tooling).

---

## 0. Prerequisites

- The local stack running (Neo4j + Qdrant + llama-server). Pick your backend:
  ```bash
  ./scripts/detect-stack.sh            # writes .env (COMPOSE_PROFILES, LLM_BASE_URL)
  docker compose --profile gpu-nvidia up -d   # or cpu / gpu-amd / host
  ```
- `uv sync --extra dev --extra observability` (observability = Weave).
- The corpus is committed at `tests/fixtures/wandb_support_corpus/` (13 real
  docs). To refresh/extend it from GitHub (optional, extraction-free):
  ```bash
  uv run python scripts/fetch_wandb_issues.py
  ```

## 1. Turn on Weave tracing (optional but recommended for the demo)

```bash
export WEAVE_PROJECT="wandb-support-demo"   # any W&B project; enables tracing
```
Set this in the shell that runs ingestion and the server. With it set, extraction
and retrieval calls show up as Weave traces you can pull up during the demo.

## 2. Seed the graph from the real corpus

> Uses the app's own CLI (`ingest-dir`) — no demo-specific code path. This is the
> one slow step (local LLM extraction over 13 docs); run it before the meeting.

```bash
uv run landscape wipe --confirm     # clears the local default graph
uv run landscape ingest-dir tests/fixtures/wandb_support_corpus --glob "*.md"
```
`--glob "*.md"` ingests only the docs and skips `manifest.json`.

Sanity-check what landed:
```bash
uv run landscape status --verbose
uv run landscape query "What login errors have users reported and how were they resolved?"
```

## 3. Start the server and connect Claude

```bash
uv run uvicorn landscape.main:app --host 127.0.0.1 --port 8000
```
In Claude Code (or Cursor), add the MCP server:
```
http://localhost:8000/mcp
```
First connect runs the OAuth flow in your browser (default `agent` scope). Claude
now has `search`, `lookup_entity`, `status`, `conversation_history`, etc.

## 4. Drive the demo (ask Claude)

Realistic support questions over the corpus — let Claude answer via the tools:

- "A user gets `Object has no attribute 'disabled'` running `wandb login` — what's
  going on?"  *(→ issue #9580)*
- "Why might `wandb.init()` hang right after logging in?"  *(→ #11532: entity
  differs from the logged-in user)*
- "A user sees `user is not logged in (401)` — what causes that?"  *(→ #7118)*
- "Is logging a non-JSON-serializable object as an artifact a known issue, and
  was it fixed?"  *(→ #4500 + CHANGELOG serialization/orjson work)* — a question
  whose full answer naturally spans an **issue + a CHANGELOG fix**.
- "What login/auth problems have users hit, across reports?"  *(broad — pulls
  several real issues together)*

**What to point out while answering:**
- **Provenance** — each fact traces to a real issue # / CHANGELOG line (in
  `manifest.json` / the returned chunks).
- **Connection across sources** — note when an answer stitches a user report to a
  CHANGELOG fix or a related issue (that's the multi-hop value).
- **Weave traces** — open the `WEAVE_PROJECT` to show the extraction/retrieval
  calls that produced the answer.

## 5. (Optional) hybrid-vs-vector backstop

To make the differentiator concrete beyond the live Q&A, run the labeled
hybrid-vs-vector comparison over the corpus (see `scripts/bench_retrieval.py`
pattern) and show hybrid's edge across the query set.

## 6. Talking points

- **Useful now:** real support recall over real W&B knowledge, in the agent the
  engineer already uses.
- **Differentiator:** graph traversal connects symptom → related issue → fix /
  fixed-in-version, which similarity search alone misses; this grows stronger as
  the corpus scales.
- **Temporal:** fixed-in-version facts come from the real CHANGELOG.
- **W&B-native:** the system is Weave-instrumented end to end.
- **Roadmap (honest):** the current entity/relation vocabulary is generic
  (org-domain); support relations like `FIXED_IN` / `WORKAROUND_FOR` collapse to
  generic ones today. Planned **user-defined domain vocabularies** would give
  support (and legal, research, …) first-class families — i.e. the same engine
  generalizes across domains.

## Notes / honesty

- Corpus is **real, public** closed-issue / public-CHANGELOG content only — every
  doc's source URL is in `manifest.json`. No fabricated problems.
- Extraction runs on a local ~9B model (fast, fully offline; occasional missed
  relations — expected, and what the retrieval layer is designed to tolerate).
- `wandb wipe --confirm` clears the local default graph; run the demo on a
  machine where that's fine.

## Cleanup

```bash
uv run landscape wipe --confirm     # optional, clears the demo graph
```
