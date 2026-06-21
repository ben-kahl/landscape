#!/usr/bin/env python3
"""Discriminating extraction benchmark over the extraction_bench_corpus.

Unlike scripts/bench_extraction.py (killer-demo: tiny, easy, substring-scored),
this corpus stresses coreference, implicit/inferred relations, distractor noise,
aliases/acronyms, temporal supersession, negation, quantified facts, and
vector-twin ambiguity — scored with EXACT (canonical, alias-aware) matching and
reported per difficulty axis so two models can actually be told apart.

Layer A (pure extract(), no DB): entity / relation / negation / alias-extraction
precision & recall per axis.
Layer B (full pipeline.ingest() into the TEST stack): end-to-end alias
persistence + temporal supersession correctness.

Numeric value scoring (counts/durations/dates) is out of scope for v1 — the
quantified axis is graded on its entities + relations only.

Usage:
    # Layer A only (no DB needed), both local profiles:
    uv run python scripts/bench_extraction_hard.py --layer a

    # One profile, both layers (needs the test stack up + the model served):
    uv run python scripts/bench_extraction_hard.py --profile local_qwen35 --layer both

    uv run python scripts/bench_extraction_hard.py --list
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

os.environ.setdefault("NEO4J_URI", "bolt://localhost:17687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-test")
os.environ.setdefault("QDRANT_URL", "http://localhost:16333")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:8080/v1")

from landscape.config import LLM_PROFILES, settings  # noqa: E402
from landscape.extraction.entity_type_coercion import coerce_entity_type  # noqa: E402
from landscape.extraction.llm import extract  # noqa: E402
from landscape.extraction.schema import normalize_relation_type  # noqa: E402

CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "../tests/fixtures/extraction_bench_corpus"
LABELS: dict = json.loads((CORPUS_DIR / "labels.json").read_text())


# -- normalization helpers --------------------------------------------------
def _n(s: str) -> str:
    return s.strip().lower()


def _ctype(t: str) -> str:
    return coerce_entity_type(t)[0]


def _crel(r: str) -> str:
    return normalize_relation_type(r)


def _name_map(entities: list[dict]) -> dict[str, str]:
    """norm(any surface form) -> norm(canonical name), for alias-aware matching."""
    m: dict[str, str] = {}
    for e in entities:
        cn = _n(e["name"])
        m[cn] = cn
        for a in e.get("aliases", []):
            m[_n(a)] = cn
    return m


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# -- Layer A scoring (per doc) ----------------------------------------------
class Counts:
    __slots__ = ("rh", "rt", "ph", "pt")  # recall hits/total, precision hits/total

    def __init__(self):
        self.rh = self.rt = self.ph = self.pt = 0

    def add(self, rh, rt, ph, pt):
        self.rh += rh
        self.rt += rt
        self.ph += ph
        self.pt += pt

    @property
    def recall(self):
        return self.rh / self.rt if self.rt else 0.0

    @property
    def precision(self):
        return self.ph / self.pt if self.pt else 0.0

    @property
    def f1(self):
        return _f1(self.precision, self.recall)


def _score_entities(extracted, expected):
    exp = [
        ({_n(e["name"])} | {_n(a) for a in e.get("aliases", [])}, _ctype(e["type"]))
        for e in expected
    ]
    ext = [(_n(x.name), _ctype(x.type)) for x in extracted]
    rh = sum(1 for names, et in exp if any(xn in names and xt == et for xn, xt in ext))
    ph = sum(1 for xn, xt in ext if any(xn in names and xt == et for names, et in exp))
    return rh, len(exp), ph, len(ext)


def _triples(rels, name_map, *, negated: bool):
    def canon(x):
        return name_map.get(_n(x), _n(x))

    return {
        (canon(r.subject), _crel(r.relation_type), canon(r.object))
        for r in rels
        if bool(getattr(r, "negated", False)) == negated
    }


def _score_relations(extracted, expected_rels, name_map):
    def canon(x):
        return name_map.get(_n(x), _n(x))

    exp = {(canon(r["subject"]), _crel(r["rel_type"]), canon(r["object"])) for r in expected_rels}
    ext = _triples(extracted, name_map, negated=False)
    inter = len(exp & ext)
    return inter, len(exp), inter, len(ext)


def _score_negation(extracted, expected_neg, name_map):
    def canon(x):
        return name_map.get(_n(x), _n(x))

    exp = {(canon(r["subject"]), _crel(r["rel_type"]), canon(r["object"])) for r in expected_neg}
    ext = _triples(extracted, name_map, negated=True)
    inter = len(exp & ext)
    return inter, len(exp), inter, len(ext)


def _score_alias_extraction(extracted, expected):
    pairs = [(e["name"], a) for e in expected for a in e.get("aliases", [])]
    ext_map: dict[str, set[str]] = defaultdict(set)
    for x in extracted:
        ext_map[_n(x.name)].update(_n(a) for a in x.aliases)
    hits = sum(1 for canonical, alias in pairs if _n(alias) in ext_map.get(_n(canonical), set()))
    return hits, len(pairs)


def run_layer_a(profile_name: str) -> dict:
    original = settings.llm_profile
    settings.llm_profile = profile_name
    by_axis: dict[str, dict[str, Counts]] = defaultdict(
        lambda: {"entities": Counts(), "relations": Counts()}
    )
    neg = Counts()
    alias = Counts()
    errors: list[str] = []
    t0 = time.perf_counter()
    try:
        for fname in sorted(LABELS):
            label = LABELS[fname]
            axis = label["axis"]
            text = (CORPUS_DIR / fname).read_text()
            try:
                ex = extract(text)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{fname}: {e}")
                continue
            nm = _name_map(label["entities"])
            by_axis[axis]["entities"].add(*_score_entities(ex.entities, label["entities"]))
            by_axis[axis]["relations"].add(*_score_relations(ex.relations, label["relations"], nm))
            nh, nt = _score_negation(ex.relations, label.get("negated_relations", []), nm)[:2]
            neg.add(nh, nt, 0, 0)
            ah, at = _score_alias_extraction(ex.entities, label["entities"])
            alias.add(ah, at, 0, 0)
    finally:
        settings.llm_profile = original
    return {
        "profile": profile_name,
        "model": LLM_PROFILES[profile_name].model,
        "seconds": round(time.perf_counter() - t0, 1),
        "by_axis": {
            ax: {
                "ent_p": c["entities"].precision, "ent_r": c["entities"].recall,
                "ent_f1": c["entities"].f1,
                "rel_p": c["relations"].precision, "rel_r": c["relations"].recall,
                "rel_f1": c["relations"].f1,
            }
            for ax, c in sorted(by_axis.items())
        },
        "entities_overall": _overall(by_axis, "entities"),
        "relations_overall": _overall(by_axis, "relations"),
        "negation_recall": neg.recall,
        "alias_extraction_recall": alias.recall,
        "errors": errors,
    }


def _overall(by_axis, kind):
    agg = Counts()
    for c in by_axis.values():
        agg.add(c[kind].rh, c[kind].rt, c[kind].ph, c[kind].pt)
    return {"p": agg.precision, "r": agg.recall, "f1": agg.f1}


# -- Layer B: end-to-end alias persistence + supersession -------------------
async def run_layer_b(profile_name: str) -> dict:
    from landscape import pipeline
    from landscape.cli.wipe import wipe_state
    from landscape.embeddings import encoder
    from landscape.storage import neo4j_store, qdrant_store

    original = settings.llm_profile
    settings.llm_profile = profile_name
    encoder.load_model()
    await wipe_state()
    await qdrant_store.init_collection()
    await qdrant_store.init_chunks_collection()

    try:
        for fname in sorted(LABELS):
            await pipeline.ingest((CORPUS_DIR / fname).read_text(), title=fname)

        # Alias persistence: expected (canonical, alias) pairs across the corpus.
        pairs = [
            (e["name"], a)
            for label in LABELS.values()
            for e in label["entities"]
            for a in e.get("aliases", [])
        ]
        driver = neo4j_store.get_driver()
        async with driver.session() as s:
            res = await s.run(
                "MATCH (a:Alias)-[:SAME_AS]->(e:Entity) "
                "RETURN toLower(a.name) AS alias, toLower(e.name) AS canonical"
            )
            persisted = {(r["alias"], r["canonical"]) async for r in res}
        alias_hits = sum(1 for c, a in pairs if (_n(a), _n(c)) in persisted)

        # Supersession: old functional edge closed, new edge live.
        supers = [
            sup
            for label in LABELS.values()
            for sup in label.get("supersession", [])
        ]
        sup_ok = 0
        sup_detail = []
        async with driver.session() as s:
            for sup in supers:
                res = await s.run(
                    "MATCH (subj:Entity)-[r:MEMORY_REL {family: $fam}]->(o:Entity) "
                    "WHERE toLower(subj.name) CONTAINS $subj "
                    "RETURN toLower(o.name) AS target, (r.system_until IS NULL) AS current",
                    fam=_crel(sup["rel_type"]),
                    subj=_n(sup["subject"]).split()[0],
                )
                rows = [(r["target"], r["current"]) async for r in res]
                new_live = any(_n(sup["new_object"]) in t and cur for t, cur in rows)
                old_closed = any(
                    _n(sup["old_object"]) in t and not cur for t, cur in rows
                ) or not any(_n(sup["old_object"]) in t for t, cur in rows if cur)
                ok = new_live and old_closed
                sup_ok += int(ok)
                sup_detail.append(
                    f"{sup['subject']} {sup['rel_type']} "
                    f"{sup['old_object']}->{sup['new_object']}: "
                    f"{'OK' if ok else 'FAIL'} ({rows})"
                )
    finally:
        settings.llm_profile = original
        await neo4j_store.close_driver()
        await qdrant_store.close_client()

    return {
        "alias_persistence_recall": alias_hits / len(pairs) if pairs else 0.0,
        "alias_pairs": len(pairs),
        "supersession_correct": sup_ok,
        "supersession_total": len(supers),
        "supersession_detail": sup_detail,
    }


def _print_layer_a(r: dict) -> None:
    print(f"\n=== Layer A: {r['profile']} ({r['model']}) — {r['seconds']}s ===")
    hdr = (
        f"{'Axis':<14} {'Ent P':>6} {'Ent R':>6} {'Ent F1':>6}  "
        f"{'Rel P':>6} {'Rel R':>6} {'Rel F1':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ax, m in r["by_axis"].items():
        print(
            f"{ax:<14} {m['ent_p']:>6.1%} {m['ent_r']:>6.1%} {m['ent_f1']:>6.1%}  "
            f"{m['rel_p']:>6.1%} {m['rel_r']:>6.1%} {m['rel_f1']:>6.1%}"
        )
    eo, ro = r["entities_overall"], r["relations_overall"]
    print("-" * len(hdr))
    print(
        f"{'OVERALL':<14} {eo['p']:>6.1%} {eo['r']:>6.1%} {eo['f1']:>6.1%}  "
        f"{ro['p']:>6.1%} {ro['r']:>6.1%} {ro['f1']:>6.1%}"
    )
    print(f"negation recall: {r['negation_recall']:.1%} | "
          f"alias-extraction recall: {r['alias_extraction_recall']:.1%}")
    for e in r["errors"]:
        print(f"  WARNING: {e}", file=sys.stderr)


def _print_layer_b(r: dict) -> None:
    print("\n=== Layer B (end-to-end, test stack) ===")
    print(f"alias persistence recall: {r['alias_persistence_recall']:.1%} "
          f"({r['alias_pairs']} pairs)")
    print(f"supersession correct: {r['supersession_correct']}/{r['supersession_total']}")
    for d in r["supersession_detail"]:
        print(f"  {d}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Discriminating extraction benchmark")
    ap.add_argument("--profile", action="append", help="Profile(s); default both local")
    ap.add_argument("--layer", choices=["a", "b", "both"], default="a")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json-out", help="Write combined results JSON to this path")
    args = ap.parse_args()

    if args.list:
        for name, p in sorted(LLM_PROFILES.items()):
            print(f"  {name:<20} {p.model:<26} {p.notes}")
        return

    profiles = args.profile or ["local_qwen35", "local_llama31"]
    combined: dict = {}
    for p in profiles:
        if p not in LLM_PROFILES:
            print(f"Unknown profile: {p}", file=sys.stderr)
            sys.exit(1)
        entry: dict = {}
        if args.layer in ("a", "both"):
            a = run_layer_a(p)
            _print_layer_a(a)
            entry["layer_a"] = a
        if args.layer in ("b", "both"):
            b = asyncio.run(run_layer_b(p))
            _print_layer_b(b)
            entry["layer_b"] = b
        combined[p] = entry

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(combined, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
