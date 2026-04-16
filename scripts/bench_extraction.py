#!/usr/bin/env python3
"""Benchmark LLM extraction quality across profiles.

Extracts entities and relations from the killer-demo corpus using each
profile listed in LLM_PROFILES, then scores against hand-labeled
ground truth. Reports per-profile precision, recall, F1 for both
entities and relations.

Usage:
    # Run against all profiles (models must already be pulled in Ollama):
    uv run python scripts/bench_extraction.py

    # Run against a single profile:
    uv run python scripts/bench_extraction.py --profile llama31_8b

    # List available profiles:
    uv run python scripts/bench_extraction.py --list
"""
import argparse
import asyncio
import pathlib
import sys
import time
from dataclasses import dataclass, field

import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "landscape-dev")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from landscape.config import LLM_PROFILES, settings  # noqa: E402
from landscape.extraction.llm import extract  # noqa: E402
from landscape.extraction.schema import normalize_relation_type  # noqa: E402

CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "../tests/fixtures/killer_demo_corpus"


# -- Ground truth -----------------------------------------------------------
# Hand-labeled from the 7 killer-demo documents. Entity names are
# case-insensitive substrings to tolerate LLM variation (e.g. "Maya Chen"
# vs "Maya"). Relations are (subject_substring, rel_type, object_substring).

EXPECTED_ENTITIES: dict[str, set[str]] = {
    "01_org_chart": {"Helios Robotics", "Maya", "Diego", "Wei Zhang", "Anika Patel", "Platform Team", "Vision Team", "Fleet Operations", "Boston"},
    "02_platform_team_roster": {"Maya", "Tomás", "Priya", "Rahul", "Platform Team"},
    "03_project_aurora": {"Aurora", "PostgreSQL", "Priya", "Platform Team"},
    "04_aurora_db_decision": {"Maya", "PostgreSQL", "Aurora"},
    "05_project_sentinel": {"Sentinel", "PyTorch", "Diego", "Vision Team"},
    "06_project_beacon": {"Beacon", "Kafka", "Fleet Operations"},
    "07_q1_engineering_sync": {"Anika", "Aurora", "Sentinel", "Beacon"},
}

EXPECTED_RELATIONS: dict[str, set[tuple[str, str, str]]] = {
    "01_org_chart": {
        ("Maya", "LEADS", "Platform Team"),
        ("Diego", "LEADS", "Vision Team"),
        ("Wei Zhang", "LEADS", "Fleet Operations"),
        ("Helios Robotics", "LOCATED_IN", "Boston"),
    },
    "02_platform_team_roster": {
        ("Tomás", "MEMBER_OF", "Platform Team"),
        ("Priya", "MEMBER_OF", "Platform Team"),
        ("Rahul", "MEMBER_OF", "Platform Team"),
    },
    "03_project_aurora": {
        ("Aurora", "USES", "PostgreSQL"),
        ("Priya", "CREATED", "Aurora"),
        ("Aurora", "BELONGS_TO", "Platform Team"),
    },
    "04_aurora_db_decision": {
        ("Maya", "APPROVED", "PostgreSQL"),
    },
    "05_project_sentinel": {
        ("Sentinel", "USES", "PyTorch"),
        ("Diego", "LEADS", "Sentinel"),
    },
    "06_project_beacon": {
        ("Beacon", "USES", "Kafka"),
        ("Beacon", "BELONGS_TO", "Fleet Operations"),
    },
    "07_q1_engineering_sync": set(),
}


def _fuzzy_match(extracted: str, expected: str) -> bool:
    return expected.lower() in extracted.lower()


def _entity_recall(extracted_names: set[str], expected: set[str]) -> tuple[int, int]:
    hits = sum(
        1 for exp in expected
        if any(_fuzzy_match(ext, exp) for ext in extracted_names)
    )
    return hits, len(expected)


def _entity_precision(extracted_names: set[str], expected: set[str]) -> tuple[int, int]:
    hits = sum(
        1 for ext in extracted_names
        if any(_fuzzy_match(ext, exp) for exp in expected)
    )
    return hits, len(extracted_names) if extracted_names else 1


def _relation_match(
    extracted: tuple[str, str, str],
    expected: set[tuple[str, str, str]],
) -> bool:
    subj, rel, obj = extracted
    return any(
        _fuzzy_match(subj, es) and rel == er and _fuzzy_match(obj, eo)
        for es, er, eo in expected
    )


@dataclass
class ProfileResult:
    profile: str
    ollama_tag: str
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    relation_precision: float = 0.0
    relation_recall: float = 0.0
    relation_f1: float = 0.0
    total_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run_profile(profile_name: str) -> ProfileResult:
    profile = LLM_PROFILES[profile_name]
    result = ProfileResult(profile=profile_name, ollama_tag=profile.ollama_tag)

    # Temporarily swap settings to this profile so extract() sees the right
    # llm_model and llm_profile (the latter controls thinking mode).
    original_model = settings.llm_model
    original_profile = settings.llm_profile
    settings.llm_model = profile.ollama_tag
    settings.llm_profile = profile_name

    total_ent_prec_hits, total_ent_prec_total = 0, 0
    total_ent_rec_hits, total_ent_rec_total = 0, 0
    total_rel_prec_hits, total_rel_prec_total = 0, 0
    total_rel_rec_hits, total_rel_rec_total = 0, 0

    try:
        for path in sorted(CORPUS_DIR.glob("*.md")):
            doc_key = path.stem
            if doc_key not in EXPECTED_ENTITIES:
                continue

            text = path.read_text()
            t0 = time.perf_counter()
            try:
                extraction = extract(text)
            except Exception as e:
                result.errors.append(f"{doc_key}: {e}")
                continue
            result.total_time_s += time.perf_counter() - t0

            ext_names = {ent.name for ent in extraction.entities}
            exp_ents = EXPECTED_ENTITIES[doc_key]

            ph, pt = _entity_precision(ext_names, exp_ents)
            total_ent_prec_hits += ph
            total_ent_prec_total += pt

            rh, rt = _entity_recall(ext_names, exp_ents)
            total_ent_rec_hits += rh
            total_ent_rec_total += rt

            ext_rels = {
                (r.subject, normalize_relation_type(r.relation_type), r.object)
                for r in extraction.relations
            }
            exp_rels = EXPECTED_RELATIONS.get(doc_key, set())

            if exp_rels:
                rel_rec_hits = sum(
                    1 for er in exp_rels
                    if any(
                        _fuzzy_match(es, er[0])
                        and es_rel == er[1]
                        and _fuzzy_match(eo, er[2])
                        for es, es_rel, eo in ext_rels
                    )
                )
                total_rel_rec_hits += rel_rec_hits
                total_rel_rec_total += len(exp_rels)

            if ext_rels:
                all_expected = set()
                for k in EXPECTED_RELATIONS.values():
                    all_expected |= k
                rel_prec_hits = sum(
                    1 for er in ext_rels if _relation_match(er, exp_rels)
                )
                total_rel_prec_hits += rel_prec_hits
                total_rel_prec_total += len(ext_rels)

    finally:
        settings.llm_model = original_model
        settings.llm_profile = original_profile

    ep = total_ent_prec_hits / total_ent_prec_total if total_ent_prec_total else 0
    er = total_ent_rec_hits / total_ent_rec_total if total_ent_rec_total else 0
    rp = total_rel_prec_hits / total_rel_prec_total if total_rel_prec_total else 0
    rr = total_rel_rec_hits / total_rel_rec_total if total_rel_rec_total else 0

    result.entity_precision = ep
    result.entity_recall = er
    result.entity_f1 = _f1(ep, er)
    result.relation_precision = rp
    result.relation_recall = rr
    result.relation_f1 = _f1(rp, rr)
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM extraction quality benchmark")
    parser.add_argument("--profile", help="Run a single profile (default: all)")
    parser.add_argument("--list", action="store_true", help="List available profiles")
    args = parser.parse_args()

    if args.list:
        for name, p in sorted(LLM_PROFILES.items()):
            print(f"  {name:<20} {p.ollama_tag:<24} {p.notes}")
        return

    profiles = [args.profile] if args.profile else sorted(LLM_PROFILES.keys())
    for p in profiles:
        if p not in LLM_PROFILES:
            print(f"Unknown profile: {p}. Run with --list to see options.", file=sys.stderr)
            sys.exit(1)

    results: list[ProfileResult] = []
    for p in profiles:
        print(f"Benchmarking {p} ({LLM_PROFILES[p].ollama_tag})...", flush=True)
        r = run_profile(p)
        results.append(r)
        if r.errors:
            for e in r.errors:
                print(f"  WARNING: {e}", file=sys.stderr)
        print(f"  Done in {r.total_time_s:.1f}s", flush=True)

    print()
    header = (
        f"{'Profile':<20} {'Tag':<24} "
        f"{'Ent P':>6} {'Ent R':>6} {'Ent F1':>6}  "
        f"{'Rel P':>6} {'Rel R':>6} {'Rel F1':>6}  "
        f"{'Time':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.profile:<20} {r.ollama_tag:<24} "
            f"{r.entity_precision:>6.1%} {r.entity_recall:>6.1%} {r.entity_f1:>6.1%}  "
            f"{r.relation_precision:>6.1%} {r.relation_recall:>6.1%} {r.relation_f1:>6.1%}  "
            f"{r.total_time_s:>6.1f}s"
        )

    if len(results) > 1:
        best = max(results, key=lambda r: r.relation_f1)
        print(f"\nBest relation F1: {best.profile} ({best.relation_f1:.1%})")


if __name__ == "__main__":
    main()
