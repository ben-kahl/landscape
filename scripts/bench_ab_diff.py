#!/usr/bin/env python3
"""Compare two bench_ab.py result files and print a regression table.

Usage:
    uv run python scripts/bench_ab_diff.py bench_ab_main.json bench_ab_redesign.json
    uv run python scripts/bench_ab_diff.py bench_ab_main.json bench_ab_redesign.json \\
        --regression-threshold-quality 0.05 \\
        --regression-threshold-tokens 0.10

Exits 0 if no regressions exceed their thresholds, non-zero otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys


def _pct_change(a: float, b: float) -> str:
    if a == 0:
        return "N/A"
    return f"{(b - a) / a * 100:+.1f}%"


def _flag(
    name: str,
    a: float,
    b: float,
    *,
    lower_is_better: bool = False,
    threshold: float | None = None,
) -> tuple[str, bool]:
    delta = b - a
    norm_delta = delta / max(abs(a), 1e-9)
    if lower_is_better:
        regression = delta > 0 and threshold is not None and norm_delta > threshold
    else:
        regression = delta < 0 and threshold is not None and abs(norm_delta) > threshold
    symbol = "✗" if regression else "✓"
    return symbol, regression


def _row(
    label: str,
    a,
    b,
    *,
    fmt: str = ".4f",
    lower_is_better: bool = False,
    threshold: float | None = None,
    informational: bool = False,
) -> tuple[str, bool]:
    if isinstance(a, int) and isinstance(b, int):
        delta_str = f"{b - a:+d}"
        a_str = str(a)
        b_str = str(b)
    else:
        delta_str = f"{b - a:+.4f}"
        a_str = f"{a:{fmt}}"
        b_str = f"{b:{fmt}}"

    if informational:
        symbol = "~"
        regression = False
    else:
        symbol, regression = _flag(
            label, float(a), float(b),
            lower_is_better=lower_is_better,
            threshold=threshold,
        )
    line = f"  {label:<30} {a_str:>8}  →  {b_str:>8}   {delta_str:>8}  {symbol}"
    return line, regression


def _main(args: argparse.Namespace) -> int:
    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.candidate) as f:
        candidate = json.load(f)

    branch_a = baseline.get("branch", args.baseline)
    branch_b = candidate.get("branch", args.candidate)

    lines = [f"Landscape A/B Comparison: {branch_a} → {branch_b}", ""]
    regressions: list[str] = []

    qt = args.regression_threshold_quality
    tt = args.regression_threshold_tokens

    # --- LongMemEval ---
    if "longmemeval" in baseline and "longmemeval" in candidate:
        lines.append("RETRIEVAL QUALITY")
        for metric in ("hit_at_1", "hit_at_5", "hit_at_10"):
            a = baseline["longmemeval"].get(metric, 0.0)
            b = candidate["longmemeval"].get(metric, 0.0)
            line, reg = _row(metric, a, b, threshold=qt)
            lines.append(line)
            if reg:
                regressions.append(f"longmemeval.{metric}")

        for metric in ("avg_ingest_s", "avg_query_s"):
            a = baseline["longmemeval"].get(metric, 0.0)
            b = candidate["longmemeval"].get(metric, 0.0)
            line, _ = _row(metric, a, b, lower_is_better=True, informational=True)
            lines.append(line)

        ollama_a = baseline["longmemeval"].get("ollama_extraction_tokens", {}).get("avg", 0.0)
        ollama_b = candidate["longmemeval"].get("ollama_extraction_tokens", {}).get("avg", 0.0)
        line, reg = _row(
            "ollama_extraction_tokens.avg", ollama_a, ollama_b,
            lower_is_better=True, threshold=tt,
        )
        lines.append(line)
        if reg:
            regressions.append("longmemeval.ollama_extraction_tokens.avg")

        resp_a = baseline["longmemeval"].get("avg_response_tokens", 0.0)
        resp_b = candidate["longmemeval"].get("avg_response_tokens", 0.0)
        line, reg = _row(
            "avg_response_tokens", resp_a, resp_b,
            lower_is_better=True, threshold=tt,
        )
        lines.append(line)
        if reg:
            regressions.append("longmemeval.avg_response_tokens")
        lines.append("")

    # --- Killer demo ---
    lines.append("KILLER DEMO")
    kd_a = baseline.get("killer_demo", {})
    kd_b = candidate.get("killer_demo", {})
    pa = kd_a.get("pass_rate", 0.0)
    pb = kd_b.get("pass_rate", 0.0)
    pass_str_a = f"{kd_a.get('assertions_passed', 0)}/{kd_a.get('assertions_total', 0)}"
    pass_str_b = f"{kd_b.get('assertions_passed', 0)}/{kd_b.get('assertions_total', 0)}"
    _, reg = _flag("pass_rate", pa, pb, threshold=qt)
    symbol = "✗" if reg else "✓"
    lines.append(f"  {'pass_rate':<30} {pass_str_a:>8}  →  {pass_str_b:>8}            {symbol}")
    if reg:
        regressions.append("killer_demo.pass_rate")

    resp_a = kd_a.get("avg_response_tokens", 0.0)
    resp_b = kd_b.get("avg_response_tokens", 0.0)
    line, reg = _row(
        "avg_response_tokens", resp_a, resp_b,
        lower_is_better=True, threshold=tt,
    )
    lines.append(line)
    if reg:
        regressions.append("killer_demo.avg_response_tokens")
    lines.append("")

    # --- Supersession ---
    lines.append("SUPERSESSION")
    ss_a = baseline.get("supersession", {})
    ss_b = candidate.get("supersession", {})
    pa = ss_a.get("pass_rate", 0.0)
    pb = ss_b.get("pass_rate", 0.0)
    pass_str_a = f"{ss_a.get('scenarios_passed', 0)}/{ss_a.get('scenarios_total', 0)}"
    pass_str_b = f"{ss_b.get('scenarios_passed', 0)}/{ss_b.get('scenarios_total', 0)}"
    _, reg = _flag("pass_rate", pa, pb, threshold=qt)
    symbol = "✗" if reg else "✓"
    lines.append(f"  {'pass_rate':<30} {pass_str_a:>8}  →  {pass_str_b:>8}            {symbol}")
    if reg:
        regressions.append("supersession.pass_rate")
    lines.append("")

    # --- Token totals ---
    lines.append("TOKEN TOTALS")
    tt_a = baseline.get("token_totals", {})
    tt_b = candidate.get("token_totals", {})

    resp_a = tt_a.get("response_tokens_total", 0)
    resp_b = tt_b.get("response_tokens_total", 0)
    line, reg = _row(
        "response_tokens_total", resp_a, resp_b,
        lower_is_better=True, threshold=tt,
    )
    lines.append(line)
    if reg:
        regressions.append("token_totals.response_tokens_total")

    oll_a = tt_a.get("ollama_tokens_total", 0)
    oll_b = tt_b.get("ollama_tokens_total", 0)
    line, reg = _row(
        "ollama_tokens_total", oll_a, oll_b,
        lower_is_better=True, threshold=tt,
    )
    lines.append(line)
    if reg:
        regressions.append("token_totals.ollama_tokens_total")

    lines.append("")

    # --- Summary ---
    if regressions:
        lines.append(f"Exit: FAIL — regressions beyond threshold: {', '.join(regressions)}")
        exit_code = 1
    else:
        lines.append("Exit: PASS (no regressions beyond threshold)")
        exit_code = 0

    print("\n".join(lines))
    return exit_code


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two bench_ab.py result files")
    p.add_argument("baseline", help="Baseline JSON file (e.g. bench_ab_main.json)")
    p.add_argument("candidate", help="Candidate JSON file (e.g. bench_ab_redesign.json)")
    p.add_argument(
        "--regression-threshold-quality",
        type=float,
        default=0.05,
        dest="regression_threshold_quality",
        help="Flag if quality metric drops by more than this fraction (default: 0.05 = 5%%)",
    )
    p.add_argument(
        "--regression-threshold-tokens",
        type=float,
        default=0.10,
        dest="regression_threshold_tokens",
        help="Flag if token count increases by more than this fraction (default: 0.10 = 10%%)",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(_main(_parse_args()))
