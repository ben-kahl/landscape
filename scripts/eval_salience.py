#!/usr/bin/env python3
"""Measure salience-select precision/recall against a labeled set.

Usage:
    uv run python scripts/eval_salience.py tests/fixtures/salience_eval/labeled_sessions.json
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# Default to the host-reachable Ollama when run against a local docker stack;
# the config default (http://ollama:11434) is only resolvable inside the
# compose network. Matches the other host-run scripts in scripts/.
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from landscape.conversation_ingestion import ConversationTurn  # noqa: E402
from landscape.extraction.salience import select_salient  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: eval_salience.py LABELED_SESSIONS_JSON", file=sys.stderr)
        return 2

    path = pathlib.Path(argv[1])
    sessions = json.loads(path.read_text())
    tp = fp = fn = 0
    for session in sessions:
        turns = [
            ConversationTurn(
                session["session_id"],
                turn["turn_id"],
                turn["role"],
                turn["text"],
            )
            for turn in session["turns"]
        ]
        gold = set(session["memory_worthy_turn_ids"])
        picked = {item.turn_id for item in select_salient(turns)}
        tp += len(picked & gold)
        fp += len(picked - gold)
        fn += len(gold - picked)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"salience precision={precision:.2f} recall={recall:.2f} (tp={tp} fp={fp} fn={fn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
