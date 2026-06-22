#!/usr/bin/env python3
"""Build the W&B support demo corpus from REAL public GitHub data.

Pulls a curated set of *closed* public W&B issues + matching CHANGELOG fix
excerpts into tests/fixtures/wandb_support_corpus/, one markdown doc each, plus a
manifest.json recording every source URL. Faithful trims only — no fabricated
content. Re-run anytime to refresh/extend the corpus.

This is EXTRACTION-FREE: it only calls the GitHub API (via `gh`) and writes
files. It does NOT call the LLM, ingest, or touch Neo4j/Qdrant. Seeding the graph
is a separate step (scripts/seed_wandb_support.py).

Requires: `gh` authenticated (`gh auth status`).

Usage:
    uv run python scripts/fetch_wandb_issues.py
"""
from __future__ import annotations

import base64
import json
import pathlib
import re
import subprocess
import sys

CORPUS = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/wandb_support_corpus"

# Curated real closed issues — a difficulty/topic spread, login-leaning.
# (repo, issue_number, topic, difficulty)
ISSUES: list[tuple[str, int, str, str]] = [
    ("wandb/wandb", 11532, "login", "med-hard"),   # init hangs if entity != logged-in user
    ("wandb/wandb", 9580, "login", "med"),         # Login error: no attribute 'disabled'
    ("wandb/wandb", 5580, "auth", "med"),          # metaflow AuthenticationError
    ("wandb/wandb", 7118, "auth", "easy-med"),     # user not logged in (401)
    ("wandb/wandb", 673, "auth", "med"),           # permission denied calling W&B API
    ("wandb/wandb", 3845, "auth", "med"),          # wandb login in GitHub Action CI
    ("wandb/wandb", 4500, "serialization", "easy"),  # artifact metadata Tensor not serializable
    ("wandb/wandb", 2063, "serialization", "easy"),  # matplotlib Figure not JSON serializable
    ("wandb/wandb", 7275, "serialization", "easy-med"),  # Object of type Error not serializable
    ("wandb/wandb", 7671, "init", "med"),          # run init timed out 90s
    ("wandb/wandb", 5396, "resume", "med"),        # can't resume run with correct id
    ("wandb/weave", 4639, "serialization", "med"),  # pydantic structured-output serialization
]

BODY_CAP = 2200
COMMENTS_CAP = 2600
CHANGELOG_KEYWORDS = [
    "login", "api key", "api_key", "auth", "serializ", "artifact", "init", "relogin",
]
CHANGELOG_MAX_LINES = 30


def _gh_json(args: list[str]) -> dict:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:50]


def _trim(text: str, cap: int) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    return text if len(text) <= cap else text[:cap].rstrip() + "\n\n…(trimmed)"


def write_issue_doc(idx: int, repo: str, topic: str, difficulty: str, data: dict) -> dict:
    num = data["number"]
    title = data["title"].strip()
    url = data["url"]
    labels = [label["name"] for label in data.get("labels", [])]
    fname = f"{idx:02d}_{repo.split('/')[1]}_{num}_{_slug(title)}.md"

    comments = data.get("comments") or []
    comment_blocks = []
    running = 0
    for c in comments:
        author = (c.get("author") or {}).get("login", "user")
        body = (c.get("body") or "").strip()
        if not body:
            continue
        block = f"**{author}:** {body}"
        running += len(block)
        comment_blocks.append(block)
        if running > COMMENTS_CAP:
            comment_blocks.append("…(further comments trimmed)")
            break
    comments_md = "\n\n".join(comment_blocks) if comment_blocks else "_(no comments)_"

    doc = (
        f"# {title}\n\n"
        f"- Source: {url}\n"
        f"- Repo: {repo} · Issue #{num} · State: {data['state'].lower()} "
        f"(closed {data.get('closedAt', '')[:10]})\n"
        f"- Labels: {', '.join(labels) if labels else 'none'}\n"
        f"- Topic: {topic} · Difficulty: {difficulty}\n\n"
        f"## Report\n\n{_trim(data.get('body', ''), BODY_CAP)}\n\n"
        f"## Discussion / resolution\n\n{_trim(comments_md, COMMENTS_CAP + 400)}\n"
    )
    (CORPUS / fname).write_text(doc)
    return {
        "file": fname, "repo": repo, "issue": num, "url": url, "title": title,
        "source": "github_issue", "topic": topic, "difficulty": difficulty,
        "closed_at": data.get("closedAt"), "labels": labels,
    }


def fetch_changelog_excerpts() -> dict | None:
    """Grab real CHANGELOG fix lines (with their version header) for relevant
    keywords, so the graph has genuine fixed-in-version facts."""
    try:
        raw = subprocess.run(
            ["gh", "api", "repos/wandb/wandb/contents/CHANGELOG.md"],
            capture_output=True, text=True, check=True,
        ).stdout
        content = base64.b64decode(json.loads(raw)["content"]).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] CHANGELOG fetch failed: {e}", file=sys.stderr)
        return None

    version_re = re.compile(r"^#{1,3}\s*\[?v?(\d+\.\d+\.\d+)")
    cur_ver = None
    hits: list[tuple[str, str]] = []
    for line in content.splitlines():
        m = version_re.match(line.strip())
        if m:
            cur_ver = m.group(1)
            continue
        low = line.lower()
        is_bullet = line.strip().startswith(("-", "*"))
        if cur_ver and is_bullet and any(k in low for k in CHANGELOG_KEYWORDS):
            hits.append((cur_ver, line.strip().lstrip("-* ").strip()))
        if len(hits) >= CHANGELOG_MAX_LINES:
            break

    if not hits:
        return None
    lines_md = "\n".join(f"- **v{ver}**: {text}" for ver, text in hits)
    doc = (
        "# W&B SDK CHANGELOG — relevant fix excerpts\n\n"
        "- Source: https://github.com/wandb/wandb/blob/main/CHANGELOG.md\n"
        "- These are real, verbatim CHANGELOG bullets for login/auth/"
        "serialization/artifact/init fixes, with the release they shipped in.\n\n"
        "## Fixes by version\n\n" + lines_md + "\n"
    )
    fname = "00_wandb_changelog_fix_excerpts.md"
    (CORPUS / fname).write_text(doc)
    return {"file": fname, "repo": "wandb/wandb", "url":
            "https://github.com/wandb/wandb/blob/main/CHANGELOG.md",
            "title": "CHANGELOG fix excerpts", "source": "github_changelog",
            "topic": "release-notes", "lines": len(hits)}


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    cl = fetch_changelog_excerpts()
    if cl:
        manifest.append(cl)
        print(f"changelog: {cl['file']} ({cl['lines']} fix lines)")

    for i, (repo, num, topic, diff) in enumerate(ISSUES, start=1):
        try:
            data = _gh_json([
                "issue", "view", str(num), "--repo", repo,
                "--json", "number,title,url,state,closedAt,labels,body,comments",
            ])
        except subprocess.CalledProcessError as e:
            print(f"  [warn] {repo}#{num} fetch failed: {e.stderr.strip()[:120]}", file=sys.stderr)
            continue
        entry = write_issue_doc(i, repo, topic, diff, data)
        manifest.append(entry)
        print(f"{entry['file']}  [{topic}/{diff}]  ({len(data.get('comments') or [])} comments)")

    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(manifest)} docs + manifest.json to {CORPUS}")


if __name__ == "__main__":
    main()
