"""Defensive read-only Cypher validator for the MCP graph_query tool.

Belt-and-suspenders approach: strip comments and string literals, then
keyword-scan for any write/admin operations. The Neo4j driver also runs
the query inside a read-only transaction, so even a missed keyword gets
rejected at the DB layer.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class CypherWriteAttempted(ValueError):
    """Raised when a Cypher query contains write or admin operations."""


# ---------------------------------------------------------------------------
# Write keywords (whole-word, case-insensitive denylist)
# ---------------------------------------------------------------------------

_WRITE_KEYWORDS: tuple[str, ...] = (
    "CREATE",
    "MERGE",
    "DELETE",
    "SET",
    "REMOVE",
    "DROP",
    "FOREACH",
    "LOAD",
    "USE",
)

_WRITE_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(_WRITE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Matches dangerous CALL targets:
#   CALL db.create*  /  CALL apoc.create.*  /  CALL apoc.merge.*  etc.
_DANGEROUS_CALL_RE = re.compile(
    r"\bCALL\s+(db|apoc)\.(create|merge|delete|refactor|set|remove|trigger|periodic)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stripping helpers
# ---------------------------------------------------------------------------

# Line comments:  // ... to end of line
_LINE_COMMENT_RE = re.compile(r"//[^\n]*", re.MULTILINE)
# Block comments:  /* ... */  (non-greedy, dotall)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# Single-quoted string literals (handles \' escapes)
_SINGLE_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'", re.DOTALL)
# Double-quoted string literals (handles \" escapes)
_DOUBLE_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"', re.DOTALL)


def _strip_non_code(cypher: str) -> str:
    """Remove comments and string literals from *cypher*.

    Replacements use a space rather than empty string to preserve word
    boundaries around the surrounding tokens.
    """
    text = _LINE_COMMENT_RE.sub(" ", cypher)
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _SINGLE_QUOTED_RE.sub(" ", text)
    text = _DOUBLE_QUOTED_RE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_read_only(cypher: str) -> None:
    """Raise :exc:`CypherWriteAttempted` if *cypher* contains any write keyword.

    Strips line comments (``// ...``), block comments (``/* ... */``), and
    string literals (``'...'`` and ``"..."``) before keyword scanning, so a
    query like ``MATCH (n {name: 'CREATE me'}) RETURN n`` is allowed.

    ``CALL`` is permitted in general (read procedures such as
    ``CALL db.labels()`` are valid), but a ``CALL`` that targets a known
    write/admin namespace (``db.create*``, ``apoc.merge.*``, etc.) is
    rejected.
    """
    sanitized = _strip_non_code(cypher)

    # 1. Check for dangerous CALL targets first (more specific check)
    m = _DANGEROUS_CALL_RE.search(sanitized)
    if m:
        raise CypherWriteAttempted(
            f"Write operation not allowed in read-only Cypher: '{m.group(0).strip()}'"
        )

    # 2. Check for write keywords
    m = _WRITE_KEYWORD_RE.search(sanitized)
    if m:
        raise CypherWriteAttempted(
            f"Write operation not allowed in read-only Cypher: '{m.group(0).upper()}'"
        )
