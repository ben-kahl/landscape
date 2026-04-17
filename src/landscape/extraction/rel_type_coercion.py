"""Embedding-based coercion of agent-supplied rel_type strings to the
canonical vocabulary.

Small local LLMs (Llama 3.1) are non-deterministic about relation
phrasing. The string-synonym map in schema.normalize_relation_type
catches obvious cases (EMPLOYED_BY -> WORKS_FOR) but misses semantic
near-misses (LOCATED_IN for an org change). This module embeds the
incoming rel_type, compares against precomputed embeddings of the 10
canonical types, and coerces to the nearest canonical type above a
cosine threshold.

Algorithm
---------
1. String-synonym fast path: call normalize_relation_type(raw). If the
   result is in RELATION_VOCAB AND it changed, the synonym map fired —
   return (canonical, 1.0) immediately without touching the encoder.

2. Canonical override path: if the string-synonym result is already a
   canonical type, embed the raw string + context phrase and compare
   against each canonical embedding. If the top-scoring alternative
   beats the LLM-supplied canonical's score by >= COERCION_MARGIN,
   coerce. This catches "LOCATED_IN" used to mean employment.

3. Novel type path: if the string-synonym result is NOT in the canonical
   vocab (LLM invented something new), embed and compare. If top
   similarity >= COERCION_THRESHOLD, coerce to that canonical. Otherwise
   preserve the novel type.

Design note on COERCION_MARGIN (0.05)
--------------------------------------
We only OVERRIDE the LLM's explicit choice when an alternative canonical
type is *meaningfully* closer in semantic space, not when scores are
essentially tied. This protects cases where the LLM picked the right
type and the embeddings are ambiguous.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from landscape.extraction.schema import RELATION_VOCAB, normalize_relation_type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Minimum cosine similarity for a novel (non-canonical) type to be coerced
# to a canonical. Below this threshold we preserve the novel type.
COERCION_THRESHOLD: float = 0.55

# Minimum similarity margin by which an alternative canonical must beat the
# LLM-supplied canonical before we override the LLM's choice.
COERCION_MARGIN: float = 0.05

# ---------------------------------------------------------------------------
# Descriptor phrases for richer embedding signal
# ---------------------------------------------------------------------------
# Bare "WORKS_FOR" and "LOCATED_IN" have very similar token distributions.
# Embedding a short human-readable descriptor gives the sentence-transformer
# more semantic signal to distinguish employment from location.

CANONICAL_DESCRIPTORS: dict[str, str] = {
    "WORKS_FOR": (
        "WORKS_FOR — employment, works at, employed by, joined company, "
        "hired by, affiliated with, staff at"
    ),
    "LEADS": (
        "LEADS — manages, runs, heads, directs, oversees, in charge of team"
    ),
    "MEMBER_OF": (
        "MEMBER_OF — member of group, part of team, belongs to community, "
        "on the team, in the group"
    ),
    "REPORTS_TO": (
        "REPORTS_TO — direct manager, supervisor, direct report, reports to"
    ),
    "APPROVED": (
        "APPROVED — sign-off, authorized, approved, green-lit, ratified, "
        "gave approval for"
    ),
    "USES": (
        "USES — uses technology, depends on, built on, runs on, powered by, "
        "technology dependency"
    ),
    "BELONGS_TO": (
        "BELONGS_TO — subsidiary of, owned by, division of, parent organization, "
        "belongs to parent"
    ),
    "LOCATED_IN": (
        "LOCATED_IN — physical location, office in city, headquartered in, "
        "based in geographic place"
    ),
    "CREATED": (
        "CREATED — authored, built, developed, wrote, founded, invented, "
        "created artifact"
    ),
    "RELATED_TO": (
        "RELATED_TO — general relationship, connected to, associated with, "
        "related, linked"
    ),
}


# ---------------------------------------------------------------------------
# Singleton lazy-init canonical embeddings
# ---------------------------------------------------------------------------


class _CanonicalEmbeddings:
    """Lazily computes and caches embeddings for all canonical rel types.

    Deferred until first use so import time stays fast and the encoder
    model does not need to be loaded before the app lifespan starts.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] | None = None

    def _load(self) -> None:
        from landscape.embeddings import encoder

        phrases = [CANONICAL_DESCRIPTORS[t] for t in sorted(RELATION_VOCAB)]
        canonical_sorted = sorted(RELATION_VOCAB)
        vectors = encoder.embed_documents(phrases)
        self._embeddings = dict(zip(canonical_sorted, vectors))
        logger.debug("Canonical rel_type embeddings loaded for %d types", len(self._embeddings))

    def get(self) -> dict[str, list[float]]:
        if self._embeddings is None:
            self._load()
        return self._embeddings  # type: ignore[return-value]


_canonical_embeddings = _CanonicalEmbeddings()


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two already-normalized or raw float vectors.

    HuggingFaceEmbeddings is configured with normalize_embeddings=True, so
    dot product == cosine similarity. We compute both ways gracefully.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coerce_rel_type(raw: str) -> tuple[str, float]:
    """Coerce an LLM-supplied rel_type to the canonical vocabulary.

    Returns ``(canonical_type, confidence)`` where confidence is in (0, 1].

    Confidence meanings:
        1.0 — string-synonym exact hit (no embedding needed)
        0.0–1.0 — embedding cosine similarity of the best match

    See module docstring for the full three-path algorithm.
    """
    if not raw:
        return "RELATED_TO", 0.0

    # ------------------------------------------------------------------
    # Path 1: string-synonym fast path
    # ------------------------------------------------------------------
    normalized = normalize_relation_type(raw)
    upper_raw = raw.strip().upper().replace(" ", "_")

    # If normalize_relation_type changed the value AND it's now canonical,
    # the synonym map fired — return immediately without embeddings.
    if normalized != upper_raw and normalized in RELATION_VOCAB:
        logger.debug("coerce_rel_type: string-synonym hit %r -> %r", raw, normalized)
        return normalized, 1.0

    # ------------------------------------------------------------------
    # Path 2: canonical override — LLM supplied a valid canonical type
    #         but may have picked the wrong one semantically.
    # ------------------------------------------------------------------
    canonical_embeddings = _canonical_embeddings.get()

    if normalized in RELATION_VOCAB:
        # Embed the raw token itself (no generic context suffix — adding
        # "relation between two entities" pulls every token toward RELATED_TO
        # because that canonical's descriptor contains "general relationship").
        from landscape.embeddings import encoder

        probe_phrase = upper_raw
        probe_vec = encoder.encode(probe_phrase)

        # Score all canonicals
        scores = {t: _cosine(probe_vec, v) for t, v in canonical_embeddings.items()}
        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]
        llm_score = scores[normalized]

        if best_type != normalized and (best_score - llm_score) >= COERCION_MARGIN:
            logger.debug(
                "coerce_rel_type: canonical override %r -> %r "
                "(best=%.3f llm_canonical=%.3f margin=%.3f)",
                raw,
                best_type,
                best_score,
                llm_score,
                best_score - llm_score,
            )
            return best_type, best_score

        # LLM's choice holds
        return normalized, llm_score

    # ------------------------------------------------------------------
    # Path 3: novel type — LLM invented something not in the vocab.
    # Embed the raw token; for truly novel tokens (e.g. JOINED_AS_EMPLOYEE)
    # the token itself carries enough semantic signal.
    # ------------------------------------------------------------------
    from landscape.embeddings import encoder

    probe_phrase = upper_raw
    probe_vec = encoder.encode(probe_phrase)

    scores = {t: _cosine(probe_vec, v) for t, v in canonical_embeddings.items()}
    best_type = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]

    if best_score >= COERCION_THRESHOLD:
        logger.debug(
            "coerce_rel_type: novel type %r coerced to %r (sim=%.3f)",
            raw,
            best_type,
            best_score,
        )
        return best_type, best_score

    # Novel type with no close canonical match — preserve it.
    logger.debug(
        "coerce_rel_type: novel type %r kept (best_sim=%.3f < threshold=%.3f)",
        raw,
        best_score,
        COERCION_THRESHOLD,
    )
    return normalized, best_score
