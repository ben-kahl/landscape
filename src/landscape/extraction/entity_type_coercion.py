"""Embedding-based coercion of agent-supplied entity_type strings to the
canonical vocabulary.

Small local LLMs (Llama 3.1) are non-deterministic about entity type
phrasing. The string-synonym map catches obvious cases (COMPANY -> Organization,
USER -> Person) but misses semantic near-misses (e.g. "SoftwareEngineer" which
clearly maps to Person). This module embeds the incoming entity_type, compares
against precomputed embeddings of the 8 canonical types, and coerces to the
nearest canonical type above a cosine threshold.

Algorithm
---------
1. Canonical fast path: if ``raw`` (case-insensitive, stripped) matches any
   canonical type exactly, return ``(canonical, 1.0)`` immediately.

2. String-synonym fast path: check a hard-coded synonym map. If a match is
   found, return ``(canonical, 1.0)`` without touching the encoder.

3. Novel type path: embed the raw type string + its descriptor phrase, compare
   against canonical embeddings. If top similarity >= COERCION_THRESHOLD (0.55),
   coerce to that canonical. Otherwise preserve the novel type — return
   ``(raw, 0.0)``.

Note: there is no "canonical override" path unlike rel_type coercion. For
entity types, a direct case-insensitive match to the vocab wins outright; we
don't second-guess it.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Minimum cosine similarity for a novel type to be coerced to a canonical.
# Below this threshold the novel type is preserved.
COERCION_THRESHOLD: float = 0.55

# ---------------------------------------------------------------------------
# Canonical entity-type vocabulary
# ---------------------------------------------------------------------------

ENTITY_TYPE_VOCAB: frozenset[str] = frozenset(
    [
        "Person",
        "Organization",
        "Project",
        "Technology",
        "Location",
        "Concept",
        "Event",
        "Document",
    ]
)

# Lowercase → canonical mapping for the fast path
_LOWER_TO_CANONICAL: dict[str, str] = {t.lower(): t for t in ENTITY_TYPE_VOCAB}

# ---------------------------------------------------------------------------
# Descriptor phrases for richer embedding signal
# ---------------------------------------------------------------------------
# Bare type names like "Person" and "Concept" can have overlapping embeddings.
# Appending a short human-readable descriptor gives the sentence-transformer
# more semantic signal to distinguish them.

CANONICAL_DESCRIPTORS: dict[str, str] = {
    "Person": (
        "Person — individual human, employee, leader, author, founder, member"
    ),
    "Organization": (
        "Organization — company, team, department, group, agency, institution"
    ),
    "Project": (
        "Project — named initiative, codename, product effort, program of work"
    ),
    "Technology": (
        "Technology — software, framework, library, language, tool, platform, database"
    ),
    "Location": (
        "Location — physical place, office, city, country, region, building"
    ),
    "Concept": (
        "Concept — abstract idea, methodology, principle, theory, pattern"
    ),
    "Event": (
        "Event — meeting, conference, incident, milestone, decision point"
    ),
    "Document": (
        "Document — report, memo, spec, contract, paper, file, document"
    ),
}

# ---------------------------------------------------------------------------
# String-synonym map
# ---------------------------------------------------------------------------
# Maps lowercase synonyms → canonical entity type.
# Only string-obvious cases belong here; semantic near-misses are handled by
# the embedding path.

_SYNONYMS: dict[str, str] = {
    # Person
    "individual": "Person",
    "user": "Person",
    "human": "Person",
    "employee": "Person",
    "engineer": "Person",
    "softwareengineer": "Person",
    "softwaredeveloper": "Person",
    "developer": "Person",
    "manager": "Person",
    "founder": "Person",
    "author": "Person",
    "member": "Person",
    "leader": "Person",
    "director": "Person",
    "executive": "Person",
    "ceo": "Person",
    "cto": "Person",
    "cfo": "Person",
    "researcher": "Person",
    "scientist": "Person",
    "analyst": "Person",
    # Organization
    "org": "Organization",
    "organisation": "Organization",
    "company": "Organization",
    "corp": "Organization",
    "corporation": "Organization",
    "firm": "Organization",
    "team": "Organization",
    "department": "Organization",
    "group": "Organization",
    "agency": "Organization",
    "institution": "Organization",
    "startup": "Organization",
    "enterprise": "Organization",
    # Project
    "initiative": "Project",
    "program": "Project",
    "programme": "Project",
    "effort": "Project",
    "product": "Project",
    "codename": "Project",
    # Technology
    "tech": "Technology",
    "framework": "Technology",
    "library": "Technology",
    "language": "Technology",
    "tool": "Technology",
    "platform": "Technology",
    "database": "Technology",
    "software": "Technology",
    "service": "Technology",
    "system": "Technology",
    "api": "Technology",
    # Location
    "place": "Location",
    "city": "Location",
    "country": "Location",
    "region": "Location",
    "office": "Location",
    "building": "Location",
    "address": "Location",
    # Concept
    "idea": "Concept",
    "methodology": "Concept",
    "principle": "Concept",
    "theory": "Concept",
    "pattern": "Concept",
    "approach": "Concept",
    "strategy": "Concept",
    # Event
    "meeting": "Event",
    "conference": "Event",
    "incident": "Event",
    "milestone": "Event",
    "decision": "Event",
    "ceremony": "Event",
    # Document
    "report": "Document",
    "memo": "Document",
    "spec": "Document",
    "specification": "Document",
    "contract": "Document",
    "paper": "Document",
    "file": "Document",
    "doc": "Document",
    "article": "Document",
}


# ---------------------------------------------------------------------------
# Singleton lazy-init canonical embeddings
# ---------------------------------------------------------------------------


class _CanonicalEmbeddings:
    """Lazily computes and caches embeddings for all canonical entity types.

    Deferred until first use so import time stays fast and the encoder
    model does not need to be loaded before the app lifespan starts.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] | None = None

    def _load(self) -> None:
        from landscape.embeddings import encoder

        canonical_sorted = sorted(ENTITY_TYPE_VOCAB)
        phrases = [CANONICAL_DESCRIPTORS[t] for t in canonical_sorted]
        vectors = encoder.embed_documents(phrases)
        self._embeddings = dict(zip(canonical_sorted, vectors))
        logger.debug(
            "Canonical entity_type embeddings loaded for %d types", len(self._embeddings)
        )

    def get(self) -> dict[str, list[float]]:
        if self._embeddings is None:
            self._load()
        return self._embeddings  # type: ignore[return-value]


_canonical_embeddings = _CanonicalEmbeddings()


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coerce_entity_type(raw: str) -> tuple[str, float]:
    """Coerce an agent- or LLM-supplied entity_type to the canonical vocabulary.

    Returns ``(canonical_type, confidence)`` where confidence is in [0, 1].

    Confidence meanings:
        1.0 — exact canonical match or string-synonym hit (no embedding needed)
        0.55–1.0 — embedding cosine similarity of the best canonical match
        0.0 — no match above threshold; raw type preserved unchanged

    See module docstring for the full algorithm.
    """
    if not raw:
        return "Concept", 0.0

    stripped = raw.strip()

    # ------------------------------------------------------------------
    # Path 1: case-insensitive canonical match — wins outright
    # ------------------------------------------------------------------
    lower = stripped.lower()
    if lower in _LOWER_TO_CANONICAL:
        canonical = _LOWER_TO_CANONICAL[lower]
        logger.debug("coerce_entity_type: canonical match %r -> %r", raw, canonical)
        return canonical, 1.0

    # ------------------------------------------------------------------
    # Path 2: string-synonym fast path
    # ------------------------------------------------------------------
    if lower in _SYNONYMS:
        canonical = _SYNONYMS[lower]
        logger.debug("coerce_entity_type: synonym hit %r -> %r", raw, canonical)
        return canonical, 1.0

    # ------------------------------------------------------------------
    # Path 3: embedding-based coercion
    # Embed the raw type string; if top similarity >= COERCION_THRESHOLD,
    # coerce. Otherwise preserve the novel type.
    # ------------------------------------------------------------------
    from landscape.embeddings import encoder

    probe_phrase = stripped
    probe_vec = encoder.encode(probe_phrase)

    canonical_embeddings = _canonical_embeddings.get()
    scores = {t: _cosine(probe_vec, v) for t, v in canonical_embeddings.items()}
    best_type = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]

    if best_score >= COERCION_THRESHOLD:
        logger.debug(
            "coerce_entity_type: embedding coercion %r -> %r (sim=%.3f)",
            raw,
            best_type,
            best_score,
        )
        return best_type, best_score

    # Novel type with no close canonical match — preserve it.
    logger.debug(
        "coerce_entity_type: novel type %r kept (best_sim=%.3f < threshold=%.3f)",
        raw,
        best_score,
        COERCION_THRESHOLD,
    )
    return stripped, 0.0
