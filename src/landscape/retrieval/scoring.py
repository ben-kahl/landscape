import math
from dataclasses import dataclass
from datetime import datetime

from landscape.config import settings


@dataclass(frozen=True)
class ScoringWeights:
    alpha: float  # vector similarity
    beta: float  # graph proximity
    gamma: float  # reinforcement
    delta: float  # edge confidence
    decay_lambda: float
    reinforcement_cap: float

    @classmethod
    def from_settings(cls) -> "ScoringWeights":
        return cls(
            alpha=settings.scoring_alpha,
            beta=settings.scoring_beta,
            gamma=settings.scoring_gamma,
            delta=settings.scoring_delta,
            decay_lambda=settings.decay_lambda,
            reinforcement_cap=settings.reinforcement_cap,
        )


def reinforcement_score(
    access_count: int,
    last_accessed: datetime | None,
    now: datetime,
    weights: ScoringWeights,
) -> float:
    """Decayed, log-scaled, capped reinforcement contribution.

    Bounded in [0, weights.reinforcement_cap] by construction."""
    if access_count <= 0 or last_accessed is None:
        return 0.0
    age_seconds = max(0.0, (now - last_accessed).total_seconds())
    decay = math.exp(-weights.decay_lambda * age_seconds)
    raw = math.log1p(access_count) * decay
    return min(raw, weights.reinforcement_cap)


def score_candidate(
    vector_sim: float,
    graph_distance: int,
    edge_confidence: float,
    reinforcement: float,
    weights: ScoringWeights,
) -> float:
    """Combine the four signals into a final score under multiplicative gating.

    Base relevance is additive over vec_sim, proximity, and edge_confidence.
    Reinforcement acts as a bounded multiplier: score = base * (1 + γ·reinforcement).
    Max possible: (α + β + δ) · (1 + γ·cap)."""
    proximity = 1.0 / (1.0 + max(0, graph_distance))
    base = (
        weights.alpha * max(0.0, min(1.0, vector_sim))
        + weights.beta * proximity
        + weights.delta * max(0.0, min(1.0, edge_confidence))
    )
    return base * (1.0 + weights.gamma * reinforcement)


def max_possible_score(weights: ScoringWeights) -> float:
    """The theoretical ceiling under multiplicative gating. Used as an invariant
    in rumination tests."""
    base = weights.alpha + weights.beta + weights.delta
    return base * (1.0 + weights.gamma * weights.reinforcement_cap)


def parse_neo4j_datetime(value: object) -> datetime | None:
    """Neo4j returns ISO strings for .isoformat() writes, or DateTime
    objects for server-side now() writes. Normalize to Python datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        return to_native()
    return None
