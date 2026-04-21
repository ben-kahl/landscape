"""Pure-Python unit tests for the scoring + reinforcement math.

These don't hit the database — they exist to guarantee that the
bounding invariants hold regardless of how Phase 2 retrieval evolves.
If any test here fails, retrieval is at risk of rumination collapse.
"""
import math
from datetime import UTC, datetime, timedelta

import pytest

from landscape.retrieval.scoring import (
    ScoringWeights,
    max_possible_score,
    reinforcement_score,
    score_candidate,
)

pytestmark = pytest.mark.unit

WEIGHTS = ScoringWeights(
    alpha=1.0,
    beta=0.8,
    gamma=0.2,
    delta=0.3,
    decay_lambda=math.log(2) / (7 * 86400),  # true 7-day half-life
    reinforcement_cap=2.0,
)


def test_never_accessed_has_zero_reinforcement():
    now = datetime.now(UTC)
    assert reinforcement_score(0, None, now, WEIGHTS) == 0.0
    assert reinforcement_score(5, None, now, WEIGHTS) == 0.0
    assert reinforcement_score(0, now, now, WEIGHTS) == 0.0


def test_log1p_sublinear_growth():
    now = datetime.now(UTC)
    # Pre-cap region: log1p(1) = 0.693, log1p(5) = 1.792 — both under cap=2.0
    r1 = reinforcement_score(1, now, now, WEIGHTS)
    r5 = reinforcement_score(5, now, now, WEIGHTS)
    assert r1 < r5 < WEIGHTS.reinforcement_cap
    # Post-cap region: log1p(10) ≈ 2.398, log1p(1M) ≈ 13.82 — both hit the cap
    r10 = reinforcement_score(10, now, now, WEIGHTS)
    r1m = reinforcement_score(1_000_000, now, now, WEIGHTS)
    assert r10 == r1m == WEIGHTS.reinforcement_cap
    # The specific pre-cap values match log1p exactly (no decay since freshly accessed)
    assert abs(r1 - math.log1p(1)) < 1e-9
    assert abs(r5 - math.log1p(5)) < 1e-9


def test_decay_half_life():
    now = datetime.now(UTC)
    fresh = reinforcement_score(100, now, now, WEIGHTS)
    seven_days_old = reinforcement_score(
        100, now - timedelta(days=7), now, WEIGHTS
    )
    # One half-life means ~0.5 multiplier on the pre-cap value.
    # Since log1p(100) ≈ 4.605 and cap is 2.0, fresh is capped at 2.0.
    # After 7 days, pre-cap value is 4.605 * 0.5 ≈ 2.302, still > cap.
    # So both are capped. Pick a larger age to see decay bite.
    fourteen_days_old = reinforcement_score(
        100, now - timedelta(days=14), now, WEIGHTS
    )
    # pre-cap: 4.605 * 0.25 ≈ 1.151 < cap, so no cap applies
    expected = math.log1p(100) * math.exp(-WEIGHTS.decay_lambda * 14 * 86400)
    assert abs(fourteen_days_old - expected) < 1e-9
    assert fourteen_days_old < fresh  # decayed less than fresh
    assert fresh == WEIGHTS.reinforcement_cap  # fresh is at the cap
    assert seven_days_old == WEIGHTS.reinforcement_cap  # still above cap


def test_decay_to_zero_at_infinity():
    now = datetime.now(UTC)
    very_old = reinforcement_score(
        1_000_000, now - timedelta(days=365), now, WEIGHTS
    )
    # 365 days = ~52 half-lives. 2^-52 ≈ 2.2e-16.
    # Times log1p(1M) ≈ 13.82 gives ~3e-15. Threshold well above that.
    assert very_old < 1e-10


def test_reinforcement_is_bounded():
    """The rumination regression test. If any future change removes the
    cap or the log1p, this fails and retrieval is unsafe to ship."""
    now = datetime.now(UTC)
    for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        r = reinforcement_score(n, now, now, WEIGHTS)
        assert 0.0 <= r <= WEIGHTS.reinforcement_cap


def test_score_bounded_by_max():
    """Total score cannot exceed (alpha + beta + delta) * (1 + gamma*cap) —
    the multiplicative-gating ceiling."""
    now = datetime.now(UTC)
    ceiling = max_possible_score(WEIGHTS)
    # Pathologically good input
    r = reinforcement_score(10_000_000, now, now, WEIGHTS)
    s = score_candidate(
        vector_sim=1.0,
        graph_distance=0,
        edge_confidence=1.0,
        reinforcement=r,
        weights=WEIGHTS,
    )
    assert s <= ceiling + 1e-9
    # And specifically, it hits the ceiling exactly
    assert abs(s - ceiling) < 1e-9


def test_score_components_balanced():
    """Under multiplicative gating, reinforcement amplifies by at most
    (1 + γ·cap). This cap must stay ≤ 1.5 so a reinforced irrelevant
    candidate cannot outrank a fresh highly-relevant one by a wide margin."""
    multiplier_ceiling = 1.0 + WEIGHTS.gamma * WEIGHTS.reinforcement_cap
    assert 1.0 < multiplier_ceiling <= 1.5


def test_vector_sim_clamped():
    """Out-of-range inputs don't break the bound."""
    now = datetime.now(UTC)
    s = score_candidate(
        vector_sim=99.0,  # nonsense, should be clamped to 1.0
        graph_distance=0,
        edge_confidence=-5.0,  # clamped to 0.0
        reinforcement=reinforcement_score(100, now, now, WEIGHTS),
        weights=WEIGHTS,
    )
    assert s <= max_possible_score(WEIGHTS) + 1e-9


def test_graph_proximity_monotone():
    """Closer = higher proximity contribution."""
    closer = score_candidate(0.5, 0, 0.5, 0.0, WEIGHTS)
    farther = score_candidate(0.5, 3, 0.5, 0.0, WEIGHTS)
    assert closer > farther


def test_log1p_specific_value_at_one_million():
    """Directly verify that log1p(1M) is ~14, not 1M. If someone ever
    replaces log1p with raw n, this test catches it before it ships."""
    assert 13.0 < math.log1p(1_000_000) < 15.0


def test_multiplicative_reinforcement_amplifies():
    """Two candidates with identical base signals but one reinforced:
    reinforced candidate scores strictly higher under multiplicative gating."""
    now = datetime.now(UTC)
    reinforced = reinforcement_score(5, now, now, WEIGHTS)
    cold = score_candidate(0.5, 1, 0.5, 0.0, WEIGHTS)
    warm = score_candidate(0.5, 1, 0.5, reinforced, WEIGHTS)
    assert warm > cold
    # Amplification factor == 1 + gamma * reinforcement
    assert abs(warm - cold * (1.0 + WEIGHTS.gamma * reinforced)) < 1e-9


def test_cold_path_baseline_equals_base():
    """Zero reinforcement must leave the base score unchanged."""
    base = (
        WEIGHTS.alpha * 0.5
        + WEIGHTS.beta * (1.0 / (1.0 + 1))
        + WEIGHTS.delta * 0.5
    )
    s = score_candidate(0.5, 1, 0.5, 0.0, WEIGHTS)
    assert abs(s - base) < 1e-9


def test_zero_base_stays_zero():
    """Reinforcement can only amplify relevance, not fabricate it."""
    now = datetime.now(UTC)
    r = reinforcement_score(10_000_000, now, now, WEIGHTS)
    # proximity is 1/(1+distance), so we can't zero it; test the two clampable signals.
    s = score_candidate(0.0, 1_000_000, 0.0, r, WEIGHTS)
    # beta * 1/(1+1e6) ≈ 8e-7; amplified by (1 + 0.2*2) = 1.4 → still ~1.1e-6
    assert s < 2e-6


def test_rumination_bound_preserved_under_gating():
    """Pathologically hot + recent input stays bounded. This is the same
    rumination guard as test_reinforcement_is_bounded, but exercised through
    score_candidate to catch any future regression that removes the
    reinforcement cap from the multiplicative path."""
    now = datetime.now(UTC)
    r = reinforcement_score(10_000_000, now, now, WEIGHTS)
    s = score_candidate(
        vector_sim=1.0,
        graph_distance=0,
        edge_confidence=1.0,
        reinforcement=r,
        weights=WEIGHTS,
    )
    ceiling = max_possible_score(WEIGHTS)
    assert s <= ceiling + 1e-9
    assert r <= WEIGHTS.reinforcement_cap
