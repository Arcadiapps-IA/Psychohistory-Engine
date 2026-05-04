"""
Unit and property-based tests for TrajectoryPredictor.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.7
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.exceptions import InsufficientDataError, InvalidHorizonError
from psychohistory.models import SocialPattern
from psychohistory.trajectory_predictor import TrajectoryPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(name: str = "test", confidence: float = 0.8) -> SocialPattern:
    return SocialPattern(
        id=str(uuid.uuid4()),
        name=name,
        confidence_score=confidence,
        causality_graph=nx.DiGraph(),
        supporting_events=[str(uuid.uuid4()) for _ in range(3)],
        is_quantum_detected=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_patterns(n: int = 3) -> list[SocialPattern]:
    return [_make_pattern(f"pattern_{i}", 0.5 + i * 0.1) for i in range(n)]


# Hypothesis strategy for valid patterns list
@st.composite
def valid_patterns(draw: st.DrawFn) -> list[SocialPattern]:
    n = draw(st.integers(min_value=1, max_value=10))
    return [
        SocialPattern(
            id=str(uuid.uuid4()),
            name=draw(st.text(min_size=1, max_size=20)),
            confidence_score=draw(st.floats(min_value=0.3, max_value=1.0, allow_nan=False)),
            causality_graph=nx.DiGraph(),
            supporting_events=[str(uuid.uuid4()) for _ in range(draw(st.integers(min_value=1, max_value=5)))],
            is_quantum_detected=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Propiedad 9: Predicción retorna exactamente tres trayectorias
# Validates: Requirement 3.1
# ---------------------------------------------------------------------------


@given(
    patterns=valid_patterns(),
    horizon=st.integers(min_value=1, max_value=1000),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_predict_returns_exactly_three_trajectories(
    patterns: list[SocialPattern],
    horizon: int,
    seed: int,
) -> None:
    """
    **Validates: Requirements 3.1**

    For any valid patterns and horizon, predict() must return exactly 3 trajectories.
    """
    predictor = TrajectoryPredictor()
    trajectories = predictor.predict(patterns, horizon_years=horizon, seed=seed)
    assert len(trajectories) == 3, (
        f"Expected exactly 3 trajectories, got {len(trajectories)}"
    )


# ---------------------------------------------------------------------------
# Propiedad 10: Confidence_Score == producto de probabilidades individuales
# Validates: Requirement 3.2
# ---------------------------------------------------------------------------


@given(
    patterns=valid_patterns(),
    horizon=st.integers(min_value=1, max_value=100),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_confidence_score_equals_product_of_probabilities(
    patterns: list[SocialPattern],
    horizon: int,
    seed: int,
) -> None:
    """
    **Validates: Requirements 3.2**

    For any trajectory, confidence_score must equal the product of individual
    node probabilities (tolerance 1e-9).
    """
    predictor = TrajectoryPredictor()
    trajectories = predictor.predict(patterns, horizon_years=horizon, seed=seed)

    for traj in trajectories:
        expected = 1.0
        for node in traj.nodes:
            expected *= node.probability

        assert math.isclose(traj.confidence_score, expected, rel_tol=1e-9, abs_tol=1e-12), (
            f"confidence_score={traj.confidence_score} != product={expected} "
            f"(diff={abs(traj.confidence_score - expected)})"
        )


# ---------------------------------------------------------------------------
# Propiedad 11: Determinismo con la misma semilla
# Validates: Requirements 3.7, 5.9
# ---------------------------------------------------------------------------


@given(
    patterns=valid_patterns(),
    horizon=st.integers(min_value=1, max_value=100),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_predict_is_deterministic_with_same_seed(
    patterns: list[SocialPattern],
    horizon: int,
    seed: int,
) -> None:
    """
    **Validates: Requirements 3.7, 5.9**

    Calling predict() twice with the same seed must produce identical results.
    """
    predictor = TrajectoryPredictor()
    result1 = predictor.predict(patterns, horizon_years=horizon, seed=seed)
    result2 = predictor.predict(patterns, horizon_years=horizon, seed=seed)

    assert len(result1) == len(result2)
    for t1, t2 in zip(result1, result2):
        assert math.isclose(t1.confidence_score, t2.confidence_score, rel_tol=1e-9), (
            f"confidence_score differs: {t1.confidence_score} vs {t2.confidence_score}"
        )
        assert len(t1.nodes) == len(t2.nodes)
        for n1, n2 in zip(t1.nodes, t2.nodes):
            assert math.isclose(n1.probability, n2.probability, rel_tol=1e-9), (
                f"Node probability differs: {n1.probability} vs {n2.probability}"
            )
        assert math.isclose(
            t1.uncertainty_bound.product,
            t2.uncertainty_bound.product,
            rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_insufficient_data_error_with_small_corpus() -> None:
    """predict() must raise InsufficientDataError when corpus < 1000 events."""
    mock_repo = MagicMock()
    mock_repo.count.return_value = 500

    predictor = TrajectoryPredictor(repository=mock_repo)
    patterns = _make_patterns(3)

    with pytest.raises(InsufficientDataError):
        predictor.predict(patterns, horizon_years=10, seed=42)


def test_invalid_horizon_error_below_minimum() -> None:
    """predict() must raise InvalidHorizonError for horizon < 1."""
    predictor = TrajectoryPredictor()
    patterns = _make_patterns(3)

    with pytest.raises(InvalidHorizonError):
        predictor.predict(patterns, horizon_years=0, seed=42)


def test_invalid_horizon_error_above_maximum() -> None:
    """predict() must raise InvalidHorizonError for horizon > 1000."""
    predictor = TrajectoryPredictor()
    patterns = _make_patterns(3)

    with pytest.raises(InvalidHorizonError):
        predictor.predict(patterns, horizon_years=1001, seed=42)


def test_predict_with_valid_repository() -> None:
    """predict() must succeed when repository has >= 1000 events."""
    mock_repo = MagicMock()
    mock_repo.count.return_value = 1500
    mock_repo.get_snapshot_hash.return_value = "abc123"

    predictor = TrajectoryPredictor(repository=mock_repo)
    patterns = _make_patterns(3)

    trajectories = predictor.predict(patterns, horizon_years=10, seed=42)
    assert len(trajectories) == 3


def test_trajectories_have_reasoning_trace() -> None:
    """Each trajectory must have a non-None reasoning_trace with steps."""
    predictor = TrajectoryPredictor()
    patterns = _make_patterns(3)

    trajectories = predictor.predict(patterns, horizon_years=10, seed=42)

    for traj in trajectories:
        assert traj.reasoning_trace is not None
        assert len(traj.reasoning_trace.steps) > 0


def test_trajectories_have_uncertainty_bound() -> None:
    """Each trajectory must have a non-None uncertainty_bound."""
    predictor = TrajectoryPredictor()
    patterns = _make_patterns(3)

    trajectories = predictor.predict(patterns, horizon_years=10, seed=42)

    for traj in trajectories:
        assert traj.uncertainty_bound is not None
        assert traj.uncertainty_bound.product >= 0.01 - 1e-12


def test_trajectories_sorted_by_descending_confidence() -> None:
    """Trajectories must be sorted by descending confidence_score."""
    predictor = TrajectoryPredictor()
    patterns = _make_patterns(5)

    trajectories = predictor.predict(patterns, horizon_years=10, seed=42)

    for i in range(len(trajectories) - 1):
        assert trajectories[i].confidence_score >= trajectories[i + 1].confidence_score, (
            f"Trajectory {i} confidence ({trajectories[i].confidence_score}) < "
            f"trajectory {i+1} confidence ({trajectories[i+1].confidence_score})"
        )


def test_nodes_have_supporting_patterns() -> None:
    """Each node must reference at least one supporting pattern."""
    predictor = TrajectoryPredictor()
    patterns = _make_patterns(3)

    trajectories = predictor.predict(patterns, horizon_years=10, seed=42)

    for traj in trajectories:
        for node in traj.nodes:
            assert len(node.supporting_patterns) >= 1, (
                f"Node {node.sequence_index} has no supporting patterns"
            )


def test_compute_confidence_score_log_space() -> None:
    """_compute_confidence_score must use log-space to avoid underflow."""
    from psychohistory.models import PredictedEvent, TrajectoryNode

    predictor = TrajectoryPredictor()

    # Create nodes with small probabilities that would underflow in direct multiplication
    nodes = []
    for i in range(10):
        node = TrajectoryNode(
            sequence_index=i,
            predicted_event=PredictedEvent(
                description=f"Event {i}",
                estimated_date=datetime(2030 + i, 1, 1),
                category=EventCategory.SOCIAL,
            ),
            probability=0.5,
        )
        nodes.append(node)

    score = predictor._compute_confidence_score(nodes)
    expected = 0.5 ** 10
    assert math.isclose(score, expected, rel_tol=1e-9), (
        f"Expected {expected}, got {score}"
    )


def test_invalidate_stale_removes_trajectories_with_large_variation() -> None:
    """_invalidate_stale must remove trajectories with confidence variation > 0.05."""
    from psychohistory.models import PredictedEvent, ReasoningTrace, TrajectoryNode, UncertaintyBound

    predictor = TrajectoryPredictor()
    patterns = _make_patterns(3)

    # Build a trajectory manually with known confidence
    node = TrajectoryNode(
        sequence_index=0,
        predicted_event=PredictedEvent(
            description="Test event",
            estimated_date=datetime(2030, 1, 1),
            category=EventCategory.SOCIAL,
        ),
        probability=0.9,
        supporting_patterns=[patterns[0].id],
    )
    ub = UncertaintyBound(
        sigma_state=0.1, sigma_momentum=0.1, product=0.01,
        h_social=0.01, was_adjusted=False
    )
    from psychohistory.models import Trajectory
    traj = Trajectory(
        id=str(uuid.uuid4()),
        nodes=[node],
        confidence_score=0.9,
        horizon_years=10,
        uncertainty_bound=ub,
        reasoning_trace=ReasoningTrace(trajectory_id="t1", steps=[], uncertainty_adjustments=[]),
        created_at=datetime.now(timezone.utc),
        corpus_snapshot_hash="hash",
        seed=42,
    )
    predictor._active_trajectories = [traj]

    # Update patterns with very low confidence → large variation
    updated_patterns = [
        SocialPattern(
            id=patterns[0].id,
            name="updated",
            confidence_score=0.1,  # was 0.5, now 0.1 → big change
            causality_graph=nx.DiGraph(),
            supporting_events=[],
            is_quantum_detected=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    ]

    stale = predictor._invalidate_stale(updated_patterns)
    assert len(stale) == 1
    assert len(predictor._active_trajectories) == 0
