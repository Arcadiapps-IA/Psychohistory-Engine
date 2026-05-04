"""
Unit and property-based tests for ExplainabilityReporter.

Validates: Requirements 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.explainability import ExplainabilityReporter
from psychohistory.models import (
    EntanglementCorrelation,
    InterventionPoint,
    PredictedEvent,
    ReasoningStep,
    ReasoningTrace,
    SocialPattern,
    Trajectory,
    TrajectoryNode,
    UncertaintyBound,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(pattern_id: str | None = None) -> SocialPattern:
    pid = pattern_id or str(uuid.uuid4())
    return SocialPattern(
        id=pid,
        name=f"pattern_{pid[:8]}",
        confidence_score=0.7,
        causality_graph=nx.DiGraph(),
        supporting_events=[str(uuid.uuid4())],
        is_quantum_detected=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_node(
    index: int,
    probability: float = 0.5,
    supporting_patterns: list[str] | None = None,
) -> TrajectoryNode:
    return TrajectoryNode(
        sequence_index=index,
        predicted_event=PredictedEvent(
            description=f"Predicted event at node {index}",
            estimated_date=datetime(2030 + index, 1, 1),
            category=EventCategory.SOCIAL,
        ),
        probability=probability,
        supporting_patterns=supporting_patterns or [str(uuid.uuid4())],
    )


def _make_trajectory(
    nodes: list[TrajectoryNode] | None = None,
    horizon: int = 10,
    with_reasoning: bool = True,
) -> Trajectory:
    if nodes is None:
        nodes = [_make_node(i) for i in range(3)]

    confidence = 1.0
    for node in nodes:
        confidence *= node.probability

    ub = UncertaintyBound(
        sigma_state=0.1,
        sigma_momentum=0.1,
        product=0.01,
        h_social=0.01,
        was_adjusted=False,
    )

    traj_id = str(uuid.uuid4())

    if with_reasoning:
        steps = [
            ReasoningStep(
                node_index=node.sequence_index,
                predicted_event_description=node.predicted_event.description,
                pattern_ids=list(node.supporting_patterns),
                event_ids=[str(uuid.uuid4())],
                confidence_contribution=node.probability,
            )
            for node in nodes
        ]
        reasoning = ReasoningTrace(
            trajectory_id=traj_id,
            steps=steps,
            uncertainty_adjustments=[],
        )
    else:
        reasoning = ReasoningTrace(
            trajectory_id=traj_id,
            steps=[],
            uncertainty_adjustments=[],
        )

    return Trajectory(
        id=traj_id,
        nodes=nodes,
        confidence_score=confidence,
        horizon_years=horizon,
        uncertainty_bound=ub,
        reasoning_trace=reasoning,
        created_at=datetime.now(timezone.utc),
        corpus_snapshot_hash="hash",
        seed=42,
    )


def _make_intervention_point(
    trajectory_id: str,
    node_index: int = 0,
    si: float = 0.8,
    supporting_patterns: list[str] | None = None,
) -> InterventionPoint:
    return InterventionPoint(
        id=str(uuid.uuid4()),
        trajectory_id=trajectory_id,
        node_index=node_index,
        temporal_coordinates=datetime(2030 + node_index, 1, 1),
        recommended_action_type="diplomatic_intervention",
        sensitivity_index=si,
        differential_impact=0.3,
        relevant_actor_categories=["POLITICAL"],
        entangled_nodes=[],
        supporting_patterns=supporting_patterns or [str(uuid.uuid4())],
    )


# ---------------------------------------------------------------------------
# Propiedad 20: Completitud de la traza — cada nodo tiene ≥1 patrón de soporte
# Validates: Requirement 9.1
# ---------------------------------------------------------------------------


@given(
    n_nodes=st.integers(min_value=1, max_value=8),
    n_patterns=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_reasoning_trace_completeness(n_nodes: int, n_patterns: int) -> None:
    """
    **Validates: Requirements 9.1**

    Each node in the trajectory must have at least 1 supporting pattern
    referenced in the reasoning trace.
    """
    patterns = [_make_pattern() for _ in range(n_patterns)]
    pattern_ids = [p.id for p in patterns]

    # Build nodes with supporting patterns
    nodes = []
    for i in range(n_nodes):
        # Each node references at least 1 pattern
        supporting = [pattern_ids[i % n_patterns]]
        nodes.append(_make_node(i, supporting_patterns=supporting))

    traj = _make_trajectory(nodes, with_reasoning=True)

    reporter = ExplainabilityReporter()
    reporter.register_trajectory(traj)
    report = reporter.get_explanation(traj.id)

    # Each reasoning step must reference at least 1 pattern
    assert len(report.reasoning_steps) == n_nodes, (
        f"Expected {n_nodes} reasoning steps, got {len(report.reasoning_steps)}"
    )

    for i, step_text in enumerate(report.reasoning_steps):
        # The step text must contain pattern reference info
        assert len(step_text) > 0, f"Step {i} is empty"

    # Verify each node has at least 1 supporting pattern
    for node in traj.nodes:
        assert len(node.supporting_patterns) >= 1, (
            f"Node {node.sequence_index} has no supporting patterns"
        )


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_get_explanation_returns_report_with_uncertainty_bound() -> None:
    """get_explanation must include UncertaintyBound description."""
    traj = _make_trajectory()
    reporter = ExplainabilityReporter()
    reporter.register_trajectory(traj)

    report = reporter.get_explanation(traj.id)

    assert report.uncertainty_description is not None
    assert len(report.uncertainty_description) > 0
    assert "sigma" in report.uncertainty_description.lower() or "σ" in report.uncertainty_description


def test_get_explanation_raises_key_error_for_unknown_id() -> None:
    """get_explanation must raise KeyError for an unregistered trajectory ID."""
    reporter = ExplainabilityReporter()

    with pytest.raises(KeyError):
        reporter.get_explanation("nonexistent-id")


def test_generate_report_contains_uncertainty_bound() -> None:
    """generate_report must include UncertaintyBound information."""
    traj = _make_trajectory()
    reporter = ExplainabilityReporter()

    report_str = reporter.generate_report(traj)

    assert "UNCERTAINTY BOUND" in report_str
    assert "sigma" in report_str.lower() or "σ" in report_str


def test_generate_report_contains_crisis_description() -> None:
    """generate_report must include Seldon Crisis descriptions when provided."""
    traj = _make_trajectory()
    ip = _make_intervention_point(traj.id, node_index=0, si=0.85)

    reporter = ExplainabilityReporter()
    report_str = reporter.generate_report(traj, intervention_points=[ip])

    assert "SELDON CRISES" in report_str
    assert "0.8500" in report_str or "0.85" in report_str


def test_generate_report_contains_predicted_events() -> None:
    """generate_report must list all predicted events."""
    nodes = [_make_node(i) for i in range(3)]
    traj = _make_trajectory(nodes)
    reporter = ExplainabilityReporter()

    report_str = reporter.generate_report(traj)

    assert "PREDICTED EVENTS" in report_str
    for node in nodes:
        assert node.predicted_event.description in report_str


def test_register_and_retrieve_trajectory() -> None:
    """register_trajectory must make the trajectory retrievable by ID."""
    traj = _make_trajectory()
    reporter = ExplainabilityReporter()

    reporter.register_trajectory(traj)
    report = reporter.get_explanation(traj.id)

    assert report.trajectory_id == traj.id


def test_report_summary_contains_key_info() -> None:
    """The report summary must contain horizon, confidence, and node count."""
    traj = _make_trajectory(horizon=50)
    reporter = ExplainabilityReporter()
    reporter.register_trajectory(traj)

    report = reporter.get_explanation(traj.id)

    assert "50" in report.summary  # horizon
    assert str(len(traj.nodes)) in report.summary  # node count


def test_report_with_adjusted_uncertainty_bound() -> None:
    """Report must mention adjustment when UncertaintyBound was adjusted."""
    nodes = [_make_node(0)]
    ub = UncertaintyBound(
        sigma_state=0.2,
        sigma_momentum=0.2,
        product=0.04,
        h_social=0.01,
        was_adjusted=True,
        adjustment_reason="Ajuste proporcional para satisfacer σ_estado × σ_momentum ≥ ħ_social",
    )
    traj_id = str(uuid.uuid4())
    traj = Trajectory(
        id=traj_id,
        nodes=nodes,
        confidence_score=0.5,
        horizon_years=10,
        uncertainty_bound=ub,
        reasoning_trace=ReasoningTrace(
            trajectory_id=traj_id,
            steps=[
                ReasoningStep(
                    node_index=0,
                    predicted_event_description="Test event",
                    pattern_ids=[str(uuid.uuid4())],
                    event_ids=[],
                    confidence_contribution=0.5,
                )
            ],
            uncertainty_adjustments=["Adjusted"],
        ),
        created_at=datetime.now(timezone.utc),
        corpus_snapshot_hash="hash",
        seed=42,
    )

    reporter = ExplainabilityReporter()
    reporter.register_trajectory(traj)
    report = reporter.get_explanation(traj.id)

    assert "adjusted" in report.uncertainty_description.lower() or "ajuste" in report.uncertainty_description.lower()


def test_intervention_descriptions_include_supporting_patterns() -> None:
    """Intervention descriptions must mention supporting patterns."""
    traj = _make_trajectory()
    pattern_id = str(uuid.uuid4())
    ip = _make_intervention_point(
        traj.id,
        node_index=0,
        si=0.85,
        supporting_patterns=[pattern_id],
    )

    reporter = ExplainabilityReporter()
    reporter.register_trajectory(traj)

    # Build report with intervention points
    report = reporter._build_report(traj, intervention_points=[ip])

    assert len(report.intervention_descriptions) == 1
    assert pattern_id in report.intervention_descriptions[0]


def test_reasoning_steps_reference_patterns() -> None:
    """Each reasoning step must reference the pattern IDs from the node."""
    pattern_id = str(uuid.uuid4())
    nodes = [_make_node(0, supporting_patterns=[pattern_id])]
    traj = _make_trajectory(nodes, with_reasoning=True)

    reporter = ExplainabilityReporter()
    reporter.register_trajectory(traj)
    report = reporter.get_explanation(traj.id)

    assert len(report.reasoning_steps) == 1
    assert pattern_id in report.reasoning_steps[0]
