"""
Unit and property-based tests for PatternAnalyzer.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.models import HistoricalEvent, SocialPattern
from psychohistory.pattern_analyzer import CorpusSnapshot, PatternAnalyzer
from psychohistory.quantum_engine import VQCResult

# Re-use the shared strategy from conftest
from tests.conftest import historical_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(events: list[HistoricalEvent]) -> CorpusSnapshot:
    return CorpusSnapshot(
        events=events,
        snapshot_hash="test",
        created_at=datetime.now(),
    )


def _make_event(
    category: EventCategory,
    year: int,
    actors: list[str] | None = None,
    magnitude: float | None = None,
) -> HistoricalEvent:
    return HistoricalEvent(
        id=str(uuid.uuid4()),
        date=datetime(year, 1, 1),
        description=f"Event in {year}",
        category=category,
        actors=actors or [],
        magnitude=magnitude,
    )


# ---------------------------------------------------------------------------
# Propiedad 5: Confidence_Score de patrón está en rango válido [0.0, 1.0]
# Validates: Requirement 2.2
# ---------------------------------------------------------------------------


@given(events=st.lists(historical_events(), min_size=10, max_size=100))
@settings(max_examples=50)
def test_pattern_confidence_score_in_valid_range(events: list[HistoricalEvent]) -> None:
    """
    **Validates: Requirements 2.2**

    For any corpus, all active patterns must have confidence_score in [0.0, 1.0].
    """
    analyzer = PatternAnalyzer()
    snapshot = _make_snapshot(events)
    patterns = analyzer.analyze(snapshot)

    for pattern in patterns:
        assert 0.0 <= pattern.confidence_score <= 1.0, (
            f"Pattern '{pattern.name}' has confidence_score={pattern.confidence_score} "
            f"outside [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# Propiedad 6: Patrones con Confidence_Score bajo son descartados
# Validates: Requirement 2.3
# ---------------------------------------------------------------------------


@given(events=st.lists(historical_events(), min_size=10, max_size=100))
@settings(max_examples=50)
def test_low_confidence_patterns_are_discarded(events: list[HistoricalEvent]) -> None:
    """
    **Validates: Requirements 2.3**

    No active pattern should have confidence_score < 0.3.
    """
    analyzer = PatternAnalyzer()
    snapshot = _make_snapshot(events)
    patterns = analyzer.analyze(snapshot)

    for pattern in patterns:
        assert pattern.confidence_score >= PatternAnalyzer.CONFIDENCE_THRESHOLD, (
            f"Pattern '{pattern.name}' has confidence_score={pattern.confidence_score} "
            f"below threshold {PatternAnalyzer.CONFIDENCE_THRESHOLD}"
        )


# ---------------------------------------------------------------------------
# Propiedad 7: Actualización del Corpus preserva patrones no relacionados
# Validates: Requirement 2.5
# ---------------------------------------------------------------------------


@given(
    initial_events=st.lists(historical_events(), min_size=20, max_size=80),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_corpus_update_preserves_unrelated_patterns(
    initial_events: list[HistoricalEvent],
) -> None:
    """
    **Validates: Requirements 2.5**

    After analyzing an initial corpus, adding events of a single category
    (POLITICAL) must not change patterns whose causality graph contains only
    NATURAL category nodes.
    """
    analyzer = PatternAnalyzer()

    # Analyze initial corpus
    snapshot = _make_snapshot(initial_events)
    analyzer.analyze(snapshot)

    # Capture patterns that are purely NATURAL (no POLITICAL nodes in graph)
    natural_only_patterns_before: dict[str, float] = {}
    for pid, pattern in analyzer._active_patterns.items():
        graph_nodes = set(pattern.causality_graph.nodes())
        if (
            EventCategory.NATURAL in graph_nodes
            and EventCategory.POLITICAL not in graph_nodes
        ):
            natural_only_patterns_before[pid] = pattern.confidence_score

    if not natural_only_patterns_before:
        # Nothing to verify — skip this example
        return

    # Add new events of only POLITICAL category
    new_political_events = [
        _make_event(EventCategory.POLITICAL, 2000 + i)
        for i in range(5)
    ]

    # recalculate_affected should NOT touch NATURAL-only patterns
    analyzer.recalculate_affected(new_political_events)

    # Verify NATURAL-only patterns are unchanged
    for pid, original_confidence in natural_only_patterns_before.items():
        if pid in analyzer._active_patterns:
            current_confidence = analyzer._active_patterns[pid].confidence_score
            assert current_confidence == original_confidence, (
                f"Pattern {pid} (NATURAL-only) changed confidence from "
                f"{original_confidence} to {current_confidence} after adding POLITICAL events"
            )


# ---------------------------------------------------------------------------
# Propiedad 8: Delegación cuántica por dimensionalidad (mock del QuantumEngine)
# Validates: Requirement 2.7
# ---------------------------------------------------------------------------


def test_quantum_delegation_by_dimensionality() -> None:
    """
    **Validates: Requirements 2.7**

    When feature_matrix has > 50 columns, train_vqc must be called.
    When feature_matrix has <= 50 columns, train_vqc must NOT be called.
    """
    # --- Case 1: feature_matrix with > 50 columns → quantum delegation ---
    mock_qe_quantum = MagicMock()
    # train_vqc must return a valid VQCResult
    mock_qe_quantum.MAX_VQC_QUBITS = 50
    mock_qe_quantum.train_vqc.return_value = VQCResult(
        patterns=[
            {"pattern_id": "p0", "confidence_score": 0.8, "feature_indices": [0, 1]},
        ],
        n_qubits=25,
        n_iterations=10,
        execution_ms=1.0,
        backend="classical",
        seed=42,
    )

    analyzer_quantum = PatternAnalyzer(quantum_engine=mock_qe_quantum)

    # Build events that produce a feature matrix with > 50 columns
    # The default feature matrix has 9 columns (6 one-hot + 3 scalars).
    # We need to patch _build_feature_matrix to return a wide matrix.
    wide_matrix = np.random.default_rng(0).standard_normal((20, 60))

    original_build = analyzer_quantum._build_feature_matrix
    analyzer_quantum._build_feature_matrix = lambda events: wide_matrix  # type: ignore[method-assign]

    events = [_make_event(EventCategory.POLITICAL, 1900 + i) for i in range(20)]
    snapshot = _make_snapshot(events)
    analyzer_quantum.analyze(snapshot)

    mock_qe_quantum.train_vqc.assert_called_once(), (
        "train_vqc should have been called when feature_matrix has > 50 columns"
    )

    # Restore
    analyzer_quantum._build_feature_matrix = original_build  # type: ignore[method-assign]

    # --- Case 2: feature_matrix with <= 50 columns → classical path ---
    mock_qe_classical = MagicMock()
    mock_qe_classical.MAX_VQC_QUBITS = 50

    analyzer_classical = PatternAnalyzer(quantum_engine=mock_qe_classical)

    # Default feature matrix has 9 columns (< 50) → classical path
    events2 = [_make_event(EventCategory.ECONOMIC, 1800 + i) for i in range(20)]
    snapshot2 = _make_snapshot(events2)
    analyzer_classical.analyze(snapshot2)

    mock_qe_classical.train_vqc.assert_not_called(), (
        "train_vqc should NOT have been called when feature_matrix has <= 50 columns"
    )


# ---------------------------------------------------------------------------
# Additional unit tests
# ---------------------------------------------------------------------------


def test_analyze_empty_corpus_returns_empty_list() -> None:
    """analyze() on an empty corpus must return an empty list."""
    analyzer = PatternAnalyzer()
    snapshot = _make_snapshot([])
    patterns = analyzer.analyze(snapshot)
    assert patterns == []


def test_analyze_small_corpus_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """analyze() on a corpus smaller than MIN_CORPUS_SIZE must log a warning."""
    import logging

    analyzer = PatternAnalyzer()
    events = [_make_event(EventCategory.POLITICAL, 1900 + i) for i in range(5)]
    snapshot = _make_snapshot(events)

    with caplog.at_level(logging.WARNING, logger="psychohistory.pattern_analyzer"):
        analyzer.analyze(snapshot)

    assert any("only" in record.message.lower() for record in caplog.records), (
        "Expected a warning about small corpus size"
    )


def test_get_active_patterns_filters_below_threshold() -> None:
    """get_active_patterns() must only return patterns with confidence >= 0.3."""
    analyzer = PatternAnalyzer()

    # Manually inject patterns with various confidence scores
    low_pattern = SocialPattern(
        id="low",
        name="low_confidence",
        confidence_score=0.1,
        causality_graph=nx.DiGraph(),
    )
    high_pattern = SocialPattern(
        id="high",
        name="high_confidence",
        confidence_score=0.8,
        causality_graph=nx.DiGraph(),
    )
    analyzer._active_patterns = {"low": low_pattern, "high": high_pattern}

    active = analyzer.get_active_patterns()
    assert len(active) == 1
    assert active[0].id == "high"


def test_build_feature_matrix_shape() -> None:
    """_build_feature_matrix must return shape (n_events, 9)."""
    analyzer = PatternAnalyzer()
    events = [_make_event(EventCategory.POLITICAL, 1900 + i) for i in range(10)]
    matrix = analyzer._build_feature_matrix(events)
    assert matrix.shape == (10, 9), f"Expected shape (10, 9), got {matrix.shape}"


def test_build_feature_matrix_one_hot_encoding() -> None:
    """One-hot encoding must set exactly one category bit per event."""
    analyzer = PatternAnalyzer()
    events = [_make_event(cat, 1900) for cat in EventCategory]
    matrix = analyzer._build_feature_matrix(events)
    # First 6 columns are one-hot; each row must have exactly one 1 in those columns
    for i, event in enumerate(events):
        one_hot = matrix[i, :6]
        assert one_hot.sum() == 1.0, f"Row {i} one-hot sum is {one_hot.sum()}"
        assert one_hot.max() == 1.0


def test_build_causality_graph_directed_edges() -> None:
    """_build_causality_graph must produce a directed graph with correct edges."""
    analyzer = PatternAnalyzer()

    # Two events: POLITICAL in 1900, ECONOMIC in 1905 (within 10-year window)
    e1 = _make_event(EventCategory.POLITICAL, 1900)
    e2 = _make_event(EventCategory.ECONOMIC, 1905)
    events = [e1, e2]

    pattern = SocialPattern(
        id="test",
        name="test_pattern",
        confidence_score=0.5,
        causality_graph=nx.DiGraph(),
        supporting_events=[e1.id, e2.id],
    )

    graph = analyzer._build_causality_graph(pattern, events)

    assert isinstance(graph, nx.DiGraph)
    assert EventCategory.POLITICAL in graph.nodes
    assert EventCategory.ECONOMIC in graph.nodes
    assert graph.has_edge(EventCategory.POLITICAL, EventCategory.ECONOMIC)


def test_detect_cyclic_patterns_finds_regular_cycle() -> None:
    """_detect_cyclic_patterns must detect a regular cycle with low std."""
    analyzer = PatternAnalyzer()

    # POLITICAL events every 50 years: 1000, 1050, 1100, 1150, 1200
    events = [_make_event(EventCategory.POLITICAL, 1000 + i * 50) for i in range(5)]

    patterns = analyzer._detect_cyclic_patterns(events)

    assert len(patterns) == 1
    p = patterns[0]
    assert p.recurrence_period_years == pytest.approx(50.0, abs=1.0)
    assert p.confidence_score >= 0.3
    assert p.confidence_score <= 1.0


def test_detect_cyclic_patterns_ignores_irregular_intervals() -> None:
    """_detect_cyclic_patterns must not detect a pattern with high std."""
    analyzer = PatternAnalyzer()

    # Irregular intervals: 1000, 1010, 1100, 1200, 1210
    events = [
        _make_event(EventCategory.ECONOMIC, 1000),
        _make_event(EventCategory.ECONOMIC, 1010),
        _make_event(EventCategory.ECONOMIC, 1100),
        _make_event(EventCategory.ECONOMIC, 1200),
        _make_event(EventCategory.ECONOMIC, 1210),
    ]

    patterns = analyzer._detect_cyclic_patterns(events)
    assert len(patterns) == 0, "Irregular intervals should not produce a cyclic pattern"


def test_classical_correlation_returns_patterns_above_threshold() -> None:
    """_classical_correlation must only return patterns with confidence >= 0.3."""
    analyzer = PatternAnalyzer()

    # Create events where POLITICAL and ECONOMIC co-occur in many periods
    events: list[HistoricalEvent] = []
    for period in range(10):
        base_year = 1000 + period * 50
        events.append(_make_event(EventCategory.POLITICAL, base_year))
        events.append(_make_event(EventCategory.ECONOMIC, base_year + 1))

    patterns = analyzer._classical_correlation(events)

    for p in patterns:
        assert p.confidence_score >= PatternAnalyzer.CONFIDENCE_THRESHOLD


def test_recalculate_affected_preserves_unrelated_patterns() -> None:
    """recalculate_affected must not modify patterns unrelated to new event categories."""
    analyzer = PatternAnalyzer()

    # Inject a NATURAL-only pattern manually
    natural_graph = nx.DiGraph()
    natural_graph.add_node(EventCategory.NATURAL)
    natural_pattern = SocialPattern(
        id="natural_only",
        name="cyclic_NATURAL",
        confidence_score=0.9,
        causality_graph=natural_graph,
        supporting_events=["e1", "e2"],
    )
    analyzer._active_patterns = {"natural_only": natural_pattern}

    # Add POLITICAL events — should not affect NATURAL-only pattern
    new_events = [_make_event(EventCategory.POLITICAL, 2000 + i) for i in range(5)]
    analyzer.recalculate_affected(new_events)

    # NATURAL-only pattern must still be present and unchanged
    assert "natural_only" in analyzer._active_patterns
    assert analyzer._active_patterns["natural_only"].confidence_score == pytest.approx(0.9)


def test_analyze_updates_active_patterns() -> None:
    """analyze() must update _active_patterns with the detected patterns."""
    analyzer = PatternAnalyzer()

    events = [
        _make_event(EventCategory.POLITICAL, 1000 + i * 50)
        for i in range(5)
    ] + [
        _make_event(EventCategory.ECONOMIC, 1000 + i * 50 + 1)
        for i in range(5)
    ]

    snapshot = _make_snapshot(events)
    patterns = analyzer.analyze(snapshot)

    assert len(analyzer._active_patterns) == len(patterns)
    for p in patterns:
        assert p.id in analyzer._active_patterns


def test_social_pattern_has_causality_graph() -> None:
    """Every pattern returned by analyze() must have a non-None causality_graph."""
    analyzer = PatternAnalyzer()

    events = [
        _make_event(EventCategory.POLITICAL, 1000 + i * 50)
        for i in range(5)
    ] + [
        _make_event(EventCategory.MILITARY, 1000 + i * 50 + 2)
        for i in range(5)
    ]

    snapshot = _make_snapshot(events)
    patterns = analyzer.analyze(snapshot)

    for p in patterns:
        assert p.causality_graph is not None
        assert isinstance(p.causality_graph, nx.DiGraph)
