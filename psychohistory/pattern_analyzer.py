"""
Pattern_Analyzer: detects statistical correlations in the Corpus and builds
causality graphs between event categories.

Delegates to QuantumEngine via VQC when the feature space exceeds 50 dimensions.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import networkx as nx
import numpy as np

from psychohistory.enums import EventCategory
from psychohistory.models import HistoricalEvent, SocialPattern
from psychohistory.quantum_engine import QuantumEngine


# ---------------------------------------------------------------------------
# CorpusSnapshot
# ---------------------------------------------------------------------------


@dataclass
class CorpusSnapshot:
    """Immutable snapshot of the Corpus used for pattern analysis."""

    events: list[HistoricalEvent]
    snapshot_hash: str
    created_at: datetime


# ---------------------------------------------------------------------------
# PatternAnalyzer
# ---------------------------------------------------------------------------

# All EventCategory members, in a stable order for one-hot encoding
_ALL_CATEGORIES: list[EventCategory] = list(EventCategory)
_N_CATEGORIES: int = len(_ALL_CATEGORIES)
_CAT_INDEX: dict[EventCategory, int] = {cat: i for i, cat in enumerate(_ALL_CATEGORIES)}


class PatternAnalyzer:
    """
    Detects statistical correlations in the Corpus and builds causality graphs
    between event categories.

    Classical correlation is used when the feature space has <= 50 dimensions.
    When the feature space exceeds 50 dimensions the analysis is delegated to
    the QuantumEngine via VQC.
    """

    CONFIDENCE_THRESHOLD = 0.3          # patterns below this are discarded
    QUANTUM_DIMENSION_THRESHOLD = 50    # delegate to QE if features > 50
    MIN_CORPUS_SIZE = 1000              # minimum for analysis

    def __init__(self, quantum_engine: QuantumEngine | None = None) -> None:
        self._qe: QuantumEngine = quantum_engine or QuantumEngine()
        self._active_patterns: dict[str, SocialPattern] = {}   # id -> pattern
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, corpus: CorpusSnapshot) -> list[SocialPattern]:
        """Analyze the full corpus and return the detected patterns."""
        events = corpus.events

        if len(events) < self.MIN_CORPUS_SIZE:
            self._logger.warning(
                "Corpus has only %d events (minimum recommended: %d). "
                "Proceeding with analysis anyway.",
                len(events),
                self.MIN_CORPUS_SIZE,
            )

        if not events:
            self._active_patterns = {}
            return []

        feature_matrix = self._build_feature_matrix(events)

        if feature_matrix.shape[1] > self.QUANTUM_DIMENSION_THRESHOLD:
            patterns = self._quantum_correlation(feature_matrix, events)
        else:
            patterns = self._classical_correlation(events)

        # Update active patterns dict
        self._active_patterns = {p.id: p for p in patterns}
        return list(self._active_patterns.values())

    def recalculate_affected(self, new_events: list[HistoricalEvent]) -> list[SocialPattern]:
        """
        Recalculate only the patterns affected by the categories of the new events.
        Preserves patterns from unrelated categories.
        Returns the patterns that changed.
        """
        if not new_events:
            return []

        # Identify categories present in the new events
        new_categories: set[EventCategory] = {e.category for e in new_events}

        # Find active patterns whose causality graph contains nodes from those categories
        affected_ids: list[str] = []
        for pid, pattern in self._active_patterns.items():
            graph_nodes: set = set(pattern.causality_graph.nodes())
            if graph_nodes & new_categories:
                affected_ids.append(pid)

        if not affected_ids:
            return []

        # Collect events relevant to the affected categories
        affected_categories: set[EventCategory] = set()
        for pid in affected_ids:
            affected_categories.update(self._active_patterns[pid].causality_graph.nodes())

        relevant_events = [e for e in new_events if e.category in affected_categories]
        if not relevant_events:
            relevant_events = new_events

        # Recalculate only the affected patterns
        recalculated = self._classical_correlation(relevant_events)

        changed: list[SocialPattern] = []
        for new_pattern in recalculated:
            # Match by name (category pair) to find the old pattern
            old_pattern = next(
                (p for p in self._active_patterns.values() if p.name == new_pattern.name),
                None,
            )
            if old_pattern is None or abs(old_pattern.confidence_score - new_pattern.confidence_score) > 1e-9:
                # Replace or add
                self._active_patterns[new_pattern.id] = new_pattern
                changed.append(new_pattern)

        return changed

    def get_active_patterns(self) -> list[SocialPattern]:
        """Return all active patterns (confidence >= threshold)."""
        return [
            p for p in self._active_patterns.values()
            if p.confidence_score >= self.CONFIDENCE_THRESHOLD
        ]

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, events: list[HistoricalEvent]) -> np.ndarray:
        """
        Build the feature matrix for analysis.

        Each event is encoded as a numeric vector:
          - Category: one-hot encoding (6 dimensions, one per EventCategory)
          - Normalized year: (year - 1000) / 1000.0
          - Magnitude: direct value or 0.5 if None
          - Normalized actor count: min(len(actors), 10) / 10.0

        Returns a numpy array of shape (n_events, n_features).
        """
        n_features = _N_CATEGORIES + 3  # 6 one-hot + year + magnitude + actors
        matrix = np.zeros((len(events), n_features), dtype=float)

        for i, event in enumerate(events):
            # One-hot category
            cat_idx = _CAT_INDEX.get(event.category, 0)
            matrix[i, cat_idx] = 1.0

            # Normalized year
            year = event.date.year if event.date is not None else 1000
            matrix[i, _N_CATEGORIES] = (year - 1000) / 1000.0

            # Magnitude
            matrix[i, _N_CATEGORIES + 1] = event.magnitude if event.magnitude is not None else 0.5

            # Normalized actor count
            matrix[i, _N_CATEGORIES + 2] = min(len(event.actors), 10) / 10.0

        return matrix

    # ------------------------------------------------------------------
    # Classical correlation
    # ------------------------------------------------------------------

    def _classical_correlation(self, events: list[HistoricalEvent]) -> list[SocialPattern]:
        """
        Classical pattern detection using statistical correlations.

        Calculates category co-occurrence in 50-year periods.
        For each category pair (A, B): confidence = co-occurrences / max_possible,
        normalized to [0.0, 1.0].
        """
        if not events:
            return []

        # Group events by 50-year period
        periods: dict[int, list[EventCategory]] = {}
        for event in events:
            period = event.date.year // 50
            periods.setdefault(period, []).append(event.category)

        n_periods = len(periods)
        if n_periods == 0:
            return []

        # Count co-occurrences per period for each category pair
        co_occurrences: dict[tuple[EventCategory, EventCategory], int] = {}
        for cats in periods.values():
            cat_set = set(cats)
            for cat_a in cat_set:
                for cat_b in cat_set:
                    if cat_a != cat_b:
                        pair = (cat_a, cat_b)
                        co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

        patterns: list[SocialPattern] = []
        seen_pairs: set[frozenset] = set()

        for (cat_a, cat_b), count in co_occurrences.items():
            pair_key = frozenset({cat_a, cat_b})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Confidence = co-occurrences / n_periods, clamped to [0, 1]
            confidence = float(np.clip(count / n_periods, 0.0, 1.0))

            if confidence < self.CONFIDENCE_THRESHOLD:
                self._logger.debug(
                    "Discarding pattern (%s, %s) with confidence %.3f < %.3f",
                    cat_a.value,
                    cat_b.value,
                    confidence,
                    self.CONFIDENCE_THRESHOLD,
                )
                continue

            # Supporting events: events whose category is cat_a or cat_b
            supporting = [
                e.id for e in events if e.category in (cat_a, cat_b)
            ]

            pattern = SocialPattern(
                id=str(uuid.uuid4()),
                name=f"{cat_a.value}-{cat_b.value}",
                confidence_score=confidence,
                causality_graph=nx.DiGraph(),
                supporting_events=supporting,
                is_quantum_detected=False,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            pattern.causality_graph = self._build_causality_graph(pattern, events)
            patterns.append(pattern)

        # Also detect cyclic patterns
        cyclic = self._detect_cyclic_patterns(events)
        patterns.extend(cyclic)

        return patterns

    # ------------------------------------------------------------------
    # Quantum correlation
    # ------------------------------------------------------------------

    def _quantum_correlation(
        self,
        feature_matrix: np.ndarray,
        events: list[HistoricalEvent],
    ) -> list[SocialPattern]:
        """
        Quantum pattern detection using VQC from the QuantumEngine.

        Delegates to QE when feature_matrix.shape[1] > QUANTUM_DIMENSION_THRESHOLD.
        """
        n_qubits = min(feature_matrix.shape[1] // 2, QuantumEngine.MAX_VQC_QUBITS)
        if n_qubits < 1:
            n_qubits = 1

        vqc_result = self._qe.train_vqc(feature_matrix, n_qubits=n_qubits, seed=42)

        patterns: list[SocialPattern] = []
        for raw_pattern in vqc_result.patterns:
            confidence = float(np.clip(raw_pattern["confidence_score"], 0.0, 1.0))

            if confidence < self.CONFIDENCE_THRESHOLD:
                self._logger.debug(
                    "Discarding quantum pattern '%s' with confidence %.3f < %.3f",
                    raw_pattern["pattern_id"],
                    confidence,
                    self.CONFIDENCE_THRESHOLD,
                )
                continue

            # Map feature_indices to supporting event IDs
            feature_indices: list[int] = raw_pattern.get("feature_indices", [])
            supporting = [
                events[idx].id
                for idx in feature_indices
                if idx < len(events)
            ]

            pattern = SocialPattern(
                id=str(uuid.uuid4()),
                name=f"quantum_{raw_pattern['pattern_id']}",
                confidence_score=confidence,
                causality_graph=nx.DiGraph(),
                supporting_events=supporting,
                is_quantum_detected=True,
                qubits_used=vqc_result.n_qubits,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            pattern.causality_graph = self._build_causality_graph(pattern, events)
            patterns.append(pattern)

        return patterns

    # ------------------------------------------------------------------
    # Causality graph
    # ------------------------------------------------------------------

    def _build_causality_graph(
        self,
        pattern: SocialPattern,
        events: list[HistoricalEvent],
    ) -> nx.DiGraph:
        """
        Build a directed causality graph between event categories.

        Nodes: EventCategory values present in the supporting events.
        Edges A→B: when an event of category A precedes an event of category B
                   within a 10-year window.
        Edge weight: number of times that sequence occurs.
        """
        graph = nx.DiGraph()

        # Filter events to those supporting this pattern
        supporting_set = set(pattern.supporting_events)
        relevant_events = [e for e in events if e.id in supporting_set]

        if not relevant_events:
            return graph

        # Add nodes for each category present
        categories_present = {e.category for e in relevant_events}
        for cat in categories_present:
            graph.add_node(cat)

        # Sort events by date
        sorted_events = sorted(relevant_events, key=lambda e: e.date)

        # Add directed edges A→B when A precedes B within 10 years
        for i, event_a in enumerate(sorted_events):
            for event_b in sorted_events[i + 1:]:
                year_diff = event_b.date.year - event_a.date.year
                if year_diff > 10:
                    break
                if event_a.category == event_b.category:
                    continue
                cat_a = event_a.category
                cat_b = event_b.category
                if graph.has_edge(cat_a, cat_b):
                    graph[cat_a][cat_b]["weight"] += 1
                else:
                    graph.add_edge(cat_a, cat_b, weight=1)

        return graph

    # ------------------------------------------------------------------
    # Cyclic pattern detection
    # ------------------------------------------------------------------

    def _detect_cyclic_patterns(self, events: list[HistoricalEvent]) -> list[SocialPattern]:
        """
        Detect cyclic patterns with periods between 1 and 500 years.

        For each category with >= 3 events, compute intervals between consecutive
        events (in years). If std < 20% of mean → cyclic pattern detected.
        Confidence = 1.0 - (std / mean), normalized to [0.3, 1.0].
        """
        # Group events by category, sorted by date
        by_category: dict[EventCategory, list[HistoricalEvent]] = {}
        for event in events:
            by_category.setdefault(event.category, []).append(event)

        patterns: list[SocialPattern] = []

        for category, cat_events in by_category.items():
            if len(cat_events) < 3:
                continue

            sorted_events = sorted(cat_events, key=lambda e: e.date)
            years = [e.date.year for e in sorted_events]
            intervals = [years[i + 1] - years[i] for i in range(len(years) - 1)]

            if not intervals:
                continue

            mean_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))

            if mean_interval <= 0:
                continue

            # Only include if period is in [1, 500] years
            if not (1 <= mean_interval <= 500):
                continue

            # Cyclic if std < 20% of mean
            if std_interval >= 0.2 * mean_interval:
                continue

            # Confidence = 1.0 - (std / mean), normalized to [0.3, 1.0]
            raw_confidence = 1.0 - (std_interval / mean_interval)
            confidence = float(np.clip(raw_confidence, 0.3, 1.0))

            supporting = [e.id for e in cat_events]

            pattern = SocialPattern(
                id=str(uuid.uuid4()),
                name=f"cyclic_{category.value}",
                confidence_score=confidence,
                causality_graph=nx.DiGraph(),
                supporting_events=supporting,
                is_quantum_detected=False,
                recurrence_period_years=mean_interval,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Build a simple self-loop graph for cyclic patterns
            graph = nx.DiGraph()
            graph.add_node(category)
            graph.add_edge(category, category, weight=len(intervals))
            pattern.causality_graph = graph

            patterns.append(pattern)

        return patterns
