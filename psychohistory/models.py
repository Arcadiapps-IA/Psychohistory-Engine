"""Data models (dataclasses) for the Psychohistory Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from psychohistory.enums import EventCategory


@dataclass
class Location:
    """Geographic location associated with a historical event."""

    name: str
    latitude: float | None = None
    longitude: float | None = None
    region: str | None = None
    country: str | None = None


@dataclass
class HistoricalEvent:
    """
    Atomic unit of the Corpus. Represents a normalized historical occurrence.

    Fields ``date`` and ``description`` are mandatory; all others are optional.
    """

    id: str
    date: datetime
    description: str
    category: EventCategory
    location: Location | None = None
    actors: list[str] = field(default_factory=list)
    magnitude: float | None = None
    source_url: str | None = None
    connector_name: str | None = None
    extraction_timestamp: datetime | None = None
    raw_document_id: str | None = None
    # Calculated during analysis; not set at ingestion time
    feature_vector: np.ndarray | None = field(default=None)


@dataclass
class SocialPattern:
    """
    Statistically significant correlation between categories of historical events.

    Represented as a directed causality graph between event categories.
    """

    id: str
    name: str
    confidence_score: float
    causality_graph: Any  # nx.DiGraph — typed as Any to avoid hard networkx import here
    supporting_events: list[str] = field(default_factory=list)
    is_quantum_detected: bool = False
    recurrence_period_years: float | None = None
    qubits_used: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PredictedEvent:
    """A single predicted future event within a trajectory node."""

    description: str
    estimated_date: datetime
    category: EventCategory
    location_hint: str | None = None


@dataclass
class EntanglementCorrelation:
    """Quantum entanglement correlation between two trajectory nodes."""

    node_index: int
    entanglement_metric: float  # Von Neumann entropy, normalized to [0.0, 1.0]


@dataclass
class TrajectoryNode:
    """A single node in a predicted trajectory sequence."""

    sequence_index: int
    predicted_event: PredictedEvent
    probability: float
    supporting_patterns: list[str] = field(default_factory=list)
    entanglement_correlations: list[EntanglementCorrelation] = field(default_factory=list)


@dataclass
class UncertaintyBound:
    """
    Heisenberg-analogue uncertainty bound for social predictions.

    Invariant: ``product >= h_social`` must always hold.
    """

    sigma_state: float       # Uncertainty in Social_State
    sigma_momentum: float    # Uncertainty in Social_Momentum
    product: float           # sigma_state * sigma_momentum
    h_social: float          # Minimum system constant (default: 0.01)
    was_adjusted: bool       # True if adjusted to satisfy the constraint
    adjustment_reason: str | None = None


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace linking a predicted event to its evidence."""

    node_index: int
    predicted_event_description: str
    pattern_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    confidence_contribution: float = 0.0


@dataclass
class ReasoningTrace:
    """Full reasoning trace for a trajectory, linking predictions to historical evidence."""

    trajectory_id: str
    steps: list[ReasoningStep] = field(default_factory=list)
    uncertainty_adjustments: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """
    Ordered sequence of probable future events with associated probabilities.

    ``confidence_score`` equals the product of individual node probabilities.
    ``seed`` guarantees reproducibility for the same corpus state and parameters.
    """

    id: str
    nodes: list[TrajectoryNode]
    confidence_score: float
    horizon_years: int
    uncertainty_bound: UncertaintyBound
    reasoning_trace: ReasoningTrace
    created_at: datetime
    corpus_snapshot_hash: str
    seed: int


@dataclass
class InterventionPoint:
    """
    A Seldon Crisis: a critical bifurcation point in a trajectory where
    a minimal intervention can significantly alter the course of events.

    Classified as a crisis when ``sensitivity_index > 0.7``.
    """

    id: str
    trajectory_id: str
    node_index: int
    temporal_coordinates: datetime
    recommended_action_type: str
    sensitivity_index: float
    differential_impact: float
    relevant_actor_categories: list[str] = field(default_factory=list)
    entangled_nodes: list[EntanglementCorrelation] = field(default_factory=list)
    supporting_patterns: list[str] = field(default_factory=list)


@dataclass
class RawSourceDocument:
    """
    A document in its original format as returned by an external source,
    before transformation to the canonical HistoricalEvent schema.
    """

    id: str
    source_url: str
    connector_name: str
    raw_content: str
    content_type: str
    extraction_timestamp: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class SystemState:
    """
    Complete serializable state of the Psychohistory Engine.

    Used for export/import (round-trip serialization).
    ``integrity_hash`` is the SHA-256 of the serialized content excluding itself.
    """

    version: str
    created_at: datetime
    corpus: list[HistoricalEvent] = field(default_factory=list)
    patterns: list[SocialPattern] = field(default_factory=list)
    active_trajectories: list[Trajectory] = field(default_factory=list)
    connector_configs: list[dict] = field(default_factory=list)
    integrity_hash: str = ""
