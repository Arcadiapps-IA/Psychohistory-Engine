"""
Trajectory_Predictor: computes the three most probable future trajectories
using the Quantum_Engine (QAOA + optional Grover) and applies the
Uncertainty_Bound principle.

Guarantees determinism via a fixed seed passed to all quantum and numpy calls.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone

import numpy as np

from psychohistory.exceptions import InsufficientDataError, InvalidHorizonError
from psychohistory.models import (
    PredictedEvent,
    ReasoningStep,
    ReasoningTrace,
    SocialPattern,
    Trajectory,
    TrajectoryNode,
    UncertaintyBound,
)
from psychohistory.quantum_engine import QuantumEngine
from psychohistory.uncertainty import compute_uncertainty_bound

_logger = logging.getLogger(__name__)

# Number of trajectories to return
N_TRAJECTORIES = 3

# Threshold for invalidating stale trajectories
STALE_THRESHOLD = 0.05

# Threshold for Grover search
GROVER_SPACE_THRESHOLD = 10_000


class TrajectoryPredictor:
    """
    Computes the three most probable future trajectories for a given set of
    Social_Patterns and a prediction horizon.

    Parameters
    ----------
    quantum_engine:
        Optional QuantumEngine instance. A default one is created if not provided.
    repository:
        Optional CorpusRepository. When provided, corpus size is validated
        before prediction (must be >= 1000 events).
    """

    MIN_CORPUS_SIZE = 1_000
    MIN_HORIZON = 1
    MAX_HORIZON = 1_000

    def __init__(
        self,
        quantum_engine: QuantumEngine | None = None,
        repository=None,
    ) -> None:
        self._qe: QuantumEngine = quantum_engine or QuantumEngine()
        self._repository = repository
        self._active_trajectories: list[Trajectory] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        patterns: list[SocialPattern],
        horizon_years: int,
        seed: int = 42,
    ) -> list[Trajectory]:
        """
        Compute the three most probable trajectories.

        Parameters
        ----------
        patterns:
            Active Social_Patterns to use as basis for prediction.
        horizon_years:
            Prediction horizon in years. Must be in [1, 1000].
        seed:
            Random seed for determinism.

        Returns
        -------
        list[Trajectory]
            Exactly 3 trajectories ordered by descending confidence_score.

        Raises
        ------
        InsufficientDataError
            When a repository is provided and contains fewer than 1,000 events.
        InvalidHorizonError
            When horizon_years is outside [1, 1000].
        """
        # Validate corpus size
        if self._repository is not None:
            corpus_size = self._repository.count()
            if corpus_size < self.MIN_CORPUS_SIZE:
                raise InsufficientDataError(
                    f"Corpus contains {corpus_size} events; "
                    f"minimum required for reliable predictions is {self.MIN_CORPUS_SIZE}."
                )

        # Validate horizon
        if not (self.MIN_HORIZON <= horizon_years <= self.MAX_HORIZON):
            raise InvalidHorizonError(
                f"horizon_years={horizon_years} is outside the valid range "
                f"[{self.MIN_HORIZON}, {self.MAX_HORIZON}]."
            )

        # Determine QAOA parameters
        n_configurations = max(N_TRAJECTORIES, len(patterns) * 10)
        n_qubits = min(len(patterns) + 2, QuantumEngine.MAX_QAOA_QUBITS)
        if n_qubits < 1:
            n_qubits = 1

        # Run QAOA to explore trajectory space
        qaoa_result = self._qe.run_qaoa(
            n_configurations=n_configurations,
            n_qubits=n_qubits,
            seed=seed,
        )

        # Optionally run Grover if space is large
        if n_configurations > GROVER_SPACE_THRESHOLD:
            grover_n_qubits = min(n_qubits, QuantumEngine.MAX_GROVER_QUBITS)
            self._qe.run_grover(
                search_space_size=n_configurations,
                n_targets=N_TRAJECTORIES,
                n_qubits=grover_n_qubits,
                seed=seed,
            )

        # Build trajectories from top-3 QAOA configurations
        top_configs = qaoa_result.top_configurations[:N_TRAJECTORIES]

        # Pad to exactly N_TRAJECTORIES if fewer configs returned
        while len(top_configs) < N_TRAJECTORIES:
            rng = np.random.default_rng(seed + len(top_configs))
            top_configs.append(
                {
                    "config_index": len(top_configs),
                    "probability": float(rng.uniform(0.1, 0.5)),
                    "binary_string": "0" * n_qubits,
                }
            )

        trajectories: list[Trajectory] = []
        corpus_hash = "no_repository"
        if self._repository is not None:
            corpus_hash = self._repository.get_snapshot_hash()

        for rank, config in enumerate(top_configs):
            traj_id = str(uuid.uuid4())
            nodes = self._build_nodes(config, patterns, horizon_years, seed, rank)
            confidence = self._compute_confidence_score(nodes)
            ub = compute_uncertainty_bound(
                sigma_state=0.1,
                sigma_momentum=0.1,
                h_social=0.01,
                horizon_years=horizon_years,
            )
            reasoning = self._build_reasoning_trace(traj_id, nodes, patterns)

            traj = Trajectory(
                id=traj_id,
                nodes=nodes,
                confidence_score=confidence,
                horizon_years=horizon_years,
                uncertainty_bound=ub,
                reasoning_trace=reasoning,
                created_at=datetime.now(timezone.utc),
                corpus_snapshot_hash=corpus_hash,
                seed=seed,
            )
            traj = self._apply_uncertainty_bound(traj, horizon_years)
            trajectories.append(traj)

        # Sort by descending confidence
        trajectories.sort(key=lambda t: t.confidence_score, reverse=True)

        self._active_trajectories = trajectories
        return trajectories

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_nodes(
        self,
        config: dict,
        patterns: list[SocialPattern],
        horizon_years: int,
        seed: int,
        rank: int,
    ) -> list[TrajectoryNode]:
        """Build TrajectoryNode list from a QAOA configuration."""
        rng = np.random.default_rng(seed + rank * 1000)

        # Number of nodes proportional to horizon (1 per decade, min 1, max 10)
        n_nodes = max(1, min(10, horizon_years // 10))

        nodes: list[TrajectoryNode] = []
        base_prob = config.get("probability", 0.5)

        for i in range(n_nodes):
            # Deterministic probability variation per node
            node_prob = float(np.clip(base_prob * (1.0 - 0.05 * i), 0.01, 1.0))

            # Select supporting patterns (deterministic via seed)
            if patterns:
                n_supporting = max(1, min(len(patterns), 3))
                pattern_indices = rng.choice(len(patterns), size=n_supporting, replace=False)
                supporting_ids = [patterns[int(idx)].id for idx in pattern_indices]
            else:
                supporting_ids = []

            # Build predicted event description from patterns
            if patterns and supporting_ids:
                pattern_names = [
                    p.name for p in patterns if p.id in supporting_ids
                ]
                event_desc = f"Predicted event based on patterns: {', '.join(pattern_names[:2])}"
            else:
                event_desc = f"Predicted event at node {i} (horizon={horizon_years}y)"

            from psychohistory.enums import EventCategory

            predicted = PredictedEvent(
                description=event_desc,
                estimated_date=datetime(
                    min(9999, datetime.now(timezone.utc).year + (horizon_years * (i + 1) // n_nodes)),
                    1,
                    1,
                ),
                category=EventCategory.SOCIAL,
                location_hint=None,
            )

            nodes.append(
                TrajectoryNode(
                    sequence_index=i,
                    predicted_event=predicted,
                    probability=node_prob,
                    supporting_patterns=supporting_ids,
                )
            )

        return nodes

    def _apply_uncertainty_bound(
        self,
        trajectory: Trajectory,
        horizon_years: int,
    ) -> Trajectory:
        """Recompute and attach the UncertaintyBound to the trajectory."""
        ub = compute_uncertainty_bound(
            sigma_state=0.1,
            sigma_momentum=0.1,
            h_social=0.01,
            horizon_years=horizon_years,
        )
        trajectory.uncertainty_bound = ub
        if ub.was_adjusted and trajectory.reasoning_trace is not None:
            trajectory.reasoning_trace.uncertainty_adjustments.append(
                ub.adjustment_reason or "Uncertainty adjusted"
            )
        return trajectory

    def _invalidate_stale(
        self,
        updated_patterns: list[SocialPattern],
    ) -> list[Trajectory]:
        """
        Discard trajectories whose confidence_score would vary by more than
        STALE_THRESHOLD (0.05) given the updated patterns.

        Returns the list of stale (discarded) trajectories.
        """
        stale: list[Trajectory] = []
        valid: list[Trajectory] = []

        for traj in self._active_trajectories:
            # Recompute confidence using updated pattern confidence scores
            new_score = self._recompute_confidence_with_patterns(traj, updated_patterns)
            if abs(new_score - traj.confidence_score) > STALE_THRESHOLD:
                stale.append(traj)
            else:
                valid.append(traj)

        self._active_trajectories = valid
        return stale

    def _recompute_confidence_with_patterns(
        self,
        trajectory: Trajectory,
        patterns: list[SocialPattern],
    ) -> float:
        """Estimate new confidence score given updated patterns."""
        pattern_map = {p.id: p for p in patterns}
        adjusted_nodes: list[TrajectoryNode] = []

        for node in trajectory.nodes:
            # Average confidence of supporting patterns
            supporting_confidences = [
                pattern_map[pid].confidence_score
                for pid in node.supporting_patterns
                if pid in pattern_map
            ]
            if supporting_confidences:
                avg_conf = float(np.mean(supporting_confidences))
                new_prob = float(np.clip(node.probability * avg_conf, 0.01, 1.0))
            else:
                new_prob = node.probability

            adjusted_nodes.append(
                TrajectoryNode(
                    sequence_index=node.sequence_index,
                    predicted_event=node.predicted_event,
                    probability=new_prob,
                    supporting_patterns=node.supporting_patterns,
                )
            )

        return self._compute_confidence_score(adjusted_nodes)

    def _compute_confidence_score(self, nodes: list[TrajectoryNode]) -> float:
        """
        Compute the global confidence score as the product of individual
        node probabilities, calculated in log-space to avoid underflow.
        """
        if not nodes:
            return 0.0

        log_sum = 0.0
        for node in nodes:
            prob = node.probability
            if prob > 0:
                log_sum += math.log(prob)
            else:
                return 0.0

        return float(math.exp(log_sum))

    def _build_reasoning_trace(
        self,
        trajectory_id: str,
        nodes: list[TrajectoryNode],
        patterns: list[SocialPattern],
    ) -> ReasoningTrace:
        """Build a ReasoningTrace linking each node to its supporting patterns."""
        pattern_map = {p.id: p for p in patterns}
        steps: list[ReasoningStep] = []

        for node in nodes:
            # Collect supporting event IDs from the patterns
            event_ids: list[str] = []
            for pid in node.supporting_patterns:
                if pid in pattern_map:
                    event_ids.extend(pattern_map[pid].supporting_events[:5])

            step = ReasoningStep(
                node_index=node.sequence_index,
                predicted_event_description=node.predicted_event.description,
                pattern_ids=list(node.supporting_patterns),
                event_ids=event_ids,
                confidence_contribution=node.probability,
            )
            steps.append(step)

        return ReasoningTrace(
            trajectory_id=trajectory_id,
            steps=steps,
            uncertainty_adjustments=[],
        )
