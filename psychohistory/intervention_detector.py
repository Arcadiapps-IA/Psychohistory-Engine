"""
Intervention_Detector: identifies Seldon Crises in a Trajectory by computing
the Sensitivity_Index for each node.

SI(node) = 0.6 * D(node) + 0.4 * E(node)

where:
  D = divergence (average distance between original and perturbed trajectories)
  E = average Entanglement_Metric (Von Neumann entropy) between the node and
      all other nodes in the trajectory

A node is classified as a Seldon_Crisis when SI > 0.7.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from psychohistory.models import (
    EntanglementCorrelation,
    InterventionPoint,
    Trajectory,
    TrajectoryNode,
)
from psychohistory.quantum_engine import QuantumEngine

_logger = logging.getLogger(__name__)


class InterventionDetector:
    """
    Identifies Seldon Crises in a Trajectory.

    Parameters
    ----------
    quantum_engine:
        Optional QuantumEngine instance. A default one is created if not provided.
    """

    CRISIS_THRESHOLD = 0.7
    ENTANGLEMENT_THRESHOLD = 0.6
    ALPHA = 0.6  # weight of divergence vs entanglement

    def __init__(self, quantum_engine: QuantumEngine | None = None) -> None:
        self._qe: QuantumEngine = quantum_engine or QuantumEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, trajectory: Trajectory) -> list[InterventionPoint]:
        """
        Analyse each node of the trajectory and return Seldon Crises.

        Returns
        -------
        list[InterventionPoint]
            Crisis points sorted by descending Sensitivity_Index.
            Empty list if no node exceeds the crisis threshold.
        """
        if not trajectory.nodes:
            return []

        intervention_points: list[InterventionPoint] = []

        for node_index, node in enumerate(trajectory.nodes):
            si = self._compute_sensitivity_index(node_index, trajectory)

            if si > self.CRISIS_THRESHOLD:
                # Find entangled nodes (Entanglement_Metric > 0.6)
                entangled: list[EntanglementCorrelation] = []
                for j, other_node in enumerate(trajectory.nodes):
                    if j == node_index:
                        continue
                    sv, total_q = self._build_state_vector(node, other_node)
                    ent_result = self._qe.von_neumann_entropy(
                        state_vector=sv,
                        subsystem_qubits=[0],
                        total_qubits=total_q,
                    )
                    if ent_result.entropy > self.ENTANGLEMENT_THRESHOLD:
                        entangled.append(
                            EntanglementCorrelation(
                                node_index=j,
                                entanglement_metric=ent_result.entropy,
                            )
                        )

                diff_impact = self._compute_differential_impact(node_index, trajectory)

                # Collect supporting patterns from the node
                supporting_patterns = list(node.supporting_patterns)

                ip = InterventionPoint(
                    id=str(uuid.uuid4()),
                    trajectory_id=trajectory.id,
                    node_index=node_index,
                    temporal_coordinates=node.predicted_event.estimated_date,
                    recommended_action_type="intervention",
                    sensitivity_index=si,
                    differential_impact=diff_impact,
                    relevant_actor_categories=[],
                    entangled_nodes=entangled,
                    supporting_patterns=supporting_patterns,
                )
                intervention_points.append(ip)

        # Sort by descending Sensitivity_Index
        intervention_points.sort(key=lambda ip: ip.sensitivity_index, reverse=True)
        return intervention_points

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_sensitivity_index(
        self,
        node_index: int,
        trajectory: Trajectory,
    ) -> float:
        """
        Compute SI = ALPHA * D + (1 - ALPHA) * E.

        D = divergence with N=10 perturbations ±10%
        E = average Entanglement_Metric via QE.von_neumann_entropy()
        """
        d = self._compute_divergence(node_index, trajectory)
        e = self._compute_entanglement_avg(node_index, trajectory)
        si = self.ALPHA * d + (1.0 - self.ALPHA) * e
        return float(np.clip(si, 0.0, 1.0))

    def _compute_divergence(
        self,
        node_index: int,
        trajectory: Trajectory,
    ) -> float:
        """
        Compute divergence D using N=10 perturbations of ±10% on the node probability.

        Returns the average distance between the original trajectory confidence
        and the perturbed ones.
        """
        N = 10
        original_node = trajectory.nodes[node_index]
        original_prob = original_node.probability

        distances: list[float] = []
        for i in range(N):
            # Deterministic perturbation: ±10% alternating
            perturbation = 1.0 + (0.1 if i % 2 == 0 else -0.1) * ((i // 2 + 1) / (N // 2))
            perturbed_prob = float(np.clip(original_prob * perturbation, 0.01, 1.0))

            # Compute perturbed trajectory confidence
            perturbed_probs = [
                perturbed_prob if j == node_index else node.probability
                for j, node in enumerate(trajectory.nodes)
            ]
            original_probs = [node.probability for node in trajectory.nodes]

            # Wasserstein-like distance: mean absolute difference
            dist = float(np.mean(np.abs(np.array(perturbed_probs) - np.array(original_probs))))
            distances.append(dist)

        return float(np.clip(np.mean(distances), 0.0, 1.0))

    def _compute_entanglement_avg(
        self,
        node_index: int,
        trajectory: Trajectory,
    ) -> float:
        """
        Compute the average Entanglement_Metric between node_index and all
        other nodes in the trajectory.
        """
        if len(trajectory.nodes) <= 1:
            return 0.0

        entropies: list[float] = []
        node = trajectory.nodes[node_index]

        for j, other_node in enumerate(trajectory.nodes):
            if j == node_index:
                continue
            sv, total_q = self._build_state_vector(node, other_node)
            result = self._qe.von_neumann_entropy(
                state_vector=sv,
                subsystem_qubits=[0],
                total_qubits=total_q,
            )
            entropies.append(result.entropy)

        if not entropies:
            return 0.0

        return float(np.mean(entropies))

    def _compute_differential_impact(
        self,
        node_index: int,
        trajectory: Trajectory,
    ) -> float:
        """
        Compute the divergence between the original confidence_score and a
        simulated post-intervention confidence_score.

        Simulates intervention by setting the node probability to 1.0.
        """
        original_score = trajectory.confidence_score

        # Simulate intervention: set node probability to 1.0
        intervened_probs = [
            1.0 if j == node_index else node.probability
            for j, node in enumerate(trajectory.nodes)
        ]

        # Compute new confidence in log-space
        import math
        log_sum = 0.0
        for p in intervened_probs:
            if p > 0:
                log_sum += math.log(p)
            else:
                return 0.0

        intervened_score = math.exp(log_sum)
        return float(abs(intervened_score - original_score))

    def _build_state_vector(
        self,
        node: TrajectoryNode,
        other_node: TrajectoryNode,
    ) -> tuple[np.ndarray, int]:
        """
        Build a 2-qubit state vector from the probabilities of two nodes.

        The state is constructed as a product state |ψ_A⟩ ⊗ |ψ_B⟩ where:
          |ψ_A⟩ = sqrt(p_A)|0⟩ + sqrt(1-p_A)|1⟩
          |ψ_B⟩ = sqrt(p_B)|0⟩ + sqrt(1-p_B)|1⟩

        Returns
        -------
        tuple[np.ndarray, int]
            (state_vector of shape (4,), total_qubits=2)
        """
        p_a = float(np.clip(node.probability, 0.0, 1.0))
        p_b = float(np.clip(other_node.probability, 0.0, 1.0))

        # Single-qubit states
        psi_a = np.array([np.sqrt(p_a), np.sqrt(1.0 - p_a)], dtype=complex)
        psi_b = np.array([np.sqrt(p_b), np.sqrt(1.0 - p_b)], dtype=complex)

        # Tensor product: |ψ_A⟩ ⊗ |ψ_B⟩
        state_vector = np.kron(psi_a, psi_b)

        # Normalize
        norm = np.linalg.norm(state_vector)
        if norm > 1e-12:
            state_vector = state_vector / norm

        return state_vector, 2
