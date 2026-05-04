"""
Explainability_Reporter: generates human-readable explanations for Trajectories
and Intervention_Points.

Provides:
  - ExplanabilityReport dataclass
  - ExplainabilityReporter class with register_trajectory(), get_explanation(),
    and generate_report() methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from psychohistory.models import InterventionPoint, Trajectory

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExplanabilityReport:
    """Human-readable explanation for a Trajectory."""

    trajectory_id: str
    summary: str
    reasoning_steps: list[str]
    uncertainty_description: str
    intervention_descriptions: list[str]
    generated_at: datetime


# ---------------------------------------------------------------------------
# ExplainabilityReporter
# ---------------------------------------------------------------------------


class ExplainabilityReporter:
    """
    Generates human-readable explanations for Trajectories and Intervention_Points.

    Parameters
    ----------
    repository:
        Optional CorpusRepository. When provided, pattern lookups can be
        enriched with corpus data.
    """

    def __init__(self, repository=None) -> None:
        self._trajectories: dict[str, Trajectory] = {}
        self._repository = repository

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_trajectory(self, trajectory: Trajectory) -> None:
        """Register a trajectory so it can be retrieved by ID."""
        self._trajectories[trajectory.id] = trajectory

    def get_explanation(self, trajectory_id: str) -> ExplanabilityReport:
        """
        Generate an ExplanabilityReport for the given trajectory ID.

        Parameters
        ----------
        trajectory_id:
            ID of the trajectory to explain.

        Returns
        -------
        ExplanabilityReport
            Human-readable report with reasoning steps, uncertainty description,
            and intervention descriptions.

        Raises
        ------
        KeyError
            When the trajectory_id is not registered.
        """
        trajectory = self._trajectories[trajectory_id]
        return self._build_report(trajectory, intervention_points=None)

    def generate_report(
        self,
        trajectory: Trajectory,
        intervention_points: list[InterventionPoint] | None = None,
    ) -> str:
        """
        Generate a human-readable string report for a trajectory.

        Parameters
        ----------
        trajectory:
            The trajectory to explain.
        intervention_points:
            Optional list of detected Seldon Crises for this trajectory.

        Returns
        -------
        str
            Multi-line human-readable report.
        """
        lines: list[str] = []

        # Header
        lines.append("=" * 60)
        lines.append("PSYCHOHISTORY ENGINE — TRAJECTORY REPORT")
        lines.append("=" * 60)
        lines.append(f"Trajectory ID : {trajectory.id}")
        lines.append(f"Horizon       : {trajectory.horizon_years} years")
        lines.append(f"Confidence    : {trajectory.confidence_score:.6f}")
        lines.append(f"Seed          : {trajectory.seed}")
        lines.append(f"Created at    : {trajectory.created_at.isoformat()}")
        lines.append("")

        # Uncertainty Bound
        ub = trajectory.uncertainty_bound
        lines.append("UNCERTAINTY BOUND")
        lines.append("-" * 40)
        lines.append(f"  σ_state    = {ub.sigma_state:.6f}")
        lines.append(f"  σ_momentum = {ub.sigma_momentum:.6f}")
        lines.append(f"  product    = {ub.product:.6f}  (ħ_social = {ub.h_social})")
        if ub.was_adjusted:
            lines.append(f"  ⚠ Adjusted : {ub.adjustment_reason}")
        else:
            lines.append("  ✓ No adjustment required")
        lines.append("")

        # Trajectory nodes
        lines.append("PREDICTED EVENTS")
        lines.append("-" * 40)
        for node in trajectory.nodes:
            pe = node.predicted_event
            lines.append(
                f"  [{node.sequence_index}] {pe.estimated_date.strftime('%Y-%m-%d')} "
                f"(p={node.probability:.4f})"
            )
            lines.append(f"      {pe.description}")
            if node.supporting_patterns:
                lines.append(f"      Patterns: {', '.join(node.supporting_patterns[:3])}")
        lines.append("")

        # Reasoning trace
        if trajectory.reasoning_trace and trajectory.reasoning_trace.steps:
            lines.append("REASONING TRACE")
            lines.append("-" * 40)
            for step in trajectory.reasoning_trace.steps:
                lines.append(
                    f"  Step {step.node_index}: {step.predicted_event_description[:80]}"
                )
                if step.pattern_ids:
                    lines.append(f"    Patterns: {', '.join(step.pattern_ids[:3])}")
                if step.event_ids:
                    lines.append(f"    Events  : {', '.join(step.event_ids[:3])}")
            lines.append("")

        # Seldon Crises
        if intervention_points:
            lines.append("SELDON CRISES IDENTIFIED")
            lines.append("-" * 40)
            for ip in intervention_points:
                lines.append(
                    f"  Crisis at node {ip.node_index} "
                    f"({ip.temporal_coordinates.strftime('%Y-%m-%d')})"
                )
                lines.append(f"    Sensitivity Index : {ip.sensitivity_index:.4f}")
                lines.append(f"    Differential Impact: {ip.differential_impact:.4f}")
                lines.append(f"    Recommended action: {ip.recommended_action_type}")
                if ip.supporting_patterns:
                    lines.append(
                        f"    Elevating patterns: {', '.join(ip.supporting_patterns[:3])}"
                    )
                if ip.entangled_nodes:
                    entangled_str = ", ".join(
                        f"node {ec.node_index} (E={ec.entanglement_metric:.3f})"
                        for ec in ip.entangled_nodes
                    )
                    lines.append(f"    Entangled nodes   : {entangled_str}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_report(
        self,
        trajectory: Trajectory,
        intervention_points: list[InterventionPoint] | None,
    ) -> ExplanabilityReport:
        """Build an ExplanabilityReport from a trajectory."""
        # Summary
        summary = (
            f"Trajectory {trajectory.id[:8]}... | "
            f"Horizon: {trajectory.horizon_years} years | "
            f"Confidence: {trajectory.confidence_score:.6f} | "
            f"Nodes: {len(trajectory.nodes)}"
        )

        # Reasoning steps
        reasoning_steps: list[str] = []
        if trajectory.reasoning_trace:
            for step in trajectory.reasoning_trace.steps:
                step_desc = (
                    f"Node {step.node_index}: {step.predicted_event_description}"
                )
                if step.pattern_ids:
                    step_desc += f" [patterns: {', '.join(step.pattern_ids[:3])}]"
                if step.event_ids:
                    step_desc += f" [events: {', '.join(step.event_ids[:3])}]"
                reasoning_steps.append(step_desc)

        # Uncertainty description
        ub = trajectory.uncertainty_bound
        if ub.was_adjusted:
            uncertainty_description = (
                f"Uncertainty bound adjusted: σ_state={ub.sigma_state:.4f}, "
                f"σ_momentum={ub.sigma_momentum:.4f}, product={ub.product:.4f} "
                f"(≥ ħ_social={ub.h_social}). Reason: {ub.adjustment_reason}"
            )
        else:
            uncertainty_description = (
                f"Uncertainty bound satisfied: σ_state={ub.sigma_state:.4f}, "
                f"σ_momentum={ub.sigma_momentum:.4f}, product={ub.product:.4f} "
                f"(≥ ħ_social={ub.h_social}). No adjustment required."
            )

        # Intervention descriptions
        intervention_descriptions: list[str] = []
        if intervention_points:
            for ip in intervention_points:
                desc = (
                    f"Seldon Crisis at node {ip.node_index} "
                    f"(SI={ip.sensitivity_index:.4f}): "
                    f"{ip.recommended_action_type}"
                )
                if ip.supporting_patterns:
                    desc += f". Elevating patterns: {', '.join(ip.supporting_patterns[:3])}"
                intervention_descriptions.append(desc)

        return ExplanabilityReport(
            trajectory_id=trajectory.id,
            summary=summary,
            reasoning_steps=reasoning_steps,
            uncertainty_description=uncertainty_description,
            intervention_descriptions=intervention_descriptions,
            generated_at=datetime.now(timezone.utc),
        )
