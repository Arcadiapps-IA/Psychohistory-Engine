"""
Serialization and deserialization of the Psychohistory Engine state.

Uses MessagePack for binary serialization and SHA-256 for integrity verification.

Supported types:
  - datetime → ISO 8601 string
  - np.ndarray → nested list (or None)
  - nx.DiGraph → {"nodes": [...], "edges": [[src, tgt, weight], ...]}
  - EventCategory → string value
  - All dataclasses → recursive dicts
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import msgpack
import networkx as nx
import numpy as np

from psychohistory.enums import EventCategory
from psychohistory.exceptions import StateImportError, StateIntegrityError
from psychohistory.models import (
    EntanglementCorrelation,
    HistoricalEvent,
    InterventionPoint,
    Location,
    PredictedEvent,
    ReasoningStep,
    ReasoningTrace,
    SocialPattern,
    SystemState,
    Trajectory,
    TrajectoryNode,
    UncertaintyBound,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_datetime(dt: datetime | None) -> str | None:
    """Convert datetime to ISO 8601 string."""
    if dt is None:
        return None
    return dt.isoformat()


def _deserialize_datetime(s: str | None) -> datetime | None:
    """Convert ISO 8601 string back to datetime."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _serialize_ndarray(arr: np.ndarray | None) -> list | None:
    """Convert numpy array to nested list."""
    if arr is None:
        return None
    return arr.tolist()


def _deserialize_ndarray(data: list | None) -> np.ndarray | None:
    """Convert nested list back to numpy array."""
    if data is None:
        return None
    return np.array(data)


def _serialize_graph(graph: Any | None) -> dict | None:
    """Convert nx.DiGraph to JSON-compatible dict."""
    if graph is None:
        return None
    if not isinstance(graph, nx.DiGraph):
        return None

    nodes = []
    for node in graph.nodes():
        if isinstance(node, EventCategory):
            nodes.append(node.value)
        else:
            nodes.append(str(node))

    edges = []
    for src, tgt, data in graph.edges(data=True):
        src_val = src.value if isinstance(src, EventCategory) else str(src)
        tgt_val = tgt.value if isinstance(tgt, EventCategory) else str(tgt)
        weight = data.get("weight", 1)
        edges.append([src_val, tgt_val, weight])

    return {"nodes": nodes, "edges": edges}


def _deserialize_graph(data: dict | None) -> nx.DiGraph:
    """Convert dict back to nx.DiGraph."""
    graph = nx.DiGraph()
    if data is None:
        return graph

    for node_val in data.get("nodes", []):
        # Try to convert back to EventCategory
        try:
            node = EventCategory(node_val)
        except (ValueError, KeyError):
            node = node_val
        graph.add_node(node)

    for edge in data.get("edges", []):
        if len(edge) >= 2:
            src_val, tgt_val = edge[0], edge[1]
            weight = edge[2] if len(edge) > 2 else 1
            try:
                src = EventCategory(src_val)
            except (ValueError, KeyError):
                src = src_val
            try:
                tgt = EventCategory(tgt_val)
            except (ValueError, KeyError):
                tgt = tgt_val
            graph.add_edge(src, tgt, weight=weight)

    return graph


# ---------------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------------


def _serialize_location(loc: Location | None) -> dict | None:
    if loc is None:
        return None
    return {
        "name": loc.name,
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "region": loc.region,
        "country": loc.country,
    }


def _deserialize_location(data: dict | None) -> Location | None:
    if data is None:
        return None
    return Location(
        name=data["name"],
        latitude=data.get("latitude"),
        longitude=data.get("longitude"),
        region=data.get("region"),
        country=data.get("country"),
    )


def _serialize_historical_event(event: HistoricalEvent) -> dict:
    return {
        "id": event.id,
        "date": _serialize_datetime(event.date),
        "description": event.description,
        "category": event.category.value if event.category else None,
        "location": _serialize_location(event.location),
        "actors": event.actors,
        "magnitude": event.magnitude,
        "source_url": event.source_url,
        "connector_name": event.connector_name,
        "extraction_timestamp": _serialize_datetime(event.extraction_timestamp),
        "raw_document_id": event.raw_document_id,
        "feature_vector": _serialize_ndarray(event.feature_vector),
    }


def _deserialize_historical_event(data: dict) -> HistoricalEvent:
    cat_val = data.get("category")
    try:
        category = EventCategory(cat_val) if cat_val else EventCategory.SOCIAL
    except ValueError:
        category = EventCategory.SOCIAL

    return HistoricalEvent(
        id=data["id"],
        date=_deserialize_datetime(data["date"]),
        description=data["description"],
        category=category,
        location=_deserialize_location(data.get("location")),
        actors=data.get("actors") or [],
        magnitude=data.get("magnitude"),
        source_url=data.get("source_url"),
        connector_name=data.get("connector_name"),
        extraction_timestamp=_deserialize_datetime(data.get("extraction_timestamp")),
        raw_document_id=data.get("raw_document_id"),
        feature_vector=_deserialize_ndarray(data.get("feature_vector")),
    )


def _serialize_social_pattern(pattern: SocialPattern) -> dict:
    return {
        "id": pattern.id,
        "name": pattern.name,
        "confidence_score": pattern.confidence_score,
        "causality_graph": _serialize_graph(pattern.causality_graph),
        "supporting_events": pattern.supporting_events,
        "is_quantum_detected": pattern.is_quantum_detected,
        "recurrence_period_years": pattern.recurrence_period_years,
        "qubits_used": pattern.qubits_used,
        "created_at": _serialize_datetime(pattern.created_at),
        "updated_at": _serialize_datetime(pattern.updated_at),
    }


def _deserialize_social_pattern(data: dict) -> SocialPattern:
    return SocialPattern(
        id=data["id"],
        name=data["name"],
        confidence_score=data["confidence_score"],
        causality_graph=_deserialize_graph(data.get("causality_graph")),
        supporting_events=data.get("supporting_events") or [],
        is_quantum_detected=data.get("is_quantum_detected", False),
        recurrence_period_years=data.get("recurrence_period_years"),
        qubits_used=data.get("qubits_used"),
        created_at=_deserialize_datetime(data.get("created_at")),
        updated_at=_deserialize_datetime(data.get("updated_at")),
    )


def _serialize_predicted_event(pe: PredictedEvent) -> dict:
    return {
        "description": pe.description,
        "estimated_date": _serialize_datetime(pe.estimated_date),
        "category": pe.category.value if pe.category else None,
        "location_hint": pe.location_hint,
    }


def _deserialize_predicted_event(data: dict) -> PredictedEvent:
    cat_val = data.get("category")
    try:
        category = EventCategory(cat_val) if cat_val else EventCategory.SOCIAL
    except ValueError:
        category = EventCategory.SOCIAL
    return PredictedEvent(
        description=data["description"],
        estimated_date=_deserialize_datetime(data["estimated_date"]),
        category=category,
        location_hint=data.get("location_hint"),
    )


def _serialize_entanglement_correlation(ec: EntanglementCorrelation) -> dict:
    return {
        "node_index": ec.node_index,
        "entanglement_metric": ec.entanglement_metric,
    }


def _deserialize_entanglement_correlation(data: dict) -> EntanglementCorrelation:
    return EntanglementCorrelation(
        node_index=data["node_index"],
        entanglement_metric=data["entanglement_metric"],
    )


def _serialize_trajectory_node(node: TrajectoryNode) -> dict:
    return {
        "sequence_index": node.sequence_index,
        "predicted_event": _serialize_predicted_event(node.predicted_event),
        "probability": node.probability,
        "supporting_patterns": node.supporting_patterns,
        "entanglement_correlations": [
            _serialize_entanglement_correlation(ec)
            for ec in node.entanglement_correlations
        ],
    }


def _deserialize_trajectory_node(data: dict) -> TrajectoryNode:
    return TrajectoryNode(
        sequence_index=data["sequence_index"],
        predicted_event=_deserialize_predicted_event(data["predicted_event"]),
        probability=data["probability"],
        supporting_patterns=data.get("supporting_patterns") or [],
        entanglement_correlations=[
            _deserialize_entanglement_correlation(ec)
            for ec in data.get("entanglement_correlations") or []
        ],
    )


def _serialize_uncertainty_bound(ub: UncertaintyBound) -> dict:
    return {
        "sigma_state": ub.sigma_state,
        "sigma_momentum": ub.sigma_momentum,
        "product": ub.product,
        "h_social": ub.h_social,
        "was_adjusted": ub.was_adjusted,
        "adjustment_reason": ub.adjustment_reason,
    }


def _deserialize_uncertainty_bound(data: dict) -> UncertaintyBound:
    return UncertaintyBound(
        sigma_state=data["sigma_state"],
        sigma_momentum=data["sigma_momentum"],
        product=data["product"],
        h_social=data["h_social"],
        was_adjusted=data["was_adjusted"],
        adjustment_reason=data.get("adjustment_reason"),
    )


def _serialize_reasoning_step(step: ReasoningStep) -> dict:
    return {
        "node_index": step.node_index,
        "predicted_event_description": step.predicted_event_description,
        "pattern_ids": step.pattern_ids,
        "event_ids": step.event_ids,
        "confidence_contribution": step.confidence_contribution,
    }


def _deserialize_reasoning_step(data: dict) -> ReasoningStep:
    return ReasoningStep(
        node_index=data["node_index"],
        predicted_event_description=data["predicted_event_description"],
        pattern_ids=data.get("pattern_ids") or [],
        event_ids=data.get("event_ids") or [],
        confidence_contribution=data.get("confidence_contribution", 0.0),
    )


def _serialize_reasoning_trace(rt: ReasoningTrace) -> dict:
    return {
        "trajectory_id": rt.trajectory_id,
        "steps": [_serialize_reasoning_step(s) for s in rt.steps],
        "uncertainty_adjustments": rt.uncertainty_adjustments,
    }


def _deserialize_reasoning_trace(data: dict) -> ReasoningTrace:
    return ReasoningTrace(
        trajectory_id=data["trajectory_id"],
        steps=[_deserialize_reasoning_step(s) for s in data.get("steps") or []],
        uncertainty_adjustments=data.get("uncertainty_adjustments") or [],
    )


def _serialize_trajectory(traj: Trajectory) -> dict:
    return {
        "id": traj.id,
        "nodes": [_serialize_trajectory_node(n) for n in traj.nodes],
        "confidence_score": traj.confidence_score,
        "horizon_years": traj.horizon_years,
        "uncertainty_bound": _serialize_uncertainty_bound(traj.uncertainty_bound),
        "reasoning_trace": _serialize_reasoning_trace(traj.reasoning_trace),
        "created_at": _serialize_datetime(traj.created_at),
        "corpus_snapshot_hash": traj.corpus_snapshot_hash,
        "seed": traj.seed,
    }


def _deserialize_trajectory(data: dict) -> Trajectory:
    return Trajectory(
        id=data["id"],
        nodes=[_deserialize_trajectory_node(n) for n in data.get("nodes") or []],
        confidence_score=data["confidence_score"],
        horizon_years=data["horizon_years"],
        uncertainty_bound=_deserialize_uncertainty_bound(data["uncertainty_bound"]),
        reasoning_trace=_deserialize_reasoning_trace(data["reasoning_trace"]),
        created_at=_deserialize_datetime(data["created_at"]),
        corpus_snapshot_hash=data["corpus_snapshot_hash"],
        seed=data["seed"],
    )


def _serialize_system_state(state: SystemState) -> dict:
    """Serialize SystemState to a JSON-compatible dict (without integrity_hash)."""
    return {
        "version": state.version,
        "created_at": _serialize_datetime(state.created_at),
        "corpus": [_serialize_historical_event(e) for e in state.corpus],
        "patterns": [_serialize_social_pattern(p) for p in state.patterns],
        "active_trajectories": [_serialize_trajectory(t) for t in state.active_trajectories],
        "connector_configs": state.connector_configs,
    }


def _deserialize_system_state(data: dict) -> SystemState:
    """Reconstruct SystemState from a dict."""
    try:
        return SystemState(
            version=data["version"],
            created_at=_deserialize_datetime(data["created_at"]),
            corpus=[_deserialize_historical_event(e) for e in data.get("corpus") or []],
            patterns=[_deserialize_social_pattern(p) for p in data.get("patterns") or []],
            active_trajectories=[
                _deserialize_trajectory(t) for t in data.get("active_trajectories") or []
            ],
            connector_configs=data.get("connector_configs") or [],
            integrity_hash=data.get("integrity_hash", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateImportError(f"Invalid state format: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_state(engine_state: SystemState, path: str) -> None:
    """
    Serialize the engine state to a MessagePack file with SHA-256 integrity hash.

    The hash is computed over the serialized content *excluding* the
    ``integrity_hash`` field itself, then added before the final write.

    Parameters
    ----------
    engine_state:
        The SystemState to serialize.
    path:
        File path to write the binary MessagePack data.
    """
    # Serialize without integrity_hash
    state_dict = _serialize_system_state(engine_state)
    serialized_without_hash = msgpack.packb(state_dict, use_bin_type=True)

    # Compute SHA-256 of the content without hash
    integrity_hash = hashlib.sha256(serialized_without_hash).hexdigest()

    # Add hash and re-serialize
    state_dict["integrity_hash"] = integrity_hash
    final_bytes = msgpack.packb(state_dict, use_bin_type=True)

    with open(path, "wb") as f:
        f.write(final_bytes)


def import_state(path: str) -> SystemState:
    """
    Deserialize a MessagePack state file and verify its SHA-256 integrity hash.

    Parameters
    ----------
    path:
        File path to read the binary MessagePack data from.

    Returns
    -------
    SystemState
        The restored engine state.

    Raises
    ------
    StateIntegrityError
        When the stored hash does not match the recomputed hash.
    StateImportError
        When the file format is invalid or cannot be parsed.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
        data = msgpack.unpackb(raw, raw=False)
    except Exception as exc:
        raise StateImportError(f"Cannot read or parse state file: {exc}") from exc

    if not isinstance(data, dict):
        raise StateImportError("State file does not contain a valid dict.")

    # Extract and verify hash
    stored_hash = data.pop("integrity_hash", None)
    if stored_hash is None:
        raise StateIntegrityError("State file is missing the integrity_hash field.")

    # Recompute hash over content without integrity_hash
    recalculated_bytes = msgpack.packb(data, use_bin_type=True)
    recalculated_hash = hashlib.sha256(recalculated_bytes).hexdigest()

    if stored_hash != recalculated_hash:
        raise StateIntegrityError(
            f"State file integrity check failed: "
            f"stored={stored_hash}, computed={recalculated_hash}"
        )

    return _deserialize_system_state(data)
