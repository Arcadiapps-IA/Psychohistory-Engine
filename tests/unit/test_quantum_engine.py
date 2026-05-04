"""
Unit and property-based tests for QuantumEngine.

Validates: Requirements 4.4, 5.1–5.9
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psychohistory.exceptions import QuantumExecutionError
from psychohistory.quantum_engine import EntanglementResult, QuantumEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_normalized_state(n_qubits: int, seed: int) -> np.ndarray:
    """Return a random normalized state vector for *n_qubits* qubits."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def _random_feature_matrix(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    """Return a random feature matrix."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


# ---------------------------------------------------------------------------
# Propiedad 13: Entanglement_Metric está en rango válido [0.0, 1.0]
# Validates: Requirement 4.4
# ---------------------------------------------------------------------------


@given(
    n_qubits=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_entanglement_metric_in_valid_range(n_qubits: int, seed: int) -> None:
    """
    **Validates: Requirements 4.4**

    For any random normalized state vector and any subsystem of half the
    qubits, the Von Neumann entropy must be in [0.0, 1.0].
    """
    engine = QuantumEngine(seed=seed)
    state_vector = _random_normalized_state(n_qubits, seed)

    # Use the first half of qubits as the subsystem
    subsystem_size = n_qubits // 2
    subsystem_qubits = list(range(subsystem_size))

    result: EntanglementResult = engine.von_neumann_entropy(
        state_vector, subsystem_qubits, n_qubits
    )

    assert isinstance(result.entropy, float), (
        f"entropy must be float, got {type(result.entropy)}"
    )
    assert 0.0 <= result.entropy <= 1.0, (
        f"entropy {result.entropy} is outside [0.0, 1.0] for "
        f"n_qubits={n_qubits}, seed={seed}"
    )


# ---------------------------------------------------------------------------
# Propiedad 11 (parte cuántica): Determinismo con la misma semilla
# Validates: Requirements 3.7, 5.9
# ---------------------------------------------------------------------------


@given(
    n_qubits=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_quantum_engine_determinism(n_qubits: int, seed: int) -> None:
    """
    **Validates: Requirements 3.7, 5.9**

    Running run_qaoa twice with the same seed must produce identical results.
    """
    engine = QuantumEngine(seed=seed)
    n_configurations = max(3, n_qubits * 2)

    result1 = engine.run_qaoa(n_configurations, n_qubits, p_layers=2, seed=seed)
    result2 = engine.run_qaoa(n_configurations, n_qubits, p_layers=2, seed=seed)

    assert result1.top_configurations == result2.top_configurations, (
        f"QAOA results differ for n_qubits={n_qubits}, seed={seed}:\n"
        f"  run1: {result1.top_configurations}\n"
        f"  run2: {result2.top_configurations}"
    )
    assert result1.seed == result2.seed
    assert result1.backend == result2.backend


# ---------------------------------------------------------------------------
# Tests de ejemplo: validación de límites de qubits
# ---------------------------------------------------------------------------


def test_vqc_raises_on_too_many_qubits() -> None:
    """train_vqc must raise QuantumExecutionError when n_qubits > 50."""
    engine = QuantumEngine()
    feature_matrix = _random_feature_matrix(10, 5, seed=0)

    with pytest.raises(QuantumExecutionError) as exc_info:
        engine.train_vqc(feature_matrix, n_qubits=51)

    err = exc_info.value
    assert err.circuit_type == "VQC"
    assert err.invalid_param == "n_qubits"
    assert "50" in err.accepted_range


def test_qaoa_raises_on_too_many_qubits() -> None:
    """run_qaoa must raise QuantumExecutionError when n_qubits > 30."""
    engine = QuantumEngine()

    with pytest.raises(QuantumExecutionError) as exc_info:
        engine.run_qaoa(n_configurations=10, n_qubits=31)

    err = exc_info.value
    assert err.circuit_type == "QAOA"
    assert err.invalid_param == "n_qubits"
    assert "30" in err.accepted_range


def test_grover_raises_on_too_many_qubits() -> None:
    """run_grover must raise QuantumExecutionError when n_qubits > 20."""
    engine = QuantumEngine()

    with pytest.raises(QuantumExecutionError) as exc_info:
        engine.run_grover(search_space_size=100, n_targets=5, n_qubits=21)

    err = exc_info.value
    assert err.circuit_type == "GROVER"
    assert err.invalid_param == "n_qubits"
    assert "20" in err.accepted_range


# ---------------------------------------------------------------------------
# Tests de Von Neumann entropy: casos especiales
# ---------------------------------------------------------------------------


def test_von_neumann_entropy_pure_state() -> None:
    """
    For a product (non-entangled) state |00⟩, the Von Neumann entropy of
    any subsystem must be 0.0.
    """
    engine = QuantumEngine()
    # |00⟩ = [1, 0, 0, 0]
    state_vector = np.array([1.0, 0.0, 0.0, 0.0])
    result = engine.von_neumann_entropy(
        state_vector, subsystem_qubits=[0], total_qubits=2
    )
    assert result.entropy == pytest.approx(0.0, abs=1e-10), (
        f"Expected entropy=0.0 for product state |00⟩, got {result.entropy}"
    )


def test_von_neumann_entropy_maximally_entangled() -> None:
    """
    For the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2, the Von Neumann entropy
    of either single-qubit subsystem must be 1.0 (maximally entangled).
    """
    engine = QuantumEngine()
    # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    state_vector = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)])
    result = engine.von_neumann_entropy(
        state_vector, subsystem_qubits=[0], total_qubits=2
    )
    assert result.entropy == pytest.approx(1.0, abs=1e-10), (
        f"Expected entropy=1.0 for Bell state |Φ+⟩, got {result.entropy}"
    )


# ---------------------------------------------------------------------------
# Additional sanity tests
# ---------------------------------------------------------------------------


def test_vqc_returns_valid_result() -> None:
    """train_vqc must return a VQCResult with valid fields."""
    engine = QuantumEngine(seed=42)
    feature_matrix = _random_feature_matrix(20, 10, seed=42)
    result = engine.train_vqc(feature_matrix, n_qubits=4, seed=42)

    assert result.n_qubits == 4
    assert result.seed == 42
    assert result.execution_ms >= 0.0
    assert result.backend in ("hardware", "simulator", "classical")
    assert isinstance(result.patterns, list)
    for p in result.patterns:
        assert "pattern_id" in p
        assert "confidence_score" in p
        assert "feature_indices" in p
        assert 0.0 <= p["confidence_score"] <= 1.0


def test_qaoa_returns_top_3_configurations() -> None:
    """run_qaoa must return at most 3 top configurations."""
    engine = QuantumEngine(seed=42)
    result = engine.run_qaoa(n_configurations=20, n_qubits=4, p_layers=2, seed=42)

    assert len(result.top_configurations) <= 3
    assert result.n_qubits == 4
    assert result.p_layers == 2
    for cfg in result.top_configurations:
        assert "config_index" in cfg
        assert "probability" in cfg
        assert "binary_string" in cfg
        assert 0.0 <= cfg["probability"] <= 1.0


def test_grover_returns_optimal_indices() -> None:
    """run_grover must return at most n_targets indices."""
    engine = QuantumEngine(seed=42)
    n_targets = 3
    result = engine.run_grover(
        search_space_size=100, n_targets=n_targets, n_qubits=7, seed=42
    )

    assert len(result.optimal_indices) <= n_targets
    assert result.n_qubits == 7
    for idx in result.optimal_indices:
        assert 0 <= idx < 100


def test_detect_backend_returns_valid_string() -> None:
    """_detect_backend must return one of the three valid backend names."""
    engine = QuantumEngine()
    assert engine._backend_name in ("pennylane", "qiskit", "classical")


def test_entanglement_result_fields() -> None:
    """von_neumann_entropy must populate all EntanglementResult fields."""
    engine = QuantumEngine()
    state_vector = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)])
    result = engine.von_neumann_entropy(state_vector, [0], total_qubits=2)

    assert isinstance(result.entropy, float)
    assert result.n_qubits == 2
    assert result.subsystem_size == 1
    assert result.execution_ms >= 0.0
    assert result.backend == "classical"
