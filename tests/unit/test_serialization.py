"""
Unit and property-based tests for serialization (export_state / import_state).

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone

import msgpack
import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.exceptions import StateImportError, StateIntegrityError
from psychohistory.models import (
    HistoricalEvent,
    Location,
    ReasoningTrace,
    SocialPattern,
    SystemState,
    Trajectory,
    UncertaintyBound,
)
from psychohistory.serialization import export_state, import_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_state(n_events: int = 3) -> SystemState:
    """Create a minimal SystemState for testing."""
    events = [
        HistoricalEvent(
            id=str(uuid.uuid4()),
            date=datetime(1900 + i, 1, 1),
            description=f"Event {i}",
            category=EventCategory.POLITICAL,
            actors=[f"actor_{i}"],
            magnitude=0.5,
        )
        for i in range(n_events)
    ]
    return SystemState(
        version="1.0",
        created_at=datetime.now(timezone.utc),
        corpus=events,
        patterns=[],
        active_trajectories=[],
        connector_configs=[],
        integrity_hash="",
    )


def _make_state_with_patterns() -> SystemState:
    """Create a SystemState with patterns."""
    graph = nx.DiGraph()
    graph.add_node(EventCategory.POLITICAL)
    graph.add_node(EventCategory.ECONOMIC)
    graph.add_edge(EventCategory.POLITICAL, EventCategory.ECONOMIC, weight=3)

    pattern = SocialPattern(
        id=str(uuid.uuid4()),
        name="test_pattern",
        confidence_score=0.75,
        causality_graph=graph,
        supporting_events=[str(uuid.uuid4())],
        is_quantum_detected=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    return SystemState(
        version="1.0",
        created_at=datetime.now(timezone.utc),
        corpus=[],
        patterns=[pattern],
        active_trajectories=[],
        connector_configs=[],
        integrity_hash="",
    )


# Hypothesis strategy for small SystemState
@st.composite
def small_system_states(draw: st.DrawFn) -> SystemState:
    """Generate a small SystemState for property-based testing."""
    n_events = draw(st.integers(min_value=0, max_value=10))
    events = [
        HistoricalEvent(
            id=str(uuid.uuid4()),
            date=datetime(
                draw(st.integers(min_value=1800, max_value=2024)),
                draw(st.integers(min_value=1, max_value=12)),
                1,
            ),
            description=draw(st.text(min_size=1, max_size=50)),
            category=draw(st.sampled_from(EventCategory)),
            actors=draw(st.lists(st.text(min_size=1, max_size=20), max_size=3)),
            magnitude=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False))),
        )
        for _ in range(n_events)
    ]
    return SystemState(
        version="1.0",
        created_at=datetime.now(timezone.utc),
        corpus=events,
        patterns=[],
        active_trajectories=[],
        connector_configs=[],
        integrity_hash="",
    )


# ---------------------------------------------------------------------------
# Propiedad 18: Round-trip de serialización
# Validates: Requirement 8.3
# ---------------------------------------------------------------------------


@given(state=small_system_states())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_serialization_round_trip(state: SystemState) -> None:
    """
    **Validates: Requirements 8.3**

    export_state + import_state must produce a functionally equivalent state:
    same number of events, same IDs, same patterns.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.msgpack")
        export_state(state, path)
        restored = import_state(path)

    # Same number of corpus events
    assert len(restored.corpus) == len(state.corpus), (
        f"Corpus size mismatch: {len(restored.corpus)} != {len(state.corpus)}"
    )

    # Same event IDs
    original_ids = {e.id for e in state.corpus}
    restored_ids = {e.id for e in restored.corpus}
    assert original_ids == restored_ids, (
        f"Event IDs mismatch: {original_ids.symmetric_difference(restored_ids)}"
    )

    # Same number of patterns
    assert len(restored.patterns) == len(state.patterns)

    # Same version
    assert restored.version == state.version


# ---------------------------------------------------------------------------
# Propiedad 19: Integridad SHA-256
# Validates: Requirement 8.5
# ---------------------------------------------------------------------------


@given(state=small_system_states())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_integrity_hash_is_correct(state: SystemState) -> None:
    """
    **Validates: Requirements 8.5**

    The integrity_hash in the exported file must be the SHA-256 of the
    content without the hash field. Modifying 1 byte must invalidate it.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.msgpack")
        export_state(state, path)

        # Read and verify the hash is present and correct
        with open(path, "rb") as f:
            raw = f.read()

        data = msgpack.unpackb(raw, raw=False)
        stored_hash = data.pop("integrity_hash")

        # Recompute hash
        recalculated = hashlib.sha256(msgpack.packb(data, use_bin_type=True)).hexdigest()
        assert stored_hash == recalculated, (
            f"Hash mismatch: stored={stored_hash}, computed={recalculated}"
        )

        # Modify 1 byte and verify import fails
        corrupted = bytearray(raw)
        # Flip a byte in the middle of the file
        mid = len(corrupted) // 2
        corrupted[mid] = (corrupted[mid] + 1) % 256

        corrupted_path = os.path.join(tmpdir, "corrupted.msgpack")
        with open(corrupted_path, "wb") as f:
            f.write(bytes(corrupted))

        with pytest.raises((StateIntegrityError, StateImportError)):
            import_state(corrupted_path)


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_export_creates_file(tmp_path) -> None:
    """export_state must create a file at the given path."""
    state = _make_minimal_state()
    path = str(tmp_path / "state.msgpack")

    assert not os.path.exists(path)
    export_state(state, path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_import_raises_integrity_error_on_corrupted_file(tmp_path) -> None:
    """import_state must raise StateIntegrityError when file is corrupted."""
    state = _make_minimal_state()
    path = str(tmp_path / "state.msgpack")
    export_state(state, path)

    # Corrupt the file by overwriting with garbage
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)

    # Tamper with the content
    data["version"] = "tampered"
    # Keep the original hash (which is now wrong)
    original_hash = data.get("integrity_hash", "")
    data["integrity_hash"] = original_hash

    corrupted_path = str(tmp_path / "corrupted.msgpack")
    with open(corrupted_path, "wb") as f:
        f.write(msgpack.packb(data, use_bin_type=True))

    with pytest.raises(StateIntegrityError):
        import_state(corrupted_path)


def test_import_raises_state_import_error_on_invalid_format(tmp_path) -> None:
    """import_state must raise StateImportError when format is invalid."""
    invalid_path = str(tmp_path / "invalid.msgpack")

    # Write invalid msgpack data
    with open(invalid_path, "wb") as f:
        f.write(b"this is not valid msgpack data at all!!!")

    with pytest.raises((StateImportError, StateIntegrityError)):
        import_state(invalid_path)


def test_round_trip_preserves_event_descriptions(tmp_path) -> None:
    """Round-trip must preserve event descriptions exactly."""
    state = _make_minimal_state(5)
    path = str(tmp_path / "state.msgpack")

    export_state(state, path)
    restored = import_state(path)

    original_descs = {e.id: e.description for e in state.corpus}
    restored_descs = {e.id: e.description for e in restored.corpus}
    assert original_descs == restored_descs


def test_round_trip_preserves_patterns(tmp_path) -> None:
    """Round-trip must preserve SocialPatterns including causality graph."""
    state = _make_state_with_patterns()
    path = str(tmp_path / "state.msgpack")

    export_state(state, path)
    restored = import_state(path)

    assert len(restored.patterns) == 1
    original_p = state.patterns[0]
    restored_p = restored.patterns[0]

    assert restored_p.id == original_p.id
    assert restored_p.name == original_p.name
    assert math.isclose(restored_p.confidence_score, original_p.confidence_score, rel_tol=1e-9)

    # Verify graph structure
    assert set(restored_p.causality_graph.nodes()) == set(original_p.causality_graph.nodes())
    assert restored_p.causality_graph.number_of_edges() == original_p.causality_graph.number_of_edges()


def test_round_trip_preserves_event_categories(tmp_path) -> None:
    """Round-trip must preserve EventCategory enum values."""
    events = [
        HistoricalEvent(
            id=str(uuid.uuid4()),
            date=datetime(1900 + i, 1, 1),
            description=f"Event {i}",
            category=cat,
        )
        for i, cat in enumerate(EventCategory)
    ]
    state = SystemState(
        version="1.0",
        created_at=datetime.now(timezone.utc),
        corpus=events,
        patterns=[],
        active_trajectories=[],
        connector_configs=[],
        integrity_hash="",
    )
    path = str(tmp_path / "state.msgpack")

    export_state(state, path)
    restored = import_state(path)

    original_cats = {e.id: e.category for e in state.corpus}
    restored_cats = {e.id: e.category for e in restored.corpus}
    assert original_cats == restored_cats


def test_import_raises_error_on_missing_hash(tmp_path) -> None:
    """import_state must raise StateIntegrityError when integrity_hash is missing."""
    state = _make_minimal_state()
    path = str(tmp_path / "state.msgpack")
    export_state(state, path)

    # Remove the integrity_hash field
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    data.pop("integrity_hash", None)

    no_hash_path = str(tmp_path / "no_hash.msgpack")
    with open(no_hash_path, "wb") as f:
        f.write(msgpack.packb(data, use_bin_type=True))

    with pytest.raises(StateIntegrityError):
        import_state(no_hash_path)
