"""
Integration tests for the full Psychohistory Engine pipeline.

Tests:
  - test_end_to_end_ingest_analyze_predict: full pipeline with synthetic corpus
  - test_determinism_with_same_seed: same seed produces identical trajectories
  - test_extraction_pipeline_with_mock_connector: ExtractionPipeline populates corpus
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from psychohistory.connectors.base import SearchQuery
from psychohistory.engine import PsychohistoryEngine, PredictionParams
from psychohistory.enums import EventCategory
from psychohistory.event_ingester import EventIngester
from psychohistory.extraction_pipeline import ExtractionPipeline
from psychohistory.models import RawSourceDocument
from psychohistory.persistence import CorpusRepository


# ---------------------------------------------------------------------------
# Synthetic event generator
# ---------------------------------------------------------------------------


def _generate_synthetic_events(n: int = 1100) -> list[dict]:
    """
    Generate n synthetic historical events with varied dates (1000–2024)
    and varied categories.
    """
    categories = [c.value for c in EventCategory]
    events = []
    for i in range(n):
        year = 1000 + (i * (1024 // n))  # spread across 1000–2024
        category = categories[i % len(categories)]
        events.append(
            {
                "date": f"{year}-01-01T00:00:00",
                "description": f"Synthetic historical event #{i}: {category} event in year {year}.",
                "category": category,
                "actors": [f"Actor_{i % 10}", f"Group_{i % 5}"],
                "magnitude": round(0.1 + (i % 10) * 0.09, 2),
                "source_url": f"https://example.com/event/{i}",
                "connector_name": "synthetic",
            }
        )
    return events


# ---------------------------------------------------------------------------
# Test 1: End-to-end pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_ingest_analyze_predict() -> None:
    """
    Full pipeline: ingest synthetic corpus → analyze patterns → predict →
    detect crises → export/import state.
    """
    engine = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)

    # 1. Ingest 1100 synthetic events
    events = _generate_synthetic_events(1100)
    report = engine.ingest_events(events, format="json")

    assert report.accepted >= 1000, (
        f"Expected at least 1000 accepted events, got {report.accepted}"
    )

    # 2. Predict
    result = engine.predict(horizon_years=50)

    assert len(result.trajectories) == 3, (
        f"Expected exactly 3 trajectories, got {len(result.trajectories)}"
    )

    # 3. Verify uncertainty bounds
    for traj in result.trajectories:
        assert traj.uncertainty_bound.product >= 0.01, (
            f"Uncertainty bound product {traj.uncertainty_bound.product} < 0.01"
        )

    # 4. Verify intervention points structure
    assert len(result.intervention_points) == 3
    for ip_list in result.intervention_points:
        assert isinstance(ip_list, list)

    # 5. Export and import state
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.msgpack")
        engine.export_state(path)

        engine2 = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)
        engine2.import_state(path)

        assert engine2.get_corpus_size() == engine.get_corpus_size(), (
            f"Corpus size mismatch after import: "
            f"{engine2.get_corpus_size()} != {engine.get_corpus_size()}"
        )


# ---------------------------------------------------------------------------
# Test 2: Determinism with same seed
# ---------------------------------------------------------------------------


def test_determinism_with_same_seed() -> None:
    """Same seed must produce identical trajectories."""
    engine = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)

    events = _generate_synthetic_events(1100)
    engine.ingest_events(events, format="json")

    params = PredictionParams(seed=42)

    result1 = engine.predict(horizon_years=20, params=params)
    result2 = engine.predict(horizon_years=20, params=params)

    assert len(result1.trajectories) == len(result2.trajectories) == 3

    for t1, t2 in zip(result1.trajectories, result2.trajectories):
        assert t1.seed == t2.seed, "Seeds must match"
        assert abs(t1.confidence_score - t2.confidence_score) < 1e-9, (
            f"Confidence scores differ: {t1.confidence_score} vs {t2.confidence_score}"
        )
        assert len(t1.nodes) == len(t2.nodes), "Node counts must match"
        for n1, n2 in zip(t1.nodes, t2.nodes):
            assert abs(n1.probability - n2.probability) < 1e-9, (
                f"Node probabilities differ: {n1.probability} vs {n2.probability}"
            )


# ---------------------------------------------------------------------------
# Test 3: ExtractionPipeline with mock connector populates corpus
# ---------------------------------------------------------------------------


def test_extraction_pipeline_with_mock_connector() -> None:
    """ExtractionPipeline with mock connector must populate corpus."""
    repository = CorpusRepository("sqlite:///:memory:")
    ingester = EventIngester(repository=repository)
    pipeline = ExtractionPipeline(event_ingester=ingester, repository=repository)

    # Create mock connector returning documents with extractable years
    docs = [
        RawSourceDocument(
            id=str(uuid.uuid4()),
            source_url=f"https://example.com/doc/{i}",
            connector_name="mock",
            raw_content=f"In {1800 + i * 10} a significant event occurred in history.",
            content_type="text/plain",
            extraction_timestamp=datetime.now(timezone.utc),
            metadata={},
        )
        for i in range(10)
    ]

    mock_connector = MagicMock()
    mock_connector.connector_name = "mock"
    mock_connector.search.return_value = docs

    query = SearchQuery(text="history")
    report = pipeline.run(mock_connector, query)

    assert report.documents_retrieved == 10
    assert report.events_generated == 10
    assert report.documents_discarded == 0

    # Verify corpus was populated
    corpus_size = repository.count()
    assert corpus_size == 10, f"Expected 10 events in corpus, got {corpus_size}"


# ---------------------------------------------------------------------------
# Test 4: Corpus size check before prediction
# ---------------------------------------------------------------------------


def test_predict_raises_on_insufficient_corpus() -> None:
    """predict() raises InsufficientDataError when corpus < 1000 events."""
    from psychohistory.exceptions import InsufficientDataError

    engine = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)

    # Ingest only 10 events
    events = _generate_synthetic_events(10)
    engine.ingest_events(events, format="json")

    with pytest.raises(InsufficientDataError):
        engine.predict(horizon_years=10)


# ---------------------------------------------------------------------------
# Test 5: configure_connector and trigger_extraction
# ---------------------------------------------------------------------------


def test_configure_and_trigger_extraction() -> None:
    """configure_connector + trigger_extraction populates corpus via pipeline."""
    engine = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)

    docs = [
        RawSourceDocument(
            id=str(uuid.uuid4()),
            source_url=f"https://example.com/doc/{i}",
            connector_name="mock_wiki",
            raw_content=f"In {1900 + i} a major political event occurred.",
            content_type="text/plain",
            extraction_timestamp=datetime.now(timezone.utc),
            metadata={},
        )
        for i in range(5)
    ]

    mock_connector = MagicMock()
    mock_connector.connector_name = "mock_wiki"
    mock_connector.search.return_value = docs

    engine.configure_connector("wiki", mock_connector)
    report = engine.trigger_extraction("wiki", SearchQuery(text="politics"))

    assert report.connector_name == "mock_wiki"
    assert report.documents_retrieved == 5
    assert report.events_generated == 5


# ---------------------------------------------------------------------------
# Test 6: export/import state round-trip preserves corpus
# ---------------------------------------------------------------------------


def test_export_import_state_round_trip() -> None:
    """Export and import state preserves corpus size and patterns."""
    engine = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)

    events = _generate_synthetic_events(100)
    engine.ingest_events(events, format="json")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state.msgpack")
        engine.export_state(path)

        engine2 = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)
        engine2.import_state(path)

        assert engine2.get_corpus_size() == engine.get_corpus_size()


# ---------------------------------------------------------------------------
# Test 7: query_corpus returns filtered results
# ---------------------------------------------------------------------------


def test_query_corpus_with_filters() -> None:
    """query_corpus returns events matching the given filters."""
    from psychohistory.persistence import CorpusQuery

    engine = PsychohistoryEngine(db_url="sqlite:///:memory:", seed=42)

    events = [
        {
            "date": "1789-07-14T00:00:00",
            "description": "Storming of the Bastille",
            "category": "POLITICAL",
        },
        {
            "date": "1929-10-24T00:00:00",
            "description": "Black Thursday",
            "category": "ECONOMIC",
        },
    ]
    engine.ingest_events(events, format="json")

    # Filter by category
    results = engine.query_corpus(
        CorpusQuery(categories=[EventCategory.POLITICAL])
    )
    assert len(results) == 1
    assert results[0].description == "Storming of the Bastille"
