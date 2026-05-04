"""
Property-based tests for EventIngester.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.event_ingester import BatchIngestionReport, EventIngester, IngestionResult
from psychohistory.models import HistoricalEvent

# Import shared strategies from conftest
from tests.conftest import historical_events, invalid_historical_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_to_dict(event: HistoricalEvent) -> dict:
    """Serialize a HistoricalEvent to a plain dict suitable for re-ingestion."""
    d: dict = {
        "id": event.id,
        "date": event.date.isoformat() if event.date is not None else None,
        "description": event.description,
        "category": event.category.value,
        "actors": event.actors,
    }
    if event.magnitude is not None:
        d["magnitude"] = event.magnitude
    if event.source_url is not None:
        d["source_url"] = event.source_url
    if event.connector_name is not None:
        d["connector_name"] = event.connector_name
    return d


# ---------------------------------------------------------------------------
# Propiedad 1: Normalización preserva el esquema canónico
# Validates: Requirement 1.2
# ---------------------------------------------------------------------------


@given(event=historical_events())
@settings(max_examples=100)
def test_normalization_preserves_canonical_schema(event: HistoricalEvent) -> None:
    """
    **Validates: Requirements 1.2**

    Para cualquier HistoricalEvent válido, después de serializar a dict y
    re-normalizar, el resultado debe tener:
    - date: datetime (no None)
    - description: str no vacío
    - id: str
    - category: EventCategory
    """
    ingester = EventIngester()
    raw = _event_to_dict(event)

    result: IngestionResult = ingester.ingest(raw, "json")

    assert result.success is True, (
        f"Expected successful ingestion but got error: {result.error}"
    )
    assert result.event is not None

    normalized = result.event

    # date must be a datetime
    assert isinstance(normalized.date, datetime), (
        f"Expected date to be datetime, got {type(normalized.date)}"
    )

    # description must be a non-empty string
    assert isinstance(normalized.description, str), (
        f"Expected description to be str, got {type(normalized.description)}"
    )
    assert len(normalized.description) > 0, "description must not be empty"

    # id must be a string
    assert isinstance(normalized.id, str), (
        f"Expected id to be str, got {type(normalized.id)}"
    )
    assert len(normalized.id) > 0, "id must not be empty"

    # category must be an EventCategory enum member
    assert isinstance(normalized.category, EventCategory), (
        f"Expected category to be EventCategory, got {type(normalized.category)}"
    )


# ---------------------------------------------------------------------------
# Propiedad 2: Rechazo de eventos con campos obligatorios ausentes
# Validates: Requirement 1.3
# ---------------------------------------------------------------------------


@given(raw=invalid_historical_events())
@settings(max_examples=100)
def test_rejection_of_events_with_missing_required_fields(raw: dict) -> None:
    """
    **Validates: Requirements 1.3**

    Para cualquier evento con date o description ausente:
    - ingest() debe retornar success=False
    - El error debe identificar explícitamente cada campo faltante
    """
    ingester = EventIngester()

    result: IngestionResult = ingester.ingest(raw, "json")

    assert result.success is False, (
        f"Expected rejection for event missing required fields, but got success=True. "
        f"Raw: {raw}"
    )
    assert result.error is not None, "Expected a ValidationError on rejection"

    missing = result.error.missing_fields

    # Determine which fields are actually missing in the raw dict
    date_missing = raw.get("date") is None
    description_missing = (
        raw.get("description") is None or raw.get("description") == ""
    )

    if date_missing:
        assert "date" in missing, (
            f"Expected 'date' in missing_fields but got: {missing}"
        )

    if description_missing:
        assert "description" in missing, (
            f"Expected 'description' in missing_fields but got: {missing}"
        )


# ---------------------------------------------------------------------------
# Propiedad 3: Ingesta produce IDs únicos
# Validates: Requirement 1.4
# ---------------------------------------------------------------------------


@given(events=st.lists(historical_events(), min_size=2, max_size=50))
@settings(max_examples=100)
def test_ingestion_produces_unique_ids(events: list[HistoricalEvent]) -> None:
    """
    **Validates: Requirements 1.4**

    Para cualquier lote de N eventos válidos, todos los IDs en los resultados
    exitosos deben ser distintos entre sí.
    """
    ingester = EventIngester()

    ids: list[str] = []
    for event in events:
        raw = _event_to_dict(event)
        result: IngestionResult = ingester.ingest(raw, "json")
        assert result.success is True, (
            f"Expected successful ingestion but got: {result.error}"
        )
        assert result.event is not None
        ids.append(result.event.id)

    # All IDs must be unique
    assert len(ids) == len(set(ids)), (
        f"Duplicate IDs found among {len(ids)} ingested events: "
        f"{[id_ for id_ in ids if ids.count(id_) > 1]}"
    )


# ---------------------------------------------------------------------------
# Propiedad 4: Consistencia del reporte de ingesta por lotes
# Validates: Requirement 1.6
# ---------------------------------------------------------------------------


@given(
    valid_events=st.lists(historical_events(), min_size=0, max_size=50),
    invalid_events=st.lists(invalid_historical_events(), min_size=0, max_size=20),
)
@settings(max_examples=100)
def test_batch_report_consistency(
    valid_events: list[HistoricalEvent],
    invalid_events: list[dict],
) -> None:
    """
    **Validates: Requirements 1.6**

    Para cualquier combinación de eventos válidos e inválidos:
    - report.accepted + report.rejected == total eventos procesados
    - report.accepted == número de eventos válidos
    - report.rejected == número de eventos inválidos
    - len(report.rejection_reasons) == report.rejected
    - Cada rejection_reason contiene 'event_index' y 'missing_fields'
    """
    ingester = EventIngester()

    # Build a mixed batch: valid events serialized to dict, then invalid dicts
    batch: list[dict] = [_event_to_dict(e) for e in valid_events] + invalid_events
    total = len(batch)

    report: BatchIngestionReport = ingester.ingest_batch(iter(batch), "json")

    # accepted + rejected == total
    assert report.accepted + report.rejected == total, (
        f"accepted ({report.accepted}) + rejected ({report.rejected}) "
        f"!= total ({total})"
    )

    # accepted == number of valid events
    assert report.accepted == len(valid_events), (
        f"Expected {len(valid_events)} accepted, got {report.accepted}"
    )

    # rejected == number of invalid events
    assert report.rejected == len(invalid_events), (
        f"Expected {len(invalid_events)} rejected, got {report.rejected}"
    )

    # rejection_reasons count matches rejected count
    assert len(report.rejection_reasons) == report.rejected, (
        f"len(rejection_reasons) ({len(report.rejection_reasons)}) "
        f"!= rejected ({report.rejected})"
    )

    # Each rejection reason has the required keys
    for reason in report.rejection_reasons:
        assert "event_index" in reason, (
            f"rejection_reason missing 'event_index': {reason}"
        )
        assert "missing_fields" in reason, (
            f"rejection_reason missing 'missing_fields': {reason}"
        )
        assert isinstance(reason["missing_fields"], list), (
            f"'missing_fields' should be a list, got {type(reason['missing_fields'])}"
        )
