"""Event ingestion, normalization and validation for the Psychohistory Engine."""

from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Literal

from psychohistory.enums import EventCategory
from psychohistory.exceptions import ValidationError
from psychohistory.models import HistoricalEvent, Location

if TYPE_CHECKING:
    from psychohistory.persistence import CorpusRepository


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Result of a single event ingestion attempt."""

    success: bool
    event: HistoricalEvent | None = None
    error: ValidationError | None = None


@dataclass
class ValidationResult:
    """Result of validating a HistoricalEvent."""

    valid: bool
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class BatchIngestionReport:
    """Summary report for a batch ingestion operation."""

    accepted: int
    rejected: int
    rejection_reasons: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EventIngester
# ---------------------------------------------------------------------------

_MAX_BATCH_SIZE = 100_000


class EventIngester:
    """
    Validates, normalizes and (optionally) persists HistoricalEvents.

    Pass a ``CorpusRepository`` instance to enable automatic persistence
    of successfully ingested events.
    """

    def __init__(self, repository: CorpusRepository | None = None) -> None:
        self.repository = repository

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        raw: dict | str,
        format: Literal["json", "csv", "text"],  # noqa: A002
    ) -> IngestionResult:
        """
        Parse, normalize and validate a single raw event.

        Returns an IngestionResult — never raises.
        If a repository is configured and the event is valid, it is persisted.
        """
        try:
            raw_dict = self._parse(raw, format)
        except Exception as exc:  # noqa: BLE001
            error = ValidationError(
                missing_fields=[],
                message=f"Parse error ({format}): {exc}",
            )
            return IngestionResult(success=False, error=error)

        event = self._normalize(raw_dict)
        validation = self._validate(event)

        if not validation.valid:
            error = ValidationError(missing_fields=validation.missing_fields)
            return IngestionResult(success=False, error=error)

        # Assign a fresh UUID v4 (overwrite any id that came in the raw data)
        event.id = str(uuid.uuid4())

        # Persist if a repository is wired in
        if self.repository is not None:
            self.repository.save(event)

        return IngestionResult(success=True, event=event)

    def ingest_batch(
        self,
        source: Iterable,
        format: str,  # noqa: A002
    ) -> BatchIngestionReport:
        """
        Ingest up to 100,000 events from an iterable source.

        Processing continues even when individual events are rejected.
        """
        accepted = 0
        rejected = 0
        rejection_reasons: list[dict] = []

        for index, raw in enumerate(source):
            if index >= _MAX_BATCH_SIZE:
                break

            result = self.ingest(raw, format)  # type: ignore[arg-type]
            if result.success:
                accepted += 1
            else:
                rejected += 1
                missing = result.error.missing_fields if result.error else []
                rejection_reasons.append(
                    {"event_index": index, "missing_fields": missing}
                )

        return BatchIngestionReport(
            accepted=accepted,
            rejected=rejected,
            rejection_reasons=rejection_reasons,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: dict | str, format: str) -> dict:  # noqa: A002
        """Convert raw input to a plain dict according to the declared format."""
        if format == "json":
            if isinstance(raw, dict):
                return raw
            return json.loads(raw)

        if format == "csv":
            if isinstance(raw, dict):
                return raw
            # Treat the string as a single CSV row; first row is the header
            reader = csv.DictReader(io.StringIO(raw))
            rows = list(reader)
            if rows:
                return dict(rows[0])
            # Fallback: try to parse as a headerless single-line CSV
            # by splitting on commas — not reliable, but keeps the contract
            return {}

        if format == "text":
            if isinstance(raw, dict):
                return raw
            # Parse "key: value" lines
            result: dict = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    result[key.strip()] = value.strip()
            return result

        raise ValueError(f"Unsupported format: {format!r}")

    def _normalize(self, raw: dict) -> HistoricalEvent:
        """
        Map a raw dict to a HistoricalEvent dataclass.

        - ``date``: converted from ISO string to datetime if necessary.
        - ``category``: converted to EventCategory enum (case-insensitive);
          defaults to SOCIAL if unrecognised.
        - ``magnitude``: converted to float if present.
        - ``id``: generated as UUID v4 if absent.
        """
        # --- id ---
        event_id = str(raw.get("id") or uuid.uuid4())

        # --- date ---
        date_raw = raw.get("date")
        date: datetime | None = None
        if date_raw is not None:
            if isinstance(date_raw, datetime):
                date = date_raw
            else:
                try:
                    date = datetime.fromisoformat(str(date_raw))
                except (ValueError, TypeError):
                    date = None

        # --- description ---
        description_raw = raw.get("description")
        description: str | None = (
            str(description_raw) if description_raw is not None else None
        )

        # --- category ---
        category_raw = raw.get("category")
        category = EventCategory.SOCIAL  # default
        if category_raw is not None:
            try:
                category = EventCategory(str(category_raw).upper())
            except ValueError:
                category = EventCategory.SOCIAL

        # --- magnitude ---
        magnitude_raw = raw.get("magnitude")
        magnitude: float | None = None
        if magnitude_raw is not None:
            try:
                magnitude = float(magnitude_raw)
            except (ValueError, TypeError):
                magnitude = None

        # --- location ---
        location_raw = raw.get("location")
        location: Location | None = None
        if isinstance(location_raw, dict):
            location = Location(
                name=str(location_raw.get("name", "")),
                latitude=_to_float_or_none(location_raw.get("latitude")),
                longitude=_to_float_or_none(location_raw.get("longitude")),
                region=location_raw.get("region"),
                country=location_raw.get("country"),
            )
        elif isinstance(location_raw, str) and location_raw:
            location = Location(name=location_raw)

        # --- actors ---
        actors_raw = raw.get("actors", [])
        if isinstance(actors_raw, list):
            actors = [str(a) for a in actors_raw]
        elif isinstance(actors_raw, str):
            actors = [a.strip() for a in actors_raw.split(",") if a.strip()]
        else:
            actors = []

        # --- optional metadata ---
        source_url = raw.get("source_url") or None
        connector_name = raw.get("connector_name") or None
        raw_document_id = raw.get("raw_document_id") or None

        extraction_ts_raw = raw.get("extraction_timestamp")
        extraction_timestamp: datetime | None = None
        if extraction_ts_raw is not None:
            if isinstance(extraction_ts_raw, datetime):
                extraction_timestamp = extraction_ts_raw
            else:
                try:
                    extraction_timestamp = datetime.fromisoformat(
                        str(extraction_ts_raw)
                    )
                except (ValueError, TypeError):
                    extraction_timestamp = None

        return HistoricalEvent(
            id=event_id,
            date=date,  # type: ignore[arg-type]  # may be None; _validate catches it
            description=description,  # type: ignore[arg-type]
            category=category,
            location=location,
            actors=actors,
            magnitude=magnitude,
            source_url=source_url,
            connector_name=connector_name,
            extraction_timestamp=extraction_timestamp,
            raw_document_id=raw_document_id,
            feature_vector=None,
        )

    def _validate(self, event: HistoricalEvent) -> ValidationResult:
        """
        Verify that mandatory fields are present.

        Required: ``date`` (not None) and ``description`` (not None, not empty).
        """
        missing: list[str] = []

        if event.date is None:
            missing.append("date")

        if event.description is None or event.description == "":
            missing.append("description")

        if missing:
            return ValidationResult(valid=False, missing_fields=missing)
        return ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
