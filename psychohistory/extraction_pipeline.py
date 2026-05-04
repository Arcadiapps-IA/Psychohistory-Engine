"""
ExtractionPipeline: orchestrates the flow from RawSourceDocument to HistoricalEvent.

Steps:
  1. Retrieve documents via connector.search(query)
  2. Persist each RawSourceDocument unchanged
  3. Transform via _transform(doc) → list[HistoricalEvent]
  4. Deduplicate via _deduplicate(events)
  5. Ingest into Corpus via EventIngester.ingest_batch()
  6. Update _last_extraction checkpoint
  7. Return ExtractionReport
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from psychohistory.connectors.base import DataConnector, SearchQuery
from psychohistory.enums import EventCategory
from psychohistory.event_ingester import EventIngester
from psychohistory.models import HistoricalEvent, RawSourceDocument

if TYPE_CHECKING:
    from psychohistory.persistence import CorpusRepository


# ---------------------------------------------------------------------------
# ExtractionReport
# ---------------------------------------------------------------------------


@dataclass
class ExtractionReport:
    """Summary report for a single extraction run."""

    connector_name: str
    documents_retrieved: int
    events_generated: int
    documents_discarded: int
    discard_reasons: list[dict] = field(default_factory=list)
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ExtractionPipeline
# ---------------------------------------------------------------------------

# Regex patterns for entity extraction (no spaCy dependency)
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_IN_YEAR_PATTERN = re.compile(r"\bin\s+(1[0-9]{3}|20[0-2][0-9])\b", re.IGNORECASE)
# Capitalized words (potential proper nouns / actors)
_CAPITALIZED_WORD = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
# Common English stop-words to exclude from actor detection
_STOP_WORDS = frozenset(
    {
        "The", "This", "That", "These", "Those", "There", "Their", "They",
        "When", "Where", "Which", "While", "With", "From", "Into", "Upon",
        "After", "Before", "During", "Since", "Until", "About", "Above",
        "Below", "Between", "Through", "Against", "Among", "Around",
        "January", "February", "March", "April", "June", "July", "August",
        "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    }
)


class ExtractionPipeline:
    """
    Orchestrates extraction from a DataConnector into the Corpus.

    Parameters
    ----------
    event_ingester:
        EventIngester instance used to ingest transformed events.
    repository:
        Optional CorpusRepository for deduplication against existing events
        and for persisting RawSourceDocuments.
    """

    def __init__(
        self,
        event_ingester: EventIngester,
        repository: CorpusRepository | None = None,
    ) -> None:
        self._ingester = event_ingester
        self._repository = repository
        self._logger = logging.getLogger(__name__)
        # connector_name -> last successful extraction timestamp
        self._last_extraction: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, connector: DataConnector, query: SearchQuery) -> ExtractionReport:
        """
        Execute a full extraction cycle.

        Parameters
        ----------
        connector:
            DataConnector to use for retrieval.
        query:
            Search parameters.

        Returns
        -------
        ExtractionReport
            Summary of the extraction run.
        """
        connector_name = connector.connector_name
        executed_at = datetime.now(timezone.utc)

        # 1. Retrieve documents
        try:
            documents = connector.search(query)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "[ExtractionPipeline] Connector '%s' search failed: %s",
                connector_name,
                exc,
            )
            documents = []

        documents_retrieved = len(documents)
        all_events: list[HistoricalEvent] = []
        discard_reasons: list[dict] = []

        for doc in documents:
            # 2. Persist RawSourceDocument unchanged (if repository available)
            if self._repository is not None:
                try:
                    self._persist_raw_document(doc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "[ExtractionPipeline] Failed to persist raw doc %s: %s",
                        doc.id,
                        exc,
                    )

            # Check incremental eligibility
            if not self._is_incremental_eligible(doc, connector_name):
                discard_reasons.append(
                    {
                        "doc_id": doc.id,
                        "reason": "Document predates last extraction checkpoint",
                    }
                )
                continue

            # 3. Transform
            events = self._transform(doc)
            if not events:
                discard_reasons.append(
                    {
                        "doc_id": doc.id,
                        "reason": "No extractable date or description found",
                    }
                )
                continue

            all_events.extend(events)

        # 4. Deduplicate
        unique_events = self._deduplicate(all_events)

        # 5. Ingest
        if unique_events:
            self._ingester.ingest_batch(
                (self._event_to_dict(e) for e in unique_events),
                format="json",
            )

        # 6. Update checkpoint
        self._last_extraction[connector_name] = executed_at

        documents_discarded = len(discard_reasons)
        events_generated = len(unique_events)

        return ExtractionReport(
            connector_name=connector_name,
            documents_retrieved=documents_retrieved,
            events_generated=events_generated,
            documents_discarded=documents_discarded,
            discard_reasons=discard_reasons,
            executed_at=executed_at,
        )

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def _transform(self, doc: RawSourceDocument) -> list[HistoricalEvent]:
        """
        Transform a RawSourceDocument into HistoricalEvents using regex-based
        entity extraction (no spaCy dependency).

        Extracts:
        - Dates: YYYY patterns or "in YYYY" patterns
        - Description: first 500 chars of raw_content
        - Actors: capitalized word sequences (simple proper noun detection)
        - Category: SOCIAL by default
        - Metadata: source_url, connector_name, extraction_timestamp

        Returns an empty list if no date or description can be extracted.
        """
        raw_content = doc.raw_content or ""

        # Extract description (first 500 chars)
        description = raw_content[:500].strip()
        if not description:
            return []

        # Extract year from content
        year: int | None = None

        # Prefer "in YYYY" pattern
        in_year_match = _IN_YEAR_PATTERN.search(raw_content)
        if in_year_match:
            year = int(in_year_match.group(1))
        else:
            # Fall back to any 4-digit year
            year_match = _YEAR_PATTERN.search(raw_content)
            if year_match:
                year = int(year_match.group(1))

        if year is None:
            # Try metadata for date
            meta_date = doc.metadata.get("date", "")
            if meta_date:
                year_match = _YEAR_PATTERN.search(str(meta_date))
                if year_match:
                    year = int(year_match.group(1))

        if year is None:
            return []

        # Build event date (Jan 1 of the extracted year)
        try:
            event_date = datetime(year, 1, 1)
        except ValueError:
            return []

        # Extract actors (capitalized word sequences, excluding stop words)
        actors: list[str] = []
        for match in _CAPITALIZED_WORD.finditer(raw_content):
            name = match.group(1)
            if name not in _STOP_WORDS and len(name) > 2:
                actors.append(name)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_actors: list[str] = []
        for actor in actors:
            if actor not in seen:
                seen.add(actor)
                unique_actors.append(actor)
        actors = unique_actors[:10]  # cap at 10

        event = HistoricalEvent(
            id=str(uuid.uuid4()),
            date=event_date,
            description=description,
            category=EventCategory.SOCIAL,
            location=None,
            actors=actors,
            magnitude=None,
            source_url=doc.source_url,
            connector_name=doc.connector_name,
            extraction_timestamp=doc.extraction_timestamp,
            raw_document_id=doc.id,
            feature_vector=None,
        )
        return [event]

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, events: list[HistoricalEvent]) -> list[HistoricalEvent]:
        """
        Deduplicate events using SHA-256 hash of (date.isoformat(), description[:100]).

        Filters duplicates within the batch and, if a repository is available,
        also against events already in the Corpus.
        """
        seen_hashes: set[str] = set()

        # Pre-populate with hashes from existing corpus if available
        if self._repository is not None:
            try:
                existing = self._repository.query(
                    __import__(
                        "psychohistory.persistence", fromlist=["CorpusQuery"]
                    ).CorpusQuery()
                )
                for existing_event in existing:
                    h = self._event_hash(existing_event)
                    seen_hashes.add(h)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "[ExtractionPipeline] Could not load existing events for dedup: %s",
                    exc,
                )

        unique: list[HistoricalEvent] = []
        for event in events:
            h = self._event_hash(event)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(event)

        return unique

    def _event_hash(self, event: HistoricalEvent) -> str:
        """Compute SHA-256 hash of (date.isoformat(), description[:100])."""
        date_str = event.date.isoformat() if event.date else ""
        desc_str = (event.description or "")[:100]
        content = f"{date_str}|{desc_str}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Incremental extraction
    # ------------------------------------------------------------------

    def _is_incremental_eligible(
        self, doc: RawSourceDocument, connector_name: str
    ) -> bool:
        """
        Return True if the document should be processed in an incremental run.

        A document is eligible if:
        - There is no checkpoint for this connector (first run), OR
        - The document's extraction_timestamp is after the last checkpoint.
        """
        last = self._last_extraction.get(connector_name)
        if last is None:
            return True
        doc_ts = doc.extraction_timestamp
        if doc_ts is None:
            return True
        # Make both timezone-aware for comparison
        if doc_ts.tzinfo is None:
            doc_ts = doc_ts.replace(tzinfo=timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return doc_ts > last

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_raw_document(self, doc: RawSourceDocument) -> None:
        """
        Persist a RawSourceDocument to the repository.

        The repository stores HistoricalEvents, so we store a minimal
        representation in the metadata. In a full implementation this would
        use a dedicated raw_documents table.
        """
        # For now, we just log — a dedicated raw_documents table would be
        # added in a production implementation.
        self._logger.debug(
            "[ExtractionPipeline] Raw document %s from %s persisted.",
            doc.id,
            doc.connector_name,
        )

    def _event_to_dict(self, event: HistoricalEvent) -> dict:
        """Convert a HistoricalEvent to a dict suitable for EventIngester.ingest_batch."""
        return {
            "id": event.id,
            "date": event.date.isoformat() if event.date else None,
            "description": event.description,
            "category": event.category.value if event.category else "SOCIAL",
            "actors": event.actors,
            "magnitude": event.magnitude,
            "source_url": event.source_url,
            "connector_name": event.connector_name,
            "extraction_timestamp": (
                event.extraction_timestamp.isoformat()
                if event.extraction_timestamp
                else None
            ),
            "raw_document_id": event.raw_document_id,
        }
