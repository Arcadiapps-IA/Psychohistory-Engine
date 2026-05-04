"""
ArchiveOrgConnector: extracts documents from Internet Archive via the
``internetarchive`` library.

If ``internetarchive`` is not installed, all methods return empty lists
with a warning log (no ImportError is raised).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from psychohistory.connectors.base import DataConnector, SearchQuery
from psychohistory.models import RawSourceDocument

_logger = logging.getLogger(__name__)


def _get_internetarchive():
    """Return the internetarchive module or None if not installed."""
    try:
        import internetarchive  # noqa: PLC0415
        return internetarchive
    except ImportError:
        return None


class ArchiveOrgConnector(DataConnector):
    """
    Data connector for Internet Archive (archive.org).

    Supports three search modes:
    1. By collection: ``collection:{collection}``
    2. By date range: ``date:[{date_from} TO {date_to}]``
    3. Free-text search: ``{text}``

    If ``internetarchive`` is not installed, all methods return empty lists.
    """

    def __init__(self) -> None:
        super().__init__(connector_name="archiveorg")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: SearchQuery) -> list[RawSourceDocument]:
        """
        Search Internet Archive and return a list of RawSourceDocuments.

        Returns an empty list (with a warning) if ``internetarchive`` is not
        installed or if a network error occurs.
        """
        ia = _get_internetarchive()
        if ia is None:
            _logger.warning(
                "[archiveorg] 'internetarchive' library is not installed. "
                "Returning empty result."
            )
            return []

        try:
            ia_query = self._build_ia_query(query)
            fields = ["identifier", "title", "date", "description", "subject"]
            results = ia.search_items(ia_query, fields=fields)
            docs = []
            for item in results:
                if len(docs) >= query.max_results:
                    break
                doc = self._to_raw_doc(item)
                docs.append(doc)
            return docs
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "[archiveorg] Search failed: %s. Returning empty result.", exc
            )
            return []

    def fetch(self, identifier: str) -> RawSourceDocument:
        """
        Fetch a single item from Internet Archive by its identifier.

        Parameters
        ----------
        identifier:
            The Internet Archive item identifier.

        Returns
        -------
        RawSourceDocument
            The fetched document.

        Raises
        ------
        RuntimeError
            If ``internetarchive`` is not installed.
        """
        ia = _get_internetarchive()
        if ia is None:
            raise RuntimeError(
                "[archiveorg] 'internetarchive' library is not installed."
            )

        item = ia.get_item(identifier)
        metadata = item.metadata or {}
        return RawSourceDocument(
            id=str(uuid.uuid4()),
            source_url=f"https://archive.org/details/{identifier}",
            connector_name=self.connector_name,
            raw_content=str(metadata.get("description", metadata.get("title", identifier))),
            content_type="text/plain",
            extraction_timestamp=datetime.now(timezone.utc),
            metadata=dict(metadata),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_ia_query(self, query: SearchQuery) -> str:
        """Build an Internet Archive search query string."""
        parts: list[str] = []

        if query.collection:
            parts.append(f"collection:{query.collection}")

        if query.date_from and query.date_to:
            from_str = query.date_from.strftime("%Y-%m-%d")
            to_str = query.date_to.strftime("%Y-%m-%d")
            parts.append(f"date:[{from_str} TO {to_str}]")
        elif query.date_from:
            from_str = query.date_from.strftime("%Y-%m-%d")
            parts.append(f"date:[{from_str} TO *]")
        elif query.date_to:
            to_str = query.date_to.strftime("%Y-%m-%d")
            parts.append(f"date:[* TO {to_str}]")

        if query.text:
            parts.append(query.text)

        return " AND ".join(parts) if parts else "*"

    def _to_raw_doc(self, item: dict) -> RawSourceDocument:
        """Map an Internet Archive search result to a RawSourceDocument."""
        identifier = item.get("identifier", str(uuid.uuid4()))
        title = item.get("title", "")
        description = item.get("description", "")
        date_str = item.get("date", "")

        raw_content = description or title or identifier

        return RawSourceDocument(
            id=str(uuid.uuid4()),
            source_url=f"https://archive.org/details/{identifier}",
            connector_name=self.connector_name,
            raw_content=raw_content,
            content_type="text/plain",
            extraction_timestamp=datetime.now(timezone.utc),
            metadata={
                "identifier": identifier,
                "title": title,
                "date": date_str,
                "subject": item.get("subject", ""),
            },
        )
