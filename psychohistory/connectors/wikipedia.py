"""
WikipediaConnector: extracts articles from Wikipedia via the MediaWiki REST API.

If the ``requests`` library is not installed, all methods return empty lists
with a warning log (no ImportError is raised).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from psychohistory.connectors.base import DataConnector, SearchQuery
from psychohistory.models import RawSourceDocument

_logger = logging.getLogger(__name__)


def _get_requests():
    """Return the requests module or None if not installed."""
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError:
        return None


class WikipediaConnector(DataConnector):
    """
    Data connector for Wikipedia via the MediaWiki REST API.

    Supports three search modes:
    1. By title: ``action=query&titles={title}&prop=revisions&rvprop=content``
    2. By category: ``action=query&list=categorymembers&cmtitle=Category:{category}``
    3. Free-text search: ``action=query&list=search&srsearch={text}``

    If ``requests`` is not installed, all methods return empty lists.
    """

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self) -> None:
        super().__init__(connector_name="wikipedia")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: SearchQuery) -> list[RawSourceDocument]:
        """
        Search Wikipedia and return a list of RawSourceDocuments.

        Returns an empty list (with a warning) if ``requests`` is not installed
        or if a network error occurs.
        """
        requests = _get_requests()
        if requests is None:
            _logger.warning(
                "[wikipedia] 'requests' library is not installed. "
                "Returning empty result."
            )
            return []

        try:
            if query.title:
                return self._search_by_title(requests, query)
            elif query.category:
                return self._search_by_category(requests, query)
            else:
                return self._search_by_text(requests, query)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "[wikipedia] Search failed: %s. Returning empty result.", exc
            )
            return []

    def fetch(self, identifier: str) -> RawSourceDocument:
        """
        Fetch a single Wikipedia page by page ID or title.

        Parameters
        ----------
        identifier:
            Either a numeric page ID (as string) or a page title.

        Returns
        -------
        RawSourceDocument
            The fetched document.

        Raises
        ------
        ConnectorTimeoutError
            If the request times out after MAX_RETRIES attempts.
        RuntimeError
            If ``requests`` is not installed.
        """
        requests = _get_requests()
        if requests is None:
            raise RuntimeError(
                "[wikipedia] 'requests' library is not installed."
            )

        if identifier.isdigit():
            params = {
                "action": "query",
                "pageids": identifier,
                "prop": "revisions",
                "rvprop": "content",
                "format": "json",
            }
        else:
            params = {
                "action": "query",
                "titles": identifier,
                "prop": "revisions",
                "rvprop": "content",
                "format": "json",
            }

        def _do_request():
            resp = requests.get(self.BASE_URL, params=params, timeout=self.TIMEOUT_SECONDS)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._handle_rate_limit(retry_after)
                # Re-raise to trigger retry
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()

        data = self._request_with_retry(_do_request)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            raise ValueError(f"No page found for identifier: {identifier!r}")
        page_data = next(iter(pages.values()))
        return self._to_raw_doc(page_data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_by_title(self, requests, query: SearchQuery) -> list[RawSourceDocument]:
        """Search by article title."""
        params = {
            "action": "query",
            "titles": query.title,
            "prop": "revisions",
            "rvprop": "content",
            "format": "json",
        }

        def _do_request():
            resp = requests.get(self.BASE_URL, params=params, timeout=self.TIMEOUT_SECONDS)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._handle_rate_limit(retry_after)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()

        data = self._request_with_retry(_do_request)
        pages = data.get("query", {}).get("pages", {})
        docs = []
        for page_data in pages.values():
            if page_data.get("pageid", -1) == -1:
                continue  # page not found
            docs.append(self._to_raw_doc(page_data))
        return docs[: query.max_results]

    def _search_by_category(self, requests, query: SearchQuery) -> list[RawSourceDocument]:
        """Search by Wikipedia category."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{query.category}",
            "cmlimit": min(query.max_results, 500),
            "format": "json",
        }

        def _do_request():
            resp = requests.get(self.BASE_URL, params=params, timeout=self.TIMEOUT_SECONDS)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._handle_rate_limit(retry_after)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()

        data = self._request_with_retry(_do_request)
        members = data.get("query", {}).get("categorymembers", [])
        docs = []
        for member in members[: query.max_results]:
            # Build a minimal RawSourceDocument from category member info
            page_id = str(member.get("pageid", ""))
            title = member.get("title", "")
            doc = RawSourceDocument(
                id=str(uuid.uuid4()),
                source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                connector_name=self.connector_name,
                raw_content=title,
                content_type="text/plain",
                extraction_timestamp=datetime.now(timezone.utc),
                metadata={"pageid": page_id, "title": title},
            )
            docs.append(doc)
        return docs

    def _search_by_text(self, requests, query: SearchQuery) -> list[RawSourceDocument]:
        """Free-text search."""
        search_text = query.text or ""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_text,
            "srlimit": min(query.max_results, 500),
            "format": "json",
        }

        def _do_request():
            resp = requests.get(self.BASE_URL, params=params, timeout=self.TIMEOUT_SECONDS)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._handle_rate_limit(retry_after)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()

        data = self._request_with_retry(_do_request)
        results = data.get("query", {}).get("search", [])
        docs = []
        for result in results[: query.max_results]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            page_id = str(result.get("pageid", ""))
            doc = RawSourceDocument(
                id=str(uuid.uuid4()),
                source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                connector_name=self.connector_name,
                raw_content=snippet or title,
                content_type="text/html",
                extraction_timestamp=datetime.now(timezone.utc),
                metadata={"pageid": page_id, "title": title},
            )
            docs.append(doc)
        return docs

    def _to_raw_doc(self, page_data: dict) -> RawSourceDocument:
        """Map a Wikipedia API page dict to a RawSourceDocument."""
        page_id = str(page_data.get("pageid", ""))
        title = page_data.get("title", "")

        # Extract wikitext content from revisions
        revisions = page_data.get("revisions", [])
        raw_content = ""
        if revisions:
            rev = revisions[0]
            # MediaWiki API returns content under "*" or "content" key
            raw_content = rev.get("*", rev.get("content", ""))

        source_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

        return RawSourceDocument(
            id=str(uuid.uuid4()),
            source_url=source_url,
            connector_name=self.connector_name,
            raw_content=raw_content or title,
            content_type="text/plain",
            extraction_timestamp=datetime.now(timezone.utc),
            metadata={"pageid": page_id, "title": title},
        )
