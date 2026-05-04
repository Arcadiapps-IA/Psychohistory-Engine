"""
Base DataConnector abstract class for the Psychohistory Engine.

Provides:
  - SearchQuery dataclass
  - DataConnector abstract base class with retry/rate-limit/timeout handling
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from psychohistory.exceptions import ConnectorTimeoutError
from psychohistory.models import RawSourceDocument


@dataclass
class SearchQuery:
    """Query parameters for a DataConnector search."""

    text: str | None = None
    title: str | None = None
    category: str | None = None
    collection: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    max_results: int = 50


class DataConnector(ABC):
    """
    Abstract base class for all data connectors.

    Subclasses must implement ``search()`` and ``fetch()``.
    Provides retry logic with exponential backoff and rate-limit handling.
    """

    TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 3
    # Backoff delays in seconds for each retry attempt (1s, 2s, 4s)
    _BACKOFF_DELAYS: tuple[int, ...] = (1, 2, 4)

    def __init__(self, connector_name: str) -> None:
        self.connector_name = connector_name
        self._logger = logging.getLogger(__name__)
        self._last_successful_extraction: datetime | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def search(self, query: SearchQuery) -> list[RawSourceDocument]:
        """Search for documents matching the query."""
        ...

    @abstractmethod
    def fetch(self, identifier: str) -> RawSourceDocument:
        """Fetch a single document by its identifier."""
        ...

    # ------------------------------------------------------------------
    # Rate-limit and timeout helpers
    # ------------------------------------------------------------------

    def _handle_rate_limit(self, retry_after: int) -> None:
        """
        Handle HTTP 429 Too Many Requests by sleeping for ``retry_after`` seconds.

        Parameters
        ----------
        retry_after:
            Number of seconds to wait before retrying, as indicated by the
            ``Retry-After`` response header.
        """
        self._logger.warning(
            "[%s] Rate limit hit. Sleeping for %d seconds (Retry-After).",
            self.connector_name,
            retry_after,
        )
        time.sleep(retry_after)

    def _handle_timeout(self, attempt: int) -> None:
        """
        Log a timeout failure and raise ConnectorTimeoutError.

        Parameters
        ----------
        attempt:
            The current attempt number (1-based).

        Raises
        ------
        ConnectorTimeoutError
            Always raised after logging.
        """
        self._logger.error(
            "[%s] Timeout at %s after %d attempt(s).",
            self.connector_name,
            datetime.now(timezone.utc).isoformat(),
            attempt,
        )
        raise ConnectorTimeoutError(
            connector=self.connector_name,
            timeout_seconds=self.TIMEOUT_SECONDS,
            attempts=attempt,
        )

    def _request_with_retry(self, func, *args, **kwargs):
        """
        Execute ``func(*args, **kwargs)`` with exponential backoff retry.

        Retries up to MAX_RETRIES times on transient errors.
        Handles HTTP 429 via ``_handle_rate_limit``.
        Handles timeouts via ``_handle_timeout``.

        Parameters
        ----------
        func:
            Callable to execute. Should raise ``requests.exceptions.Timeout``
            on timeout and ``requests.exceptions.HTTPError`` on HTTP errors.

        Returns
        -------
        Any
            The return value of ``func``.

        Raises
        ------
        ConnectorTimeoutError
            When all retries are exhausted due to timeouts.
        Exception
            Any non-retryable exception from ``func``.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                exc_type = type(exc).__name__
                exc_str = str(exc)

                # Check for HTTP 429 (rate limit)
                # requests raises HTTPError; we check the response status code
                response = getattr(exc, "response", None)
                if response is not None and getattr(response, "status_code", None) == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    self._handle_rate_limit(retry_after)
                    last_exc = exc
                    continue

                # Check for timeout
                if "Timeout" in exc_type or "timeout" in exc_str.lower():
                    self._logger.warning(
                        "[%s] Attempt %d/%d timed out: %s",
                        self.connector_name,
                        attempt,
                        self.MAX_RETRIES,
                        exc_str,
                    )
                    last_exc = exc
                    if attempt >= self.MAX_RETRIES:
                        self._handle_timeout(attempt)
                    # Backoff before retry
                    delay = self._BACKOFF_DELAYS[min(attempt - 1, len(self._BACKOFF_DELAYS) - 1)]
                    time.sleep(delay)
                    continue

                # Non-retryable error
                raise

        # All retries exhausted
        if last_exc is not None:
            self._handle_timeout(self.MAX_RETRIES)
        raise RuntimeError("Unexpected exit from retry loop")
