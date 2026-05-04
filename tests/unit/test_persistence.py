"""
Property-based and unit tests for CorpusRepository (persistence layer).

Validates: Requirement 7.4
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.models import HistoricalEvent
from psychohistory.persistence import CorpusQuery, CorpusRepository

# Import shared strategies from conftest
from tests.conftest import historical_events


# ---------------------------------------------------------------------------
# Local strategy: corpus_queries()
# ---------------------------------------------------------------------------


@st.composite
def corpus_queries(draw: st.DrawFn) -> CorpusQuery:
    """Generate arbitrary CorpusQuery instances with random filter combinations."""
    _MIN_DT = datetime(1, 1, 2)
    _MAX_DT = datetime(2024, 12, 31)

    date_from = draw(st.one_of(st.none(), st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT)))
    # date_to must be >= date_from when both are set
    if date_from is not None:
        date_to = draw(
            st.one_of(
                st.none(),
                st.datetimes(min_value=date_from, max_value=_MAX_DT),
            )
        )
    else:
        date_to = draw(st.one_of(st.none(), st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT)))

    categories = draw(
        st.one_of(
            st.just([]),
            st.lists(st.sampled_from(EventCategory), min_size=1, max_size=3, unique=True),
        )
    )
    location_country = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    location_name = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    actors_include = draw(
        st.one_of(
            st.just([]),
            st.lists(
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        blacklist_categories=("Cc", "Cs"),  # exclude control chars
                    ),
                ),
                min_size=1,
                max_size=3,
            ),
        )
    )
    operator: Literal["AND", "OR"] = draw(st.sampled_from(["AND", "OR"]))

    return CorpusQuery(
        date_from=date_from,
        date_to=date_to,
        categories=categories,
        location_country=location_country,
        location_name=location_name,
        actors_include=actors_include,
        operator=operator,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo() -> CorpusRepository:
    """Create a fresh in-memory SQLite repository for each test."""
    return CorpusRepository(db_url="sqlite:///:memory:")


def _event_satisfies_filter(event: HistoricalEvent, filters: CorpusQuery) -> list[bool]:
    """
    Return a list of boolean checks, one per active filter condition.

    This mirrors the repository logic so we can verify results independently.
    Actor matching uses substring search (LIKE) to match the SQL behavior.
    """
    checks: list[bool] = []

    if filters.date_from is not None:
        checks.append(event.date >= filters.date_from)

    if filters.date_to is not None:
        checks.append(event.date <= filters.date_to)

    if filters.categories:
        checks.append(event.category in filters.categories)

    if filters.location_country is not None:
        country = event.location.country if event.location else None
        checks.append(country == filters.location_country)

    if filters.location_name is not None:
        name = event.location.name if event.location else None
        checks.append(name == filters.location_name)

    for actor in filters.actors_include:
        # The DB stores actors as a JSON string and uses LIKE '%actor%'.
        # SQLite LIKE is case-insensitive only for ASCII characters.
        # We replicate by checking substring match in the JSON-serialized actors list.
        actors_json = json.dumps(event.actors) if event.actors else "[]"
        # Use case-insensitive match for ASCII to align with SQLite LIKE behavior
        checks.append(actor.lower() in actors_json.lower())

    return checks


def _satisfies_query(event: HistoricalEvent, filters: CorpusQuery) -> bool:
    """Return True if *event* satisfies the full CorpusQuery (AND or OR)."""
    checks = _event_satisfies_filter(event, filters)

    if not checks:
        # No active filters → every event matches
        return True

    if filters.operator == "AND":
        return all(checks)
    else:  # OR
        return any(checks)


def _has_active_filters(filters: CorpusQuery) -> bool:
    """Return True if at least one filter field is set."""
    return any([
        filters.date_from is not None,
        filters.date_to is not None,
        bool(filters.categories),
        filters.location_country is not None,
        filters.location_name is not None,
        bool(filters.actors_include),
    ])


# ---------------------------------------------------------------------------
# Propiedad 17: Corrección de filtros en consultas del Corpus
# Validates: Requirement 7.4
# ---------------------------------------------------------------------------


@given(
    events=st.lists(historical_events(), min_size=1, max_size=30),
    query=corpus_queries(),
)
@settings(max_examples=100, deadline=None)
def test_query_filters_correctness(
    events: list[HistoricalEvent],
    query: CorpusQuery,
) -> None:
    """
    **Validates: Requirements 7.4**

    Propiedad 17: Corrección de filtros en consultas del Corpus.

    For AND operator: every returned event must satisfy ALL active filters.
    For OR operator: every returned event must satisfy AT LEAST ONE active filter.
    If no filters are active, all events are returned.
    """
    # Ensure unique IDs (hypothesis may generate duplicates across the list)
    seen_ids: set[str] = set()
    unique_events: list[HistoricalEvent] = []
    for evt in events:
        if evt.id not in seen_ids:
            seen_ids.add(evt.id)
            unique_events.append(evt)

    repo = _make_repo()
    repo.save_batch(unique_events)

    results = repo.query(query)

    if not _has_active_filters(query):
        # No filters → all events must be returned
        assert len(results) == len(unique_events), (
            f"With no active filters, expected {len(unique_events)} events "
            f"but got {len(results)}"
        )
        return

    # Build a set of IDs that were saved
    saved_ids = {e.id for e in unique_events}

    for result_event in results:
        # The returned event must have been saved
        assert result_event.id in saved_ids, (
            f"Returned event {result_event.id!r} was not in the saved corpus"
        )

        checks = _event_satisfies_filter(result_event, query)

        if query.operator == "AND":
            assert all(checks), (
                f"Event {result_event.id!r} was returned but does NOT satisfy "
                f"all AND filters. Checks: {checks}, Query: {query}"
            )
        else:  # OR
            assert any(checks), (
                f"Event {result_event.id!r} was returned but does NOT satisfy "
                f"any OR filter. Checks: {checks}, Query: {query}"
            )


# ---------------------------------------------------------------------------
# Concrete example test: empty result when no events match
# ---------------------------------------------------------------------------


def test_query_returns_empty_list_when_no_results(
    small_corpus: list[HistoricalEvent],
) -> None:
    """
    Querying with filters that match no events must return an empty list,
    not raise an error.
    """
    repo = _make_repo()
    repo.save_batch(small_corpus)

    # Use a date range far in the future — no event in small_corpus matches
    filters = CorpusQuery(
        date_from=datetime(2100, 1, 1),
        date_to=datetime(2200, 12, 31),
    )
    results = repo.query(filters)

    assert results == [], (
        f"Expected empty list but got {len(results)} events: {results}"
    )


# ---------------------------------------------------------------------------
# Additional unit tests for CorpusRepository
# ---------------------------------------------------------------------------


def test_save_and_find_by_id(small_corpus: list[HistoricalEvent]) -> None:
    """save() followed by find_by_id() must return an equivalent event."""
    repo = _make_repo()
    event = small_corpus[0]
    repo.save(event)

    found = repo.find_by_id(event.id)
    assert found is not None
    assert found.id == event.id
    assert found.description == event.description
    assert found.category == event.category


def test_find_by_id_returns_none_for_missing(small_corpus: list[HistoricalEvent]) -> None:
    """find_by_id() must return None when the ID does not exist."""
    repo = _make_repo()
    repo.save_batch(small_corpus)

    result = repo.find_by_id("non-existent-id-xyz")
    assert result is None


def test_count_reflects_saved_events(small_corpus: list[HistoricalEvent]) -> None:
    """count() must equal the number of events saved."""
    repo = _make_repo()
    assert repo.count() == 0

    repo.save_batch(small_corpus)
    assert repo.count() == len(small_corpus)


def test_delete_all_empties_corpus(small_corpus: list[HistoricalEvent]) -> None:
    """delete_all() must remove every event from the repository."""
    repo = _make_repo()
    repo.save_batch(small_corpus)
    assert repo.count() == len(small_corpus)

    repo.delete_all()
    assert repo.count() == 0


def test_get_snapshot_hash_changes_after_save() -> None:
    """get_snapshot_hash() must return a different value after adding an event."""
    repo = _make_repo()
    hash_empty = repo.get_snapshot_hash()

    event = HistoricalEvent(
        id=str(uuid.uuid4()),
        date=datetime(2000, 1, 1),
        description="Test event",
        category=EventCategory.POLITICAL,
    )
    repo.save(event)
    hash_with_event = repo.get_snapshot_hash()

    assert hash_empty != hash_with_event


def test_get_snapshot_hash_is_deterministic(small_corpus: list[HistoricalEvent]) -> None:
    """get_snapshot_hash() must return the same value for the same corpus state."""
    repo = _make_repo()
    repo.save_batch(small_corpus)

    hash1 = repo.get_snapshot_hash()
    hash2 = repo.get_snapshot_hash()
    assert hash1 == hash2


def test_save_batch_single_transaction(small_corpus: list[HistoricalEvent]) -> None:
    """save_batch() must persist all events."""
    repo = _make_repo()
    repo.save_batch(small_corpus)
    assert repo.count() == len(small_corpus)


def test_query_by_category(small_corpus: list[HistoricalEvent]) -> None:
    """Filtering by category must return only events of that category."""
    repo = _make_repo()
    repo.save_batch(small_corpus)

    filters = CorpusQuery(categories=[EventCategory.POLITICAL])
    results = repo.query(filters)

    assert all(e.category == EventCategory.POLITICAL for e in results)
    expected_count = sum(1 for e in small_corpus if e.category == EventCategory.POLITICAL)
    assert len(results) == expected_count


def test_query_by_date_range(small_corpus: list[HistoricalEvent]) -> None:
    """Filtering by date range must return only events within that range."""
    repo = _make_repo()
    repo.save_batch(small_corpus)

    date_from = datetime(1900, 1, 1)
    date_to = datetime(1999, 12, 31)
    filters = CorpusQuery(date_from=date_from, date_to=date_to)
    results = repo.query(filters)

    for event in results:
        assert event.date >= date_from
        assert event.date <= date_to


def test_query_or_operator(small_corpus: list[HistoricalEvent]) -> None:
    """OR operator must return events matching any of the specified filters."""
    repo = _make_repo()
    repo.save_batch(small_corpus)

    filters = CorpusQuery(
        categories=[EventCategory.POLITICAL],
        location_country="Japan",
        operator="OR",
    )
    results = repo.query(filters)

    for event in results:
        is_political = event.category == EventCategory.POLITICAL
        is_japan = event.location is not None and event.location.country == "Japan"
        assert is_political or is_japan, (
            f"Event {event.id!r} matches neither filter: "
            f"category={event.category}, country={event.location}"
        )


def test_event_ingester_with_repository() -> None:
    """EventIngester with a repository must persist valid events automatically."""
    from psychohistory.event_ingester import EventIngester

    repo = _make_repo()
    ingester = EventIngester(repository=repo)

    raw = {
        "date": "1789-07-14",
        "description": "Storming of the Bastille",
        "category": "POLITICAL",
    }
    result = ingester.ingest(raw, "json")

    assert result.success is True
    assert result.event is not None
    assert repo.count() == 1

    found = repo.find_by_id(result.event.id)
    assert found is not None
    assert found.description == "Storming of the Bastille"


def test_event_ingester_without_repository_does_not_persist() -> None:
    """EventIngester without a repository must not raise and must not persist."""
    from psychohistory.event_ingester import EventIngester

    ingester = EventIngester()  # no repository

    raw = {
        "date": "1789-07-14",
        "description": "Storming of the Bastille",
        "category": "POLITICAL",
    }
    result = ingester.ingest(raw, "json")

    assert result.success is True
    assert result.event is not None
    # No repository → nothing to check for persistence, just no error
