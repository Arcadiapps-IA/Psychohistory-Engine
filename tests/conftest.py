"""
Shared pytest fixtures and Hypothesis strategies for the Psychohistory Engine test suite.

Strategies
----------
- ``locations()``            — generates arbitrary Location instances
- ``historical_events()``    — generates valid HistoricalEvent instances
- ``invalid_historical_events()`` — generates events missing date or description
- ``valid_corpus()``         — generates a list of ≥1000 valid HistoricalEvents

Fixtures
--------
- ``empty_corpus``           — empty list
- ``small_corpus``           — list of 10 concrete synthetic HistoricalEvents
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from hypothesis import strategies as st

from psychohistory.enums import EventCategory
from psychohistory.models import HistoricalEvent, Location


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_MIN_DATETIME = datetime(1, 1, 2, tzinfo=timezone.utc)   # avoid year-0 edge cases
_MAX_DATETIME = datetime(2024, 12, 31, tzinfo=timezone.utc)


@st.composite
def locations(draw: st.DrawFn) -> Location:
    """Generate an arbitrary Location with valid field types."""
    return Location(
        name=draw(st.text(min_size=1, max_size=100)),
        latitude=draw(st.one_of(st.none(), st.floats(min_value=-90.0, max_value=90.0, allow_nan=False))),
        longitude=draw(st.one_of(st.none(), st.floats(min_value=-180.0, max_value=180.0, allow_nan=False))),
        region=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        country=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
    )


@st.composite
def historical_events(draw: st.DrawFn) -> HistoricalEvent:
    """
    Generate a valid HistoricalEvent.

    Both ``date`` and ``description`` are always present (non-null).
    """
    return HistoricalEvent(
        id=str(uuid.uuid4()),
        date=draw(st.datetimes(min_value=_MIN_DATETIME.replace(tzinfo=None), max_value=_MAX_DATETIME.replace(tzinfo=None))),
        description=draw(st.text(min_size=1, max_size=500)),
        category=draw(st.sampled_from(EventCategory)),
        location=draw(st.one_of(st.none(), locations())),
        actors=draw(st.lists(st.text(min_size=1, max_size=50), max_size=10)),
        magnitude=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False))),
        source_url=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200))),
        connector_name=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        extraction_timestamp=None,
        raw_document_id=None,
        feature_vector=None,
    )


@st.composite
def invalid_historical_events(draw: st.DrawFn) -> dict:
    """
    Generate a raw event dict that is missing ``date``, ``description``, or both.

    Returns a plain dict (as would be received before normalization) so that
    the Event_Ingester validation logic can be exercised.
    """
    # Decide which mandatory fields to omit (at least one must be missing)
    omit_date = draw(st.booleans())
    omit_description = draw(st.booleans())

    # Ensure at least one field is missing
    if not omit_date and not omit_description:
        omit_date = True

    event: dict = {
        "id": str(uuid.uuid4()),
        "category": draw(st.sampled_from(EventCategory)).value,
        "actors": [],
    }

    if not omit_date:
        event["date"] = draw(
            st.datetimes(
                min_value=_MIN_DATETIME.replace(tzinfo=None),
                max_value=_MAX_DATETIME.replace(tzinfo=None),
            )
        ).isoformat()

    if not omit_description:
        event["description"] = draw(st.text(min_size=1, max_size=500))

    return event


@st.composite
def valid_corpus(draw: st.DrawFn) -> list[HistoricalEvent]:
    """
    Generate a list of at least 1,000 valid HistoricalEvents.

    Capped at 1,100 to keep test execution time reasonable while still
    satisfying the minimum corpus size requirement.
    """
    return draw(st.lists(historical_events(), min_size=1000, max_size=1100))


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_corpus() -> list[HistoricalEvent]:
    """Return an empty corpus (no events)."""
    return []


@pytest.fixture
def small_corpus() -> list[HistoricalEvent]:
    """Return a list of 10 concrete synthetic HistoricalEvents for unit tests."""
    base_events = [
        HistoricalEvent(
            id="evt-0001",
            date=datetime(1789, 7, 14),
            description="Storming of the Bastille marks the beginning of the French Revolution.",
            category=EventCategory.POLITICAL,
            location=Location(name="Paris", country="France"),
            actors=["Third Estate", "National Assembly"],
            magnitude=0.95,
        ),
        HistoricalEvent(
            id="evt-0002",
            date=datetime(1815, 6, 18),
            description="Battle of Waterloo ends the Napoleonic Wars.",
            category=EventCategory.MILITARY,
            location=Location(name="Waterloo", country="Belgium"),
            actors=["Napoleon Bonaparte", "Duke of Wellington"],
            magnitude=0.90,
        ),
        HistoricalEvent(
            id="evt-0003",
            date=datetime(1848, 3, 18),
            description="Revolutions of 1848 sweep across Europe.",
            category=EventCategory.SOCIAL,
            location=Location(name="Europe", region="Western Europe"),
            actors=["Working class", "Liberal reformers"],
            magnitude=0.80,
        ),
        HistoricalEvent(
            id="evt-0004",
            date=datetime(1929, 10, 24),
            description="Black Thursday triggers the Great Depression.",
            category=EventCategory.ECONOMIC,
            location=Location(name="New York", country="United States"),
            actors=["Wall Street investors", "Federal Reserve"],
            magnitude=0.98,
        ),
        HistoricalEvent(
            id="evt-0005",
            date=datetime(1945, 8, 6),
            description="Atomic bomb dropped on Hiroshima.",
            category=EventCategory.MILITARY,
            location=Location(name="Hiroshima", country="Japan"),
            actors=["United States Army Air Forces"],
            magnitude=1.0,
        ),
        HistoricalEvent(
            id="evt-0006",
            date=datetime(1969, 7, 20),
            description="Apollo 11 lands on the Moon.",
            category=EventCategory.CULTURAL,
            location=Location(name="Sea of Tranquility", region="Moon"),
            actors=["NASA", "Neil Armstrong", "Buzz Aldrin"],
            magnitude=0.85,
        ),
        HistoricalEvent(
            id="evt-0007",
            date=datetime(1989, 11, 9),
            description="Fall of the Berlin Wall.",
            category=EventCategory.POLITICAL,
            location=Location(name="Berlin", country="Germany"),
            actors=["East German citizens", "SED government"],
            magnitude=0.92,
        ),
        HistoricalEvent(
            id="evt-0008",
            date=datetime(1991, 12, 25),
            description="Dissolution of the Soviet Union.",
            category=EventCategory.POLITICAL,
            location=Location(name="Moscow", country="Russia"),
            actors=["Mikhail Gorbachev", "Boris Yeltsin"],
            magnitude=0.97,
        ),
        HistoricalEvent(
            id="evt-0009",
            date=datetime(2008, 9, 15),
            description="Lehman Brothers files for bankruptcy, triggering the global financial crisis.",
            category=EventCategory.ECONOMIC,
            location=Location(name="New York", country="United States"),
            actors=["Lehman Brothers", "US Treasury"],
            magnitude=0.93,
        ),
        HistoricalEvent(
            id="evt-0010",
            date=datetime(2020, 3, 11),
            description="WHO declares COVID-19 a global pandemic.",
            category=EventCategory.NATURAL,
            location=Location(name="Geneva", country="Switzerland"),
            actors=["World Health Organization"],
            magnitude=0.99,
        ),
    ]
    return base_events
