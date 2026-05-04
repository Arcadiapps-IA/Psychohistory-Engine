"""Persistence layer for the Psychohistory Engine using SQLAlchemy 2.0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy import (
    DateTime,
    Float,
    String,
    Text,
    create_engine,
    delete,
    func,
    or_,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from psychohistory.enums import EventCategory
from psychohistory.models import HistoricalEvent, Location


# ---------------------------------------------------------------------------
# SQLAlchemy ORM Base
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------


class HistoricalEventORM(_Base):
    """SQLAlchemy ORM model for the historical_events table."""

    __tablename__ = "historical_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), index=True, nullable=True)
    location_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    location_country: Mapped[str | None] = mapped_column(
        String(100), index=True, nullable=True
    )
    location_region: Mapped[str | None] = mapped_column(String(100), nullable=True)
    location_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    actors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    magnitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    connector_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extraction_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    raw_document_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


# ---------------------------------------------------------------------------
# CorpusQuery dataclass
# ---------------------------------------------------------------------------


@dataclass
class CorpusQuery:
    """Filters for querying the Corpus."""

    date_from: datetime | None = None
    date_to: datetime | None = None
    categories: list[EventCategory] = field(default_factory=list)
    location_country: str | None = None
    location_name: str | None = None
    actors_include: list[str] = field(default_factory=list)
    operator: Literal["AND", "OR"] = "AND"


# ---------------------------------------------------------------------------
# CorpusRepository
# ---------------------------------------------------------------------------


class CorpusRepository:
    """
    Repository for persisting and querying HistoricalEvents.

    Supports SQLite (development) and PostgreSQL (production) via SQLAlchemy 2.0.
    """

    def __init__(self, db_url: str = "sqlite:///corpus.db") -> None:
        self._engine = create_engine(db_url)
        _Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save(self, event: HistoricalEvent) -> None:
        """Persist a single HistoricalEvent."""
        orm = _model_to_orm(event)
        with Session(self._engine) as session:
            # Use merge so that re-saving the same ID is an upsert
            session.merge(orm)
            session.commit()

    def save_batch(self, events: list[HistoricalEvent]) -> None:
        """Persist a list of HistoricalEvents in a single transaction."""
        with Session(self._engine) as session:
            for event in events:
                orm = _model_to_orm(event)
                session.merge(orm)
            session.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def find_by_id(self, id: str) -> HistoricalEvent | None:  # noqa: A002
        """Return the HistoricalEvent with the given ID, or None if not found."""
        with Session(self._engine) as session:
            orm = session.get(HistoricalEventORM, id)
            if orm is None:
                return None
            return _orm_to_model(orm)

    def query(self, filters: CorpusQuery) -> list[HistoricalEvent]:
        """
        Return events matching the given filters.

        With operator="AND": all active filters must be satisfied.
        With operator="OR": at least one active filter must be satisfied.
        If no filters are active, all events are returned.
        """
        conditions = _build_conditions(filters)

        with Session(self._engine) as session:
            stmt = select(HistoricalEventORM)

            if conditions:
                if filters.operator == "AND":
                    for cond in conditions:
                        stmt = stmt.where(cond)
                else:  # OR
                    stmt = stmt.where(or_(*conditions))

            rows = session.execute(stmt).scalars().all()
            return [_orm_to_model(row) for row in rows]

    def count(self) -> int:
        """Return the total number of events in the Corpus."""
        with Session(self._engine) as session:
            result = session.execute(
                select(func.count()).select_from(HistoricalEventORM)
            )
            return result.scalar_one()

    def get_snapshot_hash(self) -> str:
        """Return a SHA-256 hash of the current Corpus state (sorted IDs)."""
        with Session(self._engine) as session:
            stmt = select(HistoricalEventORM.id).order_by(HistoricalEventORM.id)
            ids = session.execute(stmt).scalars().all()

        content = ",".join(ids)
        return hashlib.sha256(content.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def delete_all(self) -> None:
        """Delete all events from the Corpus (useful for tests)."""
        with Session(self._engine) as session:
            session.execute(delete(HistoricalEventORM))
            session.commit()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _orm_to_model(orm: HistoricalEventORM) -> HistoricalEvent:
    """Convert a HistoricalEventORM instance to a HistoricalEvent dataclass."""
    location: Location | None = None
    if orm.location_name is not None:
        location = Location(
            name=orm.location_name,
            country=orm.location_country,
            region=orm.location_region,
            latitude=orm.location_latitude,
            longitude=orm.location_longitude,
        )

    actors: list[str] = []
    if orm.actors is not None:
        try:
            actors = json.loads(orm.actors)
        except (json.JSONDecodeError, TypeError):
            actors = []

    category = EventCategory.SOCIAL
    if orm.category is not None:
        try:
            category = EventCategory(orm.category)
        except ValueError:
            category = EventCategory.SOCIAL

    return HistoricalEvent(
        id=orm.id,
        date=orm.date,
        description=orm.description,
        category=category,
        location=location,
        actors=actors,
        magnitude=orm.magnitude,
        source_url=orm.source_url,
        connector_name=orm.connector_name,
        extraction_timestamp=orm.extraction_timestamp,
        raw_document_id=orm.raw_document_id,
        feature_vector=None,
    )


def _model_to_orm(event: HistoricalEvent) -> HistoricalEventORM:
    """Convert a HistoricalEvent dataclass to a HistoricalEventORM instance."""
    location_name: str | None = None
    location_country: str | None = None
    location_region: str | None = None
    location_latitude: float | None = None
    location_longitude: float | None = None

    if event.location is not None:
        location_name = event.location.name
        location_country = event.location.country
        location_region = event.location.region
        location_latitude = event.location.latitude
        location_longitude = event.location.longitude

    actors_json: str | None = None
    if event.actors:
        actors_json = json.dumps(event.actors)

    return HistoricalEventORM(
        id=event.id,
        date=event.date,
        description=event.description,
        category=event.category.value if event.category is not None else None,
        location_name=location_name,
        location_country=location_country,
        location_region=location_region,
        location_latitude=location_latitude,
        location_longitude=location_longitude,
        actors=actors_json,
        magnitude=event.magnitude,
        source_url=event.source_url,
        connector_name=event.connector_name,
        extraction_timestamp=event.extraction_timestamp,
        raw_document_id=event.raw_document_id,
    )


# ---------------------------------------------------------------------------
# Internal: build SQLAlchemy filter conditions from CorpusQuery
# ---------------------------------------------------------------------------


def _build_conditions(filters: CorpusQuery) -> list:
    """Return a list of SQLAlchemy column expressions for the active filters."""
    conditions = []

    if filters.date_from is not None:
        conditions.append(HistoricalEventORM.date >= filters.date_from)

    if filters.date_to is not None:
        conditions.append(HistoricalEventORM.date <= filters.date_to)

    if filters.categories:
        category_values = [c.value for c in filters.categories]
        conditions.append(HistoricalEventORM.category.in_(category_values))

    if filters.location_country is not None:
        conditions.append(
            HistoricalEventORM.location_country == filters.location_country
        )

    if filters.location_name is not None:
        conditions.append(
            HistoricalEventORM.location_name == filters.location_name
        )

    for actor in filters.actors_include:
        conditions.append(HistoricalEventORM.actors.like(f"%{actor}%"))

    return conditions
