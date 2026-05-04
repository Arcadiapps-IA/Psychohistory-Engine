"""
PsychohistoryEngine: the main orchestrator for the Psychohistory Engine.

Exposes the public API and coordinates all subsystems:
  - EventIngester
  - PatternAnalyzer
  - TrajectoryPredictor
  - InterventionDetector
  - QuantumEngine
  - ExtractionPipeline
  - CorpusRepository
  - ExplainabilityReporter
  - Serialization (export/import state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from psychohistory.connectors.base import DataConnector, SearchQuery
from psychohistory.event_ingester import BatchIngestionReport, EventIngester
from psychohistory.explainability import ExplainabilityReporter, ExplanabilityReport
from psychohistory.extraction_pipeline import ExtractionPipeline, ExtractionReport
from psychohistory.intervention_detector import InterventionDetector
from psychohistory.models import (
    HistoricalEvent,
    InterventionPoint,
    SocialPattern,
    SystemState,
    Trajectory,
    UncertaintyBound,
)
from psychohistory.pattern_analyzer import CorpusSnapshot, PatternAnalyzer
from psychohistory.persistence import CorpusQuery, CorpusRepository
from psychohistory.quantum_engine import QuantumEngine
from psychohistory.trajectory_predictor import TrajectoryPredictor
from psychohistory.uncertainty import SIGMA_STATE_MIN_WARNING

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PredictionParams and PredictionResult
# ---------------------------------------------------------------------------


@dataclass
class PredictionParams:
    """Parameters for a prediction request."""

    seed: int = 42
    sigma_state: float = 0.1
    sigma_momentum: float = 0.1


@dataclass
class PredictionResult:
    """Result of a prediction request."""

    trajectories: list[Trajectory]
    intervention_points: list[list[InterventionPoint]]  # one list per trajectory
    uncertainty_bounds: list[UncertaintyBound]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# PsychohistoryEngine
# ---------------------------------------------------------------------------


class PsychohistoryEngine:
    """
    Main orchestrator for the Psychohistory Engine.

    Initializes all subsystems and exposes the public API.

    Parameters
    ----------
    db_url:
        SQLAlchemy database URL. Defaults to ``sqlite:///corpus.db``.
    seed:
        Random seed for determinism. Defaults to 42.
    use_hardware:
        Whether to attempt hardware quantum backend. Defaults to False.
    """

    def __init__(
        self,
        db_url: str = "sqlite:///corpus.db",
        seed: int = 42,
        use_hardware: bool = False,
    ) -> None:
        self._seed = seed

        # Initialize subsystems
        self._repository = CorpusRepository(db_url)
        self._quantum_engine = QuantumEngine(seed=seed, use_hardware=use_hardware)
        self._event_ingester = EventIngester(repository=self._repository)
        self._pattern_analyzer = PatternAnalyzer(quantum_engine=self._quantum_engine)
        self._trajectory_predictor = TrajectoryPredictor(
            quantum_engine=self._quantum_engine,
            repository=self._repository,
        )
        self._intervention_detector = InterventionDetector(
            quantum_engine=self._quantum_engine
        )
        self._explainability_reporter = ExplainabilityReporter(
            repository=self._repository
        )
        self._connectors: dict[str, DataConnector] = {}
        self._extraction_pipeline = ExtractionPipeline(
            event_ingester=self._event_ingester,
            repository=self._repository,
        )

        # Cache of active patterns (populated on first predict or explicit analyze)
        self._active_patterns: list[SocialPattern] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_events(
        self, events: list[dict], format: str = "json"  # noqa: A002
    ) -> BatchIngestionReport:
        """
        Ingest a list of raw event dicts into the Corpus.

        Parameters
        ----------
        events:
            List of raw event dicts.
        format:
            Input format: "json", "csv", or "text". Defaults to "json".

        Returns
        -------
        BatchIngestionReport
            Summary with accepted/rejected counts and rejection reasons.
        """
        return self._event_ingester.ingest_batch(iter(events), format)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        horizon_years: int,
        params: PredictionParams | None = None,
    ) -> PredictionResult:
        """
        Generate probabilistic trajectory predictions.

        Parameters
        ----------
        horizon_years:
            Prediction horizon in years. Must be in [1, 1000].
        params:
            Optional prediction parameters (seed, sigma_state, sigma_momentum).

        Returns
        -------
        PredictionResult
            Three trajectories with intervention points and uncertainty bounds.

        Raises
        ------
        InsufficientDataError
            When the Corpus contains fewer than 1,000 events.
        InvalidHorizonError
            When horizon_years is outside [1, 1000].
        """
        if params is None:
            params = PredictionParams(seed=self._seed)

        # Warn if sigma_state is below minimum recommended value
        if params.sigma_state < SIGMA_STATE_MIN_WARNING:
            _logger.warning(
                "sigma_state=%.4f is below the minimum recommended value of %.2f. "
                "Increasing state precision implies greater uncertainty in Social_Momentum.",
                params.sigma_state,
                SIGMA_STATE_MIN_WARNING,
            )

        # 1. Get or compute active patterns
        if not self._active_patterns:
            self._analyze_corpus()

        patterns = self._active_patterns

        # 2. Predict trajectories
        trajectories = self._trajectory_predictor.predict(
            patterns=patterns,
            horizon_years=horizon_years,
            seed=params.seed,
        )

        # 3. Detect intervention points for each trajectory
        all_intervention_points: list[list[InterventionPoint]] = []
        for trajectory in trajectories:
            # Register trajectory for explainability
            self._explainability_reporter.register_trajectory(trajectory)
            # Detect crises
            ips = self._intervention_detector.detect(trajectory)
            all_intervention_points.append(ips)

        # 4. Collect uncertainty bounds
        uncertainty_bounds = [t.uncertainty_bound for t in trajectories]

        return PredictionResult(
            trajectories=trajectories,
            intervention_points=all_intervention_points,
            uncertainty_bounds=uncertainty_bounds,
            generated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Corpus query
    # ------------------------------------------------------------------

    def query_corpus(self, filters: CorpusQuery) -> list[HistoricalEvent]:
        """
        Query the Corpus with the given filters.

        Parameters
        ----------
        filters:
            CorpusQuery with optional date range, categories, location, actors.

        Returns
        -------
        list[HistoricalEvent]
            Matching events (empty list if none found).
        """
        return self._repository.query(filters)

    def get_corpus_size(self) -> int:
        """Return the total number of events in the Corpus."""
        return self._repository.count()

    # ------------------------------------------------------------------
    # State export / import
    # ------------------------------------------------------------------

    def export_state(self, path: str) -> None:
        """
        Export the complete engine state to a MessagePack file.

        Parameters
        ----------
        path:
            File path to write the state to.
        """
        from psychohistory import serialization  # noqa: PLC0415

        # Build SystemState
        corpus = self._repository.query(CorpusQuery())
        state = SystemState(
            version="1.0",
            created_at=datetime.now(timezone.utc),
            corpus=corpus,
            patterns=list(self._active_patterns),
            active_trajectories=list(self._trajectory_predictor._active_trajectories),
            connector_configs=[],
            integrity_hash="",
        )
        serialization.export_state(state, path)
        _logger.info("[PsychohistoryEngine] State exported to %s", path)

    def import_state(self, path: str) -> None:
        """
        Import engine state from a MessagePack file.

        Does not modify the current state if the import fails.

        Parameters
        ----------
        path:
            File path to read the state from.

        Raises
        ------
        StateIntegrityError
            When the file's integrity hash does not match.
        StateImportError
            When the file format is invalid.
        """
        from psychohistory import serialization  # noqa: PLC0415

        # Load state (raises on error — current state is not modified)
        state = serialization.import_state(path)

        # Restore corpus
        if state.corpus:
            self._repository.save_batch(state.corpus)

        # Restore patterns
        self._active_patterns = list(state.patterns)
        self._pattern_analyzer._active_patterns = {
            p.id: p for p in state.patterns
        }

        # Restore trajectories
        self._trajectory_predictor._active_trajectories = list(
            state.active_trajectories
        )
        for traj in state.active_trajectories:
            self._explainability_reporter.register_trajectory(traj)

        _logger.info("[PsychohistoryEngine] State imported from %s", path)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def get_explanation(self, trajectory_id: str) -> ExplanabilityReport:
        """
        Get a human-readable explanation for a trajectory.

        Parameters
        ----------
        trajectory_id:
            ID of the trajectory to explain.

        Returns
        -------
        ExplanabilityReport
            Human-readable report with reasoning steps and uncertainty description.

        Raises
        ------
        KeyError
            When the trajectory_id is not registered.
        """
        return self._explainability_reporter.get_explanation(trajectory_id)

    # ------------------------------------------------------------------
    # Connectors
    # ------------------------------------------------------------------

    def configure_connector(
        self, connector_id: str, connector: DataConnector
    ) -> None:
        """
        Register a DataConnector under the given ID.

        Parameters
        ----------
        connector_id:
            Unique identifier for this connector.
        connector:
            DataConnector instance to register.
        """
        self._connectors[connector_id] = connector
        _logger.info(
            "[PsychohistoryEngine] Connector '%s' configured as '%s'.",
            connector.connector_name,
            connector_id,
        )

    def trigger_extraction(
        self, connector_id: str, query: SearchQuery
    ) -> ExtractionReport:
        """
        Trigger a manual extraction using the registered connector.

        Parameters
        ----------
        connector_id:
            ID of the connector to use (must be registered via configure_connector).
        query:
            Search parameters.

        Returns
        -------
        ExtractionReport
            Summary of the extraction run.

        Raises
        ------
        KeyError
            When connector_id is not registered.
        """
        connector = self._connectors[connector_id]
        return self._extraction_pipeline.run(connector, query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_corpus(self) -> None:
        """Run pattern analysis on the current corpus and cache results."""
        corpus = self._repository.query(CorpusQuery())
        snapshot_hash = self._repository.get_snapshot_hash()
        snapshot = CorpusSnapshot(
            events=corpus,
            snapshot_hash=snapshot_hash,
            created_at=datetime.now(timezone.utc),
        )
        self._active_patterns = self._pattern_analyzer.analyze(snapshot)
        _logger.info(
            "[PsychohistoryEngine] Pattern analysis complete: %d patterns detected.",
            len(self._active_patterns),
        )
