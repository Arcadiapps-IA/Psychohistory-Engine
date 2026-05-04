"""Exception hierarchy for the Psychohistory Engine."""


class PsychohistoryError(Exception):
    """Base exception for all Psychohistory Engine errors."""


class ValidationError(PsychohistoryError):
    """Raised when a historical event fails validation due to missing required fields."""

    def __init__(self, missing_fields: list[str], message: str | None = None) -> None:
        self.missing_fields = missing_fields
        if message is None:
            fields_str = ", ".join(missing_fields)
            message = f"Missing required fields: {fields_str}"
        super().__init__(message)


class BatchIngestionError(PsychohistoryError):
    """Raised when a batch ingestion operation fails at the batch level."""


class InsufficientDataError(PsychohistoryError):
    """Raised when the corpus contains fewer than 1,000 events, making predictions unreliable."""


class InvalidHorizonError(PsychohistoryError):
    """Raised when the prediction horizon is outside the valid range [1, 1000] years."""


class QuantumExecutionError(PsychohistoryError):
    """Raised when a quantum circuit execution request has parameters outside the supported range."""

    def __init__(
        self,
        circuit_type: str,
        invalid_param: str,
        accepted_range: str,
        message: str | None = None,
    ) -> None:
        self.circuit_type = circuit_type
        self.invalid_param = invalid_param
        self.accepted_range = accepted_range
        if message is None:
            message = (
                f"Invalid parameter '{invalid_param}' for circuit type '{circuit_type}'. "
                f"Accepted range: {accepted_range}"
            )
        super().__init__(message)


class ConnectorTimeoutError(PsychohistoryError):
    """Raised when a data connector does not receive a response within the timeout period."""

    def __init__(
        self,
        connector: str,
        timeout_seconds: int,
        attempts: int,
        message: str | None = None,
    ) -> None:
        self.connector = connector
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        if message is None:
            message = (
                f"Connector '{connector}' timed out after {timeout_seconds}s "
                f"({attempts} attempt(s))"
            )
        super().__init__(message)


class ConnectorRateLimitError(PsychohistoryError):
    """Raised when a data connector receives an HTTP 429 Too Many Requests response."""

    def __init__(self, retry_after: int, message: str | None = None) -> None:
        self.retry_after = retry_after
        if message is None:
            message = f"Rate limit exceeded. Retry after {retry_after} seconds."
        super().__init__(message)


class StateIntegrityError(PsychohistoryError):
    """Raised when a state file is corrupted or has been tampered with (hash mismatch)."""


class StateImportError(PsychohistoryError):
    """Raised when a state file has an invalid or unrecognized format."""
