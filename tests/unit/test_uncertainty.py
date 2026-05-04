"""
Unit and property-based tests for compute_uncertainty_bound.

Validates: Requirements 6.1, 6.2, 6.3, 6.7
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from psychohistory.uncertainty import SIGMA_STATE_MIN_WARNING, compute_uncertainty_bound


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_positive_floats = st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False)
_h_social_floats = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)
_horizon_ints = st.integers(min_value=1, max_value=1000)


# ---------------------------------------------------------------------------
# Propiedad 15: Invariante del Uncertainty_Bound (product >= h_social)
# Validates: Requirements 6.1, 6.2, 6.3
# ---------------------------------------------------------------------------


@given(
    sigma_state=_positive_floats,
    sigma_momentum=_positive_floats,
    h_social=_h_social_floats,
    horizon=_horizon_ints,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_uncertainty_bound_invariant(
    sigma_state: float,
    sigma_momentum: float,
    h_social: float,
    horizon: int,
) -> None:
    """
    **Validates: Requirements 6.1, 6.2, 6.3**

    For any valid inputs, the resulting UncertaintyBound must satisfy
    product >= h_social.
    """
    result = compute_uncertainty_bound(sigma_state, sigma_momentum, h_social, horizon)
    assert result.product >= h_social - 1e-12, (
        f"Invariant violated: product={result.product} < h_social={h_social} "
        f"(sigma_state={sigma_state}, sigma_momentum={sigma_momentum}, horizon={horizon})"
    )


# ---------------------------------------------------------------------------
# Propiedad 16: Monotonía del Uncertainty_Bound con el horizonte temporal
# Validates: Requirement 6.7
# ---------------------------------------------------------------------------


@given(
    sigma_state=_positive_floats,
    sigma_momentum=_positive_floats,
    h1=st.integers(min_value=102, max_value=1000),
    h2=st.integers(min_value=101, max_value=999),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_uncertainty_bound_monotone_with_horizon(
    sigma_state: float,
    sigma_momentum: float,
    h1: int,
    h2: int,
) -> None:
    """
    **Validates: Requirement 6.7**

    For any h1 > h2 > 100 with the same sigma_state and sigma_momentum,
    UB(h1).product > UB(h2).product must hold.
    """
    # Ensure h1 > h2 > 100
    if h1 <= h2:
        h1, h2 = h2 + 1, h2
    if h2 <= 100:
        h2 = 101
    if h1 <= h2:
        h1 = h2 + 1

    ub1 = compute_uncertainty_bound(sigma_state, sigma_momentum, horizon_years=h1)
    ub2 = compute_uncertainty_bound(sigma_state, sigma_momentum, horizon_years=h2)

    assert ub1.product > ub2.product - 1e-12, (
        f"Monotonicity violated: UB({h1}).product={ub1.product} "
        f"should be > UB({h2}).product={ub2.product} "
        f"(sigma_state={sigma_state}, sigma_momentum={sigma_momentum})"
    )


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_adjustment_when_product_below_h_social() -> None:
    """When sigma_state * sigma_momentum < h_social, was_adjusted must be True."""
    # product = 0.001 * 0.001 = 1e-6 < h_social=0.01
    result = compute_uncertainty_bound(
        sigma_state=0.001,
        sigma_momentum=0.001,
        h_social=0.01,
        horizon_years=1,
    )
    assert result.was_adjusted is True
    assert result.adjustment_reason is not None
    assert result.product >= 0.01 - 1e-12


def test_no_adjustment_when_product_above_h_social() -> None:
    """When sigma_state * sigma_momentum >= h_social, was_adjusted must be False."""
    # product = 0.5 * 0.5 = 0.25 >= h_social=0.01
    result = compute_uncertainty_bound(
        sigma_state=0.5,
        sigma_momentum=0.5,
        h_social=0.01,
        horizon_years=1,
    )
    assert result.was_adjusted is False
    assert result.adjustment_reason is None
    assert math.isclose(result.product, 0.25, rel_tol=1e-9)


def test_warning_when_sigma_state_below_minimum(caplog: pytest.LogCaptureFixture) -> None:
    """When sigma_state < 0.05, a warning must be logged."""
    import logging

    with caplog.at_level(logging.WARNING, logger="psychohistory.uncertainty"):
        compute_uncertainty_bound(
            sigma_state=0.01,
            sigma_momentum=0.5,
            h_social=0.01,
            horizon_years=1,
        )

    assert any(
        "sigma_state" in record.message.lower() or "0.01" in record.message
        for record in caplog.records
    ), "Expected a warning about sigma_state being below minimum"


def test_no_warning_when_sigma_state_above_minimum(caplog: pytest.LogCaptureFixture) -> None:
    """When sigma_state >= 0.05, no warning should be logged."""
    import logging

    with caplog.at_level(logging.WARNING, logger="psychohistory.uncertainty"):
        compute_uncertainty_bound(
            sigma_state=0.1,
            sigma_momentum=0.5,
            h_social=0.01,
            horizon_years=1,
        )

    assert not any(
        "sigma_state" in record.message.lower()
        for record in caplog.records
    ), "No warning expected when sigma_state >= 0.05"


def test_horizon_scaling_above_100() -> None:
    """For horizon > 100, the product must be larger than for horizon = 1."""
    sigma_state = 0.1
    sigma_momentum = 0.1

    ub_short = compute_uncertainty_bound(sigma_state, sigma_momentum, horizon_years=1)
    ub_long = compute_uncertainty_bound(sigma_state, sigma_momentum, horizon_years=200)

    assert ub_long.product > ub_short.product, (
        f"Long-horizon product ({ub_long.product}) should exceed "
        f"short-horizon product ({ub_short.product})"
    )


def test_horizon_exactly_100_no_scaling() -> None:
    """For horizon == 100, no scaling should be applied (scale_factor = 1.0)."""
    sigma_state = 0.2
    sigma_momentum = 0.3
    h_social = 0.01

    result = compute_uncertainty_bound(sigma_state, sigma_momentum, h_social, horizon_years=100)
    # No horizon scaling, product = 0.2 * 0.3 = 0.06 >= 0.01 → no adjustment
    assert result.was_adjusted is False
    assert math.isclose(result.product, 0.06, rel_tol=1e-9)


def test_returned_fields_are_consistent() -> None:
    """The returned UncertaintyBound fields must be internally consistent."""
    result = compute_uncertainty_bound(
        sigma_state=0.3,
        sigma_momentum=0.4,
        h_social=0.01,
        horizon_years=50,
    )
    assert math.isclose(result.product, result.sigma_state * result.sigma_momentum, rel_tol=1e-9)
    assert result.h_social == 0.01
