"""
Uncertainty_Bound computation for the Psychohistory Engine.

Implements the Heisenberg-analogue uncertainty principle:
    σ_estado × σ_momentum ≥ ħ_social

For horizons > 100 years, both sigmas are scaled proportionally to reflect
the accumulation of uncertainty in long-range predictions.
"""

from __future__ import annotations

import logging
import math

from psychohistory.models import UncertaintyBound

_logger = logging.getLogger(__name__)

# Minimum sigma_state below which a warning is issued to the user
SIGMA_STATE_MIN_WARNING = 0.05


def compute_uncertainty_bound(
    sigma_state: float,
    sigma_momentum: float,
    h_social: float = 0.01,
    horizon_years: int = 1,
) -> UncertaintyBound:
    """
    Compute the Uncertainty_Bound for a social prediction.

    Parameters
    ----------
    sigma_state:
        Uncertainty in the Social_State (σ_estado).
    sigma_momentum:
        Uncertainty in the Social_Momentum (σ_momentum).
    h_social:
        Minimum system constant (ħ_social). Default: 0.01.
    horizon_years:
        Prediction horizon in years. When > 100, both sigmas are scaled
        proportionally to reflect long-range uncertainty accumulation.

    Returns
    -------
    UncertaintyBound
        Fully populated bound with ``product >= h_social`` guaranteed.

    Notes
    -----
    - When ``sigma_state < SIGMA_STATE_MIN_WARNING`` (0.05), a warning is
      logged to inform the user that increasing state precision implies
      greater momentum uncertainty.
    - When ``horizon_years > 100``, both sigmas are scaled by
      ``1.0 + (horizon_years - 100) / 100.0``.
    - When the product still violates the constraint after horizon scaling,
      both sigmas are further scaled by ``sqrt(h_social / product)`` and
      ``was_adjusted`` is set to ``True``.
    """
    if sigma_state < SIGMA_STATE_MIN_WARNING:
        _logger.warning(
            "sigma_state=%.4f is below the minimum recommended value of %.2f. "
            "Increasing state precision implies greater uncertainty in Social_Momentum "
            "according to the Uncertainty_Bound principle.",
            sigma_state,
            SIGMA_STATE_MIN_WARNING,
        )

    # Step 1: Scale by horizon when > 100 years
    if horizon_years > 100:
        scale_factor = 1.0 + (horizon_years - 100) / 100.0
        sigma_state = sigma_state * scale_factor
        sigma_momentum = sigma_momentum * scale_factor

    # Step 2: Compute product
    product = sigma_state * sigma_momentum

    # Step 3: Adjust if product < h_social
    was_adjusted = False
    adjustment_reason: str | None = None

    if product < h_social:
        adjustment = math.sqrt(h_social / product)
        sigma_state = sigma_state * adjustment
        sigma_momentum = sigma_momentum * adjustment
        product = sigma_state * sigma_momentum
        was_adjusted = True
        adjustment_reason = (
            "Ajuste proporcional para satisfacer σ_estado × σ_momentum ≥ ħ_social"
        )

    return UncertaintyBound(
        sigma_state=sigma_state,
        sigma_momentum=sigma_momentum,
        product=product,
        h_social=h_social,
        was_adjusted=was_adjusted,
        adjustment_reason=adjustment_reason,
    )
