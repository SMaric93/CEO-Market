"""Reduced-form regressions for realized fit and fit-gap outcomes."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.models.succession_outcomes import estimate_main_models


def estimate_fit_models(
    event_panel: pl.DataFrame,
    config: WhoGetsConsideredConfig,
    *,
    logger: logging.Logger,
) -> pl.DataFrame:
    """Convenience wrapper for fit-related regressions."""

    return estimate_main_models(event_panel, config, logger=logger).filter(
        pl.col("model_group") == "fit_outcomes"
    )
