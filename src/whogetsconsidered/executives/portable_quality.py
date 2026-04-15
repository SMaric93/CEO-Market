"""Portable candidate-quality features used in choice and fit models."""

from __future__ import annotations

import polars as pl


def basic_portable_quality(executive_year_panel: pl.DataFrame) -> pl.DataFrame:
    """Build minimal portable-quality features from observable public-firm histories."""

    histories = executive_year_panel.sort(["person_id", "fyear"]).group_by("person_id").agg(
        pl.max("is_ceo").cast(pl.Int8).alias("prior_public_ceo_flag"),
        pl.sum("is_ceo").alias("years_as_public_ceo"),
        pl.n_unique("gvkey").alias("num_prior_public_firms"),
        pl.n_unique("ff49").alias("num_prior_industries"),
        (pl.n_unique("gvkey") > 1).cast(pl.Int8).alias("mover_flag"),
        pl.mean("log_assets").alias("avg_prior_firm_log_assets"),
        pl.mean("roa_raw").alias("avg_prior_firm_roa"),
        pl.mean("tobin_q_raw").alias("avg_prior_firm_tobin_q"),
    )
    return histories
