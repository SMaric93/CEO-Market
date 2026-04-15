"""Compustat adapters for cleaned firm-year panels."""

from __future__ import annotations

import polars as pl

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.firm.industries import attach_ff_industries
from whogetsconsidered.firm.performance import compute_raw_outcomes, residualize_outcomes
from whogetsconsidered.geography.market_defs import assign_market_id, compute_local_density, expand_hq_history


def build_firm_year_panel(
    compustat_firm_year: pl.DataFrame,
    hq_history: pl.DataFrame,
    ff_industry_map: pl.DataFrame,
    noncompete_state_year: pl.DataFrame,
    config: WhoGetsConsideredConfig,
) -> pl.DataFrame:
    """Build the canonical firm-year panel with outcomes, geography, and local controls."""

    min_year = int(compustat_firm_year["fyear"].min())
    max_year = int(compustat_firm_year["fyear"].max())
    expanded_hq = expand_hq_history(hq_history, min_year=min_year, max_year=max_year)
    merged = compustat_firm_year.join(expanded_hq, on=["gvkey", "fyear"], how="left").filter(
        (pl.col("fyear") >= config.sample.start_year)
        & (pl.col("fyear") <= config.sample.end_year)
        & (pl.col("at") >= config.sample.min_assets)
    )
    if config.sample.exclude_financials:
        merged = merged.filter(~pl.col("sic").is_between(6000, 6999, closed="both"))
    if config.sample.exclude_utilities:
        merged = merged.filter(~pl.col("sic").is_between(4900, 4999, closed="both"))
    if config.sample.exclude_public_sector:
        merged = merged.filter((pl.col("sic").is_null()) | (pl.col("sic") <= 9000))
    if config.sample.require_hq_coordinates:
        merged = merged.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    merged = merged.with_columns(pl.coalesce("state_hq", "state").alias("state"))
    merged = attach_ff_industries(merged, ff_industry_map)
    merged = assign_market_id(merged, config.market)
    merged = compute_raw_outcomes(merged)
    merged = compute_local_density(merged, config.market)
    merged = merged.join(
        noncompete_state_year.rename({"year": "fyear"}),
        left_on=["state", "fyear"],
        right_on=["state", "fyear"],
        how="left",
    ).rename({"nca_score": "noncompete_score"}).with_columns(
        pl.col("fyear").min().over("gvkey").alias("first_seen_year"),
        pl.coalesce("state_hq", "state").alias("state_hq"),
    )
    return residualize_outcomes(merged, config.residualization)
