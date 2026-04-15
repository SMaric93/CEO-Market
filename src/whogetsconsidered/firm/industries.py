"""Industry crosswalk utilities and Fama-French mappings."""

from __future__ import annotations

import polars as pl


def attach_ff_industries(firm_year_panel: pl.DataFrame, ff_industry_map: pl.DataFrame) -> pl.DataFrame:
    """Attach Fama-French industry labels using SIC codes."""

    return firm_year_panel.join(ff_industry_map, on="sic", how="left")
