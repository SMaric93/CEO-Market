"""CEO-ready status construction from title flags and optional robustness rules."""

from __future__ import annotations

import polars as pl


def ceo_ready_candidates(executive_year_panel: pl.DataFrame, *, robust: bool = False) -> pl.DataFrame:
    """Filter the executive panel to CEO-ready candidates."""

    flag = "is_ceo_ready_robust" if robust else "is_ceo_ready"
    return executive_year_panel.filter(pl.col(flag))
