"""Career-history feature construction using only pre-event information."""

from __future__ import annotations

import polars as pl


def add_internal_tenure(executive_year_panel: pl.DataFrame) -> pl.DataFrame:
    """Add within-firm tenure counts based on observed executive spells."""

    return executive_year_panel.sort(["person_id", "gvkey", "fyear"]).with_columns(
        pl.int_range(1, pl.len() + 1).over(["person_id", "gvkey"]).alias("firm_tenure_years")
    )
