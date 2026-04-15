"""Shared scoring helpers for fit aggregation and decomposition."""

from __future__ import annotations

import polars as pl


def add_zscores(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Add standardized versions of numeric columns, filling undefined z-scores with zero."""

    return df.with_columns(
        [
            ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std(ddof=0))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"{column}_z")
            for column in columns
        ]
    )
