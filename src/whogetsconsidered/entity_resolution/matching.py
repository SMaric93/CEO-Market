"""Conservative candidate-match proposals that never silently merge executives."""

from __future__ import annotations

import polars as pl


def build_exact_name_crosswalk(executive_names: pl.DataFrame) -> pl.DataFrame:
    """Create deterministic person identifiers from exact normalized names only."""

    normalized = executive_names.select("normalized_name").unique().sort("normalized_name")
    return normalized.with_columns(
        pl.format("AUTO_{}", pl.col("normalized_name")).alias("person_id"),
        pl.lit("exact_name").alias("person_id_source"),
    )
