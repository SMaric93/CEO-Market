"""Reviewed crosswalk ingestion for stable person identifiers."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.entity_resolution.matching import build_exact_name_crosswalk


def attach_person_ids(
    executive_df: pl.DataFrame,
    reviewed_crosswalk: pl.DataFrame | None,
    *,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach stable person identifiers using a reviewed crosswalk when available."""

    if reviewed_crosswalk is not None:
        reviewed = reviewed_crosswalk.select("normalized_name", "person_id").unique()
        resolved = executive_df.join(reviewed, on="normalized_name", how="left")
        unresolved = resolved.filter(pl.col("person_id").is_null()).select("normalized_name").unique()
        if unresolved.height > 0:
            logger.warning(
                "reviewed crosswalk missing %s normalized names; assigning exact-name AUTO ids",
                unresolved.height,
            )
            auto = build_exact_name_crosswalk(unresolved)
            resolved = resolved.join(auto, on="normalized_name", how="left", suffix="_auto").with_columns(
                pl.coalesce("person_id", "person_id_auto").alias("person_id"),
                pl.when(pl.col("person_id").is_not_null())
                .then(pl.lit("reviewed_crosswalk"))
                .otherwise(pl.col("person_id_source"))
                .alias("person_id_source"),
            ).drop("person_id_auto")
        else:
            resolved = resolved.with_columns(pl.lit("reviewed_crosswalk").alias("person_id_source"))
        return resolved, unresolved

    logger.warning("no reviewed person crosswalk supplied; using exact normalized-name ids only")
    auto = build_exact_name_crosswalk(executive_df.select("normalized_name").unique())
    resolved = executive_df.join(auto, on="normalized_name", how="left")
    return resolved, pl.DataFrame(schema={"normalized_name": pl.String})
