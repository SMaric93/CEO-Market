"""Accessible-set summary metrics such as max fit and realized-fit gaps."""

from __future__ import annotations

import polars as pl


def build_fit_event_summary(accessible_candidate_set: pl.DataFrame) -> pl.DataFrame:
    """Aggregate event-level realized-fit and fit-gap metrics."""

    fit_col = "task_fit_z_if" if "task_fit_z_if" in accessible_candidate_set.columns else "task_alignment_fit_score"
    released = accessible_candidate_set.filter(pl.col("released_candidate_flag") == 1)
    global_q75 = float(released[fit_col].quantile(0.75) or 0.0) if released.height else 0.0
    if released.height > 0:
        relevant = released.group_by("event_id", maintain_order=True).agg(
            pl.max(fit_col).alias("max_released_task_fit_z"),
            pl.mean(fit_col).alias("avg_released_task_fit_z"),
            (pl.col(fit_col) >= global_q75).cast(pl.Int64).sum().alias("topquartile_released_count"),
        )
    else:
        relevant = pl.DataFrame(
            schema={
                "event_id": pl.String,
                "max_released_task_fit_z": pl.Float64,
                "avg_released_task_fit_z": pl.Float64,
                "topquartile_released_count": pl.Int64,
            }
        )

    summary = accessible_candidate_set.group_by("event_id", maintain_order=True).agg(
        pl.max(fit_col).alias("max_accessible_task_fit_z"),
        pl.mean(fit_col).alias("avg_accessible_task_fit_z"),
        pl.when(pl.col("chosen_flag") == 1)
        .then(pl.col(fit_col))
        .otherwise(None)
        .max()
        .alias("realized_task_fit_z"),
        pl.max("predictive_fit_score").alias("MaxPredictiveFit_e"),
        pl.mean("predictive_fit_score").alias("AvgPredictiveFit_e"),
        pl.when(pl.col("chosen_flag") == 1)
        .then(pl.col("predictive_fit_score"))
        .otherwise(None)
        .max()
        .alias("RealizedPredictiveFit_e"),
        pl.when(pl.col("chosen_flag") == 1)
        .then(pl.col("known_to_board_flag"))
        .otherwise(None)
        .max()
        .alias("known_to_board_flag"),
        pl.when(pl.col("chosen_flag") == 1)
        .then(pl.col("board_tie_flag"))
        .otherwise(None)
        .max()
        .alias("board_tie_flag"),
        pl.when(pl.col("chosen_flag") == 1)
        .then(pl.col("employment_tie_flag"))
        .otherwise(None)
        .max()
        .alias("employment_tie_flag"),
        pl.when(pl.col("chosen_flag") == 1)
        .then(pl.col("portable_quality_z_i"))
        .otherwise(None)
        .max()
        .alias("portable_quality_z_i"),
    )
    return summary.join(relevant, on="event_id", how="left").with_columns(
        pl.col("max_released_task_fit_z").fill_null(0.0),
        pl.col("avg_released_task_fit_z").fill_null(0.0),
        pl.col("topquartile_released_count").fill_null(0),
        (pl.col("max_accessible_task_fit_z") - pl.col("realized_task_fit_z")).alias("gap_accessible_task_fit_z"),
        (pl.col("MaxPredictiveFit_e") - pl.col("RealizedPredictiveFit_e")).alias("GapPredictiveFit_e"),
        pl.col("realized_task_fit_z").alias("RealizedTaskFit_e"),
        pl.col("max_released_task_fit_z").alias("MaxRelevantFit_e"),
        pl.col("avg_released_task_fit_z").alias("AvgRelevantFit_e"),
        pl.col("topquartile_released_count").alias("TopQuartileRelevantSupply_e"),
        pl.col("max_accessible_task_fit_z").alias("MaxTaskFit_e"),
        pl.col("avg_accessible_task_fit_z").alias("AvgTaskFit_e"),
        (pl.col("max_accessible_task_fit_z") - pl.col("realized_task_fit_z")).alias("GapTaskFit_e"),
    )
