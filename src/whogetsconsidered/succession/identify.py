"""CEO year-panel construction and succession-event identification."""

from __future__ import annotations

import polars as pl


def identify_ceo_year_panel(executive_year_panel: pl.DataFrame) -> pl.DataFrame:
    """Identify the CEO for each firm-year using the companion-paper default rule."""

    ordered = executive_year_panel.sort(
        ["gvkey", "fyear", "is_ceo_rule_candidate", "is_interim", "title_seniority_score", "exec_rank"],
        descending=[False, False, True, False, True, False],
    )
    ceo_year = ordered.group_by(["gvkey", "fyear"], maintain_order=True).first()
    return ceo_year.select(
        "gvkey",
        "fyear",
        "person_id",
        "exec_name_raw",
        "title_raw",
        "is_interim",
        "is_ceo_rule_candidate",
        "title_seniority_score",
    ).rename(
        {
            "exec_name_raw": "ceo_name",
            "title_raw": "ceo_title_raw",
            "is_interim": "ceo_interim_flag",
        }
    )


def detect_successions(ceo_year_panel: pl.DataFrame) -> pl.DataFrame:
    """Detect CEO changes between consecutive firm-years."""

    ordered = ceo_year_panel.sort(["gvkey", "fyear"]).with_columns(
        pl.col("person_id").shift(1).over("gvkey").alias("predecessor_person_id"),
        pl.col("ceo_name").shift(1).over("gvkey").alias("predecessor_name"),
        pl.col("fyear").shift(1).over("gvkey").alias("previous_fyear"),
    )
    events = ordered.filter(
        pl.col("predecessor_person_id").is_not_null()
        & (pl.col("previous_fyear") == pl.col("fyear") - 1)
        & (pl.col("predecessor_person_id") != pl.col("person_id"))
    )
    return events.select(
        pl.format("{}_{}", pl.col("gvkey"), pl.col("fyear")).alias("event_id"),
        "gvkey",
        pl.col("fyear").alias("succession_year"),
        "predecessor_person_id",
        pl.col("person_id").alias("successor_person_id"),
        "predecessor_name",
        pl.col("ceo_name").alias("successor_name"),
        pl.col("ceo_interim_flag").alias("interim_flag"),
    )
