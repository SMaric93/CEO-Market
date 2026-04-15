"""Succession panel pipeline stage for CEO changes and internal bench measures."""

from __future__ import annotations

from datetime import date
import logging

import polars as pl

from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_artifact, write_csv
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.succession.bench import build_internal_bench
from whogetsconsidered.succession.classify import classify_successions
from whogetsconsidered.succession.identify import detect_successions


def build_succession_panel(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Build succession events and t-1 internal bench measures."""

    with log_stage(logger, "build-succession-panel"):
        registry = ArtifactRegistry(config)
        executive_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL))
        ceo_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.CEO_YEAR_PANEL))
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        announcement_dates = (
            read_input_table("ceo_announcement_dates", config.inputs.ceo_announcement_dates)
            if config.inputs.ceo_announcement_dates is not None
            else None
        )

        succession_events = detect_successions(ceo_year_panel)
        succession_events = classify_successions(
            succession_events,
            executive_year_panel,
            firm_year_panel,
            config.market,
        )
        if announcement_dates is not None:
            join_keys = ["gvkey"]
            if "successor_person_id" in announcement_dates.columns:
                join_keys.append("successor_person_id")
            elif "successor_name" in announcement_dates.columns:
                succession_events = succession_events.join(
                    announcement_dates,
                    on=["gvkey", "successor_name"],
                    how="left",
                )
            else:
                succession_events = succession_events.join(announcement_dates, on=["gvkey"], how="left")
            if "announcement_date" not in succession_events.columns:
                succession_events = succession_events.join(announcement_dates, on=join_keys, how="left")
        if "announcement_date" not in succession_events.columns:
            succession_events = succession_events.with_columns(
                pl.lit(None, dtype=pl.Date).alias("announcement_date"),
                pl.lit(None, dtype=pl.Boolean).alias("source_quality_flag"),
            )
        if "announcement_date_right" in succession_events.columns:
            succession_events = succession_events.with_columns(
                pl.coalesce("announcement_date_right", "announcement_date").alias("announcement_date")
            ).drop("announcement_date_right")
        missing_announcement = succession_events.filter(pl.col("announcement_date").is_null()).select(
            "event_id",
            "gvkey",
            "succession_year",
            "successor_name",
        )
        succession_events = succession_events.with_columns(
            pl.when(pl.col("announcement_date").is_null())
            .then(
                pl.struct("succession_year").map_elements(
                    lambda row: date(int(row["succession_year"]), 6, 30),
                    return_dtype=pl.Date,
                )
            )
            .otherwise(pl.col("announcement_date"))
            .alias("announcement_date"),
            pl.col("announcement_date").is_null().cast(pl.Int8).alias("announcement_imputed_flag"),
        ).with_columns(
            pl.lit(int(config.features.boardex_enabled)).cast(pl.Int8).alias("boardex_sample_flag"),
            pl.when(pl.lit(config.inputs.crsp_daily is not None))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("car_sample_flag"),
            pl.when(pl.lit(config.inputs.tfp_inputs is not None))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("tfp_sample_flag"),
        )
        internal_bench = build_internal_bench(succession_events, executive_year_panel)
        write_csv(registry.output_path("logs", "missing_announcement_dates.csv"), missing_announcement)

        write_artifact(
            registry.artifact_path(ArtifactName.SUCCESSION_EVENTS),
            succession_events,
            lineage={
                "outsider_flag": "1 if successor absent from focal firm's top-four executive roster at t-1",
                "local_external_flag": "1 if outsider comes from within 60 miles of focal HQ",
                "distance_miles": "great-circle miles from successor source HQ to focal HQ",
            },
        )
        write_artifact(
            registry.artifact_path(ArtifactName.INTERNAL_BENCH),
            internal_bench,
            lineage={
                "num_ceo_ready_insiders_tminus1": "count of non-CEO insiders with CEO-ready titles at t-1",
                "heir_apparent_proxy_tminus1": "single President/COO without similarly senior competing CEO-ready insider",
                "bench_index_z": "mean of standardized t-1 bench components used in heterogeneity analyses",
            },
        )
