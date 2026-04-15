"""Successor-origin and event-quality classification rules."""

from __future__ import annotations

import math

import polars as pl

from whogetsconsidered.config import MarketConfig
from whogetsconsidered.geography.market_defs import classify_market_relation


def classify_successions(
    succession_events: pl.DataFrame,
    executive_year_panel: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
    market_config: MarketConfig,
) -> pl.DataFrame:
    """Classify successor origin, source firm, and search breadth for each CEO change."""

    internal_candidates = executive_year_panel.select(
        "gvkey",
        (pl.col("fyear") + 1).alias("succession_year"),
        pl.col("person_id").alias("successor_person_id"),
        "exec_rank",
    ).filter(pl.col("exec_rank").fill_null(99) <= 4).unique(
        subset=["gvkey", "succession_year", "successor_person_id"]
    )
    events = succession_events.join(
        internal_candidates.with_columns(pl.lit(1).alias("internal_prior_flag")),
        on=["gvkey", "succession_year", "successor_person_id"],
        how="left",
    ).with_columns(
        pl.col("internal_prior_flag").fill_null(0).cast(pl.Int8).alias("internal_prior_flag"),
        (pl.col("internal_prior_flag").fill_null(0) == 0).cast(pl.Int8).alias("outsider_flag"),
    )

    histories = executive_year_panel.select(
        pl.col("person_id").alias("successor_person_id"),
        pl.col("gvkey").alias("source_firm_gvkey"),
        pl.col("fyear").alias("source_year"),
        pl.col("title_raw").alias("successor_source_title"),
    )
    prior_sources = (
        events.join(histories, on="successor_person_id", how="left")
        .filter(pl.col("source_year") < pl.col("succession_year"))
        .sort(["event_id", "source_year"], descending=[False, True])
        .group_by("event_id", maintain_order=True)
        .first()
        .select("event_id", "source_firm_gvkey", "source_year", "successor_source_title")
    )

    focal_locs = firm_year_panel.select(
        "gvkey",
        "fyear",
        pl.col("lat").alias("focal_lat"),
        pl.col("lon").alias("focal_lon"),
        pl.col("market_id").alias("focal_market_id"),
        "focal_market_id_60mi",
        "focal_market_id_100mi",
        "focal_msa_code",
        "focal_state",
    )
    source_locs = firm_year_panel.select(
        pl.col("gvkey").alias("source_firm_gvkey"),
        pl.col("fyear").alias("source_year"),
        pl.col("state").alias("source_state"),
        pl.col("msa_code").alias("source_msa_code"),
        pl.col("lat").alias("source_lat"),
        pl.col("lon").alias("source_lon"),
    )

    enriched = (
        events.join(prior_sources, on="event_id", how="left")
        .join(
            focal_locs,
            left_on=["gvkey", "succession_year"],
            right_on=["gvkey", "fyear"],
            how="left",
        )
        .join(source_locs, on=["source_firm_gvkey", "source_year"], how="left")
    )

    classified_rows: list[dict[str, object]] = []
    for row in enriched.to_dicts():
        if row["outsider_flag"] == 0:
            relation = "internal"
            is_local = False
            distance_km = 0.0
        elif row["source_firm_gvkey"] is None:
            relation = "unknown"
            is_local = False
            distance_km = None
        else:
            result = classify_market_relation(
                source_msa=row["source_msa_code"],
                target_msa=row["focal_msa_code"],
                source_state=row["source_state"],
                target_state=row["focal_state"],
                source_lat=row["source_lat"],
                source_lon=row["source_lon"],
                target_lat=row["focal_lat"],
                target_lon=row["focal_lon"],
                config=market_config,
            )
            relation = result.relation
            is_local = result.is_local
            distance_km = result.distance_km
        row["source_market_relation"] = relation
        row["local_external_flag"] = int(bool(row["outsider_flag"]) and is_local)
        distance_miles = (distance_km or 0.0) / 1.609344 if distance_km is not None else None
        row["source_hq_distance_km"] = distance_km
        row["distance_miles"] = distance_miles
        row["log1p_distance_miles"] = 0.0 if distance_miles is None else math.log1p(distance_miles)
        row["distant_external_flag"] = int(bool(row["outsider_flag"]) and (distance_miles or 0.0) > 60.0)
        row["classification_quality_flag"] = int(
            row["predecessor_person_id"] is not None and row["successor_person_id"] is not None
        )
        row["announcement_date"] = None
        row["event_sample_flag"] = int(row["classification_quality_flag"] == 1)
        row["boardex_sample_flag"] = 0
        row["car_sample_flag"] = 0
        row["tfp_sample_flag"] = 0
        classified_rows.append(row)
    return pl.DataFrame(classified_rows).select(
        "event_id",
        "gvkey",
        "succession_year",
        "predecessor_person_id",
        "successor_person_id",
        "successor_name",
        "predecessor_name",
        "outsider_flag",
        "local_external_flag",
        "distant_external_flag",
        "interim_flag",
        "source_firm_gvkey",
        "successor_source_title",
        pl.col("source_lat").alias("successor_source_hq_lat"),
        pl.col("source_lon").alias("successor_source_hq_lon"),
        pl.col("source_msa_code").alias("successor_source_msa"),
        "source_hq_distance_km",
        "distance_miles",
        "log1p_distance_miles",
        "source_market_relation",
        "focal_market_id_60mi",
        "focal_market_id_100mi",
        "focal_msa_code",
        "focal_state",
        "announcement_date",
        "event_sample_flag",
        "boardex_sample_flag",
        "car_sample_flag",
        "tfp_sample_flag",
        "classification_quality_flag",
    )
