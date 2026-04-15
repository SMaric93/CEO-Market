"""Released-candidate construction and event-level access-shock variables."""

from __future__ import annotations

from datetime import date
import math

import polars as pl

from whogetsconsidered.config import MarketConfig
from whogetsconsidered.geography.distance import haversine_km


def _release_tier(event_type: str) -> str:
    normalized = event_type.casefold()
    if any(token in normalized for token in ("merger", "acquisition", "consolidation", "hq_exit")):
        return "A" if "distress" not in normalized else "C"
    if "tier_b" in normalized or "hq exit" in normalized:
        return "B"
    if any(token in normalized for token in ("bankruptcy", "distress", "liquidation", "tier_c")):
        return "C"
    return "B"


def classify_release_tiers(release_events: pl.DataFrame) -> pl.DataFrame:
    """Assign transparent release-event quality tiers."""

    return release_events.with_columns(
        pl.col("event_type").map_elements(_release_tier, return_dtype=pl.String).alias("release_tier")
    )


def build_released_candidates(
    release_events: pl.DataFrame,
    executive_year_panel: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
) -> pl.DataFrame:
    """Build the canonical released-candidate table from clean firm release events."""

    releases = classify_release_tiers(release_events).with_columns((pl.col("event_year") - 1).alias("eligibility_year"))
    source_firm_attrs = firm_year_panel.select(
        pl.col("gvkey").alias("source_gvkey"),
        pl.col("fyear").alias("eligibility_year"),
        "ff10",
        "ff49",
        pl.col("state").alias("source_state"),
    )
    source_roles = executive_year_panel.select(
        pl.col("gvkey").alias("source_gvkey"),
        pl.col("fyear").alias("eligibility_year"),
        pl.col("person_id").alias("candidate_person_id"),
        pl.col("exec_name_raw").alias("candidate_name"),
        pl.col("title_raw").alias("source_title"),
        "is_ceo_ready",
        "is_ceo_ready_robust",
        "is_ceo",
        "is_president",
        "is_coo",
        "is_cfo",
    )
    candidates = (
        releases.join(source_roles, on=["source_gvkey", "eligibility_year"], how="left")
        .join(source_firm_attrs, on=["source_gvkey", "eligibility_year"], how="left")
        .filter(pl.col("candidate_person_id").is_not_null())
        .filter(pl.col("is_ceo_ready"))
    )

    post_release_same_firm = executive_year_panel.select(
        pl.col("person_id").alias("candidate_person_id"),
        pl.col("gvkey").alias("source_gvkey"),
        pl.col("fyear").alias("post_year"),
        "title_raw",
    )
    candidates = candidates.join(post_release_same_firm, on=["candidate_person_id", "source_gvkey"], how="left")
    candidates = candidates.with_columns(
        (
            (pl.col("post_year") >= pl.col("event_year"))
            & (pl.col("title_raw") == pl.col("source_title"))
        )
        .cast(pl.Int8)
        .alias("same_role_after_release_flag")
    )
    person_histories = executive_year_panel.select(
        pl.col("person_id").alias("candidate_person_id"),
        "fyear",
        "is_ceo",
    )
    released = (
        candidates.group_by(
            [
                "candidate_person_id",
                "candidate_name",
                "source_gvkey",
                "eligibility_year",
                "event_date",
                "event_year",
                "event_type",
                "clean_release_flag",
                "release_tier",
                "source_hq_lat",
                "source_hq_lon",
                "source_msa_code",
                "source_state",
                "source_title",
                "ff10",
                "ff49",
                "is_ceo",
                "is_president",
                "is_coo",
                "is_cfo",
            ],
            maintain_order=True,
        )
        .agg(pl.max("same_role_after_release_flag").alias("same_role_after_release_flag"))
        .filter(pl.col("same_role_after_release_flag") == 0)
        .drop("same_role_after_release_flag")
        .rename({"ff10": "source_ff10", "ff49": "source_ff49"})
        .with_columns(
            pl.col("event_date").alias("release_event_date"),
            pl.col("event_year").alias("release_event_year"),
            pl.col("eligibility_year").alias("candidate_year"),
            (pl.col("release_tier") == "A").cast(pl.Int8).alias("main_sample_release_flag"),
            (
                pl.col("clean_release_flag")
                & pl.col("event_date").is_not_null()
                & pl.col("source_hq_lat").is_not_null()
                & pl.col("source_hq_lon").is_not_null()
            )
            .cast(pl.Int8)
            .alias("release_quality_flag"),
        )
    )
    prior_ceo = (
        released.select("candidate_person_id", "release_event_year")
        .join(person_histories, on="candidate_person_id", how="left")
        .filter(pl.col("fyear") < pl.col("release_event_year"))
        .group_by(["candidate_person_id", "release_event_year"], maintain_order=True)
        .agg(pl.col("is_ceo").cast(pl.Int8).max().alias("prior_public_ceo_flag"))
    )
    return released.join(
        prior_ceo,
        on=["candidate_person_id", "release_event_year"],
        how="left",
    ).with_columns(
        pl.col("prior_public_ceo_flag").fill_null(0),
        pl.col("main_sample_release_flag").fill_null(0),
        pl.col("release_quality_flag").fill_null(0),
        pl.lit(1).alias("baseline_release_eligibility_flag"),
        (pl.col("is_ceo") | pl.col("is_president") | pl.col("is_coo") | pl.col("is_cfo"))
        .cast(pl.Int8)
        .alias("robust_release_eligibility_flag"),
        pl.col("candidate_name").alias("exec_name_raw"),
    )


def _event_reference_date(event: dict[str, object]) -> date:
    announcement_date = event.get("announcement_date")
    if announcement_date is not None:
        return announcement_date
    return date(int(event["succession_year"]), 6, 30)


def _distance_miles(
    source_lat: float | None,
    source_lon: float | None,
    target_lat: float | None,
    target_lon: float | None,
) -> float | None:
    if None in (source_lat, source_lon, target_lat, target_lon):
        return None
    return haversine_km(float(source_lat), float(source_lon), float(target_lat), float(target_lon)) / 1.609344


def _is_local_candidate(
    *,
    source_msa: str | None,
    target_msa: str | None,
    source_state: str | None,
    target_state: str | None,
    distance_miles: float | None,
    config: MarketConfig,
    radius_miles: float,
) -> bool:
    if distance_miles is not None and distance_miles <= radius_miles:
        return True
    if config.definition == "msa" and source_msa is not None and target_msa is not None:
        return source_msa == target_msa
    if config.definition == "state" and source_state is not None and target_state is not None:
        return source_state == target_state
    return False


def build_release_supply_metrics(
    succession_events: pl.DataFrame,
    released_candidates: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
    market_config: MarketConfig,
    *,
    window_years: int,
) -> pl.DataFrame:
    """Construct exact released-pool measures for each succession event."""

    focal_attrs = firm_year_panel.select(
        "gvkey",
        "fyear",
        "ff10",
        "ff49",
        pl.col("lat").alias("focal_lat"),
        pl.col("lon").alias("focal_lon"),
        pl.col("msa_code").alias("focal_msa_code"),
        pl.col("state").alias("focal_state"),
    )
    events = succession_events.join(
        focal_attrs,
        left_on=["gvkey", "succession_year"],
        right_on=["gvkey", "fyear"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    released_rows = released_candidates.to_dicts()
    for event in events.to_dicts():
        event_date = _event_reference_date(event)
        released_pool_730_60_outind: list[dict[str, object]] = []
        released_pool_365_60_outind: list[dict[str, object]] = []
        released_pool_1095_60_outind: list[dict[str, object]] = []
        released_pool_730_100_outind: list[dict[str, object]] = []
        released_pool_730_60_allind: list[dict[str, object]] = []
        released_pool_730_60_tier_a: list[dict[str, object]] = []
        released_pool_730_60_tier_ab: list[dict[str, object]] = []
        for candidate in released_rows:
            if int(candidate["release_quality_flag"]) != 1:
                continue
            day_gap = (event_date - candidate["release_event_date"]).days
            if day_gap <= 0:
                continue
            distance_miles = _distance_miles(
                candidate["source_hq_lat"],
                candidate["source_hq_lon"],
                event["focal_lat"],
                event["focal_lon"],
            )
            if distance_miles is None:
                continue
            local_60 = _is_local_candidate(
                source_msa=candidate["source_msa_code"],
                target_msa=event["focal_msa_code"],
                source_state=candidate["source_state"],
                target_state=event["focal_state"],
                distance_miles=distance_miles,
                config=market_config,
                radius_miles=60.0,
            )
            local_100 = _is_local_candidate(
                source_msa=candidate["source_msa_code"],
                target_msa=event["focal_msa_code"],
                source_state=candidate["source_state"],
                target_state=event["focal_state"],
                distance_miles=distance_miles,
                config=market_config,
                radius_miles=100.0,
            )
            out_of_industry = candidate["source_ff10"] != event["ff10"]
            if day_gap <= 730 and local_60 and out_of_industry:
                released_pool_730_60_outind.append(candidate)
                if candidate["release_tier"] == "A":
                    released_pool_730_60_tier_a.append(candidate)
                if candidate["release_tier"] in {"A", "B"}:
                    released_pool_730_60_tier_ab.append(candidate)
            if day_gap <= 365 and local_60 and out_of_industry:
                released_pool_365_60_outind.append(candidate)
            if day_gap <= 1095 and local_60 and out_of_industry:
                released_pool_1095_60_outind.append(candidate)
            if day_gap <= 730 and local_100 and out_of_industry:
                released_pool_730_100_outind.append(candidate)
            if day_gap <= 730 and local_60:
                released_pool_730_60_allind.append(candidate)
        successor_id = str(event["successor_person_id"])
        rows.append(
            {
                "event_id": event["event_id"],
                "release_count_730d_60mi_outind": len(released_pool_730_60_outind),
                "release_count_365d_60mi_outind": len(released_pool_365_60_outind),
                "release_count_1095d_60mi_outind": len(released_pool_1095_60_outind),
                "release_count_730d_100mi_outind": len(released_pool_730_100_outind),
                "release_count_730d_60mi_allind": len(released_pool_730_60_allind),
                "release_count_730d_60mi_tier_a": len(released_pool_730_60_tier_a),
                "release_count_730d_60mi_tier_ab": len(released_pool_730_60_tier_ab),
                "from_release_pool_flag": int(
                    any(str(candidate["candidate_person_id"]) == successor_id for candidate in released_pool_730_60_outind)
                ),
                "generic_supply": len(released_pool_730_60_allind),
                "released_source_flag": int(
                    any(str(candidate["candidate_person_id"]) == successor_id for candidate in released_pool_730_60_allind)
                ),
                "placebo_release_count_3to4y_post": len(
                    [
                        candidate
                        for candidate in released_rows
                        if -1460 <= (event_date - candidate["release_event_date"]).days <= -1095
                    ]
                ),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("release_count_730d_60mi_outind").cast(pl.Int64),
        pl.col("release_count_365d_60mi_outind").cast(pl.Int64),
        pl.col("release_count_1095d_60mi_outind").cast(pl.Int64),
        pl.col("release_count_730d_100mi_outind").cast(pl.Int64),
        pl.col("release_count_730d_60mi_allind").cast(pl.Int64),
        pl.col("release_count_730d_60mi_tier_a").cast(pl.Int64),
        pl.col("release_count_730d_60mi_tier_ab").cast(pl.Int64),
        pl.col("from_release_pool_flag").cast(pl.Int8),
        pl.col("generic_supply").cast(pl.Int64),
        pl.col("released_source_flag").cast(pl.Int8),
    )
