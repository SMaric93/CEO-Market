"""Market-definition utilities for MSA, radius, and same-state feasible sets."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from whogetsconsidered.config import MarketConfig
from whogetsconsidered.geography.distance import haversine_km, miles_to_km


@dataclass(frozen=True)
class MarketRelation:
    """Pairwise relation between two headquarters under the configured market definition."""

    relation: str
    is_local: bool
    distance_km: float | None


def expand_hq_history(hq_history: pl.DataFrame, *, min_year: int, max_year: int) -> pl.DataFrame:
    """Expand HQ history intervals into one row per firm-year."""

    rows: list[dict[str, object]] = []
    for row in hq_history.to_dicts():
        start_year = row["start_date"].year
        end_date = row["end_date"]
        end_year = end_date.year if end_date is not None else max_year
        for year in range(max(min_year, start_year), min(max_year, end_year) + 1):
            expanded = dict(row)
            expanded["fyear"] = year
            rows.append(expanded)
    return pl.DataFrame(rows)


def assign_market_id(df: pl.DataFrame, config: MarketConfig) -> pl.DataFrame:
    """Assign a stable market identifier for fixed effects and local density."""

    with_ids = df.with_columns(
        pl.when(pl.col("msa_code").is_not_null())
        .then(pl.format("radius60:{}", pl.col("msa_code")))
        .otherwise(pl.format("radius60:{}", pl.col("state")))
        .alias("focal_market_id_60mi"),
        pl.when(pl.col("msa_code").is_not_null())
        .then(pl.format("radius100:{}", pl.col("msa_code")))
        .otherwise(pl.format("radius100:{}", pl.col("state")))
        .alias("focal_market_id_100mi"),
        pl.when(pl.col("msa_code").is_not_null())
        .then(pl.format("msa:{}", pl.col("msa_code")))
        .otherwise(pl.format("msa:{}", pl.col("state")))
        .alias("focal_msa_market_id"),
        pl.format("state:{}", pl.col("state")).alias("focal_state_market_id"),
        pl.col("msa_code").alias("focal_msa_code"),
        pl.col("state").alias("focal_state"),
    )
    if config.definition == "msa":
        return with_ids.with_columns(pl.col("focal_msa_market_id").alias("market_id"))
    if config.definition == "state":
        return with_ids.with_columns(pl.col("focal_state_market_id").alias("market_id"))
    return with_ids.with_columns(pl.col("focal_market_id_60mi").alias("market_id"))


def compute_local_density(df: pl.DataFrame, config: MarketConfig) -> pl.DataFrame:
    """Add market-size and same-industry density measures to a firm-year panel."""

    rows: list[dict[str, object]] = []
    for _, year_df in df.group_by("fyear", maintain_order=True):
        records = year_df.to_dicts()
        for focal in records:
            market_count_60 = 0
            same_industry_count_60 = 0
            market_count_100 = 0
            same_industry_count_100 = 0
            for other in records:
                if focal["gvkey"] == other["gvkey"]:
                    continue
                distance = haversine_km(
                    float(focal["lat"]),
                    float(focal["lon"]),
                    float(other["lat"]),
                    float(other["lon"]),
                )
                if distance <= miles_to_km(60.0):
                    market_count_60 += 1
                    if focal["ff49"] == other["ff49"]:
                        same_industry_count_60 += 1
                if distance <= miles_to_km(100.0):
                    market_count_100 += 1
                    if focal["ff49"] == other["ff49"]:
                        same_industry_count_100 += 1
            rows.append(
                {
                    "gvkey": focal["gvkey"],
                    "fyear": focal["fyear"],
                    "market_size_60mi": market_count_60,
                    "same_industry_density_60mi": same_industry_count_60,
                    "market_size_100mi": market_count_100,
                    "same_industry_density_100mi": same_industry_count_100,
                }
            )
    density = pl.DataFrame(rows)
    density_cols = density.columns
    merged = df.join(density, on=["gvkey", "fyear"], how="left")
    alias_market = "market_size_60mi" if config.radius_miles <= 60.0 else "market_size_100mi"
    alias_same_industry = (
        "same_industry_density_60mi" if config.radius_miles <= 60.0 else "same_industry_density_100mi"
    )
    if "market_size_60mi" not in density_cols:
        return merged
    return merged.with_columns(
        pl.col(alias_market).alias("market_size_public_firms"),
        pl.col(alias_same_industry).alias("same_industry_local_density"),
    )


def classify_market_relation(
    *,
    source_msa: str | None,
    target_msa: str | None,
    source_state: str | None,
    target_state: str | None,
    source_lat: float | None,
    source_lon: float | None,
    target_lat: float | None,
    target_lon: float | None,
    config: MarketConfig,
) -> MarketRelation:
    """Classify whether a source headquarters lies in the focal firm's feasible market."""

    if None not in (source_lat, source_lon, target_lat, target_lon):
        distance_km = haversine_km(
            float(source_lat),
            float(source_lon),
            float(target_lat),
            float(target_lon),
        )
    else:
        distance_km = None

    if config.definition == "msa" and source_msa is not None and target_msa is not None:
        if source_msa == target_msa:
            return MarketRelation("same_msa", True, distance_km)
    if (
        config.definition == "radius"
        and distance_km is not None
        and distance_km <= miles_to_km(config.radius_miles)
    ):
        return MarketRelation("within_radius", True, distance_km)
    if config.same_state_fallback and source_state is not None and source_state == target_state:
        return MarketRelation("same_state", True, distance_km)
    return MarketRelation("outside_market", False, distance_km)
