"""Tests for distance calculations and market-density utilities."""

from __future__ import annotations

import polars as pl

from whogetsconsidered.config import MarketConfig
from whogetsconsidered.geography.distance import haversine_km
from whogetsconsidered.geography.market_defs import assign_market_id, classify_market_relation, compute_local_density


def test_haversine_km_is_small_within_new_york_area() -> None:
    distance = haversine_km(40.7128, -74.0060, 40.7357, -74.1724)
    assert 10 < distance < 20


def test_assign_market_id_uses_msa_when_available() -> None:
    df = pl.DataFrame({"msa_code": ["35620"], "state": ["NY"]})
    assigned = assign_market_id(df, MarketConfig(definition="msa"))
    assert assigned["market_id"].to_list() == ["msa:35620"]


def test_compute_local_density_counts_same_market_firms() -> None:
    df = pl.DataFrame(
        {
            "gvkey": ["1", "2", "3"],
            "fyear": [2020, 2020, 2020],
            "market_id": ["msa:a", "msa:a", "msa:b"],
            "ff49": ["X", "X", "Y"],
            "lat": [0.0, 0.0, 1.0],
            "lon": [0.0, 0.0, 1.0],
        }
    )
    out = compute_local_density(df, MarketConfig(definition="msa"))
    assert out.filter(pl.col("gvkey") == "1")["market_size_public_firms"].item() == 1
    assert out.filter(pl.col("gvkey") == "1")["same_industry_local_density"].item() == 1


def test_classify_market_relation_uses_same_msa() -> None:
    relation = classify_market_relation(
        source_msa="35620",
        target_msa="35620",
        source_state="NY",
        target_state="NY",
        source_lat=40.7,
        source_lon=-74.0,
        target_lat=40.8,
        target_lon=-74.1,
        config=MarketConfig(definition="msa"),
    )
    assert relation.is_local is True
    assert relation.relation == "same_msa"
