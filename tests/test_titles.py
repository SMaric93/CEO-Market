"""Tests for deterministic title parsing and CEO-ready rules."""

from __future__ import annotations

import polars as pl

from whogetsconsidered.config import TitlesConfig
from whogetsconsidered.executives.titles import add_title_features, parse_title_flags


def test_parse_title_flags_handles_interim_ceo() -> None:
    flags = parse_title_flags("Interim Chief Executive Officer and President")
    assert flags.is_ceo
    assert flags.is_president
    assert flags.is_interim


def test_add_title_features_marks_chair_president_as_robust_ready() -> None:
    df = pl.DataFrame({"title_raw": ["Chairman and President"]})
    enriched = add_title_features(df, TitlesConfig())
    row = enriched.row(0, named=True)
    assert row["is_ceo_ready"] is True
    assert row["is_ceo_ready_robust"] is True
