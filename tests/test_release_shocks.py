"""Tests for released-candidate construction and event-level supply measures."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from whogetsconsidered.config import load_config
from whogetsconsidered.executives.cri import build_executive_year_panel
from whogetsconsidered.firm.compustat import build_firm_year_panel
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.shocks.releases import build_release_supply_metrics, build_released_candidates
from whogetsconsidered.succession.classify import classify_successions
from whogetsconsidered.succession.identify import detect_successions, identify_ceo_year_panel


FIXTURE_DIR = Path("tests/fixtures")


def test_released_candidates_and_generic_supply() -> None:
    config = load_config("examples/minimal_config.yml")
    logger = logging.getLogger("test")
    reviewed = read_input_table("reviewed_person_crosswalk", FIXTURE_DIR / "synthetic_crosswalk.parquet")
    cri = read_input_table("cri_exec_panel", FIXTURE_DIR / "synthetic_cri.parquet")
    exec_panel, _ = build_executive_year_panel(cri, config, logger=logger, reviewed_crosswalk=reviewed)
    firm_panel = build_firm_year_panel(
        read_input_table("compustat_firm_year", FIXTURE_DIR / "synthetic_compustat.parquet"),
        read_input_table("hq_history", FIXTURE_DIR / "synthetic_hq.parquet"),
        read_input_table("ff_industry_map", FIXTURE_DIR / "synthetic_ff_map.parquet"),
        read_input_table("noncompete_state_year", FIXTURE_DIR / "synthetic_noncompete.parquet"),
        config,
    )
    exec_panel = exec_panel.join(
        firm_panel.select("gvkey", "fyear", "ff49", "log_assets", "roa_raw", "tobin_q_raw"),
        on=["gvkey", "fyear"],
        how="left",
    )
    ceo_year = identify_ceo_year_panel(exec_panel)
    successions = classify_successions(
        detect_successions(ceo_year),
        exec_panel,
        firm_panel,
        config.market,
    )
    released = build_released_candidates(
        read_input_table("release_events", FIXTURE_DIR / "synthetic_delist.parquet"),
        exec_panel,
        firm_panel,
    )
    metrics = build_release_supply_metrics(
        successions,
        released,
        firm_panel,
        config.market,
        window_years=config.regression.release_window_years,
    )

    assert released.filter(pl.col("candidate_person_id").is_not_null()).height >= 2
    assert released.filter((pl.col("source_gvkey") == "001004") & (pl.col("exec_name_raw") == "Frank Fox")).height == 1
    assert metrics.filter(pl.col("event_id") == "001002_2021")["generic_supply"].item() >= 1
    assert metrics.filter(pl.col("event_id") == "001002_2021")["release_count_730d_60mi_outind"].item() >= 0
    assert metrics.filter(pl.col("event_id") == "001002_2021")["released_source_flag"].item() == 1
