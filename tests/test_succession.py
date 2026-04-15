"""Tests for CEO identification, succession detection, and internal bench construction."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from whogetsconsidered.config import load_config
from whogetsconsidered.executives.cri import build_executive_year_panel
from whogetsconsidered.firm.compustat import build_firm_year_panel
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.succession.bench import build_internal_bench
from whogetsconsidered.succession.classify import classify_successions
from whogetsconsidered.succession.identify import detect_successions, identify_ceo_year_panel


FIXTURE_DIR = Path("tests/fixtures")


def test_ceo_identification_and_successions_on_synthetic_data() -> None:
    config = load_config("examples/minimal_config.yml")
    logger = logging.getLogger("test")
    cri = read_input_table("cri_exec_panel", FIXTURE_DIR / "synthetic_cri.parquet")
    reviewed = read_input_table("reviewed_person_crosswalk", FIXTURE_DIR / "synthetic_crosswalk.parquet")
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
    successions = detect_successions(ceo_year)
    classified = classify_successions(successions, exec_panel, firm_panel, config.market)
    bench = build_internal_bench(classified, exec_panel)

    assert successions.height >= 4
    assert classified.filter((pl.col("gvkey") == "001001") & (pl.col("succession_year") == 2021))["outsider_flag"].item() == 0
    assert classified.filter((pl.col("gvkey") == "001002") & (pl.col("succession_year") == 2021))["local_external_flag"].item() == 1
    assert classified.filter((pl.col("gvkey") == "001005") & (pl.col("succession_year") == 2022))["interim_flag"].item() is True
    assert bench.filter(pl.col("event_id") == "001001_2021")["num_ceo_ready_insiders_tminus1"].item() >= 2
