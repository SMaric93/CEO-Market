"""Tests for accessible candidate sets and task-alignment fit scoring."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from whogetsconsidered.candidates.accessible_set import build_accessible_candidate_set
from whogetsconsidered.candidates.pool_metrics import build_fit_event_summary
from whogetsconsidered.candidates.universe import build_candidate_universe
from whogetsconsidered.config import load_config
from whogetsconsidered.executives.cri import build_executive_year_panel
from whogetsconsidered.fit.task_alignment import score_task_alignment
from whogetsconsidered.firm.compustat import build_firm_year_panel
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.shocks.releases import build_released_candidates
from whogetsconsidered.succession.classify import classify_successions
from whogetsconsidered.succession.identify import detect_successions, identify_ceo_year_panel


FIXTURE_DIR = Path("tests/fixtures")


def test_accessible_set_keeps_chosen_candidate_and_scores_fit() -> None:
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
    successions = classify_successions(
        detect_successions(identify_ceo_year_panel(exec_panel)),
        exec_panel,
        firm_panel,
        config.market,
    )
    released = build_released_candidates(
        read_input_table("release_events", FIXTURE_DIR / "synthetic_delist.parquet"),
        exec_panel,
        firm_panel,
    )
    universe = build_candidate_universe(exec_panel, firm_panel)
    accessible = build_accessible_candidate_set(
        successions,
        exec_panel,
        released,
        universe,
        firm_panel,
        config,
    )
    scored = score_task_alignment(accessible, successions, exec_panel, firm_panel)
    summary = build_fit_event_summary(scored)

    assert accessible.filter((pl.col("event_id") == "001003_2022") & (pl.col("chosen_flag") == 1)).height == 1
    assert accessible.filter((pl.col("event_id") == "001005_2022") & (pl.col("chosen_flag") == 1)).height == 1
    assert scored["task_alignment_fit_score"].null_count() == 0
    assert summary.height == successions.height
    assert summary.filter(pl.col("event_id") == "001002_2021")["GapTaskFit_e"].item() >= 0


def test_boardex_known_to_board_flag_can_be_applied() -> None:
    config = load_config("examples/minimal_config.yml")
    config.features.boardex_enabled = True
    logger = logging.getLogger("test")
    reviewed = read_input_table("reviewed_person_crosswalk", FIXTURE_DIR / "synthetic_crosswalk.parquet")
    frank_id = reviewed.filter(pl.col("normalized_name") == "frankfox")["person_id"].item()
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
    successions = classify_successions(
        detect_successions(identify_ceo_year_panel(exec_panel)),
        exec_panel,
        firm_panel,
        config.market,
    )
    released = build_released_candidates(
        read_input_table("release_events", FIXTURE_DIR / "synthetic_delist.parquet"),
        exec_panel,
        firm_panel,
    )
    universe = build_candidate_universe(exec_panel, firm_panel)
    accessible = build_accessible_candidate_set(
        successions,
        exec_panel,
        released,
        universe,
        firm_panel,
        config,
        boardex_board_roles=read_input_table(
            "boardex_board_roles", FIXTURE_DIR / "synthetic_boardex_board_roles.parquet"
        ),
        boardex_employment=read_input_table(
            "boardex_employment", FIXTURE_DIR / "synthetic_boardex_employment.parquet"
        ),
    )
    flag = accessible.filter(
        (pl.col("event_id") == "001002_2021") & (pl.col("candidate_person_id") == frank_id)
    )["known_to_board_flag"].item()
    assert flag == 1
