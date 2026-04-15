"""Tests for regression, table, figure, and CLI wiring."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from whogetsconsidered.cli import app
from whogetsconsidered.config import load_config
from whogetsconsidered.logging_utils import configure_logging
from whogetsconsidered.outputs.figures import make_figures
from whogetsconsidered.outputs.tables import make_tables
from whogetsconsidered.pipelines.build_base_panel import build_base_panel
from whogetsconsidered.pipelines.build_candidate_sets import build_candidate_sets
from whogetsconsidered.pipelines.build_release_shocks import build_release_shocks
from whogetsconsidered.pipelines.build_succession_panel import build_succession_panel
from whogetsconsidered.pipelines.estimate_main_results import estimate_choice, estimate_iv, estimate_main
from whogetsconsidered.pipelines.score_fit import score_fit


def test_estimation_and_output_artifacts_exist() -> None:
    config = load_config("examples/minimal_config.yml")
    logger = configure_logging()
    build_base_panel(config, logger=logger)
    build_succession_panel(config, logger=logger)
    build_release_shocks(config, logger=logger)
    build_candidate_sets(config, logger=logger)
    score_fit(config, logger=logger)
    estimate_main(config, logger=logger)
    estimate_iv(config, logger=logger)
    estimate_choice(config, logger=logger)
    make_tables(config, logger=logger)
    make_figures(config, logger=logger)

    assert Path("artifacts/models/regression_results.parquet").exists()
    assert Path("artifacts/models/iv_results.parquet").exists()
    assert Path("artifacts/models/choice_results.parquet").exists()
    assert Path("artifacts/panels/event_analysis_panel.parquet").exists()
    results = pl.read_parquet("artifacts/models/regression_results.parquet")
    assert results.filter(pl.col("model_group") == "validation").height > 0
    assert Path("artifacts/tables/table_1_sample_summary.csv").exists()
    assert Path("artifacts/figures/figure_4_search_radius.png").exists()


def test_cli_run_all_wiring() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validate-inputs", "--config", "examples/minimal_config.yml"])
    assert result.exit_code == 0
