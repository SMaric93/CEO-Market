"""Typer CLI for running reproducible research stages from configuration."""

from __future__ import annotations

from pathlib import Path

import typer

from whogetsconsidered.config import load_config
from whogetsconsidered.logging_utils import configure_logging

app = typer.Typer(help="Who Gets Considered? research pipeline")
logger = configure_logging()


@app.command("validate-inputs")
def validate_inputs(config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)) -> None:
    """Validate raw input schemas without constructing derived artifacts."""

    from whogetsconsidered.pipelines.validate_inputs import validate_inputs_pipeline

    validate_inputs_pipeline(load_config(config), logger=logger)


@app.command("pull-wrds")
def pull_wrds_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Pull project-relevant WRDS tables and merge them into canonical extracts."""

    from whogetsconsidered.pipelines.pull_wrds import pull_wrds

    pull_wrds(load_config(config), logger=logger)


@app.command("build-base-panel")
def build_base_panel_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Build firm, executive, and CEO year panels."""

    from whogetsconsidered.pipelines.build_base_panel import build_base_panel

    build_base_panel(load_config(config), logger=logger)


@app.command("build-succession-panel")
def build_succession_panel_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Build succession events and internal bench measures."""

    from whogetsconsidered.pipelines.build_succession_panel import build_succession_panel

    build_succession_panel(load_config(config), logger=logger)


@app.command("build-release-shocks")
def build_release_shocks_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Build released-candidate and supply-shock artifacts."""

    from whogetsconsidered.pipelines.build_release_shocks import build_release_shocks

    build_release_shocks(load_config(config), logger=logger)


@app.command("build-candidate-sets")
def build_candidate_sets_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Construct accessible candidate sets for succession events."""

    from whogetsconsidered.pipelines.build_candidate_sets import build_candidate_sets

    build_candidate_sets(load_config(config), logger=logger)


@app.command("score-fit")
def score_fit_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Score theory-driven and optional predictive fit measures."""

    from whogetsconsidered.pipelines.score_fit import score_fit

    score_fit(load_config(config), logger=logger)


@app.command("estimate-main")
def estimate_main_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Estimate main reduced-form models."""

    from whogetsconsidered.pipelines.estimate_main_results import estimate_main

    estimate_main(load_config(config), logger=logger)


@app.command("estimate-iv")
def estimate_iv_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Estimate IV models when enabled."""

    from whogetsconsidered.pipelines.estimate_main_results import estimate_iv

    estimate_iv(load_config(config), logger=logger)


@app.command("estimate-choice")
def estimate_choice_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Estimate choice-set models when enabled."""

    from whogetsconsidered.pipelines.estimate_main_results import estimate_choice

    estimate_choice(load_config(config), logger=logger)


@app.command("make-tables")
def make_tables_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Render regression tables and descriptive tables."""

    from whogetsconsidered.outputs.tables import make_tables

    make_tables(load_config(config), logger=logger)


@app.command("make-figures")
def make_figures_command(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)
) -> None:
    """Render figures from stored artifacts and results."""

    from whogetsconsidered.outputs.figures import make_figures

    make_figures(load_config(config), logger=logger)


@app.command("run-all")
def run_all_command(config: Path = typer.Option(..., "--config", exists=True, dir_okay=False)) -> None:
    """Run the full enabled pipeline in dependency order."""

    from whogetsconsidered.pipelines.run_all import run_all

    run_all(load_config(config), logger=logger)


if __name__ == "__main__":
    app()
