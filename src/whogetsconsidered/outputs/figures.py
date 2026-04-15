"""Figure rendering for the empirical output bundle."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
import matplotlib.pyplot as plt
import polars as pl

from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.logging_utils import log_stage


def _save_figure(fig: plt.Figure, registry: ArtifactRegistry, stem: str) -> None:
    fig.savefig(registry.output_path("figures", f"{stem}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(registry.output_path("figures", f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _coefficient_plot(
    results: pl.DataFrame,
    *,
    title: str,
    spec_ids: list[str],
    terms: list[str],
    registry: ArtifactRegistry,
    stem: str,
) -> None:
    subset = results.filter(pl.col("spec_id").is_in(spec_ids) & pl.col("term").is_in(terms))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if subset.height == 0:
        ax.text(0.5, 0.5, "No estimates on current sample", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title)
        _save_figure(fig, registry, stem)
        return
    labels = [f"{row['spec_id']} | {row['dep_var']} | {row['term']}" for row in subset.to_dicts()]
    y = list(range(subset.height))
    coef = subset["coefficient"].to_list()
    lower = subset["ci_lower"].to_list()
    upper = subset["ci_upper"].to_list()
    ax.errorbar(
        coef,
        y,
        xerr=[[c - l for c, l in zip(coef, lower, strict=True)], [u - c for c, u in zip(coef, upper, strict=True)]],
        fmt="o",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    _save_figure(fig, registry, stem)


def make_figures(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Render the paper's required figure set."""

    registry = ArtifactRegistry(config)
    with log_stage(logger, "make-figures"):
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        succession_events = pl.read_parquet(registry.require_artifact(ArtifactName.SUCCESSION_EVENTS))
        released_candidates = pl.read_parquet(registry.require_artifact(ArtifactName.RELEASED_CANDIDATES))
        event_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EVENT_ANALYSIS_PANEL))
        results = pl.read_parquet(registry.require_artifact(ArtifactName.MAIN_RESULTS))

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        stages = ["Firm-years", "Exec-years", "CEO-years", "Successions", "Released candidates", "Event-candidate rows"]
        counts = [
            firm_year_panel.height,
            pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL)).height,
            pl.read_parquet(registry.require_artifact(ArtifactName.CEO_YEAR_PANEL)).height,
            succession_events.height,
            released_candidates.height,
            pl.read_parquet(registry.require_artifact(ArtifactName.ACCESSIBLE_CANDIDATE_SET)).height,
        ]
        ax1.bar(stages, counts)
        ax1.set_ylabel("Rows")
        ax1.set_title("Figure 1. Sample construction flowchart")
        ax1.tick_params(axis="x", rotation=30)
        _save_figure(fig1, registry, "figure_1_sample_flowchart")

        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        by_year = released_candidates.group_by("release_event_year", maintain_order=True).len().sort("release_event_year")
        axes[0].bar(by_year["release_event_year"].to_list(), by_year["len"].to_list())
        axes[0].set_title("Release shocks over time")
        by_market = released_candidates.group_by("source_msa_code", maintain_order=True).len()
        axes[1].bar([str(value) for value in by_market["source_msa_code"].to_list()], by_market["len"].to_list())
        axes[1].set_title("Release shocks over space")
        axes[1].tick_params(axis="x", rotation=45)
        fig2.suptitle("Figure 2. Distribution of release shocks over time and space")
        _save_figure(fig2, registry, "figure_2_release_shocks")

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        internal = event_panel.filter(pl.col("outsider_flag") == 0)["distance_miles"].fill_null(0.0).to_list()
        outsider = event_panel.filter(pl.col("outsider_flag") == 1)["distance_miles"].fill_null(0.0).to_list()
        ax3.hist([internal, outsider], bins=10, label=["Internal", "Outsider"], alpha=0.7)
        ax3.set_title("Figure 3. Hire distance by internal vs outsider successions")
        ax3.set_xlabel("Distance miles")
        ax3.legend()
        _save_figure(fig3, registry, "figure_3_hire_distance")

        _coefficient_plot(
            results,
            title="Figure 4. Coefficient plot for search-radius outcomes",
            spec_ids=["S1_main", "S2_marketyear"],
            terms=["release_count_730d_60mi_outind", "max_released_task_fit_z"],
            registry=registry,
            stem="figure_4_search_radius",
        )
        _coefficient_plot(
            results,
            title="Figure 5. Coefficient plot for fit outcomes",
            spec_ids=["F1_taskfit", "F2_fitgap", "F3_textfit"],
            terms=["release_count_730d_60mi_outind", "max_released_task_fit_z"],
            registry=registry,
            stem="figure_5_fit_outcomes",
        )
        _coefficient_plot(
            results,
            title="Figure 6. Coefficient plot for CAR and operating outcomes",
            spec_ids=["M1_car_m1_p1", "M2_car_m2_p2", "O1_delta_roa_h3", "O1_delta_q_h3"],
            terms=["release_count_730d_60mi_outind", "max_released_task_fit_z"],
            registry=registry,
            stem="figure_6_outcomes",
        )
        _coefficient_plot(
            results,
            title="Figure 7. Heterogeneity by bench strength and noncompetes",
            spec_ids=["H1_weak_bench", "H2_noncompete"],
            terms=["max_released_task_fit_z:weak_bench_flag", "max_released_task_fit_z:nca_score_l1"],
            registry=registry,
            stem="figure_7_heterogeneity",
        )
