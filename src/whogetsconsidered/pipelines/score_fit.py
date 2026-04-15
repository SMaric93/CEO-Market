"""Fit-scoring stage for task-alignment and predictive fit outputs."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.candidates.pool_metrics import build_fit_event_summary
from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.fit.predictive_fit import add_predictive_fit_scores
from whogetsconsidered.fit.task_alignment import score_task_alignment
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_artifact
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.models.succession_outcomes import build_event_analysis_panel


def score_fit(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Score task-alignment fit and build event-level realized-fit summaries."""

    with log_stage(logger, "score-fit"):
        registry = ArtifactRegistry(config)
        accessible_candidate_set = pl.read_parquet(
            registry.require_artifact(ArtifactName.ACCESSIBLE_CANDIDATE_SET)
        )
        succession_events = pl.read_parquet(registry.require_artifact(ArtifactName.SUCCESSION_EVENTS))
        executive_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL))
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        internal_bench = pl.read_parquet(registry.require_artifact(ArtifactName.INTERNAL_BENCH))
        release_supply_metrics = pl.read_parquet(registry.require_artifact(ArtifactName.RELEASE_SUPPLY_METRICS))

        scored = score_task_alignment(
            accessible_candidate_set,
            succession_events,
            executive_year_panel,
            firm_year_panel,
        )
        if config.features.predictive_fit_enabled:
            event_panel = build_event_analysis_panel(
                succession_events,
                internal_bench,
                release_supply_metrics,
                build_fit_event_summary(scored),
                firm_year_panel,
                horizons=config.regression.outcome_horizons,
            )
            scored = add_predictive_fit_scores(scored, event_panel, config)
        fit_summary = build_fit_event_summary(scored)

        write_artifact(
            registry.artifact_path(ArtifactName.ACCESSIBLE_CANDIDATE_SET),
            scored,
            lineage={
                "task_fit_z_if": "structured fit z-score based on squared distance between firm needs and candidate experience",
                "task_alignment_fit_score": "alias for task_fit_z_if retained for downstream compatibility",
                "predictive_fit_score": "reserved for cross-fitted predictive fit when enabled",
            },
        )
        write_artifact(
            registry.artifact_path(ArtifactName.FIT_EVENT_SUMMARY),
            fit_summary,
            lineage={
                "gap_accessible_task_fit_z": "best accessible structured fit minus realized structured fit",
                "realized_task_fit_z": "structured fit of the chosen successor",
            },
        )
