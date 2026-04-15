"""Estimation stages for reduced-form, IV, and choice-model outputs."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_artifact, write_csv, write_json
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.models.conditional_logit import estimate_choice_model
from whogetsconsidered.models.event_study import compute_announcement_cars
from whogetsconsidered.models.first_stage import build_reemployment_panel, estimate_reemployment_model
from whogetsconsidered.models.iv import estimate_fit_iv
from whogetsconsidered.models.succession_outcomes import build_event_analysis_panel, estimate_main_models


def estimate_main(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Estimate validation and reduced-form main results."""

    with log_stage(logger, "estimate-main"):
        registry = ArtifactRegistry(config)
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        executive_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL))
        succession_events = pl.read_parquet(registry.require_artifact(ArtifactName.SUCCESSION_EVENTS))
        internal_bench = pl.read_parquet(registry.require_artifact(ArtifactName.INTERNAL_BENCH))
        released_candidates = pl.read_parquet(registry.require_artifact(ArtifactName.RELEASED_CANDIDATES))
        release_supply_metrics = pl.read_parquet(registry.require_artifact(ArtifactName.RELEASE_SUPPLY_METRICS))
        candidate_universe = pl.read_parquet(registry.require_artifact(ArtifactName.CANDIDATE_UNIVERSE))
        fit_event_summary = pl.read_parquet(registry.require_artifact(ArtifactName.FIT_EVENT_SUMMARY))
        if config.inputs.crsp_daily is not None:
            crsp_daily = read_input_table("crsp_daily", config.inputs.crsp_daily)
            car_frame = compute_announcement_cars(succession_events, crsp_daily)
            succession_events = succession_events.drop(["car_sample_flag"], strict=False).join(
                car_frame,
                on="event_id",
                how="left",
            )
        if config.inputs.tfp_inputs is not None:
            tfp_inputs = read_input_table("tfp_inputs", config.inputs.tfp_inputs)
            firm_year_panel = firm_year_panel.join(tfp_inputs, on=["gvkey", "fyear"], how="left")

        reemployment_panel = build_reemployment_panel(
            released_candidates,
            candidate_universe,
            executive_year_panel,
            firm_year_panel,
            config.market,
        )
        first_stage_results = estimate_reemployment_model(reemployment_panel)
        event_panel = build_event_analysis_panel(
            succession_events,
            internal_bench,
            release_supply_metrics,
            fit_event_summary,
            firm_year_panel,
            horizons=config.regression.outcome_horizons,
        )
        if "text_fit_tfidf_cosine" not in event_panel.columns:
            event_panel = event_panel.with_columns(pl.lit(None, dtype=pl.Float64).alias("text_fit_tfidf_cosine"))
        if "car_m1_p1" not in event_panel.columns:
            event_panel = event_panel.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("car_m1_p1"),
                pl.lit(None, dtype=pl.Float64).alias("car_m2_p2"),
            )
        main_results, diagnostics = estimate_main_models(event_panel, config, logger=logger)
        results = pl.concat([first_stage_results, main_results], how="vertical_relaxed")

        write_artifact(
            registry.artifact_path(ArtifactName.EVENT_ANALYSIS_PANEL),
            event_panel,
            lineage={
                "release_count_730d_60mi_outind": "count of released CEO-ready candidates within 730 days, 60 miles, and outside focal FF10",
                "realized_task_fit_z": "structured fit score of the realized successor",
                "gap_accessible_task_fit_z": "best accessible structured fit minus realized structured fit",
            },
        )
        write_artifact(registry.artifact_path(ArtifactName.MAIN_RESULTS), results)
        write_csv(registry.output_path("models", "regression_results.csv"), results)
        write_json(registry.output_path("models", "regression_results.json"), results.to_dicts())
        write_json(
            registry.output_path("models", "model_metadata.json"),
            {
                "diagnostics": diagnostics,
                "regression_rows": results.height,
                "event_analysis_rows": event_panel.height,
            },
        )
        dropped_specs = pl.DataFrame(diagnostics.get("skipped_specs", []))
        if dropped_specs.height == 0:
            dropped_specs = pl.DataFrame({"spec_id": [], "dep_var": []}, schema={"spec_id": pl.String, "dep_var": pl.String})
        write_csv(registry.output_path("logs", "dropped_observations_log.csv"), dropped_specs)


def estimate_iv(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Estimate IV models for realized fit mediation."""

    with log_stage(logger, "estimate-iv"):
        registry = ArtifactRegistry(config)
        event_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EVENT_ANALYSIS_PANEL))
        iv_results = estimate_fit_iv(event_panel)
        write_artifact(registry.artifact_path(ArtifactName.IV_RESULTS), iv_results)
        write_csv(registry.output_path("models", "iv_results.csv"), iv_results)
        write_json(registry.output_path("models", "iv_results.json"), iv_results.to_dicts())


def estimate_choice(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Estimate candidate-choice models over accessible sets."""

    with log_stage(logger, "estimate-choice"):
        registry = ArtifactRegistry(config)
        accessible_candidate_set = pl.read_parquet(
            registry.require_artifact(ArtifactName.ACCESSIBLE_CANDIDATE_SET)
        )
        choice_results = estimate_choice_model(accessible_candidate_set)
        write_artifact(registry.artifact_path(ArtifactName.CHOICE_RESULTS), choice_results)
        write_csv(registry.output_path("models", "choice_results.csv"), choice_results)
        write_json(registry.output_path("models", "choice_results.json"), choice_results.to_dicts())
