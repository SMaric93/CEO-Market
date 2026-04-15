"""Release shock pipeline stage for released candidates and supply measures."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_artifact, write_csv
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.shocks.releases import build_release_supply_metrics, build_released_candidates


def build_release_shocks(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Build released-candidate and event-level released-supply artifacts."""

    with log_stage(logger, "build-release-shocks"):
        registry = ArtifactRegistry(config)
        release_events = read_input_table("release_events", config.inputs.release_events)
        executive_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL))
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        succession_events = pl.read_parquet(registry.require_artifact(ArtifactName.SUCCESSION_EVENTS))

        released_candidates = build_released_candidates(
            release_events,
            executive_year_panel,
            firm_year_panel,
        )
        release_supply_metrics = build_release_supply_metrics(
            succession_events,
            released_candidates,
            firm_year_panel,
            config.market,
            window_years=config.regression.release_window_years,
        )

        write_artifact(
            registry.artifact_path(ArtifactName.RELEASED_CANDIDATES),
            released_candidates,
            lineage={
                "baseline_release_eligibility_flag": "1 if held a CEO, President, or COO role at t-1",
                "main_sample_release_flag": "1 for tier-A clean releases",
                "release_quality_flag": "1 if event date and source HQ coordinates are observed",
            },
        )
        write_artifact(
            registry.artifact_path(ArtifactName.RELEASE_SUPPLY_METRICS),
            release_supply_metrics,
            lineage={
                "release_count_730d_60mi_outind": "main released-pool count within 730 days, 60 miles, and outside focal FF10",
                "from_release_pool_flag": "1 if the realized successor appears in the main released pool",
            },
        )
        quality_audit = released_candidates.select(
            "candidate_person_id",
            "source_gvkey",
            "release_event_year",
            "release_tier",
            "main_sample_release_flag",
            "release_quality_flag",
        )
        write_csv(registry.output_path("logs", "release_event_quality_audit.csv"), quality_audit)
