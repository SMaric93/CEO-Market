"""Candidate-set construction stage for feasible successor pools."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.candidates.accessible_set import build_accessible_candidate_set
from whogetsconsidered.candidates.universe import build_candidate_universe
from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_artifact
from whogetsconsidered.logging_utils import log_stage


def build_candidate_sets(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Build candidate-universe and accessible-set artifacts."""

    with log_stage(logger, "build-candidate-sets"):
        registry = ArtifactRegistry(config)
        executive_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL))
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        succession_events = pl.read_parquet(registry.require_artifact(ArtifactName.SUCCESSION_EVENTS))
        released_candidates = pl.read_parquet(registry.require_artifact(ArtifactName.RELEASED_CANDIDATES))
        boardex_board_roles = (
            read_input_table("boardex_board_roles", config.inputs.boardex_board_roles)
            if config.features.boardex_enabled and config.inputs.boardex_board_roles is not None
            else None
        )
        boardex_employment = (
            read_input_table("boardex_employment", config.inputs.boardex_employment)
            if config.features.boardex_enabled and config.inputs.boardex_employment is not None
            else None
        )

        candidate_universe = build_candidate_universe(executive_year_panel, firm_year_panel)
        accessible_candidate_set = build_accessible_candidate_set(
            succession_events,
            executive_year_panel,
            released_candidates,
            candidate_universe,
            firm_year_panel,
            config,
            boardex_board_roles=boardex_board_roles,
            boardex_employment=boardex_employment,
        )

        write_artifact(
            registry.artifact_path(ArtifactName.CANDIDATE_UNIVERSE),
            candidate_universe,
            lineage={
                "portable_quality_score": "standardized composite of prior CEO experience and prior firm quality",
                "candidate_year": "last year of observed CEO-ready availability used for pre-event features",
            },
        )
        write_artifact(
            registry.artifact_path(ArtifactName.ACCESSIBLE_CANDIDATE_SET),
            accessible_candidate_set,
            lineage={
                "internal_flag": "1 if candidate is a CEO-ready insider at t-1",
                "released_flag": "1 if candidate comes from the released-candidate pool",
                "distance_miles_if": "great-circle miles from candidate source to focal HQ",
                "known_to_board_flag": "BoardEx employment or board tie when enabled",
                "chosen_out_of_market_flag": "1 if chosen successor had to be injected to avoid dropping the realized hire",
            },
        )
