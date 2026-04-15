"""Base panel pipeline stage for firm, executive, and CEO panels."""

from __future__ import annotations

import logging

from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.executives.cri import build_executive_year_panel
from whogetsconsidered.firm.compustat import build_firm_year_panel
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_artifact
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.succession.identify import identify_ceo_year_panel
from whogetsconsidered.entity_resolution.review_exports import export_unresolved_names


def build_base_panel(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Build firm-year, executive-year, and CEO-year artifacts."""

    with log_stage(logger, "build-base-panel"):
        registry = ArtifactRegistry(config)
        cri = read_input_table("cri_exec_panel", config.inputs.cri_exec_panel)
        compustat = read_input_table("compustat_firm_year", config.inputs.compustat_firm_year)
        hq = read_input_table("hq_history", config.inputs.hq_history)
        ff = read_input_table("ff_industry_map", config.inputs.ff_industry_map)
        noncompete = read_input_table("noncompete_state_year", config.inputs.noncompete_state_year)
        reviewed = (
            read_input_table("reviewed_person_crosswalk", config.inputs.reviewed_person_crosswalk)
            if config.inputs.reviewed_person_crosswalk is not None
            else None
        )

        firm_year_panel = build_firm_year_panel(compustat, hq, ff, noncompete, config)
        executive_year_panel, unresolved = build_executive_year_panel(
            cri,
            config,
            logger=logger,
            reviewed_crosswalk=reviewed,
        )
        executive_year_panel = executive_year_panel.join(
            firm_year_panel.select("gvkey", "fyear", "ff49", "log_assets", "roa_raw", "tobin_q_raw"),
            on=["gvkey", "fyear"],
            how="left",
        )
        ceo_year_panel = identify_ceo_year_panel(executive_year_panel)

        write_artifact(
            registry.artifact_path(ArtifactName.FIRM_YEAR_PANEL),
            firm_year_panel,
            lineage={
                "tobin_q_raw": "(market value of equity + debt proxy) / assets",
                "roa_raw": "ebit / at",
                "market_id": "MSA when available, otherwise state fallback under config",
            },
        )
        write_artifact(
            registry.artifact_path(ArtifactName.EXECUTIVE_YEAR_PANEL),
            executive_year_panel,
            lineage={
                "normalized_name": "exact normalized name used for conservative person resolution",
                "person_id": "reviewed crosswalk id or deterministic AUTO id from exact normalized name",
                "is_ceo_ready": "CEO or President or COO baseline title rule",
            },
        )
        write_artifact(
            registry.artifact_path(ArtifactName.CEO_YEAR_PANEL),
            ceo_year_panel,
            lineage={
                "person_id": "highest-priority CEO rule candidate within firm-year",
                "ceo_interim_flag": "parsed from CEO title string",
            },
        )
        export_unresolved_names(unresolved, registry.output_path("logs", "unresolved_name_matches.csv"))
