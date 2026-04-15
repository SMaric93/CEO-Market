"""CRI executive panel adapters and canonical person-year construction."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.entity_resolution.crosswalks import attach_person_ids
from whogetsconsidered.entity_resolution.names import normalize_person_name
from whogetsconsidered.executives.career_history import add_internal_tenure
from whogetsconsidered.executives.titles import add_title_features


def build_executive_year_panel(
    cri_exec_panel: pl.DataFrame,
    config: WhoGetsConsideredConfig,
    *,
    logger: logging.Logger,
    reviewed_crosswalk: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a canonical executive-year panel from cleaned CRI-style records."""

    base = cri_exec_panel.with_columns(
        pl.col("exec_name_raw").map_elements(normalize_person_name, return_dtype=pl.String).alias(
            "normalized_name"
        )
    )
    resolved, unresolved = attach_person_ids(base, reviewed_crosswalk, logger=logger)
    enriched = add_title_features(resolved, config.titles)
    enriched = add_internal_tenure(enriched)
    return enriched.sort(["gvkey", "fyear", "exec_rank", "person_id"]), unresolved
