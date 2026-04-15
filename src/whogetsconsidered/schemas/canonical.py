"""Schemas for derived canonical tables written to the artifact registry."""

from __future__ import annotations

from typing import Mapping

from whogetsconsidered.schemas.raw import ColumnSpec, TableSchema


CANONICAL_SCHEMAS: Mapping[str, TableSchema] = {
    "firm_year_panel": TableSchema(
        name="firm_year_panel",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("fyear", "int", nullable=False),
            ColumnSpec("log_assets", "float", nullable=True),
            ColumnSpec("roa_raw", "float", nullable=True),
            ColumnSpec("tobin_q_raw", "float", nullable=True),
        ),
    ),
    "executive_year_panel": TableSchema(
        name="executive_year_panel",
        columns=(
            ColumnSpec("person_id", "str", nullable=False),
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("fyear", "int", nullable=False),
            ColumnSpec("title_raw", "str", nullable=False),
            ColumnSpec("normalized_name", "str", nullable=False),
        ),
    ),
    "ceo_year_panel": TableSchema(
        name="ceo_year_panel",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("fyear", "int", nullable=False),
            ColumnSpec("person_id", "str", nullable=False),
        ),
    ),
    "succession_events": TableSchema(
        name="succession_events",
        columns=(
            ColumnSpec("event_id", "str", nullable=False),
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("succession_year", "int", nullable=False),
            ColumnSpec("predecessor_person_id", "str", nullable=False),
            ColumnSpec("successor_person_id", "str", nullable=False),
        ),
    ),
    "internal_bench": TableSchema(
        name="internal_bench",
        columns=(
            ColumnSpec("event_id", "str", nullable=False),
            ColumnSpec("num_ceo_ready_insiders_tminus1", "int", nullable=False),
        ),
    ),
    "released_candidates": TableSchema(
        name="released_candidates",
        columns=(
            ColumnSpec("candidate_person_id", "str", nullable=False),
            ColumnSpec("release_year", "int", nullable=False),
            ColumnSpec("source_gvkey", "str", nullable=False),
        ),
    ),
    "release_supply_metrics": TableSchema(
        name="release_supply_metrics",
        columns=(
            ColumnSpec("event_id", "str", nullable=False),
            ColumnSpec("generic_supply", "int", nullable=False),
            ColumnSpec("released_source_flag", "int", nullable=False),
        ),
    ),
    "accessible_candidate_set": TableSchema(
        name="accessible_candidate_set",
        columns=(
            ColumnSpec("event_id", "str", nullable=False),
            ColumnSpec("candidate_person_id", "str", nullable=False),
            ColumnSpec("chosen_flag", "int", nullable=False),
        ),
    ),
}
