"""Schema validation for cleaned raw extracts supplied by the researcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import polars as pl


@dataclass(frozen=True)
class ColumnSpec:
    """Required column definition for a research table."""

    name: str
    dtype: str
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    """Declarative schema definition for an input or derived table."""

    name: str
    columns: tuple[ColumnSpec, ...]


def _dtype_matches(actual: pl.DataType, expected: str) -> bool:
    """Check whether a Polars dtype is compatible with a schema label."""

    if actual == pl.Null:
        return True
    if expected == "str":
        return actual == pl.Utf8 or actual == pl.String
    if expected == "int":
        return actual.is_integer()
    if expected == "float":
        return actual.is_float() or actual.is_integer()
    if expected == "bool":
        return actual == pl.Boolean
    if expected == "date":
        return actual == pl.Date or actual == pl.Datetime
    raise ValueError(f"unsupported dtype label: {expected}")


def validate_dataframe(df: pl.DataFrame, schema: TableSchema) -> list[str]:
    """Return validation errors for a dataframe against a table schema."""

    errors: list[str] = []
    columns = set(df.columns)
    for column in schema.columns:
        if column.name not in columns:
            errors.append(f"{schema.name}: missing required column `{column.name}`")
            continue
        series = df[column.name]
        if series.dtype == pl.Null and column.nullable:
            continue
        if not _dtype_matches(series.dtype, column.dtype):
            errors.append(
                f"{schema.name}: column `{column.name}` has dtype {series.dtype} "
                f"but expected {column.dtype}"
            )
        if not column.nullable and series.null_count() > 0:
            errors.append(f"{schema.name}: column `{column.name}` contains nulls")
    return errors


RAW_SCHEMAS: Mapping[str, TableSchema] = {
    "cri_exec_panel": TableSchema(
        name="cri_exec_panel",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("fyear", "int", nullable=False),
            ColumnSpec("exec_name_raw", "str", nullable=False),
            ColumnSpec("title_raw", "str", nullable=False),
            ColumnSpec("exec_rank", "int", nullable=True),
            ColumnSpec("filing_date", "date", nullable=True),
        ),
    ),
    "compustat_firm_year": TableSchema(
        name="compustat_firm_year",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("fyear", "int", nullable=False),
            ColumnSpec("sic", "int", nullable=True),
            ColumnSpec("state_hq", "str", nullable=True),
            ColumnSpec("at", "float", nullable=False),
            ColumnSpec("ebit", "float", nullable=True),
            ColumnSpec("xrd", "float", nullable=True),
            ColumnSpec("capx", "float", nullable=True),
            ColumnSpec("dltt", "float", nullable=True),
            ColumnSpec("dlc", "float", nullable=True),
            ColumnSpec("dv", "float", nullable=True),
            ColumnSpec("prcc_f", "float", nullable=True),
            ColumnSpec("csho", "float", nullable=True),
            ColumnSpec("sale", "float", nullable=True),
            ColumnSpec("ceq", "float", nullable=True),
            ColumnSpec("datadate", "date", nullable=True),
        ),
    ),
    "hq_history": TableSchema(
        name="hq_history",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("start_date", "date", nullable=False),
            ColumnSpec("end_date", "date", nullable=True),
            ColumnSpec("city", "str", nullable=False),
            ColumnSpec("state", "str", nullable=False),
            ColumnSpec("zip", "str", nullable=True),
            ColumnSpec("lat", "float", nullable=False),
            ColumnSpec("lon", "float", nullable=False),
            ColumnSpec("msa_code", "str", nullable=True),
        ),
    ),
    "release_events": TableSchema(
        name="release_events",
        columns=(
            ColumnSpec("source_gvkey", "str", nullable=False),
            ColumnSpec("event_date", "date", nullable=False),
            ColumnSpec("event_year", "int", nullable=False),
            ColumnSpec("event_type", "str", nullable=False),
            ColumnSpec("clean_release_flag", "bool", nullable=False),
            ColumnSpec("acquirer_gvkey", "str", nullable=True),
            ColumnSpec("source_hq_lat", "float", nullable=True),
            ColumnSpec("source_hq_lon", "float", nullable=True),
            ColumnSpec("source_msa_code", "str", nullable=True),
        ),
    ),
    "noncompete_state_year": TableSchema(
        name="noncompete_state_year",
        columns=(
            ColumnSpec("state", "str", nullable=False),
            ColumnSpec("year", "int", nullable=False),
            ColumnSpec("nca_score", "float", nullable=False),
        ),
    ),
    "ff_industry_map": TableSchema(
        name="ff_industry_map",
        columns=(
            ColumnSpec("sic", "int", nullable=False),
            ColumnSpec("ff10", "str", nullable=False),
            ColumnSpec("ff49", "str", nullable=False),
        ),
    ),
    "boardex_people": TableSchema(
        name="boardex_people",
        columns=(
            ColumnSpec("person_id", "str", nullable=False),
            ColumnSpec("person_name", "str", nullable=False),
        ),
    ),
    "boardex_board_roles": TableSchema(
        name="boardex_board_roles",
        columns=(
            ColumnSpec("person_id", "str", nullable=False),
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("role_start_date", "date", nullable=True),
            ColumnSpec("role_end_date", "date", nullable=True),
            ColumnSpec("role_title", "str", nullable=True),
        ),
    ),
    "boardex_employment": TableSchema(
        name="boardex_employment",
        columns=(
            ColumnSpec("person_id", "str", nullable=False),
            ColumnSpec("gvkey", "str", nullable=True),
            ColumnSpec("employer_name", "str", nullable=True),
            ColumnSpec("start_date", "date", nullable=True),
            ColumnSpec("end_date", "date", nullable=True),
            ColumnSpec("title", "str", nullable=True),
        ),
    ),
    "execucomp": TableSchema(
        name="execucomp",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("year", "int", nullable=False),
            ColumnSpec("execid", "str", nullable=False),
            ColumnSpec("exec_name_raw", "str", nullable=False),
            ColumnSpec("title_raw", "str", nullable=True),
            ColumnSpec("exec_rank", "int", nullable=True),
            ColumnSpec("pceo", "int", nullable=True),
        ),
    ),
    "blm_bridge": TableSchema(
        name="blm_bridge",
        columns=(
            ColumnSpec("event_id", "str", nullable=False),
            ColumnSpec("match_quality_score", "float", nullable=False),
        ),
    ),
    "travel_time_shocks": TableSchema(
        name="travel_time_shocks",
        columns=(
            ColumnSpec("origin_msa_code", "str", nullable=False),
            ColumnSpec("dest_msa_code", "str", nullable=False),
            ColumnSpec("year", "int", nullable=False),
            ColumnSpec("travel_time_index", "float", nullable=False),
        ),
    ),
    "crsp_daily": TableSchema(
        name="crsp_daily",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("date", "date", nullable=False),
            ColumnSpec("ret", "float", nullable=False),
            ColumnSpec("rf", "float", nullable=False),
            ColumnSpec("mktrf", "float", nullable=False),
            ColumnSpec("smb", "float", nullable=False),
            ColumnSpec("hml", "float", nullable=False),
        ),
    ),
    "ceo_announcement_dates": TableSchema(
        name="ceo_announcement_dates",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("announcement_date", "date", nullable=False),
            ColumnSpec("source_quality_flag", "bool", nullable=False),
        ),
    ),
    "tfp_inputs": TableSchema(
        name="tfp_inputs",
        columns=(
            ColumnSpec("gvkey", "str", nullable=False),
            ColumnSpec("fyear", "int", nullable=False),
            ColumnSpec("tfp_op", "float", nullable=True),
            ColumnSpec("tfp_lp", "float", nullable=True),
        ),
    ),
    "reviewed_person_crosswalk": TableSchema(
        name="reviewed_person_crosswalk",
        columns=(
            ColumnSpec("normalized_name", "str", nullable=False),
            ColumnSpec("person_id", "str", nullable=False),
        ),
    ),
}
