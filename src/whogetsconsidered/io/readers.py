"""Validated readers for cleaned CSV and Parquet inputs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from whogetsconsidered.schemas.raw import RAW_SCHEMAS, validate_dataframe


def read_input_table(name: str, path: str | Path) -> pl.DataFrame:
    """Read a cleaned input table and validate it against the declared schema."""

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(table_path)
    elif suffix in {".csv", ".txt"}:
        df = pl.read_csv(table_path, try_parse_dates=True)
    else:
        raise ValueError(f"unsupported file type for {table_path}")

    schema = RAW_SCHEMAS[name]
    errors = validate_dataframe(df, schema)
    if errors:
        raise ValueError("\n".join(errors))
    return df
