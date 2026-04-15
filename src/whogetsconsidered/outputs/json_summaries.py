"""JSON summary writers for machine-readable regression and sample metadata."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from whogetsconsidered.io.writers import write_json


def write_dataframe_summary(path: str | Path, df: pl.DataFrame) -> Path:
    """Write a dataframe as a JSON records summary."""

    return write_json(path, df.to_dicts())
