"""Exports for unresolved or ambiguous executive identities."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def export_unresolved_names(unresolved_names: pl.DataFrame, path: str | Path) -> Path:
    """Write unresolved normalized names for manual review."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_names.write_csv(output_path)
    return output_path
