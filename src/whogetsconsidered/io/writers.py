"""Artifact writers that save tabular outputs plus machine-readable lineage metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from whogetsconsidered.logging_utils import ensure_directory


def write_artifact(
    path: str | Path,
    df: pl.DataFrame,
    *,
    lineage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a parquet artifact and a JSON sidecar describing derivation lineage."""

    artifact_path = Path(path)
    ensure_directory(artifact_path.parent)
    df.write_parquet(artifact_path)
    payload = {
        "rows": df.height,
        "columns": df.columns,
        "lineage": lineage or {},
        "metadata": metadata or {},
    }
    sidecar = artifact_path.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return artifact_path


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
    """Write JSON output deterministically."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def write_csv(path: str | Path, df: pl.DataFrame) -> Path:
    """Write CSV output deterministically."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    df.write_csv(output_path)
    return output_path
