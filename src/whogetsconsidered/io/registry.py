"""Central registry for artifact locations and upstream-dependency checks."""

from __future__ import annotations

from pathlib import Path

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.logging_utils import ensure_directory


PANEL_ARTIFACTS = {
    ArtifactName.FIRM_YEAR_PANEL,
    ArtifactName.EXECUTIVE_YEAR_PANEL,
    ArtifactName.CEO_YEAR_PANEL,
    ArtifactName.SUCCESSION_EVENTS,
    ArtifactName.INTERNAL_BENCH,
    ArtifactName.RELEASED_CANDIDATES,
    ArtifactName.RELEASE_SUPPLY_METRICS,
    ArtifactName.CANDIDATE_UNIVERSE,
    ArtifactName.ACCESSIBLE_CANDIDATE_SET,
    ArtifactName.FIT_EVENT_SUMMARY,
    ArtifactName.EVENT_ANALYSIS_PANEL,
}

MODEL_ARTIFACTS = {
    ArtifactName.MAIN_RESULTS,
    ArtifactName.IV_RESULTS,
    ArtifactName.CHOICE_RESULTS,
    ArtifactName.MODEL_METADATA,
}


class ArtifactRegistry:
    """Resolve input and output paths for pipeline stages."""

    def __init__(self, config: WhoGetsConsideredConfig) -> None:
        self.config = config
        self.artifacts_dir = ensure_directory(config.paths.artifacts_dir)
        self.output_dir = ensure_directory(config.paths.output_dir)

    def artifact_path(self, name: str) -> Path:
        """Return the canonical parquet artifact path for a named stage output."""

        if name in PANEL_ARTIFACTS:
            return self.artifacts_dir / "panels" / f"{name}.parquet"
        if name in MODEL_ARTIFACTS:
            return self.artifacts_dir / "models" / f"{name}.parquet"
        return self.artifacts_dir / f"{name}.parquet"

    def output_path(self, *parts: str) -> Path:
        """Return a path inside the configured output directory."""

        path = self.output_dir.joinpath(*parts)
        ensure_directory(path.parent)
        return path

    def require_artifact(self, name: str) -> Path:
        """Ensure an upstream artifact exists before a later stage runs."""

        path = self.artifact_path(name)
        if not path.exists():
            raise FileNotFoundError(f"required artifact missing: {path}")
        return path
