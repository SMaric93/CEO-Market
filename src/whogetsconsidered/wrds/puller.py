"""Stage-oriented WRDS pull and merge orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from whogetsconsidered.config import WhoGetsConsideredConfig, WrdsPullConfig, WrdsTableConfig
from whogetsconsidered.io.writers import write_artifact, write_json
from whogetsconsidered.logging_utils import ensure_directory
from whogetsconsidered.wrds.client import WrdsClientProtocol, build_wrds_client
from whogetsconsidered.wrds.merge import (
    _to_polars,
    build_boardex_board_roles_from_bridge,
    build_boardex_capiq_bridge,
    build_boardex_employment_from_bridge,
    build_boardex_people_from_bridge,
    build_compustat_firm_year,
    build_cri_proxy_from_execucomp,
    build_hq_history,
    build_release_events,
    build_wrds_merged_company_year,
    standardize_capiq_people_analytics,
    standardize_boardex_board_roles,
    standardize_boardex_employment,
    standardize_boardex_people,
    standardize_execucomp,
)


def _wrds_specs(config: WrdsPullConfig) -> dict[str, WrdsTableConfig]:
    """Return named WRDS extraction specs from config."""

    return {
        "compustat_fundamentals": config.compustat_fundamentals,
        "company_reference": config.company_reference,
        "execucomp_anncomp": config.execucomp_anncomp,
        "ccm_link_table": config.ccm_link_table,
        "crsp_delist_table": config.crsp_delist_table,
        "boardex_people": config.boardex_people_table,
        "boardex_board_roles": config.boardex_board_roles_table,
        "boardex_employment": config.boardex_employment_table,
        "capiq_people_analytics": config.capiq_people_analytics_table,
        "boardex_ciq_link": config.boardex_ciq_link_table,
    }


def _fetch_table(
    client: WrdsClientProtocol,
    spec: WrdsTableConfig,
    *,
    row_limit: int | None,
) -> pl.DataFrame:
    """Fetch a single WRDS source table using either SQL or direct table access."""

    if spec.sql is not None:
        return _to_polars(client.raw_sql(spec.sql, date_cols=spec.date_columns or None))
    return _to_polars(
        client.get_table(
            library=spec.library,
            table=spec.table,
            columns=spec.columns or None,
            rows=row_limit,
        )
    )


def _stage_wrds_tables(
    config: WrdsPullConfig,
    *,
    client: WrdsClientProtocol,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Pull enabled WRDS tables and stage them as parquet for reproducibility."""

    staging_dir = ensure_directory(config.staging_dir)
    staged_paths: dict[str, Path] = {}
    for name, spec in _wrds_specs(config).items():
        if not spec.enabled:
            continue
        try:
            frame = _fetch_table(client, spec, row_limit=config.row_limit)
        except Exception as exc:  # noqa: BLE001
            if spec.required:
                raise RuntimeError(f"failed to pull required WRDS table `{name}`") from exc
            logger.warning("skipping optional WRDS table %s because pull failed: %s", name, exc)
            continue
        output_path = staging_dir / f"{name}.parquet"
        write_artifact(
            output_path,
            frame,
            metadata={
                "source": "wrds",
                "library": spec.library,
                "table": spec.table,
                "columns": spec.columns,
                "sql": spec.sql,
                "row_limit": config.row_limit,
            },
        )
        staged_paths[name] = output_path
        logger.info("staged WRDS table name=%s rows=%s path=%s", name, frame.height, output_path)
    return staged_paths


def _load_staged_frames(staged_paths: dict[str, Path]) -> dict[str, pl.DataFrame]:
    """Load staged parquet pulls for downstream canonicalization."""

    return {name: pl.read_parquet(path) for name, path in staged_paths.items()}


def _write_canonical(path: Path, name: str, df: pl.DataFrame, *, metadata: dict[str, Any]) -> Path:
    """Write a canonical WRDS-derived extract with lineage metadata."""

    return write_artifact(
        path,
        df,
        lineage={"canonical_input": name, "source": "wrds"},
        metadata=metadata,
    )


def pull_wrds_bundle(
    config: WhoGetsConsideredConfig,
    *,
    logger: logging.Logger,
    client: WrdsClientProtocol | None = None,
) -> dict[str, Any]:
    """Pull project-relevant WRDS data, stage it locally, and merge canonical inputs."""

    if not config.wrds.enabled:
        raise ValueError("WRDS pull requested but `wrds.enabled` is false in config")

    own_client = client is None
    wrds_client = client or build_wrds_client(config)
    canonical_dir = ensure_directory(config.wrds.canonical_dir)
    try:
        staged_paths = _stage_wrds_tables(config.wrds, client=wrds_client, logger=logger)
        staged_frames = _load_staged_frames(staged_paths)

        built_inputs: dict[str, str] = {}
        notes: list[str] = []

        compustat = build_compustat_firm_year(
            staged_frames["compustat_fundamentals"],
            staged_frames["company_reference"],
        )
        compustat_path = _write_canonical(
            canonical_dir / "compustat_firm_year.parquet",
            "compustat_firm_year",
            compustat,
            metadata={
                "filters": {
                    "indfmt": "INDL",
                    "datafmt": "STD",
                    "consol": "C",
                    "popsrc": "D",
                },
            },
        )
        built_inputs["compustat_firm_year"] = str(compustat_path)

        cri_proxy: pl.DataFrame | None = None
        if config.wrds.build_cri_proxy_from_execucomp and "execucomp_anncomp" in staged_frames:
            standardized_execucomp = standardize_execucomp(staged_frames["execucomp_anncomp"])
            execucomp_path = _write_canonical(
                canonical_dir / "execucomp.parquet",
                "execucomp",
                standardized_execucomp,
                metadata={"proxy_role": "standardized optional enrichments table"},
            )
            built_inputs["execucomp"] = str(execucomp_path)

            cri_proxy = build_cri_proxy_from_execucomp(staged_frames["execucomp_anncomp"])
            cri_path = _write_canonical(
                canonical_dir / "cri_exec_panel.parquet",
                "cri_exec_panel",
                cri_proxy,
                metadata={
                    "proxy_role": "Execucomp-based CRI proxy",
                    "coverage_note": "Execucomp is narrower than true CRI coverage",
                },
            )
            built_inputs["cri_exec_panel"] = str(cri_path)
        else:
            notes.append("CRI-style executive coverage was not built because Execucomp was unavailable or disabled.")

        hq_history: pl.DataFrame | None = None
        if config.wrds.hq_geocode_crosswalk is not None:
            hq_history = build_hq_history(
                staged_frames["company_reference"],
                compustat,
                geocode_crosswalk_path=config.wrds.hq_geocode_crosswalk,
            )
            hq_path = _write_canonical(
                canonical_dir / "hq_history.parquet",
                "hq_history",
                hq_history,
                metadata={
                    "construction_note": "single-spell HQ bootstrap from WRDS company metadata plus local geocode crosswalk",
                },
            )
            built_inputs["hq_history"] = str(hq_path)
        else:
            notes.append(
                "HQ history was not built because WRDS company tables do not include geocoded historical HQ spells; provide wrds.hq_geocode_crosswalk to bootstrap this input."
            )

        release_events: pl.DataFrame | None = None
        if (
            config.wrds.build_release_events_from_crsp
            and "ccm_link_table" in staged_frames
            and "crsp_delist_table" in staged_frames
        ):
            release_events = build_release_events(
                staged_frames["crsp_delist_table"],
                staged_frames["ccm_link_table"],
                hq_history=hq_history,
            )
            release_path = _write_canonical(
                canonical_dir / "release_events.parquet",
                "release_events",
                release_events,
                metadata={
                    "construction_note": "CRSP delistings linked to GVKEY via CCM; clean_release_flag is conservative and only marks 2xx merger-style delistings",
                },
            )
            built_inputs["release_events"] = str(release_path)
        else:
            notes.append("Release events were not built because CRSP delist and/or CCM link data were unavailable.")

        optional_outputs: dict[str, str] = {}
        capiq_people_analytics: pl.DataFrame | None = None
        if "capiq_people_analytics" in staged_frames:
            capiq_people_analytics = standardize_capiq_people_analytics(staged_frames["capiq_people_analytics"])
            capiq_people_analytics_path = _write_canonical(
                canonical_dir / "capiq_people_analytics.parquet",
                "capiq_people_analytics",
                capiq_people_analytics,
                metadata={"source": "Capital IQ People Intelligence via WRDS"},
            )
            optional_outputs["capiq_people_analytics"] = str(capiq_people_analytics_path)

        boardex_bridge: pl.DataFrame | None = None
        if (
            "boardex_board_roles" in staged_frames
            and "boardex_employment" in staged_frames
            and "boardex_ciq_link" in staged_frames
            and capiq_people_analytics is not None
        ):
            boardex_bridge = build_boardex_capiq_bridge(
                raw_boardex_people=staged_frames.get("boardex_people", staged_frames["boardex_board_roles"]),
                raw_boardex_board_roles=staged_frames["boardex_board_roles"],
                raw_boardex_employment=staged_frames["boardex_employment"],
                raw_boardex_ciq_link=staged_frames["boardex_ciq_link"],
                capiq_people_analytics=capiq_people_analytics,
            )
            boardex_bridge_path = _write_canonical(
                canonical_dir / "boardex_capiq_bridge.parquet",
                "boardex_capiq_bridge",
                boardex_bridge,
                metadata={"source": "BoardEx joined to Capital IQ People Intelligence via WRDS plink cross-file"},
            )
            optional_outputs["boardex_capiq_bridge"] = str(boardex_bridge_path)

            boardex_people_path = _write_canonical(
                canonical_dir / "boardex_people.parquet",
                "boardex_people",
                build_boardex_people_from_bridge(boardex_bridge),
                metadata={"source": "BoardEx joined to CIQ via WRDS cross-file"},
            )
            optional_outputs["boardex_people"] = str(boardex_people_path)

            boardex_roles_path = _write_canonical(
                canonical_dir / "boardex_board_roles.parquet",
                "boardex_board_roles",
                build_boardex_board_roles_from_bridge(boardex_bridge),
                metadata={"source": "BoardEx joined to CIQ via WRDS cross-file"},
            )
            optional_outputs["boardex_board_roles"] = str(boardex_roles_path)

            boardex_employment_path = _write_canonical(
                canonical_dir / "boardex_employment.parquet",
                "boardex_employment",
                build_boardex_employment_from_bridge(boardex_bridge),
                metadata={"source": "BoardEx joined to CIQ via WRDS cross-file"},
            )
            optional_outputs["boardex_employment"] = str(boardex_employment_path)

        if "boardex_people" in staged_frames:
            if "boardex_people" not in optional_outputs:
                boardex_people_path = _write_canonical(
                    canonical_dir / "boardex_people.parquet",
                    "boardex_people",
                    standardize_boardex_people(staged_frames["boardex_people"]),
                    metadata={"source": "BoardEx via WRDS"},
                )
                optional_outputs["boardex_people"] = str(boardex_people_path)
        if "boardex_board_roles" in staged_frames and "boardex_board_roles" not in optional_outputs:
            if "gvkey" in staged_frames["boardex_board_roles"].columns:
                boardex_roles_path = _write_canonical(
                    canonical_dir / "boardex_board_roles.parquet",
                    "boardex_board_roles",
                    standardize_boardex_board_roles(staged_frames["boardex_board_roles"]),
                    metadata={"source": "BoardEx via WRDS"},
                )
                optional_outputs["boardex_board_roles"] = str(boardex_roles_path)
            else:
                notes.append(
                    "Raw BoardEx board-role pull did not include gvkey; enable capiq_people_analytics_table and boardex_ciq_link_table to materialize canonical boardex_board_roles."
                )
        if "boardex_employment" in staged_frames and "boardex_employment" not in optional_outputs:
            if "gvkey" in staged_frames["boardex_employment"].columns:
                boardex_employment_path = _write_canonical(
                    canonical_dir / "boardex_employment.parquet",
                    "boardex_employment",
                    standardize_boardex_employment(staged_frames["boardex_employment"]),
                    metadata={"source": "BoardEx via WRDS"},
                )
                optional_outputs["boardex_employment"] = str(boardex_employment_path)
            else:
                notes.append(
                    "Raw BoardEx employment pull did not include gvkey; enable capiq_people_analytics_table and boardex_ciq_link_table to materialize canonical boardex_employment."
                )

        merged_panel = build_wrds_merged_company_year(
            compustat,
            cri_proxy=cri_proxy,
            release_events=release_events,
        )
        merged_panel_path = _write_canonical(
            canonical_dir / "wrds_merged_company_year.parquet",
            "wrds_merged_company_year",
            merged_panel,
            metadata={"description": "audit panel combining WRDS-derived firm-year, executive proxy, and release coverage"},
        )

        missing_required_inputs = [
            name
            for name in ("cri_exec_panel", "hq_history", "release_events", "noncompete_state_year", "ff_industry_map")
            if name not in built_inputs
        ]
        manifest: dict[str, Any] = {
            "staged_tables": {name: str(path) for name, path in staged_paths.items()},
            "built_inputs": built_inputs,
            "optional_outputs": optional_outputs,
            "diagnostic_outputs": {"wrds_merged_company_year": str(merged_panel_path)},
            "missing_required_inputs": missing_required_inputs,
            "notes": notes,
        }
        write_json(config.wrds.manifest_path, manifest)
        logger.info("wrote WRDS manifest path=%s", config.wrds.manifest_path)
        return manifest
    finally:
        if own_client:
            wrds_client.close()
