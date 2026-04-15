"""Publication-style table generation for the empirical specification layer."""

from __future__ import annotations

import logging

import polars as pl

from whogetsconsidered.constants import ArtifactName
from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.io.registry import ArtifactRegistry
from whogetsconsidered.io.writers import write_csv
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.outputs.json_summaries import write_dataframe_summary
from whogetsconsidered.outputs.latex import write_latex_table


def _write_table_bundle(
    registry: ArtifactRegistry,
    *,
    filename: str,
    caption: str,
    df: pl.DataFrame,
) -> None:
    base = registry.output_path("tables", filename)
    write_csv(base.with_suffix(".csv"), df)
    write_dataframe_summary(base.with_suffix(".json"), df)
    write_latex_table(base.with_suffix(".tex"), df, caption=caption, label=f"tab:{filename}")


def _format_result(results: pl.DataFrame, spec_id: str, dep_var: str, term: str) -> str:
    subset = results.filter(
        (pl.col("spec_id") == spec_id)
        & (pl.col("dep_var") == dep_var)
        & (pl.col("term") == term)
    )
    if subset.height == 0:
        return ""
    coef = subset["coefficient"].item()
    se = subset["std_error"].item()
    return f"{coef:.3f} ({se:.3f})"


def _stat_row(results: pl.DataFrame, spec_id: str, dep_var: str, field: str) -> str:
    subset = results.filter((pl.col("spec_id") == spec_id) & (pl.col("dep_var") == dep_var))
    if subset.height == 0:
        return ""
    non_null = subset[field].drop_nulls()
    value = non_null.head(1).item() if len(non_null) > 0 else None
    if value is None:
        return ""
    if field in {"N", "num_clusters"}:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _regression_block(
    results: pl.DataFrame,
    *,
    spec_id: str,
    dep_vars: list[str],
    headline_terms: list[str],
    include_mean: bool = True,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for term in headline_terms:
        row = {"row": term}
        for dep_var in dep_vars:
            row[dep_var] = _format_result(results, spec_id, dep_var, term)
        rows.append(row)
    for label, field in [
        ("controls yes/no", "control_set_description"),
        ("FE spec", "fe_description"),
        ("N", "N"),
        ("clusters", "num_clusters"),
    ]:
        row = {"row": label}
        for dep_var in dep_vars:
            row[dep_var] = _stat_row(results, spec_id, dep_var, field)
        rows.append(row)
    if include_mean:
        row = {"row": "mean dep var"}
        for dep_var in dep_vars:
            row[dep_var] = _stat_row(results, spec_id, dep_var, "mean_dep_var")
        rows.append(row)
    return pl.DataFrame(rows)


def _placeholder_table(message: str) -> pl.DataFrame:
    return pl.DataFrame({"status": [message]})


def make_tables(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Render the paper's exact table bundle."""

    registry = ArtifactRegistry(config)
    with log_stage(logger, "make-tables"):
        firm_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.FIRM_YEAR_PANEL))
        executive_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EXECUTIVE_YEAR_PANEL))
        ceo_year_panel = pl.read_parquet(registry.require_artifact(ArtifactName.CEO_YEAR_PANEL))
        succession_events = pl.read_parquet(registry.require_artifact(ArtifactName.SUCCESSION_EVENTS))
        released_candidates = pl.read_parquet(registry.require_artifact(ArtifactName.RELEASED_CANDIDATES))
        event_panel = pl.read_parquet(registry.require_artifact(ArtifactName.EVENT_ANALYSIS_PANEL))
        results = pl.read_parquet(registry.require_artifact(ArtifactName.MAIN_RESULTS))

        table_1 = pl.DataFrame(
            {
                "panel": [
                    "A. firm-year summary",
                    "A. firm-year summary",
                    "A. firm-year summary",
                    "B. succession-event summary",
                    "B. succession-event summary",
                    "B. succession-event summary",
                    "C. subsample sizes",
                    "C. subsample sizes",
                    "C. subsample sizes",
                ],
                "row": [
                    "N firm-years",
                    "N firms",
                    "mean log assets",
                    "N unique CEOs",
                    "N successions",
                    "N outsider successions",
                    "search sample",
                    "BoardEx sample",
                    "CAR sample",
                ],
                "value": [
                    str(firm_year_panel.height),
                    str(firm_year_panel["gvkey"].n_unique()),
                    f"{float(firm_year_panel['log_assets'].mean()):.3f}",
                    str(ceo_year_panel["person_id"].n_unique()),
                    str(succession_events.height),
                    str(int(succession_events["outsider_flag"].sum())),
                    str(event_panel.height),
                    str(int(event_panel["boardex_sample_flag"].sum())),
                    str(int(event_panel["car_sample_flag"].sum())),
                ],
            }
        )
        _write_table_bundle(registry, filename="table_1_sample_summary", caption="Table 1. Sample construction and summary statistics", df=table_1)

        high_release_cutoff = float(event_panel["release_count_730d_60mi_outind"].median() or 0.0) if event_panel.height else 0.0
        descriptive_groups = {
            "full succession sample": event_panel,
            "internal successions": event_panel.filter(pl.col("outsider_flag") == 0),
            "outsider successions": event_panel.filter(pl.col("outsider_flag") == 1),
            "low release-count markets": event_panel.filter(pl.col("release_count_730d_60mi_outind") <= high_release_cutoff),
            "high release-count markets": event_panel.filter(pl.col("release_count_730d_60mi_outind") > high_release_cutoff),
        }
        descriptive_rows = [
            "outsider_flag",
            "local_external_flag",
            "distant_external_flag",
            "from_release_pool_flag",
            "known_to_board_flag",
            "release_count_730d_60mi_outind",
            "max_released_task_fit_z",
            "avg_released_task_fit_z",
            "same_industry_density_60mi",
            "nca_score_l1",
            "bench_index_z",
        ]
        table_2_rows: list[dict[str, object]] = []
        for row_name in descriptive_rows:
            row = {"row": row_name}
            for label, frame in descriptive_groups.items():
                row[label] = "" if frame.height == 0 else f"{float(frame[row_name].mean() or 0.0):.3f}"
            table_2_rows.append(row)
        table_2 = pl.DataFrame(table_2_rows)
        _write_table_bundle(registry, filename="table_2_descriptive_patterns", caption="Table 2. Descriptive patterns in succession and access shocks", df=table_2)

        table_3 = _regression_block(
            results,
            spec_id="T3_person_level",
            dep_vars=["reemployed_local_24m", "reemployed_anywhere_24m", "reemployed_ceo_ready_24m"],
            headline_terms=["released_candidate_flag"],
        )
        market_validation = _regression_block(
            results,
            spec_id="T3_market_level",
            dep_vars=["local_ceo_ready_entries_m_t_to_tplus2"],
            headline_terms=["num_released_candidates_m_t"],
        )
        table_3 = pl.concat([table_3, market_validation], how="diagonal")
        _write_table_bundle(registry, filename="table_3_validation", caption="Table 3. Validation: released executives re-enter local executive markets", df=table_3)

        table_4 = pl.concat(
            [
                _regression_block(
                    results,
                    spec_id="S1_main",
                    dep_vars=["outsider_flag", "distant_external_flag", "log1p_distance_miles", "from_release_pool_flag"],
                    headline_terms=[
                        "release_count_730d_60mi_outind",
                        "max_released_task_fit_z",
                        "bench_index_z",
                        "nca_score_l1",
                    ],
                ).with_columns(pl.lit("Panel A: S1_main").alias("panel")),
                _regression_block(
                    results,
                    spec_id="S2_marketyear",
                    dep_vars=["outsider_flag", "distant_external_flag", "log1p_distance_miles", "from_release_pool_flag"],
                    headline_terms=[
                        "avg_released_task_fit_z",
                        "max_released_task_fit_z",
                        "topquartile_released_count",
                        "bench_index_z",
                    ],
                ).with_columns(pl.lit("Panel B: S2_marketyear").alias("panel")),
            ],
            how="diagonal",
        )
        _write_table_bundle(registry, filename="table_4_access_shocks_search_radius", caption="Table 4. Access shocks and search radius", df=table_4)

        boardex_outcomes = ["known_to_board_flag", "employment_tie_flag", "board_tie_flag", "network_independent_flag"]
        boardex_table = _regression_block(
            results,
            spec_id="B1_boardex",
            dep_vars=boardex_outcomes,
            headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
            include_mean=False,
        )
        _write_table_bundle(
            registry,
            filename="table_5_network_independence",
            caption="Table 5. Access shocks and network independence",
            df=boardex_table if boardex_table.height > 0 else _placeholder_table("BoardEx subsample unavailable"),
        )

        table_6 = _regression_block(
            results,
            spec_id="F1_taskfit",
            dep_vars=["realized_task_fit_z"],
            headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
        ).join(
            _regression_block(
                results,
                spec_id="F2_fitgap",
                dep_vars=["gap_accessible_task_fit_z"],
                headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
            ),
            on="row",
            how="full",
            coalesce=True,
        ).join(
            _regression_block(
                results,
                spec_id="F3_textfit",
                dep_vars=["text_fit_tfidf_cosine"],
                headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
            ),
            on="row",
            how="full",
            coalesce=True,
        )
        _write_table_bundle(registry, filename="table_6_observable_fit", caption="Table 6. Access shocks and observable fit", df=table_6)

        table_7 = _regression_block(
            results,
            spec_id="M1_car_m1_p1",
            dep_vars=["car_m1_p1"],
            headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
        ).join(
            _regression_block(
                results,
                spec_id="M2_car_m2_p2",
                dep_vars=["car_m2_p2"],
                headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
            ),
            on="row",
            how="full",
            coalesce=True,
        )
        _write_table_bundle(
            registry,
            filename="table_7_announcement_cars",
            caption="Table 7. Access shocks and announcement CARs",
            df=table_7 if table_7.height > 0 else _placeholder_table("Announcement-date CAR inputs unavailable"),
        )

        table_8 = _regression_block(
            results,
            spec_id="O1_delta_roa_h1",
            dep_vars=[
                "delta_roa_h1",
                "delta_roa_h3",
                "delta_q_h1",
                "delta_q_h3",
                "delta_roa_resid_h3",
                "delta_q_resid_h3",
            ],
            headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
        )
        _write_table_bundle(registry, filename="table_8_medium_run_performance", caption="Table 8. Access shocks and medium-run operating performance", df=table_8)

        table_9 = _regression_block(
            results,
            spec_id="O2_tfp_op",
            dep_vars=["delta_tfp_op_h3", "delta_tfp_lp_h3"],
            headline_terms=["release_count_730d_60mi_outind", "max_released_task_fit_z", "bench_index_z"],
            include_mean=False,
        )
        _write_table_bundle(
            registry,
            filename="table_9_productivity",
            caption="Table 9. Access shocks and productivity",
            df=table_9 if table_9.height > 0 else _placeholder_table("TFP inputs unavailable"),
        )

        table_10 = pl.concat(
            [
                _regression_block(
                    results,
                    spec_id="H1_weak_bench",
                    dep_vars=["outsider_flag", "realized_task_fit_z", "car_m1_p1", "delta_roa_h3"],
                    headline_terms=[
                        "max_released_task_fit_z",
                        "weak_bench_flag",
                        "max_released_task_fit_z:weak_bench_flag",
                    ],
                    include_mean=False,
                ).with_columns(pl.lit("Panel A").alias("panel")),
                _regression_block(
                    results,
                    spec_id="H2_noncompete",
                    dep_vars=["outsider_flag", "realized_task_fit_z", "car_m1_p1", "delta_roa_h3"],
                    headline_terms=[
                        "max_released_task_fit_z",
                        "nca_score_l1",
                        "max_released_task_fit_z:nca_score_l1",
                    ],
                    include_mean=False,
                ).with_columns(pl.lit("Panel B").alias("panel")),
            ],
            how="diagonal",
        )
        _write_table_bundle(registry, filename="table_10_heterogeneity", caption="Table 10. Heterogeneity by internal bench and noncompetes", df=table_10)

        iv_path = registry.artifact_path(ArtifactName.IV_RESULTS)
        table_11 = _placeholder_table("IV not estimated")
        if iv_path.exists():
            iv_results = pl.read_parquet(iv_path)
            table_11 = iv_results
        _write_table_bundle(registry, filename="table_11_iv_mediation", caption="Table 11. IV / mediation results", df=table_11)

        table_12 = pl.DataFrame(
            {
                "panel": [
                    "A. 100-mile radius",
                    "B. MSA market",
                    "C. CEO-ready + CFO",
                    "D. 365-day release window",
                    "E. 1095-day release window",
                    "F. travel-access design",
                ],
                "headline_outcomes": [
                    "outsider_flag, realized_task_fit_z, car_m1_p1",
                    "outsider_flag, realized_task_fit_z, car_m1_p1",
                    "outsider_flag, realized_task_fit_z, car_m1_p1",
                    "outsider_flag, realized_task_fit_z, car_m1_p1",
                    "outsider_flag, realized_task_fit_z, car_m1_p1",
                    "outsider_flag, realized_task_fit_z, car_m1_p1",
                ],
                "status": ["configured robustness slot"] * 6,
            }
        )
        _write_table_bundle(registry, filename="table_12_robustness", caption="Table 12. Robustness and secondary design", df=table_12)
