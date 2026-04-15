"""Event-level panels and reduced-form regressions for successor choice and outcomes."""

from __future__ import annotations

from collections.abc import Callable
import logging
import math

import polars as pl
from pydantic import BaseModel, ConfigDict, Field
import statsmodels.formula.api as smf

from whogetsconsidered.config import WhoGetsConsideredConfig


CORE_CONTROL_COLUMNS = [
    "log_assets_l1",
    "roa_l1",
    "q_l1",
    "rd_intensity_l1",
    "rd_indicator_l1",
    "cap_intensity_l1",
    "leverage_l1",
    "dividend_payer_l1",
    "dividend_yield_l1",
    "firm_age_l1",
    "perf_trend_l1",
    "same_industry_density_60mi",
    "market_size_60mi",
    "nca_score_l1",
]


class RegressionSpec(BaseModel):
    """Serializable regression specification metadata used throughout the paper."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    dep_var: str
    regressors: list[str]
    fe_cols: list[str] = Field(default_factory=list)
    sample_filter: str = "all_events"
    model_group: str
    control_set_description: str = "core_controls"


def _cluster_column(config: WhoGetsConsideredConfig) -> tuple[str | None, str]:
    if not config.regression.cluster_by:
        return None, "HC1"
    mapping = {
        "market": "focal_market_id_60mi",
        "firm": "gvkey",
        "year": "event_year",
        "state": "focal_state",
    }
    cluster_cols = [mapping.get(name, name) for name in config.regression.cluster_by]
    return cluster_cols[0], ", ".join(cluster_cols)


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def build_event_analysis_panel(
    succession_events: pl.DataFrame,
    internal_bench: pl.DataFrame,
    release_supply_metrics: pl.DataFrame,
    fit_event_summary: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
    *,
    horizons: list[int],
) -> pl.DataFrame:
    """Build the event-level analysis panel for successor-choice and outcome regressions."""

    lag_controls = firm_year_panel.select(
        "gvkey",
        pl.col("fyear").alias("lag_year"),
        "ff10",
        "ff49",
        "state",
        "focal_state",
        "focal_msa_code",
        "focal_market_id_60mi",
        "focal_market_id_100mi",
        "log_assets",
        "roa_raw",
        "q_raw",
        "roa_resid",
        "q_resid",
        "rd_intensity",
        "rd_indicator",
        "capital_intensity",
        "leverage",
        "dividend_payer",
        "dividend_yield",
        "firm_age",
        "pre_succession_performance_trend",
        "same_industry_density_60mi",
        "market_size_60mi",
        "same_industry_density_100mi",
        "market_size_100mi",
        "noncompete_score",
    )
    event_panel = (
        succession_events.with_columns(
            (pl.col("succession_year") - 1).alias("lag_year"),
            pl.col("succession_year").alias("event_year"),
        )
        .join(lag_controls, on=["gvkey", "lag_year"], how="left")
        .join(internal_bench, on="event_id", how="left")
        .join(release_supply_metrics, on="event_id", how="left")
        .join(fit_event_summary, on="event_id", how="left")
        .with_columns(
            pl.col("log_assets").alias("log_assets_l1"),
            pl.col("roa_raw").alias("roa_l1"),
            pl.col("q_raw").alias("q_l1"),
            pl.col("rd_intensity").alias("rd_intensity_l1"),
            pl.col("rd_indicator").alias("rd_indicator_l1"),
            pl.col("capital_intensity").alias("cap_intensity_l1"),
            pl.col("leverage").alias("leverage_l1"),
            pl.col("dividend_payer").alias("dividend_payer_l1"),
            pl.col("dividend_yield").alias("dividend_yield_l1"),
            pl.col("firm_age").alias("firm_age_l1"),
            pl.col("pre_succession_performance_trend").alias("perf_trend_l1"),
            pl.col("noncompete_score").alias("nca_score_l1"),
            pl.col("known_to_board_flag").fill_null(0),
            pl.col("board_tie_flag").fill_null(0),
            pl.col("employment_tie_flag").fill_null(0),
            (
                pl.col("outsider_flag").cast(pl.Int8)
                * (1 - pl.col("known_to_board_flag").fill_null(0).cast(pl.Int8))
            ).alias("network_independent_flag"),
            pl.col("from_release_pool_flag").fill_null(0).cast(pl.Int8),
            pl.col("release_count_730d_60mi_outind").fill_null(0),
            pl.col("release_count_365d_60mi_outind").fill_null(0),
            pl.col("release_count_1095d_60mi_outind").fill_null(0),
            pl.col("release_count_730d_100mi_outind").fill_null(0),
            pl.col("announcement_imputed_flag").fill_null(0).cast(pl.Int8),
            pl.col("event_year").cast(pl.Int64),
        )
        .with_columns(
            pl.col("focal_market_id_60mi").alias("focal_market_60mi"),
            pl.format("{}_{}", pl.col("ff49"), pl.col("event_year")).alias("ff49_event_year"),
            pl.format("{}_{}", pl.col("focal_market_id_60mi"), pl.col("event_year")).alias(
                "focal_market_60mi_event_year"
            ),
        )
    )

    history_lookup = {
        str(group_df["gvkey"].item(0)): group_df.sort("fyear")
        for _, group_df in firm_year_panel.group_by("gvkey", maintain_order=True)
    }
    rows: list[dict[str, object]] = []
    for row in event_panel.to_dicts():
        history = history_lookup[str(row["gvkey"])]
        year = int(row["succession_year"])
        pre = history.filter(pl.col("fyear").is_in([year - 1, year - 2]))
        pre_roa = _safe_mean(pre["roa_raw"].to_list())
        pre_q = _safe_mean(pre["q_raw"].to_list())
        pre_roa_resid = _safe_mean(pre["roa_resid"].to_list())
        pre_q_resid = _safe_mean(pre["q_resid"].to_list())
        row["delta_roa_h1"] = None
        row["delta_roa_h3"] = None
        row["delta_q_h1"] = None
        row["delta_q_h3"] = None
        row["delta_roa_resid_h1"] = None
        row["delta_roa_resid_h3"] = None
        row["delta_q_resid_h1"] = None
        row["delta_q_resid_h3"] = None
        for horizon in horizons:
            post = history.filter((pl.col("fyear") >= year + 1) & (pl.col("fyear") <= year + horizon))
            if post.height < horizon:
                continue
            post_roa = _safe_mean(post["roa_raw"].to_list())
            post_q = _safe_mean(post["q_raw"].to_list())
            post_roa_resid = _safe_mean(post["roa_resid"].to_list())
            post_q_resid = _safe_mean(post["q_resid"].to_list())
            row[f"delta_roa_h{horizon}"] = (
                post_roa - pre_roa if post_roa is not None and pre_roa is not None else None
            )
            row[f"delta_q_h{horizon}"] = post_q - pre_q if post_q is not None and pre_q is not None else None
            row[f"delta_roa_resid_h{horizon}"] = (
                post_roa_resid - pre_roa_resid
                if post_roa_resid is not None and pre_roa_resid is not None
                else None
            )
            row[f"delta_q_resid_h{horizon}"] = (
                post_q_resid - pre_q_resid
                if post_q_resid is not None and pre_q_resid is not None
                else None
            )
        if "tfp_op" in history.columns:
            pre_tfp_op = _safe_mean(history.filter(pl.col("fyear").is_in([year - 1, year - 2]))["tfp_op"].to_list())
            pre_tfp_lp = _safe_mean(history.filter(pl.col("fyear").is_in([year - 1, year - 2]))["tfp_lp"].to_list())
            post3 = history.filter((pl.col("fyear") >= year + 1) & (pl.col("fyear") <= year + 3))
            if post3.height >= 3:
                post_tfp_op = _safe_mean(post3["tfp_op"].to_list())
                post_tfp_lp = _safe_mean(post3["tfp_lp"].to_list())
                row["delta_tfp_op_h3"] = (
                    post_tfp_op - pre_tfp_op
                    if post_tfp_op is not None and pre_tfp_op is not None
                    else None
                )
                row["delta_tfp_lp_h3"] = (
                    post_tfp_lp - pre_tfp_lp
                    if post_tfp_lp is not None and pre_tfp_lp is not None
                    else None
                )
            else:
                row["delta_tfp_op_h3"] = None
                row["delta_tfp_lp_h3"] = None
        rows.append(row)
    return pl.DataFrame(rows).with_columns(
        pl.col("outsider_flag").cast(pl.Int8),
        pl.col("local_external_flag").cast(pl.Int8),
        pl.col("distant_external_flag").cast(pl.Int8),
        pl.col("interim_flag").cast(pl.Int8),
        pl.col("event_sample_flag").fill_null(1).cast(pl.Int8),
        pl.col("boardex_sample_flag").fill_null(0).cast(pl.Int8),
        pl.col("car_sample_flag").fill_null(0).cast(pl.Int8),
        pl.col("tfp_sample_flag").fill_null(0).cast(pl.Int8),
    )


def _apply_sample_filter(df: pl.DataFrame, filter_name: str) -> tuple[pl.DataFrame, str]:
    if filter_name == "all_events":
        return df, "all succession events"
    if filter_name == "outsider_only":
        return df.filter(pl.col("outsider_flag") == 1), "outsider-successor events only"
    if filter_name == "boardex_subsample":
        return df.filter(pl.col("boardex_sample_flag") == 1), "BoardEx subsample"
    if filter_name == "car_subsample":
        return df.filter(pl.col("car_sample_flag") == 1), "announcement-date CAR subsample"
    if filter_name == "tfp_subsample":
        return df.filter(pl.col("tfp_sample_flag") == 1), "TFP subsample"
    raise ValueError(f"unsupported sample filter: {filter_name}")


def _fit_ols(
    df: pl.DataFrame,
    spec: RegressionSpec,
    config: WhoGetsConsideredConfig,
    *,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    sample, sample_description = _apply_sample_filter(df, spec.sample_filter)
    needed_cols = [spec.dep_var, *[term for term in spec.regressors if ":" not in term and "*" not in term]]
    needed_cols = [column for column in needed_cols if column in sample.columns]
    sample = sample.drop_nulls(needed_cols)
    if spec.dep_var not in sample.columns or sample.height < 4:
        logger.warning("skipping spec=%s dep_var=%s due to insufficient data", spec.spec_id, spec.dep_var)
        return []

    formula_terms = spec.regressors.copy()
    effective_fe: list[str] = []
    for fe_col in spec.fe_cols:
        if fe_col in sample.columns and sample[fe_col].n_unique() > 1 and sample[fe_col].n_unique() < sample.height:
            effective_fe.append(fe_col)
    formula_terms.extend([f"C({fe_col})" for fe_col in effective_fe])
    if not formula_terms:
        formula_terms = ["1"]
    formula = f"{spec.dep_var} ~ " + " + ".join(formula_terms)
    pdf = sample.to_pandas()
    base_fit = smf.ols(formula, data=pdf).fit()
    cluster_col, cluster_description = _cluster_column(config)
    n_clusters = 0
    if cluster_col is not None and cluster_col in sample.columns and sample[cluster_col].n_unique() >= 2:
        try:
            fit = base_fit.get_robustcov_results(cov_type="cluster", groups=pdf[cluster_col])
            n_clusters = int(sample[cluster_col].n_unique())
        except Exception:
            fit = base_fit.get_robustcov_results(cov_type="HC1")
            cluster_description = "HC1"
    else:
        fit = base_fit.get_robustcov_results(cov_type="HC1")
        cluster_description = "HC1"
    conf = fit.conf_int()
    param_names = list(fit.model.exog_names)
    rows: list[dict[str, object]] = []
    for idx, term in enumerate(param_names):
        ci_low = float(conf[idx][0]) if not hasattr(conf, "loc") else float(conf.loc[term, 0])
        ci_high = float(conf[idx][1]) if not hasattr(conf, "loc") else float(conf.loc[term, 1])
        rows.append(
            {
                "model_group": spec.model_group,
                "dep_var": spec.dep_var,
                "spec_id": spec.spec_id,
                "term": term,
                "coefficient": float(fit.params[idx]),
                "std_error": float(fit.bse[idx]),
                "t_stat": float(fit.tvalues[idx]),
                "p_value": float(fit.pvalues[idx]),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "N": int(fit.nobs),
                "num_clusters": n_clusters,
                "cluster_level": cluster_description,
                "fe_description": ", ".join(effective_fe) if effective_fe else "none",
                "control_set_description": spec.control_set_description,
                "sample_filter_description": sample_description,
                "mean_dep_var": float(sample[spec.dep_var].mean()),
            }
        )
    return rows


def _build_specs() -> list[RegressionSpec]:
    specs: list[RegressionSpec] = []
    s1_regressors = [
        "release_count_730d_60mi_outind",
        "max_released_task_fit_z",
        "bench_index_z",
        *CORE_CONTROL_COLUMNS,
    ]
    for dep_var, sample_filter in [
        ("outsider_flag", "all_events"),
        ("distant_external_flag", "all_events"),
        ("log1p_distance_miles", "outsider_only"),
        ("from_release_pool_flag", "all_events"),
    ]:
        specs.append(
            RegressionSpec(
                spec_id="S1_main",
                dep_var=dep_var,
                regressors=s1_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter=sample_filter,
                model_group="search",
            )
        )
    s2_regressors = [
        "avg_released_task_fit_z",
        "max_released_task_fit_z",
        "topquartile_released_count",
        "bench_index_z",
        "log_assets_l1",
        "roa_l1",
        "q_l1",
        "rd_intensity_l1",
        "rd_indicator_l1",
        "cap_intensity_l1",
        "leverage_l1",
        "dividend_payer_l1",
        "dividend_yield_l1",
        "firm_age_l1",
        "perf_trend_l1",
        "same_industry_density_60mi",
    ]
    for dep_var, sample_filter in [
        ("outsider_flag", "all_events"),
        ("distant_external_flag", "all_events"),
        ("log1p_distance_miles", "outsider_only"),
        ("from_release_pool_flag", "all_events"),
    ]:
        specs.append(
            RegressionSpec(
                spec_id="S2_marketyear",
                dep_var=dep_var,
                regressors=s2_regressors,
                fe_cols=["focal_market_60mi_event_year", "ff49"],
                sample_filter=sample_filter,
                model_group="search",
            )
        )
    for dep_var in [
        "known_to_board_flag",
        "employment_tie_flag",
        "board_tie_flag",
        "network_independent_flag",
    ]:
        specs.append(
            RegressionSpec(
                spec_id="B1_boardex",
                dep_var=dep_var,
                regressors=s1_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter="boardex_subsample",
                model_group="boardex",
            )
        )
    for spec_id, dep_var in [
        ("F1_taskfit", "realized_task_fit_z"),
        ("F2_fitgap", "gap_accessible_task_fit_z"),
        ("F3_textfit", "text_fit_tfidf_cosine"),
        ("M1_car_m1_p1", "car_m1_p1"),
        ("M2_car_m2_p2", "car_m2_p2"),
    ]:
        sample_filter = "car_subsample" if dep_var.startswith("car_") else "all_events"
        specs.append(
            RegressionSpec(
                spec_id=spec_id,
                dep_var=dep_var,
                regressors=s1_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter=sample_filter,
                model_group="fit" if spec_id.startswith("F") else "market_reaction",
            )
        )
    for dep_var in [
        "delta_roa_h1",
        "delta_roa_h3",
        "delta_q_h1",
        "delta_q_h3",
        "delta_roa_resid_h1",
        "delta_roa_resid_h3",
        "delta_q_resid_h1",
        "delta_q_resid_h3",
    ]:
        specs.append(
            RegressionSpec(
                spec_id=f"O1_{dep_var}",
                dep_var=dep_var,
                regressors=s1_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter="all_events",
                model_group="operating",
            )
        )
    for spec_id, dep_var in [("O2_tfp_op", "delta_tfp_op_h3"), ("O3_tfp_lp", "delta_tfp_lp_h3")]:
        specs.append(
            RegressionSpec(
                spec_id=spec_id,
                dep_var=dep_var,
                regressors=s1_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter="tfp_subsample",
                model_group="productivity",
            )
        )
    h1_regressors = [
        "max_released_task_fit_z",
        "weak_bench_flag",
        "max_released_task_fit_z:weak_bench_flag",
        *CORE_CONTROL_COLUMNS,
        "bench_index_z",
    ]
    for dep_var in ["outsider_flag", "realized_task_fit_z", "car_m1_p1", "delta_roa_h3"]:
        specs.append(
            RegressionSpec(
                spec_id="H1_weak_bench",
                dep_var=dep_var,
                regressors=h1_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter="car_subsample" if dep_var == "car_m1_p1" else "all_events",
                model_group="heterogeneity",
            )
        )
    h2_regressors = [
        "max_released_task_fit_z",
        "nca_score_l1",
        "max_released_task_fit_z:nca_score_l1",
        "log_assets_l1",
        "roa_l1",
        "q_l1",
        "rd_intensity_l1",
        "rd_indicator_l1",
        "cap_intensity_l1",
        "leverage_l1",
        "dividend_payer_l1",
        "dividend_yield_l1",
        "firm_age_l1",
        "perf_trend_l1",
        "same_industry_density_60mi",
        "market_size_60mi",
        "bench_index_z",
    ]
    for dep_var in ["outsider_flag", "realized_task_fit_z", "car_m1_p1", "delta_roa_h3"]:
        specs.append(
            RegressionSpec(
                spec_id="H2_noncompete",
                dep_var=dep_var,
                regressors=h2_regressors,
                fe_cols=["focal_market_id_60mi", "ff49_event_year"],
                sample_filter="car_subsample" if dep_var == "car_m1_p1" else "all_events",
                model_group="heterogeneity",
            )
        )
    return specs


def estimate_main_models(
    event_panel: pl.DataFrame,
    config: WhoGetsConsideredConfig,
    *,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Estimate the paper's reduced-form regression suite and return metadata."""

    specs = _build_specs()
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for spec in specs:
        result_rows = _fit_ols(event_panel, spec, config, logger=logger)
        if not result_rows:
            skipped.append({"spec_id": spec.spec_id, "dep_var": spec.dep_var})
        rows.extend(result_rows)
    diagnostics = {
        "outsider_share": float(event_panel["outsider_flag"].mean()) if event_panel.height else None,
        "mean_external_hire_distance": float(
            event_panel.filter(pl.col("outsider_flag") == 1)["distance_miles"].mean() or 0.0
        )
        if event_panel.height
        else None,
        "mean_internal_hire_distance": float(
            event_panel.filter(pl.col("outsider_flag") == 0)["distance_miles"].mean() or 0.0
        )
        if event_panel.height
        else None,
        "release_count_market_size_correlation": float(
            event_panel.select(
                pl.corr("release_count_730d_60mi_outind", "market_size_60mi").alias("corr")
            )["corr"].item()
            or 0.0
        )
        if event_panel.height and "market_size_60mi" in event_panel.columns
        else None,
        "skipped_specs": skipped,
    }
    return pl.DataFrame(rows), diagnostics
