"""Two-stage least squares utilities for realized-fit mediation designs."""

from __future__ import annotations

import polars as pl
from linearmodels.iv import IV2SLS
import statsmodels.api as sm


def estimate_fit_iv(
    event_panel: pl.DataFrame,
    *,
    outcome: str = "delta_roa_resid_h1",
    endog: str = "realized_task_fit_z",
    instrument: str = "release_count_730d_60mi_outind",
) -> pl.DataFrame:
    """Estimate a simple 2SLS mediation specification for realized fit."""

    required = [outcome, endog, instrument, "max_released_task_fit_z", "bench_index_z"]
    if any(column not in event_panel.columns for column in required):
        return pl.DataFrame(
            {
                "model_group": ["iv"],
                "model_name": ["iv_unavailable"],
                "outcome": [outcome],
                "term": ["unavailable"],
                "coef": [None],
                "se": [None],
                "tstat": [None],
                "pvalue": [None],
                "ci_low": [None],
                "ci_high": [None],
                "nobs": [0],
                "first_stage_f": [None],
                "spec": ["missing_required_columns"],
            }
        )
    sample = event_panel.drop_nulls(required)
    if sample.height < 4:
        return pl.DataFrame(
            {
                "model_group": ["iv"],
                "model_name": ["iv_unavailable"],
                "outcome": [outcome],
                "term": ["insufficient_sample"],
                "coef": [None],
                "se": [None],
                "tstat": [None],
                "pvalue": [None],
                "ci_low": [None],
                "ci_high": [None],
                "nobs": [sample.height],
                "first_stage_f": [None],
                "spec": ["insufficient_sample"],
            }
        )
    pdf = sample.to_pandas()
    formula = (
        f"{outcome} ~ 1 + max_released_task_fit_z + bench_index_z "
        f"+ [{endog} ~ {instrument}]"
    )
    try:
        result = IV2SLS.from_formula(formula, data=pdf).fit(cov_type="robust")
        spec = "full"
    except ValueError:
        try:
            result = IV2SLS.from_formula(
                f"{outcome} ~ 1 + max_released_task_fit_z + [{endog} ~ {instrument}]",
                data=pdf,
            ).fit(cov_type="robust")
            spec = "fallback_minimal"
        except ValueError:
            first = sm.OLS(pdf[endog], sm.add_constant(pdf[[instrument]])).fit()
            fitted = first.fittedvalues
            second = sm.OLS(pdf[outcome], sm.add_constant(pl.DataFrame({"fit_hat": fitted}).to_pandas())).fit(cov_type="HC1")
            conf = second.conf_int()
            rows: list[dict[str, object]] = []
            for term in second.params.index:
                rows.append(
                    {
                        "model_group": "iv",
                        "model_name": f"iv_{outcome}_{endog}",
                        "outcome": outcome,
                        "term": term,
                        "coef": float(second.params[term]),
                        "se": float(second.bse[term]),
                        "tstat": float(second.tvalues[term]),
                        "pvalue": float(second.pvalues[term]),
                        "ci_low": float(conf.loc[term, 0]),
                        "ci_high": float(conf.loc[term, 1]),
                        "nobs": int(second.nobs),
                        "first_stage_f": float(first.fvalue if first.fvalue is not None else 0.0),
                        "spec": "manual_fallback",
                    }
                )
            return pl.DataFrame(rows)
    diagnostics = getattr(result.first_stage, "diagnostics", None)
    if diagnostics is not None and not diagnostics.empty:
        first_stage_f = float(diagnostics.iloc[0].get("f.stat", 0.0))
    else:
        first_stage_f = 0.0
    rows: list[dict[str, object]] = []
    conf = result.conf_int()
    for term in result.params.index:
        rows.append(
            {
                "model_group": "iv",
                "model_name": f"iv_{outcome}_{endog}",
                "outcome": outcome,
                "term": term,
                "coef": float(result.params[term]),
                "se": float(result.std_errors[term]),
                "tstat": float(result.tstats[term]),
                "pvalue": float(result.pvalues[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "nobs": int(result.nobs),
                "first_stage_f": first_stage_f,
                "spec": spec,
            }
        )
    return pl.DataFrame(rows)
