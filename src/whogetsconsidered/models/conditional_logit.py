"""Event-level candidate-choice models over accessible successor sets."""

from __future__ import annotations

import polars as pl
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


def estimate_choice_model(accessible_candidate_set: pl.DataFrame) -> pl.DataFrame:
    """Estimate a conditional-logit style model over event-level candidate sets."""

    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("statsmodels conditional logit is unavailable") from exc

    cols = [
        "portable_quality_z_i",
        "task_fit_z_if",
        "internal_candidate_flag",
        "released_candidate_flag",
        "distance_miles_if",
        "known_to_board_flag",
    ]
    sample = accessible_candidate_set.select("chosen_flag", "event_id", *cols).fill_null(0.0)
    pdf = sample.to_pandas()
    scaler = StandardScaler()
    pdf[cols] = scaler.fit_transform(pdf[cols])
    rows: list[dict[str, object]] = []
    try:
        model = ConditionalLogit(pdf["chosen_flag"], pdf[cols], groups=pdf["event_id"])
        result = model.fit(disp=False)
        conf = result.conf_int()
        for idx, term in enumerate(cols):
            rows.append(
                {
                    "model_group": "choice",
                    "model_name": "conditional_logit",
                    "outcome": "chosen_flag",
                    "term": term,
                    "coef": float(result.params[idx]),
                    "se": float(result.bse[idx]),
                    "tstat": float(result.tvalues[idx]),
                    "pvalue": float(result.pvalues[idx]),
                    "ci_low": float(conf[idx][0]),
                    "ci_high": float(conf[idx][1]),
                    "nobs": int(result.nobs),
                }
            )
    except Exception:
        formula = (
            "chosen_flag ~ portable_quality_z_i + task_fit_z_if + internal_candidate_flag + "
            "released_candidate_flag + distance_miles_if + known_to_board_flag + C(event_id)"
        )
        result = smf.ols(formula, data=pdf).fit(cov_type="HC1")
        conf = result.conf_int()
        for term in result.params.index:
            rows.append(
                {
                    "model_group": "choice",
                    "model_name": "choice_fallback_ols",
                    "outcome": "chosen_flag",
                    "term": term,
                    "coef": float(result.params[term]),
                    "se": float(result.bse[term]),
                    "tstat": float(result.tvalues[term]),
                    "pvalue": float(result.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "nobs": int(result.nobs),
                }
            )
    return pl.DataFrame(rows)
