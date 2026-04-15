"""Cross-fitted predictive fit models estimated on held-out succession events."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.fit.crossfit import assign_event_folds


Z_COLS = [
    "prior_public_ceo_flag",
    "years_as_public_ceo",
    "num_prior_public_firms",
    "num_prior_industries",
    "mover_flag",
    "avg_prior_firm_log_assets",
    "avg_prior_firm_roa",
    "avg_prior_firm_tobin_q",
    "avg_prior_firm_rd_intensity",
    "avg_prior_firm_leverage",
    "avg_prior_firm_capital_intensity",
    "same_industry_experience_share",
    "same_state_experience_share",
    "same_msa_experience_share",
    "current_title_seniority_score",
    "public_market_tenure_years",
]

X_COLS = [
    "log_assets",
    "roa_raw",
    "tobin_q_raw",
    "rd_intensity",
    "capital_intensity",
    "leverage",
    "dividend_payer",
    "dividend_yield",
    "firm_age",
    "pre_succession_performance_trend",
    "same_industry_local_density",
    "noncompete_score",
    "market_size_public_firms",
    "num_ceo_ready_insiders_tminus1",
]


def _interaction_matrix(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate([z[:, i : i + 1] * x for i in range(z.shape[1])], axis=1)


def add_predictive_fit_scores(
    accessible_candidate_set: pl.DataFrame,
    event_panel: pl.DataFrame,
    config: WhoGetsConsideredConfig,
) -> pl.DataFrame:
    """Add cross-fitted predictive fit scores to the accessible candidate set."""

    target_frame = event_panel.with_columns(
        (
            (
                (pl.col("Delta_roa_resid_h1") - pl.col("Delta_roa_resid_h1").mean())
                / pl.col("Delta_roa_resid_h1").std(ddof=0)
            ).fill_nan(0.0).fill_null(0.0)
            + (
                (pl.col("Delta_tobin_q_resid_h1") - pl.col("Delta_tobin_q_resid_h1").mean())
                / pl.col("Delta_tobin_q_resid_h1").std(ddof=0)
            ).fill_nan(0.0).fill_null(0.0)
        ).alias("_predictive_target_sum")
    ).with_columns((pl.col("_predictive_target_sum") / 2).alias("predictive_target"))
    folds = assign_event_folds(
        target_frame,
        n_folds=config.regression.crossfit_folds,
        time_ordered=config.regression.use_time_ordered_folds,
        random_seed=config.regression.random_seed,
    )

    chosen = accessible_candidate_set.filter(pl.col("chosen_flag") == 1).join(
        target_frame.select("event_id", "predictive_target", *X_COLS),
        on="event_id",
        how="left",
    )
    all_rows = accessible_candidate_set.join(target_frame.select("event_id", *X_COLS), on="event_id", how="left")

    predictions: list[dict[str, object]] = []
    for fold in sorted(set(folds.values())):
        train_ids = [event_id for event_id, fold_id in folds.items() if fold_id != fold]
        score_ids = [event_id for event_id, fold_id in folds.items() if fold_id == fold]
        train = chosen.filter(pl.col("event_id").is_in(train_ids))
        score = all_rows.filter(pl.col("event_id").is_in(score_ids))
        if train.height < 2 or score.height == 0:
            continue
        z_train = train.select(Z_COLS).fill_null(0.0).to_numpy()
        x_train = train.select(X_COLS).fill_null(0.0).to_numpy()
        y_train = train["predictive_target"].fill_null(0.0).to_numpy()

        z_scaler = StandardScaler().fit(z_train)
        x_scaler = StandardScaler().fit(x_train)
        z_train_scaled = z_scaler.transform(z_train)
        x_train_scaled = x_scaler.transform(x_train)

        model_z = ElasticNetCV(cv=min(3, train.height), random_state=config.regression.random_seed)
        model_z.fit(z_train_scaled, y_train)
        resid_after_z = y_train - model_z.predict(z_train_scaled)

        model_x = ElasticNetCV(cv=min(3, train.height), random_state=config.regression.random_seed)
        model_x.fit(x_train_scaled, resid_after_z)
        resid_after_x = resid_after_z - model_x.predict(x_train_scaled)

        inter_train = _interaction_matrix(z_train_scaled, x_train_scaled)
        model_inter = ElasticNetCV(cv=min(3, train.height), random_state=config.regression.random_seed)
        model_inter.fit(inter_train, resid_after_x)

        z_score = score.select(Z_COLS).fill_null(0.0).to_numpy()
        x_score = score.select(X_COLS).fill_null(0.0).to_numpy()
        z_score_scaled = z_scaler.transform(z_score)
        x_score_scaled = x_scaler.transform(x_score)
        inter_score = _interaction_matrix(z_score_scaled, x_score_scaled)

        portable = model_z.predict(z_score_scaled)
        predictive = model_inter.predict(inter_score)
        for idx, row in enumerate(score.select("event_id", "candidate_person_id").to_dicts()):
            predictions.append(
                {
                    "event_id": row["event_id"],
                    "candidate_person_id": row["candidate_person_id"],
                    "portable_quality_component": float(portable[idx]),
                    "predictive_fit_score": float(predictive[idx]),
                }
            )

    if not predictions:
        return accessible_candidate_set.with_columns(
            pl.col("portable_quality_score").alias("portable_quality_component"),
            pl.lit(0.0).alias("predictive_fit_score"),
        )
    prediction_df = pl.DataFrame(predictions)
    return accessible_candidate_set.join(
        prediction_df,
        on=["event_id", "candidate_person_id"],
        how="left",
    ).with_columns(
        pl.coalesce("portable_quality_component", "portable_quality_score").alias(
            "portable_quality_component"
        ),
        pl.coalesce("predictive_fit_score", pl.lit(0.0)).alias("predictive_fit_score"),
    )
