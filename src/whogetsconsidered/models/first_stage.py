"""First-stage validation models for release-driven local executive availability."""

from __future__ import annotations

from collections import defaultdict

import polars as pl
import statsmodels.formula.api as smf

from whogetsconsidered.config import MarketConfig
from whogetsconsidered.geography.distance import haversine_km


def _is_local(
    source: dict[str, object],
    target: dict[str, object],
    market_config: MarketConfig,
) -> bool:
    if None not in (source["lat"], source["lon"], target["lat"], target["lon"]):
        distance_miles = haversine_km(
            float(source["lat"]),
            float(source["lon"]),
            float(target["lat"]),
            float(target["lon"]),
        ) / 1.609344
        if distance_miles <= 60.0:
            return True
    if market_config.definition == "msa" and source["msa_code"] is not None and source["msa_code"] == target["msa_code"]:
        return True
    return False


def build_reemployment_panel(
    released_candidates: pl.DataFrame,
    candidate_universe: pl.DataFrame,
    executive_year_panel: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
    market_config: MarketConfig,
) -> pl.DataFrame:
    """Build person-level and market-level validation samples."""

    history = executive_year_panel.join(
        firm_year_panel.select(
            "gvkey",
            "fyear",
            "ff10",
            "ff49",
            "log_assets",
            "roa_raw",
            "q_raw",
            "msa_code",
            "state",
            "lat",
            "lon",
        ),
        on=["gvkey", "fyear"],
        how="left",
    )
    history_by_person: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in history.to_dicts():
        history_by_person[str(row["person_id"])].append(row)
    universe_lookup = {
        (str(row["candidate_person_id"]), int(row["candidate_year"]), str(row["current_gvkey"])): row
        for row in candidate_universe.to_dicts()
    }

    released_keys = {
        (str(row["candidate_person_id"]), int(row["candidate_year"]), str(row["source_gvkey"]))
        for row in released_candidates.to_dicts()
    }
    rows: list[dict[str, object]] = []
    for row in released_candidates.to_dicts():
        universe_row = universe_lookup.get((str(row["candidate_person_id"]), int(row["candidate_year"]), str(row["source_gvkey"])))
        if universe_row is None:
            continue
        outcomes = {
            "reemployed_local_24m": 0,
            "reemployed_anywhere_24m": 0,
            "reemployed_ceo_ready_24m": 0,
        }
        source_loc = {
            "msa_code": row["source_msa_code"],
            "state": row["source_state"],
            "lat": row["source_hq_lat"],
            "lon": row["source_hq_lon"],
        }
        for obs in history_by_person[str(row["candidate_person_id"])]:
            if (
                int(obs["fyear"]) > int(row["release_event_year"])
                and int(obs["fyear"]) <= int(row["release_event_year"]) + 2
                and obs["gvkey"] != row["source_gvkey"]
            ):
                outcomes["reemployed_anywhere_24m"] = 1
                if bool(obs["is_ceo_ready"]):
                    outcomes["reemployed_ceo_ready_24m"] = 1
                if _is_local(source_loc, obs, market_config):
                    outcomes["reemployed_local_24m"] = 1
        rows.append(
            {
                "candidate_person_id": row["candidate_person_id"],
                "released_candidate_flag": 1,
                "release_year": row["release_event_year"],
                "source_market": row["source_msa_code"] or row["source_state"],
                "source_title": row["source_title"],
                "source_ff10": row["source_ff10"],
                "prior_public_ceo_flag": row["prior_public_ceo_flag"],
                "source_log_assets": universe_row["avg_prior_firm_log_assets"],
                "source_roa": universe_row["avg_prior_firm_roa"],
                "source_q": universe_row["avg_prior_firm_tobin_q"],
                **outcomes,
            }
        )
    for row in candidate_universe.to_dicts():
        key = (str(row["candidate_person_id"]), int(row["candidate_year"]), str(row["current_gvkey"]))
        if key in released_keys:
            continue
        person_history = history_by_person[str(row["candidate_person_id"])]
        source_obs = next(
            (
                obs
                for obs in person_history
                if int(obs["fyear"]) == int(row["candidate_year"]) and obs["gvkey"] == row["current_gvkey"]
            ),
            None,
        )
        if source_obs is None:
            continue
        outcomes = {
            "reemployed_local_24m": 0,
            "reemployed_anywhere_24m": 0,
            "reemployed_ceo_ready_24m": 0,
        }
        for obs in person_history:
            if (
                int(obs["fyear"]) > int(row["candidate_year"])
                and int(obs["fyear"]) <= int(row["candidate_year"]) + 2
                and obs["gvkey"] != row["current_gvkey"]
            ):
                outcomes["reemployed_anywhere_24m"] = 1
                if bool(obs["is_ceo_ready"]):
                    outcomes["reemployed_ceo_ready_24m"] = 1
                if _is_local(source_obs, obs, market_config):
                    outcomes["reemployed_local_24m"] = 1
        rows.append(
            {
                "candidate_person_id": row["candidate_person_id"],
                "released_candidate_flag": 0,
                "release_year": row["candidate_year"],
                "source_market": source_obs["msa_code"] or source_obs["state"],
                "source_title": row["current_title_raw"],
                "source_ff10": source_obs["ff10"],
                "prior_public_ceo_flag": row["prior_public_ceo_flag"],
                "source_log_assets": row["avg_prior_firm_log_assets"],
                "source_roa": row["avg_prior_firm_roa"],
                "source_q": row["avg_prior_firm_tobin_q"],
                **outcomes,
            }
        )
    return pl.DataFrame(rows)


def _fit_validation_model(panel: pl.DataFrame, outcome: str, model_name: str) -> list[dict[str, object]]:
    if outcome not in panel.columns or panel.drop_nulls([outcome]).height < 4:
        return []
    pdf = panel.to_pandas()
    formula = (
        f"{outcome} ~ released_candidate_flag + prior_public_ceo_flag + source_log_assets + source_roa + source_q "
        "+ C(source_market) + C(release_year) + C(source_title) + C(source_ff10)"
    )
    fit = smf.ols(formula, data=pdf).fit(cov_type="HC1")
    conf = fit.conf_int()
    rows: list[dict[str, object]] = []
    for term in fit.params.index:
        rows.append(
            {
                "model_group": "validation",
                "dep_var": outcome,
                "spec_id": model_name,
                "term": term,
                "coefficient": float(fit.params[term]),
                "std_error": float(fit.bse[term]),
                "t_stat": float(fit.tvalues[term]),
                "p_value": float(fit.pvalues[term]),
                "ci_lower": float(conf.loc[term, 0]),
                "ci_upper": float(conf.loc[term, 1]),
                "N": int(fit.nobs),
                "num_clusters": 0,
                "cluster_level": "HC1",
                "fe_description": "source_market, release_year, source_title, source_ff10",
                "control_set_description": "validation_controls",
                "sample_filter_description": "released and matched nonreleased CEO-ready executives",
                "mean_dep_var": float(panel[outcome].mean()),
            }
        )
    return rows


def estimate_reemployment_model(reemployment_panel: pl.DataFrame) -> pl.DataFrame:
    """Estimate the first-stage local reemployment validation models."""

    rows: list[dict[str, object]] = []
    for outcome in [
        "reemployed_local_24m",
        "reemployed_anywhere_24m",
        "reemployed_ceo_ready_24m",
    ]:
        rows.extend(_fit_validation_model(reemployment_panel, outcome, "T3_person_level"))
    if reemployment_panel.height > 0:
        market_panel = reemployment_panel.group_by(["source_market", "release_year"], maintain_order=True).agg(
            pl.sum("reemployed_ceo_ready_24m").alias("local_ceo_ready_entries_m_t_to_tplus2"),
            pl.sum("released_candidate_flag").alias("num_released_candidates_m_t"),
        )
        if market_panel.height >= 4:
            pdf = market_panel.to_pandas()
            fit = smf.ols(
                "local_ceo_ready_entries_m_t_to_tplus2 ~ num_released_candidates_m_t + C(source_market) + C(release_year)",
                data=pdf,
            ).fit(cov_type="HC1")
            conf = fit.conf_int()
            for term in fit.params.index:
                rows.append(
                    {
                        "model_group": "validation",
                        "dep_var": "local_ceo_ready_entries_m_t_to_tplus2",
                        "spec_id": "T3_market_level",
                        "term": term,
                        "coefficient": float(fit.params[term]),
                        "std_error": float(fit.bse[term]),
                        "t_stat": float(fit.tvalues[term]),
                        "p_value": float(fit.pvalues[term]),
                        "ci_lower": float(conf.loc[term, 0]),
                        "ci_upper": float(conf.loc[term, 1]),
                        "N": int(fit.nobs),
                        "num_clusters": 0,
                        "cluster_level": "HC1",
                        "fe_description": "source_market, release_year",
                        "control_set_description": "market_validation_controls",
                        "sample_filter_description": "market-year aggregation",
                        "mean_dep_var": float(market_panel["local_ceo_ready_entries_m_t_to_tplus2"].mean()),
                    }
                )
    return pl.DataFrame(rows)
