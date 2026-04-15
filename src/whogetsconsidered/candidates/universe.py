"""CEO-ready candidate-universe construction by year."""

from __future__ import annotations

import polars as pl


def _safe_mean(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def build_candidate_universe(
    executive_year_panel: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
) -> pl.DataFrame:
    """Build the candidate universe with cumulative pre-event portable-quality features."""

    enriched = executive_year_panel.join(
        firm_year_panel.select(
            "gvkey",
            "fyear",
            "rd_intensity",
            "leverage",
            "capital_intensity",
            "state",
            "msa_code",
            "lat",
            "lon",
        ),
        on=["gvkey", "fyear"],
        how="left",
    ).filter(pl.col("is_ceo_ready_robust"))

    rows: list[dict[str, object]] = []
    for _, person_df in enriched.sort(["person_id", "fyear"]).group_by("person_id", maintain_order=True):
        history: list[dict[str, object]] = []
        for row in person_df.to_dicts():
            history.append(row)
            rows.append(
                {
                    "candidate_person_id": row["person_id"],
                    "candidate_year": row["fyear"],
                    "current_gvkey": row["gvkey"],
                    "current_title_raw": row["title_raw"],
                    "current_title_seniority_score": row["title_seniority_score"],
                    "current_state": row["state"],
                    "current_msa_code": row["msa_code"],
                    "current_lat": row["lat"],
                    "current_lon": row["lon"],
                    "is_ceo": int(row["is_ceo"]),
                    "is_president": int(row["is_president"]),
                    "is_coo": int(row["is_coo"]),
                    "is_cfo": int(row["is_cfo"]),
                    "prior_public_ceo_flag": int(any(bool(h["is_ceo"]) for h in history)),
                    "years_as_public_ceo": sum(int(bool(h["is_ceo"])) for h in history),
                    "num_prior_public_firms": len({str(h["gvkey"]) for h in history}),
                    "num_prior_industries": len({str(h["ff49"]) for h in history if h["ff49"] is not None}),
                    "mover_flag": int(len({str(h["gvkey"]) for h in history}) > 1),
                    "avg_prior_firm_log_assets": _safe_mean([h["log_assets"] for h in history]),
                    "avg_prior_firm_roa": _safe_mean([h["roa_raw"] for h in history]),
                    "avg_prior_firm_tobin_q": _safe_mean([h["tobin_q_raw"] for h in history]),
                    "avg_prior_firm_rd_intensity": _safe_mean([h["rd_intensity"] for h in history]),
                    "avg_prior_firm_leverage": _safe_mean([h["leverage"] for h in history]),
                    "avg_prior_firm_capital_intensity": _safe_mean(
                        [h["capital_intensity"] for h in history]
                    ),
                    "public_market_tenure_years": len(history),
                }
            )
    universe = pl.DataFrame(rows)
    score_cols = [
        "prior_public_ceo_flag",
        "years_as_public_ceo",
        "avg_prior_firm_log_assets",
        "avg_prior_firm_roa",
        "avg_prior_firm_tobin_q",
        "mover_flag",
    ]
    universe = universe.with_columns(
        [
            (
                (pl.col(column) - pl.col(column).mean()) / pl.col(column).std(ddof=0)
            ).fill_nan(0.0).fill_null(0.0).alias(f"{column}_z")
            for column in score_cols
        ]
    ).with_columns(
        (pl.sum_horizontal([pl.col(f"{column}_z") for column in score_cols]) / len(score_cols)).alias(
            "portable_quality_score"
        )
    )
    return universe
