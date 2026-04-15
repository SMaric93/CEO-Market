"""Transparent structured task-alignment fit used as the paper's main fit outcome."""

from __future__ import annotations

from collections import defaultdict
import math

import polars as pl


def _zscore(value: float | None, mean: float, std: float) -> float:
    if value is None or std == 0 or math.isnan(std):
        return 0.0
    return float((value - mean) / std)


def _build_standardizers(firm_year_panel: pl.DataFrame) -> dict[str, tuple[float, float]]:
    features = {
        "rd_intensity": firm_year_panel["rd_intensity"].fill_null(0.0),
        "neg_roa": (-firm_year_panel["roa_raw"].fill_null(0.0)),
        "leverage": firm_year_panel["leverage"].fill_null(0.0),
        "log_assets": firm_year_panel["log_assets"].fill_null(0.0),
        "capital_intensity": firm_year_panel["capital_intensity"].fill_null(0.0),
        "q_raw": firm_year_panel["q_raw"].fill_null(0.0),
    }
    return {
        name: (float(series.mean()), float(series.std(ddof=0) or 0.0))
        for name, series in features.items()
    }


def score_task_alignment(
    accessible_candidate_set: pl.DataFrame,
    succession_events: pl.DataFrame,
    executive_year_panel: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the paper's structured five-dimension task-fit score for each event-candidate pair."""

    standards = _build_standardizers(firm_year_panel)
    firm_lookup = {
        (str(row["gvkey"]), int(row["fyear"])): row
        for row in firm_year_panel.select(
            "gvkey",
            "fyear",
            "ff49",
            "rd_intensity",
            "roa_raw",
            "leverage",
            "log_assets",
            "capital_intensity",
            "q_raw",
        ).to_dicts()
    }
    event_need_lookup: dict[str, dict[str, float | str]] = {}
    for event in succession_events.to_dicts():
        need_row = firm_lookup.get((str(event["gvkey"]), int(event["succession_year"]) - 1))
        if need_row is None:
            continue
        n1 = _zscore(need_row["rd_intensity"], *standards["rd_intensity"])
        n2 = 0.5 * (
            _zscore(-float(need_row["roa_raw"] or 0.0), *standards["neg_roa"])
            + _zscore(need_row["leverage"], *standards["leverage"])
        )
        n3 = _zscore(need_row["log_assets"], *standards["log_assets"])
        n4 = _zscore(need_row["capital_intensity"], *standards["capital_intensity"])
        n5 = _zscore(need_row["q_raw"], *standards["q_raw"])
        event_need_lookup[str(event["event_id"])] = {
            "focal_ff49": str(need_row["ff49"]) if need_row["ff49"] is not None else "",
            "need_1": n1,
            "need_2": n2,
            "need_3": n3,
            "need_4": n4,
            "need_5": n5,
        }

    history_panel = executive_year_panel.join(
        firm_year_panel.select(
            "gvkey",
            "fyear",
            "ff49",
            "rd_intensity",
            "roa_raw",
            "leverage",
            "log_assets",
            "capital_intensity",
            "q_raw",
        ),
        on=["gvkey", "fyear"],
        how="left",
    )
    history_by_person: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in history_panel.sort(["person_id", "fyear"]).to_dicts():
        history_by_person[str(row["person_id"])].append(row)

    rows: list[dict[str, object]] = []
    for row in accessible_candidate_set.to_dicts():
        event_id = str(row["event_id"])
        event_need = event_need_lookup[event_id]
        history = [
            item
            for item in history_by_person.get(str(row["candidate_person_id"]), [])
            if int(item["fyear"]) <= int(row["candidate_year"]) and bool(item["is_ceo_ready_robust"])
        ]
        total_years = max(len(history), 1)
        weight = 1.0 / total_years
        e1 = sum(_zscore(item["rd_intensity"], *standards["rd_intensity"]) * weight for item in history)
        e2 = sum(
            0.5
            * (
                _zscore(-float(item["roa_raw"] or 0.0), *standards["neg_roa"])
                + _zscore(item["leverage"], *standards["leverage"])
            )
            * weight
            for item in history
        )
        e3 = sum(_zscore(item["log_assets"], *standards["log_assets"]) * weight for item in history)
        e4 = sum(_zscore(item["capital_intensity"], *standards["capital_intensity"]) * weight for item in history)
        e5 = sum(_zscore(item["q_raw"], *standards["q_raw"]) * weight for item in history)
        industry_fit = sum(str(item["ff49"]) == event_need["focal_ff49"] for item in history) / total_years
        d1 = (float(event_need["need_1"]) - e1) ** 2
        d2 = (float(event_need["need_2"]) - e2) ** 2
        d3 = (float(event_need["need_3"]) - e3) ** 2
        d4 = (float(event_need["need_4"]) - e4) ** 2
        d5 = (float(event_need["need_5"]) - e5) ** 2
        task_fit_raw = -(d1 + d2 + d3 + d4 + d5) / 5.0 + industry_fit
        rows.append(
            {
                **row,
                "need_innovation": float(event_need["need_1"]),
                "need_turnaround": float(event_need["need_2"]),
                "need_scale": float(event_need["need_3"]),
                "need_operations": float(event_need["need_4"]),
                "need_growth": float(event_need["need_5"]),
                "exp_innovation": e1,
                "exp_turnaround": e2,
                "exp_scale": e3,
                "exp_operations": e4,
                "exp_growth": e5,
                "industry_fit_if": industry_fit,
                "task_fit_component_1": -d1 / 5.0,
                "task_fit_component_2": -d2 / 5.0,
                "task_fit_component_3": -d3 / 5.0,
                "task_fit_component_4": -d4 / 5.0,
                "task_fit_component_5": -d5 / 5.0,
                "task_fit_raw_if": task_fit_raw,
            }
        )
    scored = pl.DataFrame(rows)
    mean_fit = float(scored["task_fit_raw_if"].mean()) if scored.height else 0.0
    std_fit = float(scored["task_fit_raw_if"].std(ddof=0) or 0.0) if scored.height else 0.0
    return scored.with_columns(
        pl.when(std_fit == 0.0)
        .then(pl.lit(0.0))
        .otherwise((pl.col("task_fit_raw_if") - mean_fit) / std_fit)
        .alias("task_fit_z_if"),
        pl.when(std_fit == 0.0)
        .then(pl.lit(0.0))
        .otherwise((pl.col("task_fit_raw_if") - mean_fit) / std_fit)
        .alias("task_alignment_fit_score"),
        pl.lit(None, dtype=pl.Float64).alias("predictive_fit_score"),
    )
