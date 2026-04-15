"""Event-study and announcement-CAR utilities."""

from __future__ import annotations

import polars as pl
import statsmodels.api as sm


def build_simple_event_study_series(
    succession_events: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
    *,
    outcome: str,
    window: range = range(-2, 4),
) -> pl.DataFrame:
    """Construct a simple relative-time average series around succession events."""

    rows: list[dict[str, object]] = []
    for event in succession_events.to_dicts():
        for rel in window:
            year = int(event["succession_year"]) + rel
            match = firm_year_panel.filter((pl.col("gvkey") == event["gvkey"]) & (pl.col("fyear") == year))
            if match.height == 0:
                continue
            rows.append({"event_id": event["event_id"], "relative_year": rel, outcome: match[outcome].item()})
    if not rows:
        return pl.DataFrame(schema={"relative_year": pl.Int64, outcome: pl.Float64})
    return pl.DataFrame(rows).group_by("relative_year", maintain_order=True).agg(pl.mean(outcome).alias(outcome))


def compute_announcement_cars(
    succession_events: pl.DataFrame,
    crsp_daily: pl.DataFrame,
) -> pl.DataFrame:
    """Compute FF3 announcement-window CARs for CEO succession events when daily data are available."""

    if crsp_daily.height == 0 or succession_events.height == 0:
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "car_m1_p1": pl.Float64,
                "car_m2_p2": pl.Float64,
                "car_estimation_obs": pl.Int64,
                "car_sample_flag": pl.Int8,
            }
        )
    daily_lookup: dict[str, pl.DataFrame] = {}
    for key, frame in crsp_daily.group_by("gvkey", maintain_order=True):
        gvkey = key[0] if isinstance(key, tuple) else key
        daily_lookup[str(gvkey)] = frame.sort("date")
    rows: list[dict[str, object]] = []
    for event in succession_events.to_dicts():
        if event.get("announcement_date") is None:
            rows.append(
                {
                    "event_id": event["event_id"],
                    "car_m1_p1": None,
                    "car_m2_p2": None,
                    "car_estimation_obs": 0,
                    "car_sample_flag": 0,
                }
            )
            continue
        daily = daily_lookup.get(str(event["gvkey"]))
        if daily is None:
            rows.append(
                {
                    "event_id": event["event_id"],
                    "car_m1_p1": None,
                    "car_m2_p2": None,
                    "car_estimation_obs": 0,
                    "car_sample_flag": 0,
                }
            )
            continue
        event_date = event["announcement_date"]
        estimation = daily.with_columns((pl.col("date") - pl.lit(event_date)).dt.total_days().alias("rel_day")).filter(
            (pl.col("rel_day") >= -255) & (pl.col("rel_day") <= -46)
        )
        if estimation.height < 120:
            rows.append(
                {
                    "event_id": event["event_id"],
                    "car_m1_p1": None,
                    "car_m2_p2": None,
                    "car_estimation_obs": estimation.height,
                    "car_sample_flag": 0,
                }
            )
            continue
        estimation_pdf = estimation.select(
            (pl.col("ret") - pl.col("rf")).alias("excess_ret"),
            "mktrf",
            "smb",
            "hml",
        ).to_pandas()
        fit = sm.OLS(
            estimation_pdf["excess_ret"],
            sm.add_constant(estimation_pdf[["mktrf", "smb", "hml"]]),
        ).fit()
        event_window = daily.with_columns((pl.col("date") - pl.lit(event_date)).dt.total_days().alias("rel_day")).filter(
            pl.col("rel_day").is_in([-2, -1, 0, 1, 2])
        )
        if event_window.height == 0:
            rows.append(
                {
                    "event_id": event["event_id"],
                    "car_m1_p1": None,
                    "car_m2_p2": None,
                    "car_estimation_obs": estimation.height,
                    "car_sample_flag": 0,
                }
            )
            continue
        event_pdf = event_window.select(
            "rel_day",
            (pl.col("ret") - pl.col("rf")).alias("excess_ret"),
            "mktrf",
            "smb",
            "hml",
        ).to_pandas()
        predicted = fit.predict(sm.add_constant(event_pdf[["mktrf", "smb", "hml"]], has_constant="add"))
        event_pdf["abnormal"] = event_pdf["excess_ret"] - predicted
        car_m1_p1 = float(event_pdf.loc[event_pdf["rel_day"].between(-1, 1), "abnormal"].sum())
        car_m2_p2 = float(event_pdf["abnormal"].sum())
        rows.append(
            {
                "event_id": event["event_id"],
                "car_m1_p1": car_m1_p1,
                "car_m2_p2": car_m2_p2,
                "car_estimation_obs": estimation.height,
                "car_sample_flag": 1,
            }
        )
    return pl.DataFrame(rows)
