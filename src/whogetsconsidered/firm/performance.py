"""Firm-performance metrics and residualization utilities."""

from __future__ import annotations

import polars as pl

from whogetsconsidered.config import ResidualizationConfig


def compute_raw_outcomes(firm_year_panel: pl.DataFrame) -> pl.DataFrame:
    """Construct raw valuation and profitability outcomes plus laggable controls."""

    return (
        firm_year_panel.sort(["gvkey", "fyear"])
        .with_columns(
            (pl.col("prcc_f") * pl.col("csho")).alias("market_value_equity"),
            ((pl.col("prcc_f") * pl.col("csho")) + pl.col("dltt").fill_null(0) + pl.col("dlc").fill_null(0))
            .truediv(pl.col("at"))
            .alias("tobin_q_raw"),
            ((pl.col("prcc_f") * pl.col("csho")) + pl.col("dltt").fill_null(0) + pl.col("dlc").fill_null(0))
            .truediv(pl.col("at"))
            .alias("q_raw"),
            pl.col("ebit").truediv(pl.col("at")).alias("roa_raw"),
            pl.col("at").log().alias("log_assets"),
            pl.col("xrd").fill_null(0).truediv(pl.col("at")).alias("rd_intensity"),
            (pl.col("xrd").fill_null(0) > 0).cast(pl.Int8).alias("rd_indicator"),
            pl.col("capx").fill_null(0).truediv(pl.col("at")).alias("capital_intensity"),
            (pl.col("dltt").fill_null(0) + pl.col("dlc").fill_null(0))
            .truediv(pl.col("at"))
            .alias("leverage"),
            (pl.col("dv").fill_null(0) > 0).cast(pl.Int8).alias("dividend_payer"),
            pl.when(pl.col("ceq").fill_null(0) > 0)
            .then(pl.col("dv").fill_null(0).truediv(pl.col("ceq")))
            .otherwise(pl.col("dv").fill_null(0).truediv(pl.col("at")))
            .alias("dividend_yield"),
        )
        .with_columns(
            (pl.col("fyear") - pl.col("fyear").min().over("gvkey") + 1).alias("firm_age"),
            (pl.col("roa_raw") - pl.col("roa_raw").shift(2).over("gvkey")).alias(
                "pre_succession_performance_trend"
            ),
            pl.col("roa_raw")
            .rolling_std(window_size=3, min_samples=2)
            .over("gvkey")
            .alias("performance_volatility_3y"),
        )
    )


def add_size_quartiles(firm_year_panel: pl.DataFrame) -> pl.DataFrame:
    """Add within-year size quartiles for residualization."""

    rows: list[pl.DataFrame] = []
    for _, year_df in firm_year_panel.group_by("fyear", maintain_order=True):
        year_ranked = year_df.with_columns(
            pl.col("log_assets")
            .rank(method="ordinal")
            .truediv(pl.len())
            .alias("_size_rank")
        ).with_columns(
            pl.when(pl.col("_size_rank") <= 0.25)
            .then(pl.lit(1))
            .when(pl.col("_size_rank") <= 0.50)
            .then(pl.lit(2))
            .when(pl.col("_size_rank") <= 0.75)
            .then(pl.lit(3))
            .otherwise(pl.lit(4))
            .alias("size_quartile"),
        ).drop("_size_rank")
        rows.append(year_ranked)
    return pl.concat(rows, how="vertical")


def _demean(df: pl.DataFrame, value_col: str, by: list[str], out_col: str) -> pl.Expr:
    return (pl.col(value_col) - pl.col(value_col).mean().over(by)).alias(out_col)


def residualize_outcomes(
    firm_year_panel: pl.DataFrame,
    config: ResidualizationConfig,
) -> pl.DataFrame:
    """Construct residualized ROA and Tobin's Q under the configured FE option."""

    df = add_size_quartiles(firm_year_panel)
    df = df.with_columns(
        _demean(df, "tobin_q_raw", ["fyear", "ff49"], "tobin_q_resid_industry"),
        _demean(df, "roa_raw", ["fyear", "ff49"], "roa_resid_industry"),
        _demean(df, "tobin_q_raw", ["fyear", "size_quartile"], "tobin_q_resid_size"),
        _demean(df, "roa_raw", ["fyear", "size_quartile"], "roa_resid_size"),
        _demean(df, "tobin_q_raw", ["fyear", "state"], "tobin_q_resid_state"),
        _demean(df, "roa_raw", ["fyear", "state"], "roa_resid_state"),
    )

    if config.year_industry:
        return df.with_columns(
            pl.col("tobin_q_resid_industry").alias("tobin_q_resid"),
            pl.col("roa_resid_industry").alias("roa_resid"),
            pl.col("tobin_q_resid_industry").alias("q_resid"),
        )
    if config.year_size_quartile:
        return df.with_columns(
            pl.col("tobin_q_resid_size").alias("tobin_q_resid"),
            pl.col("roa_resid_size").alias("roa_resid"),
            pl.col("tobin_q_resid_size").alias("q_resid"),
        )
    if config.year_state:
        return df.with_columns(
            pl.col("tobin_q_resid_state").alias("tobin_q_resid"),
            pl.col("roa_resid_state").alias("roa_resid"),
            pl.col("tobin_q_resid_state").alias("q_resid"),
        )
    return df.with_columns(
        pl.col("tobin_q_raw").alias("tobin_q_resid"),
        pl.col("roa_raw").alias("roa_resid"),
        pl.col("q_raw").alias("q_resid"),
    )
