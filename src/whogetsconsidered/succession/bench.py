"""Internal succession-bench measures observed before the event year."""

from __future__ import annotations

import polars as pl

from whogetsconsidered.fit.scoring import add_zscores


def build_internal_bench(
    succession_events: pl.DataFrame,
    executive_year_panel: pl.DataFrame,
) -> pl.DataFrame:
    """Construct t-1 internal bench measures for each succession event."""

    base = (
        succession_events.select("event_id", "gvkey", "succession_year")
        .join(executive_year_panel, on="gvkey", how="left")
        .filter(pl.col("fyear") == pl.col("succession_year") - 1)
        .filter(~pl.col("is_ceo"))
        .group_by("event_id", maintain_order=True)
        .agg(
            pl.col("is_president").cast(pl.Int8).max().alias("has_president_tminus1"),
            pl.col("is_coo").cast(pl.Int8).max().alias("has_coo_tminus1"),
            pl.col("is_cfo").cast(pl.Int8).max().alias("has_cfo_tminus1"),
            pl.col("is_ceo_ready").cast(pl.Int64).sum().alias("num_ceo_ready_insiders_tminus1"),
            pl.mean("firm_tenure_years").alias("avg_insider_tenure_tminus1"),
            (
                (
                    ((pl.col("is_president") | pl.col("is_coo")) & (pl.col("exec_rank").fill_null(99) <= 2))
                    .cast(pl.Int8)
                    .sum()
                    == 1
                )
                & (pl.col("is_ceo_ready").cast(pl.Int8).sum() <= 2)
            )
            .cast(pl.Int8)
            .alias("heir_apparent_proxy_tminus1"),
        )
    )
    bench_rows = succession_events.select("event_id").join(base, on="event_id", how="left").with_columns(
        pl.col("has_president_tminus1").fill_null(0),
        pl.col("has_coo_tminus1").fill_null(0),
        pl.col("has_cfo_tminus1").fill_null(0),
        pl.col("num_ceo_ready_insiders_tminus1").fill_null(0),
        pl.col("avg_insider_tenure_tminus1").fill_null(0.0),
        pl.col("heir_apparent_proxy_tminus1").fill_null(0),
    )
    bench_rows = add_zscores(
        bench_rows,
        [
            "has_president_tminus1",
            "has_coo_tminus1",
            "num_ceo_ready_insiders_tminus1",
            "avg_insider_tenure_tminus1",
        ],
    ).with_columns(
        (
            pl.sum_horizontal(
                [
                    pl.col("has_president_tminus1_z"),
                    pl.col("has_coo_tminus1_z"),
                    pl.col("num_ceo_ready_insiders_tminus1_z"),
                    pl.col("avg_insider_tenure_tminus1_z"),
                ]
            )
            / 4.0
        ).alias("bench_index_z")
    )
    median_bench = bench_rows["bench_index_z"].median() if bench_rows.height else 0.0
    return bench_rows.with_columns(
        (pl.col("bench_index_z") < median_bench).cast(pl.Int8).alias("weak_bench_flag")
    )
