"""Canonicalization helpers for WRDS source tables.

These transforms are intentionally conservative. They only materialize canonical research
inputs when WRDS tables provide a credible equivalent, and they emit explicit gaps when
WRDS alone cannot reproduce a required input such as noncompete scores or a fully geocoded
historical HQ panel.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from whogetsconsidered.schemas.raw import RAW_SCHEMAS, validate_dataframe


def _to_polars(frame: object) -> pl.DataFrame:
    """Convert a pandas- or polars-like frame to Polars."""

    if isinstance(frame, pl.DataFrame):
        return frame
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - pandas is installed via statsmodels in practice.
        pd = None  # type: ignore[assignment]
    if pd is not None and isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame, include_index=False)
    raise TypeError(f"unsupported WRDS frame type: {type(frame)!r}")


def _read_local_frame(path: Path) -> pl.DataFrame:
    """Read a local CSV or Parquet table used by the WRDS bootstrap layer."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"unsupported local table type: {path}")


def _normalize_text(expr: pl.Expr) -> pl.Expr:
    """Normalize string keys used for conservative joins."""

    return expr.cast(pl.Utf8).str.strip_chars().str.to_uppercase()


def _normalize_zip(expr: pl.Expr) -> pl.Expr:
    """Reduce ZIP codes to their comparable 5-digit core."""

    return (
        expr.cast(pl.Utf8)
        .str.replace_all(r"[^0-9]", "")
        .str.slice(0, 5)
        .str.strip_chars()
    )


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first available column among candidate names."""

    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _validate_generated_input(name: str, df: pl.DataFrame) -> pl.DataFrame:
    """Validate a generated canonical input against the declared schema."""

    errors = validate_dataframe(df, RAW_SCHEMAS[name])
    if errors:
        raise ValueError("\n".join(errors))
    return df


def _normalize_indicator(expr: pl.Expr) -> pl.Expr:
    """Convert heterogeneous WRDS indicator encodings into nullable 0/1 integers."""

    text = expr.cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
    return (
        pl.when(text.is_null() | (text == ""))
        .then(pl.lit(None, dtype=pl.Int64))
        .when(text.is_in(["1", "Y", "YES", "TRUE", "T", "CEO"]))
        .then(pl.lit(1, dtype=pl.Int64))
        .when(text.is_in(["0", "N", "NO", "FALSE", "F"]))
        .then(pl.lit(0, dtype=pl.Int64))
        .otherwise(pl.lit(0, dtype=pl.Int64))
    )


def standardize_execucomp(execucomp: pl.DataFrame) -> pl.DataFrame:
    """Standardize Execucomp columns into a reusable optional input table."""

    year_column = _first_present(execucomp.columns, ("year", "fyear"))
    name_column = _first_present(execucomp.columns, ("exec_fullname", "exec_name", "name"))
    title_column = _first_present(execucomp.columns, ("titleann", "title", "title_raw"))
    rank_column = _first_present(
        execucomp.columns,
        ("rankann", "execrank", "execrankann", "exec_rank", "rank"),
    )
    pceo_column = _first_present(execucomp.columns, ("pceo", "ceoann", "is_ceo"))
    if year_column is None or name_column is None or "gvkey" not in execucomp.columns:
        raise ValueError("Execucomp extract must contain gvkey, year/fyear, and an executive name column")

    standardized = execucomp.select(
        pl.col("gvkey").cast(pl.Utf8),
        pl.col(year_column).cast(pl.Int64).alias("year"),
        pl.coalesce(
            [
                pl.col(_first_present(execucomp.columns, ("execid", "personid")) or name_column)
                .cast(pl.Utf8),
                pl.col(name_column).cast(pl.Utf8),
            ]
        ).alias("execid"),
        pl.col(name_column).cast(pl.Utf8).alias("exec_name_raw"),
        (
            pl.col(title_column).cast(pl.Utf8)
            if title_column is not None
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("title_raw"),
        (
            pl.col(rank_column).cast(pl.Int64)
            if rank_column is not None
            else pl.lit(None, dtype=pl.Int64)
        ).alias("exec_rank"),
        (
            _normalize_indicator(pl.col(pceo_column))
            if pceo_column is not None
            else pl.lit(None, dtype=pl.Int64)
        ).alias("pceo"),
    )
    standardized = standardized.filter(
        pl.col("gvkey").is_not_null()
        & pl.col("year").is_not_null()
        & pl.col("execid").is_not_null()
        & pl.col("exec_name_raw").is_not_null()
    )
    return _validate_generated_input("execucomp", standardized)


def build_cri_proxy_from_execucomp(execucomp: pl.DataFrame) -> pl.DataFrame:
    """Build a CRI-shaped executive panel proxy from Execucomp when CRI is unavailable.

    This proxy is narrower than true CRI coverage because Execucomp only spans named
    executives at covered firms. The output keeps the CRI schema so the main package can
    use it transparently, but metadata should always mark it as a WRDS/Execucomp proxy.
    """

    standardized = standardize_execucomp(execucomp)
    cri_proxy = standardized.select(
        "gvkey",
        pl.col("year").alias("fyear"),
        "exec_name_raw",
        pl.col("title_raw").fill_null(""),
        "exec_rank",
        pl.lit(None, dtype=pl.Date).alias("filing_date"),
    )
    return _validate_generated_input("cri_exec_panel", cri_proxy)


def build_compustat_firm_year(fundamentals: pl.DataFrame, company_reference: pl.DataFrame) -> pl.DataFrame:
    """Merge Compustat fundamentals with reference metadata into the package input schema."""

    filtered = fundamentals
    if "indfmt" in filtered.columns:
        filtered = filtered.filter(pl.col("indfmt") == "INDL")
    if "datafmt" in filtered.columns:
        filtered = filtered.filter(pl.col("datafmt") == "STD")
    if "consol" in filtered.columns:
        filtered = filtered.filter(pl.col("consol") == "C")
    if "popsrc" in filtered.columns:
        filtered = filtered.filter(pl.col("popsrc") == "D")
    filtered = filtered.filter(pl.col("fyear").is_not_null() & pl.col("at").is_not_null())

    state_column = _first_present(company_reference.columns, ("state", "hq_state", "addr_state"))
    standardized_company = company_reference.select(
        pl.col("gvkey").cast(pl.Utf8),
        (
            pl.col("sic").cast(pl.Int64)
            if "sic" in company_reference.columns
            else pl.lit(None, dtype=pl.Int64)
        ).alias("sic_company"),
        (
            pl.col(state_column).cast(pl.Utf8)
            if state_column is not None
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("state_hq"),
        *[
            pl.col(column)
            for column in ("city", "addzip")
            if column in company_reference.columns
        ],
    ).unique(subset=["gvkey"], keep="last")

    firm_year = (
        filtered.join(
            standardized_company.select("gvkey", "sic_company", "state_hq"),
            on="gvkey",
            how="left",
        )
        .select(
            pl.col("gvkey").cast(pl.Utf8),
            pl.col("fyear").cast(pl.Int64),
            pl.coalesce(
                [
                    (
                        pl.col("sic")
                        if "sic" in filtered.columns
                        else pl.lit(None, dtype=pl.Int64)
                    ),
                    pl.col("sic_company"),
                ]
            )
            .cast(pl.Int64)
            .alias("sic"),
            pl.col("state_hq").cast(pl.Utf8),
            pl.col("at").cast(pl.Float64),
            pl.col("ebit").cast(pl.Float64),
            pl.col("xrd").cast(pl.Float64),
            pl.col("capx").cast(pl.Float64),
            pl.col("dltt").cast(pl.Float64),
            pl.col("dlc").cast(pl.Float64),
            pl.col("dv").cast(pl.Float64),
            pl.col("prcc_f").cast(pl.Float64),
            pl.col("csho").cast(pl.Float64),
            pl.col("sale").cast(pl.Float64),
            pl.col("ceq").cast(pl.Float64),
            pl.col("datadate").cast(pl.Date),
        )
        .sort(["gvkey", "fyear"])
    )
    return _validate_generated_input("compustat_firm_year", firm_year)


def build_hq_history(
    company_reference: pl.DataFrame,
    compustat_firm_year: pl.DataFrame,
    *,
    geocode_crosswalk_path: Path,
) -> pl.DataFrame:
    """Build a best-available HQ history from WRDS company metadata plus a local geocode map.

    WRDS company reference data do not provide a clean historical latitude/longitude panel.
    This function therefore creates a single spell per firm using the observed WRDS address
    metadata and a user-supplied geocode crosswalk. The metadata emitted by the pipeline
    flags this as a best-available bootstrap rather than a full historical HQ history.
    """

    crosswalk_raw = _read_local_frame(geocode_crosswalk_path)
    city_column = _first_present(crosswalk_raw.columns, ("city", "hq_city"))
    state_column = _first_present(crosswalk_raw.columns, ("state", "hq_state"))
    zip_column = _first_present(crosswalk_raw.columns, ("zip", "addzip", "postal_code"))
    msa_column = _first_present(crosswalk_raw.columns, ("msa_code", "msa"))
    if city_column is None or state_column is None or zip_column is None:
        raise ValueError("HQ geocode crosswalk must contain city, state, and zip-like columns")
    if "lat" not in crosswalk_raw.columns or "lon" not in crosswalk_raw.columns:
        raise ValueError("HQ geocode crosswalk must contain lat and lon columns")

    crosswalk = (
        crosswalk_raw.select(
            _normalize_text(pl.col(city_column)).alias("city_key"),
            _normalize_text(pl.col(state_column)).alias("state_key"),
            _normalize_zip(pl.col(zip_column)).alias("zip_key"),
            pl.col("lat").cast(pl.Float64),
            pl.col("lon").cast(pl.Float64),
            (
                pl.col(msa_column).cast(pl.Utf8)
                if msa_column is not None
                else pl.lit(None, dtype=pl.Utf8)
            ).alias("msa_code"),
        )
        .unique(subset=["city_key", "state_key", "zip_key"], keep="last")
    )

    company_city = _first_present(company_reference.columns, ("city", "hq_city"))
    company_state = _first_present(company_reference.columns, ("state", "hq_state", "addr_state"))
    company_zip = _first_present(company_reference.columns, ("addzip", "zip", "postal_code"))
    if company_city is None or company_state is None or company_zip is None:
        raise ValueError("WRDS company reference must contain city, state, and zip-like columns")

    first_observation = compustat_firm_year.group_by("gvkey").agg(
        pl.col("datadate").min().alias("first_datadate"),
        pl.col("fyear").min().alias("first_fyear"),
    )
    base = (
        company_reference.select(
            pl.col("gvkey").cast(pl.Utf8),
            pl.col(company_city).cast(pl.Utf8).alias("city"),
            pl.col(company_state).cast(pl.Utf8).alias("state"),
            pl.col(company_zip).cast(pl.Utf8).alias("zip"),
        )
        .unique(subset=["gvkey"], keep="last")
        .with_columns(
            _normalize_text(pl.col("city")).alias("city_key"),
            _normalize_text(pl.col("state")).alias("state_key"),
            _normalize_zip(pl.col("zip")).alias("zip_key"),
        )
        .join(first_observation, on="gvkey", how="left")
        .join(crosswalk, on=["city_key", "state_key", "zip_key"], how="left")
        .with_columns(
            pl.when(pl.col("first_datadate").is_not_null())
            .then(pl.col("first_datadate"))
            .otherwise(pl.date(pl.col("first_fyear"), pl.lit(1), pl.lit(1)))
            .alias("start_date"),
            pl.lit(None, dtype=pl.Date).alias("end_date"),
        )
    )

    hq_history = base.select(
        "gvkey",
        pl.col("start_date").cast(pl.Date),
        pl.col("end_date").cast(pl.Date),
        pl.col("city").cast(pl.Utf8),
        pl.col("state").cast(pl.Utf8),
        pl.col("zip").cast(pl.Utf8),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
        pl.col("msa_code").cast(pl.Utf8),
    )
    return _validate_generated_input("hq_history", hq_history)


def _standardize_link_table(link_table: pl.DataFrame) -> pl.DataFrame:
    """Standardize CRSP/Compustat linking columns."""

    permno_column = _first_present(link_table.columns, ("permno", "lpermno"))
    if permno_column is None or "gvkey" not in link_table.columns:
        raise ValueError("CCM link table must contain gvkey and permno/lpermno")
    linkdt_column = _first_present(link_table.columns, ("linkdt",))
    linkenddt_column = _first_present(link_table.columns, ("linkenddt",))
    return link_table.select(
        pl.col("gvkey").cast(pl.Utf8),
        pl.col(permno_column).cast(pl.Int64).alias("permno"),
        (
            pl.col(linkdt_column).cast(pl.Date)
            if linkdt_column is not None
            else pl.lit(None, dtype=pl.Date)
        ).alias("linkdt"),
        (
            pl.col(linkenddt_column).cast(pl.Date)
            if linkenddt_column is not None
            else pl.lit(None, dtype=pl.Date)
        ).alias("linkenddt"),
        (
            pl.col("usedflag").cast(pl.Int64)
            if "usedflag" in link_table.columns
            else pl.lit(1, dtype=pl.Int64)
        ).alias("usedflag"),
        (
            pl.col("linkprim").cast(pl.Utf8)
            if "linkprim" in link_table.columns
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("linkprim"),
        (
            pl.col("linktype").cast(pl.Utf8)
            if "linktype" in link_table.columns
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("linktype"),
    ).filter(
        (pl.col("usedflag") == 1)
        & (pl.col("linkprim").is_null() | pl.col("linkprim").is_in(["P", "C"]))
    )


def build_release_events(
    delist_table: pl.DataFrame,
    ccm_link_table: pl.DataFrame,
    *,
    hq_history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a conservative release-event input from CRSP delistings plus CCM links."""

    permno_column = _first_present(delist_table.columns, ("permno",))
    event_date_column = _first_present(delist_table.columns, ("dlstdt", "event_date"))
    delist_code_column = _first_present(delist_table.columns, ("dlstcd",))
    if permno_column is None or event_date_column is None or delist_code_column is None:
        raise ValueError("CRSP delisting table must contain permno, dlstdt, and dlstcd")

    delist = delist_table.select(
        pl.col(permno_column).cast(pl.Int64).alias("permno"),
        pl.col(event_date_column).cast(pl.Date).alias("event_date"),
        pl.col(delist_code_column).cast(pl.Int64).alias("dlstcd"),
    ).filter(pl.col("event_date").is_not_null())
    links = _standardize_link_table(ccm_link_table)

    matched = (
        delist.join(links, on="permno", how="left")
        .filter(
            (pl.col("linkdt").is_null() | (pl.col("event_date") >= pl.col("linkdt")))
            & (pl.col("linkenddt").is_null() | (pl.col("event_date") <= pl.col("linkenddt")))
        )
        .sort(["permno", "event_date", "linkdt"])
        .group_by(["permno", "event_date"])
        .agg(
            pl.col("gvkey").last().alias("source_gvkey"),
            pl.col("dlstcd").last().alias("dlstcd"),
        )
    )

    if hq_history is not None:
        hq_snapshot = (
            hq_history.sort(["gvkey", "start_date"])
            .group_by("gvkey")
            .agg(
                pl.col("lat").last().alias("source_hq_lat"),
                pl.col("lon").last().alias("source_hq_lon"),
                pl.col("msa_code").last().alias("source_msa_code"),
            )
            .rename({"gvkey": "source_gvkey"})
        )
    else:
        hq_snapshot = pl.DataFrame(
            schema={
                "gvkey": pl.Utf8,
                "source_hq_lat": pl.Float64,
                "source_hq_lon": pl.Float64,
                "source_msa_code": pl.Utf8,
            }
        ).rename({"gvkey": "source_gvkey"})

    release_events = (
        matched.join(hq_snapshot, on="source_gvkey", how="left")
        .filter(pl.col("source_gvkey").is_not_null())
        .with_columns(
            pl.col("event_date").dt.year().alias("event_year"),
            (pl.col("dlstcd") >= 200).and_(pl.col("dlstcd") < 300).alias("clean_release_flag"),
            pl.when((pl.col("dlstcd") >= 200) & (pl.col("dlstcd") < 300))
            .then(pl.lit("crsp_merger_delist"))
            .when((pl.col("dlstcd") >= 400) & (pl.col("dlstcd") < 500))
            .then(pl.lit("crsp_distress_delist"))
            .otherwise(pl.lit("crsp_other_delist"))
            .alias("event_type"),
            pl.lit(None, dtype=pl.Utf8).alias("acquirer_gvkey"),
        )
        .select(
            "source_gvkey",
            "event_date",
            pl.col("event_year").cast(pl.Int64),
            "event_type",
            pl.col("clean_release_flag").cast(pl.Boolean),
            "acquirer_gvkey",
            pl.col("source_hq_lat").cast(pl.Float64),
            pl.col("source_hq_lon").cast(pl.Float64),
            pl.col("source_msa_code").cast(pl.Utf8),
        )
        .sort(["source_gvkey", "event_date"])
    )
    return _validate_generated_input("release_events", release_events)


def standardize_boardex_people(raw_boardex_people: pl.DataFrame) -> pl.DataFrame:
    """Standardize BoardEx people extracts to the package schema."""

    person_id_column = _first_present(raw_boardex_people.columns, ("person_id", "personid", "directorid"))
    name_column = _first_present(raw_boardex_people.columns, ("person_name", "name", "fullname", "directorname"))
    if person_id_column is None or name_column is None:
        raise ValueError("BoardEx people table must contain person_id/personid and a name column")
    standardized = raw_boardex_people.select(
        pl.col(person_id_column).cast(pl.Utf8).alias("person_id"),
        pl.col(name_column).cast(pl.Utf8).alias("person_name"),
    )
    return _validate_generated_input("boardex_people", standardized)


def standardize_boardex_board_roles(raw_boardex_board_roles: pl.DataFrame) -> pl.DataFrame:
    """Standardize BoardEx board-role extracts to the package schema."""

    person_id_column = _first_present(raw_boardex_board_roles.columns, ("person_id", "personid", "directorid"))
    start_column = _first_present(
        raw_boardex_board_roles.columns,
        ("role_start_date", "start_date", "datestartrole", "startdate"),
    )
    end_column = _first_present(
        raw_boardex_board_roles.columns,
        ("role_end_date", "end_date", "dateendrole", "enddate"),
    )
    title_column = _first_present(
        raw_boardex_board_roles.columns,
        ("role_title", "title", "rolename", "brdposition"),
    )
    if person_id_column is None or "gvkey" not in raw_boardex_board_roles.columns:
        raise ValueError("BoardEx board roles must contain person_id/personid and gvkey")
    standardized = raw_boardex_board_roles.select(
        pl.col(person_id_column).cast(pl.Utf8).alias("person_id"),
        pl.col("gvkey").cast(pl.Utf8),
        (
            pl.col(start_column).cast(pl.Date)
            if start_column is not None
            else pl.lit(None, dtype=pl.Date)
        ).alias("role_start_date"),
        (
            pl.col(end_column).cast(pl.Date)
            if end_column is not None
            else pl.lit(None, dtype=pl.Date)
        ).alias("role_end_date"),
        (
            pl.col(title_column).cast(pl.Utf8)
            if title_column is not None
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("role_title"),
    )
    return _validate_generated_input("boardex_board_roles", standardized)


def standardize_boardex_employment(raw_boardex_employment: pl.DataFrame) -> pl.DataFrame:
    """Standardize BoardEx employment extracts to the package schema."""

    person_id_column = _first_present(raw_boardex_employment.columns, ("person_id", "personid", "directorid"))
    title_column = _first_present(
        raw_boardex_employment.columns,
        ("title", "job_title", "rolename", "brdposition"),
    )
    start_column = _first_present(
        raw_boardex_employment.columns,
        ("start_date", "employment_start_date", "datestartrole", "startdate"),
    )
    end_column = _first_present(
        raw_boardex_employment.columns,
        ("end_date", "employment_end_date", "dateendrole", "enddate"),
    )
    employer_column = _first_present(
        raw_boardex_employment.columns,
        ("employer_name", "company_name", "employer", "companyname"),
    )
    if person_id_column is None:
        raise ValueError("BoardEx employment table must contain person_id/personid")
    standardized = raw_boardex_employment.select(
        pl.col(person_id_column).cast(pl.Utf8).alias("person_id"),
        (
            pl.col("gvkey").cast(pl.Utf8)
            if "gvkey" in raw_boardex_employment.columns
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("gvkey"),
        (
            pl.col(employer_column).cast(pl.Utf8)
            if employer_column is not None
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("employer_name"),
        (
            pl.col(start_column).cast(pl.Date)
            if start_column is not None
            else pl.lit(None, dtype=pl.Date)
        ).alias("start_date"),
        (
            pl.col(end_column).cast(pl.Date)
            if end_column is not None
            else pl.lit(None, dtype=pl.Date)
        ).alias("end_date"),
        (
            pl.col(title_column).cast(pl.Utf8)
            if title_column is not None
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("title"),
    )
    return _validate_generated_input("boardex_employment", standardized)


def _make_date_from_parts(
    *,
    year: pl.Expr,
    month: pl.Expr,
    day: pl.Expr,
) -> pl.Expr:
    """Build a conservative date from nullable year/month/day integer parts."""

    return (
        pl.when(year.is_null())
        .then(pl.lit(None, dtype=pl.Date))
        .otherwise(
            pl.date(
                year.cast(pl.Int32),
                month.fill_null(1).clip(1, 12).cast(pl.Int32),
                day.fill_null(1).clip(1, 28).cast(pl.Int32),
            )
        )
    )


def standardize_capiq_people_analytics(raw_capiq_people_analytics: pl.DataFrame) -> pl.DataFrame:
    """Standardize WRDS Capital IQ People Intelligence professional data."""

    standardized = raw_capiq_people_analytics.select(
        pl.col("personid").cast(pl.Utf8).alias("personid"),
        pl.col("companyid").cast(pl.Utf8).alias("companyid"),
        (
            pl.col("gvkey").cast(pl.Utf8)
            if "gvkey" in raw_capiq_people_analytics.columns
            else pl.lit(None, dtype=pl.Utf8)
        ).alias("gvkey"),
        pl.col("proid").cast(pl.Utf8).alias("proid"),
        pl.col("companyname").cast(pl.Utf8).alias("companyname"),
        pl.col("personname").cast(pl.Utf8).alias("personname"),
        pl.col("profunctionname").cast(pl.Utf8).alias("profunctionname"),
        pl.col("title").cast(pl.Utf8).alias("title"),
        _make_date_from_parts(
            year=pl.col("startyear"),
            month=pl.col("startmonth"),
            day=pl.col("startday"),
        ).alias("start_date"),
        _make_date_from_parts(
            year=pl.col("endyear"),
            month=pl.col("endmonth"),
            day=pl.col("endday"),
        ).alias("end_date"),
        pl.col("rank").cast(pl.Int64),
        pl.col("prorank").cast(pl.Int64),
        pl.col("boardrank").cast(pl.Int64),
        _normalize_indicator(pl.col("proflag")).alias("proflag"),
        _normalize_indicator(pl.col("currentproflag")).alias("currentproflag"),
        _normalize_indicator(pl.col("boardflag")).alias("boardflag"),
        _normalize_indicator(pl.col("currentboardflag")).alias("currentboardflag"),
        _normalize_indicator(pl.col("currentflag")).alias("currentflag"),
        _normalize_indicator(pl.col("keyexecflag")).alias("keyexecflag"),
        _normalize_indicator(pl.col("topkeyexecflag")).alias("topkeyexecflag"),
        _normalize_indicator(pl.col("advisorflag")).alias("advisorflag"),
        _normalize_indicator(pl.col("graduateflag")).alias("graduateflag"),
        _normalize_indicator(pl.col("dealmakerflag")).alias("dealmakerflag"),
        _normalize_indicator(pl.col("sponsorflag")).alias("sponsorflag"),
        _normalize_indicator(pl.col("undergraduateflag")).alias("undergraduateflag"),
        _normalize_indicator(pl.col("onlyoneflag")).alias("onlyoneflag"),
        _normalize_indicator(pl.col("companyflag")).alias("companyflag"),
        _normalize_indicator(pl.col("hideflag")).alias("hideflag"),
        pl.col("country").cast(pl.Utf8).alias("country"),
        pl.col("state").cast(pl.Utf8).alias("state"),
        pl.col("committeeid").cast(pl.Utf8).alias("committeeid"),
    ).filter(pl.col("personid").is_not_null() & pl.col("companyid").is_not_null())
    return standardized


def build_boardex_capiq_bridge(
    *,
    raw_boardex_people: pl.DataFrame,
    raw_boardex_board_roles: pl.DataFrame,
    raw_boardex_employment: pl.DataFrame,
    raw_boardex_ciq_link: pl.DataFrame,
    capiq_people_analytics: pl.DataFrame,
) -> pl.DataFrame:
    """Build a merged BoardEx-Capital IQ people bridge from the WRDS cross-file.

    This artifact links BoardEx director identities to CIQ person/company identifiers,
    then enriches the crosswalk with CIQ professional analytics and BoardEx role timing.
    It is the preferred source for generating BoardEx inputs usable by the main package.
    """

    boardex_people = raw_boardex_people.select(
        pl.col(_first_present(raw_boardex_people.columns, ("directorid", "person_id", "personid")) or "directorid")
        .cast(pl.Utf8)
        .alias("directorid"),
        pl.col(_first_present(raw_boardex_people.columns, ("directorname", "person_name", "name")) or "directorname")
        .cast(pl.Utf8)
        .alias("directorname"),
    ).unique(subset=["directorid"], keep="first")

    boardex_roles = raw_boardex_board_roles.select(
        pl.col("directorid").cast(pl.Utf8).alias("directorid"),
        pl.col("companyid").cast(pl.Utf8).alias("companyid"),
        pl.col("companyname").cast(pl.Utf8).alias("boardex_companyname"),
        pl.col("datestartrole").cast(pl.Date).alias("boardex_role_start_date"),
        pl.col("dateendrole").cast(pl.Date).alias("boardex_role_end_date"),
        pl.col("startdate").cast(pl.Date).alias("boardex_start_date"),
        pl.col("enddate").cast(pl.Date).alias("boardex_end_date"),
        pl.col("title").cast(pl.Utf8).alias("boardex_title"),
        pl.col("rolename").cast(pl.Utf8).alias("boardex_rolename"),
        pl.col("brdposition").cast(pl.Utf8).alias("boardex_brdposition"),
        pl.col("fulltextdescription").cast(pl.Utf8).alias("boardex_description"),
        _normalize_indicator(pl.col("ned")).alias("boardex_ned_flag"),
        _normalize_indicator(pl.col("leadershipteam")).alias("boardex_leadershipteam_flag"),
    )

    boardex_emp = raw_boardex_employment.select(
        pl.col("directorid").cast(pl.Utf8).alias("directorid"),
        pl.col("companyid").cast(pl.Utf8).alias("companyid"),
        pl.col("companyname").cast(pl.Utf8).alias("employment_companyname"),
        pl.col("datestartrole").cast(pl.Date).alias("employment_start_date"),
        pl.col("dateendrole").cast(pl.Date).alias("employment_end_date"),
        pl.col("rolename").cast(pl.Utf8).alias("employment_rolename"),
        pl.col("brdposition").cast(pl.Utf8).alias("employment_brdposition"),
        _normalize_indicator(pl.col("ned")).alias("employment_ned_flag"),
        _normalize_indicator(pl.col("leadershipteam")).alias("employment_leadershipteam_flag"),
    )

    crosswalk = raw_boardex_ciq_link.select(
        pl.col("directorid").cast(pl.Utf8).alias("directorid"),
        pl.col("directorname").cast(pl.Utf8).alias("crosswalk_directorname"),
        pl.col("boardid").cast(pl.Utf8).alias("boardid"),
        pl.col("boardname").cast(pl.Utf8).alias("boardname"),
        pl.col("bd_ticker").cast(pl.Utf8).alias("board_ticker"),
        pl.col("isin").cast(pl.Utf8).alias("isin"),
        pl.col("cikcode").cast(pl.Utf8).alias("cikcode"),
        pl.col("title").cast(pl.Utf8).alias("crosswalk_title"),
        pl.col("personid").cast(pl.Utf8).alias("personid"),
        pl.col("companyid").cast(pl.Utf8).alias("companyid"),
        pl.col("companyname").cast(pl.Utf8).alias("crosswalk_companyname"),
        pl.col("firstname").cast(pl.Utf8).alias("ciq_firstname"),
        pl.col("middlename").cast(pl.Utf8).alias("ciq_middlename"),
        pl.col("lastname").cast(pl.Utf8).alias("ciq_lastname"),
        pl.col("prefix").cast(pl.Utf8).alias("ciq_prefix"),
        pl.col("suffix").cast(pl.Utf8).alias("ciq_suffix"),
        pl.col("salutation").cast(pl.Utf8).alias("ciq_salutation"),
        pl.col("score").cast(pl.Float64).alias("link_score"),
        pl.col("matchstyle").cast(pl.Utf8).alias("match_style"),
    )

    bridge = (
        crosswalk.join(boardex_people, on="directorid", how="left")
        .join(boardex_roles, on=["directorid", "companyid"], how="left")
        .join(boardex_emp, on=["directorid", "companyid"], how="left")
        .join(capiq_people_analytics, on=["personid", "companyid"], how="left")
        .with_columns(
            pl.coalesce([pl.col("directorname"), pl.col("crosswalk_directorname")]).alias("boardex_person_name"),
            pl.coalesce(
                [pl.col("boardex_companyname"), pl.col("employment_companyname"), pl.col("crosswalk_companyname")]
            ).alias(
                "boardex_company_name"
            ),
            pl.coalesce(
                [
                    pl.col("boardex_rolename"),
                    pl.col("boardex_title"),
                    pl.col("employment_rolename"),
                    pl.col("crosswalk_title"),
                    pl.col("title"),
                ]
            ).alias(
                "merged_role_title"
            ),
            pl.coalesce(
                [
                    pl.col("boardex_role_start_date"),
                    pl.col("boardex_start_date"),
                    pl.col("employment_start_date"),
                    pl.col("start_date"),
                ]
            ).alias(
                "merged_start_date"
            ),
            pl.coalesce(
                [
                    pl.col("boardex_role_end_date"),
                    pl.col("boardex_end_date"),
                    pl.col("employment_end_date"),
                    pl.col("end_date"),
                ]
            ).alias(
                "merged_end_date"
            ),
        )
    )
    return bridge.unique(
        subset=["directorid", "personid", "companyid", "gvkey", "merged_role_title", "merged_start_date"],
        keep="first",
    )


def build_boardex_people_from_bridge(boardex_capiq_bridge: pl.DataFrame) -> pl.DataFrame:
    """Build canonical BoardEx people output from the merged BoardEx-CIQ bridge."""

    best = boardex_capiq_bridge.sort(["directorid", "link_score"], descending=[False, True]).unique(
        subset=["directorid"], keep="first"
    )
    output = best.select(
        pl.col("directorid").alias("person_id"),
        pl.col("boardex_person_name").alias("person_name"),
        pl.col("personid").alias("ciq_personid"),
        pl.col("link_score"),
        pl.col("match_style"),
    )
    return _validate_generated_input("boardex_people", output)


def build_boardex_board_roles_from_bridge(boardex_capiq_bridge: pl.DataFrame) -> pl.DataFrame:
    """Build canonical BoardEx board-role output from the merged BoardEx-CIQ bridge."""

    output = (
        boardex_capiq_bridge.filter(pl.col("gvkey").is_not_null())
        .select(
            pl.col("directorid").alias("person_id"),
            pl.col("gvkey"),
            pl.col("merged_start_date").alias("role_start_date"),
            pl.col("merged_end_date").alias("role_end_date"),
            pl.col("merged_role_title").alias("role_title"),
            pl.col("personid").alias("ciq_personid"),
            pl.col("link_score"),
            pl.col("match_style"),
            pl.col("boardex_company_name"),
        )
        .unique(subset=["person_id", "gvkey", "role_title", "role_start_date"], keep="first")
    )
    return _validate_generated_input("boardex_board_roles", output)


def build_boardex_employment_from_bridge(boardex_capiq_bridge: pl.DataFrame) -> pl.DataFrame:
    """Build canonical BoardEx employment output from the merged BoardEx-CIQ bridge."""

    output = (
        boardex_capiq_bridge.filter(pl.col("gvkey").is_not_null())
        .select(
            pl.col("directorid").alias("person_id"),
            pl.col("gvkey"),
            pl.col("boardex_company_name").alias("employer_name"),
            pl.col("merged_start_date").alias("start_date"),
            pl.col("merged_end_date").alias("end_date"),
            pl.col("merged_role_title").alias("title"),
            pl.col("personid").alias("ciq_personid"),
            pl.col("link_score"),
            pl.col("match_style"),
        )
        .unique(subset=["person_id", "gvkey", "title", "start_date"], keep="first")
    )
    return _validate_generated_input("boardex_employment", output)


def build_wrds_merged_company_year(
    compustat_firm_year: pl.DataFrame,
    *,
    cri_proxy: pl.DataFrame | None = None,
    release_events: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Create a single merged WRDS firm-year panel for auditing staged coverage."""

    merged = compustat_firm_year
    if cri_proxy is not None:
        exec_counts = cri_proxy.group_by(["gvkey", "fyear"]).agg(
            pl.len().alias("execucomp_exec_count")
        )
        merged = merged.join(exec_counts, on=["gvkey", "fyear"], how="left")
    if release_events is not None:
        release_counts = release_events.group_by(["source_gvkey", "event_year"]).agg(
            pl.len().alias("release_event_count"),
            pl.col("clean_release_flag").cast(pl.Int64).sum().alias("clean_release_count"),
        )
        merged = merged.join(
            release_counts.rename({"source_gvkey": "gvkey", "event_year": "fyear"}),
            on=["gvkey", "fyear"],
            how="left",
        )
    exprs: list[pl.Expr] = []
    if "execucomp_exec_count" in merged.columns:
        exprs.append(pl.col("execucomp_exec_count").fill_null(0))
    else:
        exprs.append(pl.lit(0).alias("execucomp_exec_count"))
    if "release_event_count" in merged.columns:
        exprs.append(pl.col("release_event_count").fill_null(0))
    else:
        exprs.append(pl.lit(0).alias("release_event_count"))
    if "clean_release_count" in merged.columns:
        exprs.append(pl.col("clean_release_count").fill_null(0))
    else:
        exprs.append(pl.lit(0).alias("clean_release_count"))
    return merged.with_columns(*exprs)
