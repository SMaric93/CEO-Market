"""Schema-validation tests for cleaned raw inputs and config loading."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from whogetsconsidered.config import load_config
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.schemas.raw import RAW_SCHEMAS, validate_dataframe


FIXTURE_DIR = Path("tests/fixtures")


def test_minimal_config_loads() -> None:
    config = load_config("examples/minimal_config.yml")
    assert config.market.definition == "radius"
    assert config.regression.release_window_years == 2


def test_raw_fixture_tables_validate() -> None:
    fixture_map = {
        "cri_exec_panel": FIXTURE_DIR / "synthetic_cri.parquet",
        "compustat_firm_year": FIXTURE_DIR / "synthetic_compustat.parquet",
        "hq_history": FIXTURE_DIR / "synthetic_hq.parquet",
        "release_events": FIXTURE_DIR / "synthetic_delist.parquet",
        "noncompete_state_year": FIXTURE_DIR / "synthetic_noncompete.parquet",
        "ff_industry_map": FIXTURE_DIR / "synthetic_ff_map.parquet",
    }
    for name, path in fixture_map.items():
        df = read_input_table(name, path)
        assert df.height > 0


def test_validation_fails_on_missing_required_column() -> None:
    schema = RAW_SCHEMAS["cri_exec_panel"]
    df = pl.DataFrame(
        {
            "gvkey": ["001001"],
            "fyear": [2020],
            "exec_name_raw": ["Jane Doe"],
            "title_raw": ["Chief Executive Officer"],
            "exec_rank": [1],
        }
    )
    errors = validate_dataframe(df, schema)
    assert any("filing_date" in error for error in errors)
