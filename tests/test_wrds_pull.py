"""Tests for the optional WRDS bootstrap pipeline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from whogetsconsidered.config import WrdsPullConfig, WrdsTableConfig, load_config
from whogetsconsidered.logging_utils import configure_logging
from whogetsconsidered.wrds.puller import pull_wrds_bundle


class FakeWrdsClient:
    """Tiny fake WRDS client that returns deterministic synthetic tables."""

    def __init__(self, tables: dict[str, pl.DataFrame]) -> None:
        self.tables = tables

    def get_table(
        self,
        *,
        library: str,
        table: str,
        columns: list[str] | None = None,
        rows: int | None = None,
    ) -> pl.DataFrame:
        frame = self.tables[f"{library}.{table}"]
        if columns:
            frame = frame.select(columns)
        if rows is not None:
            frame = frame.head(rows)
        return frame

    def raw_sql(self, sql: str, *, date_cols: list[str] | None = None) -> pl.DataFrame:
        if "crsp.ccmxpf_linktable" in sql.lower():
            return self.tables["crsp.ccmxpf_linktable"]
        raise ValueError(f"unexpected SQL in fake WRDS client: {sql}")

    def close(self) -> None:
        return None


def _wrds_test_config(tmp_path: Path, geocode_path: Path) -> WrdsPullConfig:
    """Return a WRDS config block pointed at the pytest temp directory."""

    return WrdsPullConfig(
        enabled=True,
        staging_dir=tmp_path / "raw",
        canonical_dir=tmp_path / "canonical",
        manifest_path=tmp_path / "manifest.json",
        hq_geocode_crosswalk=geocode_path,
    )


def test_wrds_bootstrap_config_loads() -> None:
    config = load_config("examples/wrds_bootstrap_config.yml")
    assert config.wrds.enabled is True
    assert config.wrds.company_reference.table == "company"


def test_pull_wrds_bundle_builds_canonical_extracts(tmp_path: Path) -> None:
    geocode_path = tmp_path / "hq_geocode.parquet"
    pl.DataFrame(
        {
            "city": ["New York", "San Francisco"],
            "state": ["NY", "CA"],
            "zip": ["10001", "94105"],
            "lat": [40.7506, 37.7898],
            "lon": [-73.9972, -122.3942],
            "msa_code": ["35620", "41860"],
        }
    ).write_parquet(geocode_path)

    tables = {
        "comp.funda": pl.DataFrame(
            {
                "gvkey": ["001001", "001001", "001002"],
                "fyear": [2020, 2021, 2021],
                "datadate": [date(2020, 12, 31), date(2021, 12, 31), date(2021, 12, 31)],
                "sic": [3571, 3571, 2834],
                "at": [1000.0, 1100.0, 900.0],
                "ebit": [100.0, 120.0, 95.0],
                "xrd": [50.0, 60.0, 70.0],
                "capx": [20.0, 25.0, 15.0],
                "dltt": [100.0, 120.0, 80.0],
                "dlc": [10.0, 10.0, 5.0],
                "dv": [5.0, 5.0, 0.0],
                "prcc_f": [10.0, 11.0, 20.0],
                "csho": [100.0, 100.0, 30.0],
                "sale": [900.0, 1000.0, 950.0],
                "ceq": [400.0, 450.0, 300.0],
                "indfmt": ["INDL", "INDL", "INDL"],
                "datafmt": ["STD", "STD", "STD"],
                "consol": ["C", "C", "C"],
                "popsrc": ["D", "D", "D"],
            }
        ),
        "comp.company": pl.DataFrame(
            {
                "gvkey": ["001001", "001002"],
                "conm": ["Alpha Corp", "Beta Corp"],
                "city": ["New York", "San Francisco"],
                "state": ["NY", "CA"],
                "addzip": ["10001", "94105"],
                "sic": [3571, 2834],
            }
        ),
        "execcomp.anncomp": pl.DataFrame(
            {
                "gvkey": ["001001", "001002"],
                "year": [2021, 2021],
                "execid": ["E1", "E2"],
                "exec_fullname": ["Jane Doe", "John Smith"],
                "titleann": ["Chief Executive Officer", "President and COO"],
                "execrank": [1, 2],
                "pceo": [1, 0],
            }
        ),
        "crsp.ccmxpf_linktable": pl.DataFrame(
            {
                "gvkey": ["001001", "001002"],
                "permno": [10001, 10002],
                "linktype": ["LC", "LC"],
                "linkprim": ["P", "P"],
                "linkdt": [date(2019, 1, 1), date(2019, 1, 1)],
                "linkenddt": [None, None],
                "usedflag": [1, 1],
            }
        ),
        "crsp.dsedelist": pl.DataFrame(
            {
                "permno": [10001, 10002],
                "dlstdt": [date(2021, 6, 30), date(2021, 7, 31)],
                "dlstcd": [200, 500],
                "nwperm": [None, None],
                "nextdt": [None, None],
            }
        ),
    }

    config = load_config("examples/minimal_config.yml").model_copy(
        update={"wrds": _wrds_test_config(tmp_path, geocode_path)}
    )
    manifest = pull_wrds_bundle(config, logger=configure_logging(), client=FakeWrdsClient(tables))

    canonical_dir = tmp_path / "canonical"
    assert (canonical_dir / "compustat_firm_year.parquet").exists()
    assert (canonical_dir / "cri_exec_panel.parquet").exists()
    assert (canonical_dir / "hq_history.parquet").exists()
    assert (canonical_dir / "release_events.parquet").exists()
    assert manifest["missing_required_inputs"] == ["noncompete_state_year", "ff_industry_map"]

    compustat = pl.read_parquet(canonical_dir / "compustat_firm_year.parquet")
    assert compustat.select("state_hq").to_series().to_list() == ["NY", "NY", "CA"]

    releases = pl.read_parquet(canonical_dir / "release_events.parquet")
    assert releases.filter(pl.col("clean_release_flag")).height == 1

    merged = pl.read_parquet(canonical_dir / "wrds_merged_company_year.parquet")
    assert merged.filter(pl.col("release_event_count") > 0).height == 2

    manifest_payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "cri_exec_panel" in manifest_payload["built_inputs"]


def test_pull_wrds_bundle_builds_capiq_boardex_bridge(tmp_path: Path) -> None:
    geocode_path = tmp_path / "hq_geocode.parquet"
    pl.DataFrame(
        {
            "city": ["New York"],
            "state": ["NY"],
            "zip": ["10001"],
            "lat": [40.7506],
            "lon": [-73.9972],
            "msa_code": ["35620"],
        }
    ).write_parquet(geocode_path)

    tables = {
        "comp.funda": pl.DataFrame(
            {
                "gvkey": ["001001"],
                "fyear": [2021],
                "datadate": [date(2021, 12, 31)],
                "sic": [3571],
                "at": [1000.0],
                "ebit": [120.0],
                "xrd": [60.0],
                "capx": [25.0],
                "dltt": [120.0],
                "dlc": [10.0],
                "dv": [5.0],
                "prcc_f": [11.0],
                "csho": [100.0],
                "sale": [1000.0],
                "ceq": [450.0],
                "indfmt": ["INDL"],
                "datafmt": ["STD"],
                "consol": ["C"],
                "popsrc": ["D"],
            }
        ),
        "comp.company": pl.DataFrame(
            {
                "gvkey": ["001001"],
                "conm": ["Alpha Corp"],
                "city": ["New York"],
                "state": ["NY"],
                "addzip": ["10001"],
                "sic": [3571],
            }
        ),
        "execcomp.anncomp": pl.DataFrame(
            {
                "gvkey": ["001001"],
                "year": [2021],
                "execid": ["E1"],
                "exec_fullname": ["Jane Doe"],
                "titleann": ["Chief Executive Officer"],
                "execrank": [1],
                "pceo": [1],
            }
        ),
        "crsp.ccmxpf_linktable": pl.DataFrame(
            {
                "gvkey": ["001001"],
                "permno": [10001],
                "linktype": ["LC"],
                "linkprim": ["P"],
                "linkdt": [date(2019, 1, 1)],
                "linkenddt": [None],
                "usedflag": [1],
            }
        ),
        "crsp.dsedelist": pl.DataFrame(
            {
                "permno": [10001],
                "dlstdt": [date(2021, 6, 30)],
                "dlstcd": [200],
                "nwperm": [None],
                "nextdt": [None],
            }
        ),
        "boardex.na_wrds_dir_profile_all": pl.DataFrame(
            {
                "directorid": ["D1"],
                "directorname": ["Jane Director"],
                "companyid": ["C1"],
                "companyname": ["Alpha Corp"],
                "datestartrole": [date(2019, 1, 1)],
                "dateendrole": [None],
                "startdate": [date(2019, 1, 1)],
                "enddate": [None],
                "title": ["Director"],
                "rolename": ["Board Director"],
                "brdposition": ["Independent Director"],
                "fulltextdescription": ["Board seat"],
                "ned": [1],
                "leadershipteam": [0],
            }
        ),
        "boardex.na_wrds_dir_profile_emp": pl.DataFrame(
            {
                "directorid": ["D1"],
                "directorname": ["Jane Director"],
                "companyid": ["C1"],
                "companyname": ["Alpha Corp"],
                "datestartrole": [date(2019, 1, 1)],
                "dateendrole": [None],
                "rolename": ["Board Director"],
                "brdposition": ["Independent Director"],
                "ned": [1],
                "leadershipteam": [0],
                "isin": ["US0000000001"],
            }
        ),
        "ciq_pplintel.wrds_professional": pl.DataFrame(
            {
                "companyid": ["C1"],
                "personid": ["P1"],
                "proid": ["PR1"],
                "profunctionid": ["F1"],
                "companyname": ["Alpha Corp"],
                "personname": ["Jane Director"],
                "profunctionname": ["Board"],
                "yearfounded": [1990],
                "yearborn": [1970],
                "title": ["Director"],
                "countryid": [1],
                "country": ["United States"],
                "stateid": [33],
                "state": ["NY"],
                "startday": [1],
                "startmonth": [1],
                "startyear": [2019],
                "endday": [None],
                "endmonth": [None],
                "endyear": [None],
                "rank": [1],
                "prorank": [1],
                "boardrank": [1],
                "proflag": [1],
                "currentproflag": [1],
                "boardflag": [1],
                "currentboardflag": [1],
                "currentflag": [1],
                "keyexecflag": [1],
                "topkeyexecflag": [1],
                "advisorflag": [0],
                "graduateflag": [0],
                "dealmakerflag": [0],
                "sponsorflag": [0],
                "undergraduateflag": [0],
                "onlyoneflag": [0],
                "companyflag": [1],
                "hideflag": [0],
                "committeeid": ["COM1"],
                "gvkey": ["001001"],
            }
        ),
        "wrdsapps_plink_boardex_ciq.boardex_ciq": pl.DataFrame(
            {
                "directorid": ["D1"],
                "directorname": ["Jane Director"],
                "boardid": ["B1"],
                "boardname": ["Alpha Corp Board"],
                "bd_ticker": ["ALP"],
                "isin": ["US0000000001"],
                "cikcode": ["1000001"],
                "title": ["Director"],
                "personid": ["P1"],
                "companyid": ["C1"],
                "companyname": ["Alpha Corp"],
                "firstname": ["Jane"],
                "middlename": [None],
                "lastname": ["Director"],
                "prefix": [None],
                "suffix": [None],
                "salutation": [None],
                "score": [0.99],
                "matchstyle": ["exact"],
            }
        ),
    }

    wrds_config = _wrds_test_config(tmp_path, geocode_path).model_copy(
        update={
            "boardex_people_table": WrdsTableConfig(
                enabled=True,
                required=False,
                library="boardex",
                table="na_wrds_dir_profile_all",
                columns=["directorid", "directorname"],
            ),
            "boardex_board_roles_table": WrdsTableConfig(
                enabled=True,
                required=False,
                library="boardex",
                table="na_wrds_dir_profile_all",
                columns=[
                    "directorid",
                    "directorname",
                    "companyid",
                    "companyname",
                    "datestartrole",
                    "dateendrole",
                    "startdate",
                    "enddate",
                    "title",
                    "rolename",
                    "brdposition",
                    "fulltextdescription",
                    "ned",
                    "leadershipteam",
                ],
                date_columns=["datestartrole", "dateendrole", "startdate", "enddate"],
            ),
            "boardex_employment_table": WrdsTableConfig(
                enabled=True,
                required=False,
                library="boardex",
                table="na_wrds_dir_profile_emp",
                columns=[
                    "directorid",
                    "directorname",
                    "companyid",
                    "companyname",
                    "datestartrole",
                    "dateendrole",
                    "rolename",
                    "brdposition",
                    "ned",
                    "leadershipteam",
                    "isin",
                ],
                date_columns=["datestartrole", "dateendrole"],
            ),
            "capiq_people_analytics_table": WrdsTableConfig(
                enabled=True,
                required=False,
                library="ciq_pplintel",
                table="wrds_professional",
                columns=[
                    "companyid",
                    "personid",
                    "proid",
                    "profunctionid",
                    "companyname",
                    "personname",
                    "profunctionname",
                    "yearfounded",
                    "yearborn",
                    "title",
                    "countryid",
                    "country",
                    "stateid",
                    "state",
                    "startday",
                    "startmonth",
                    "startyear",
                    "endday",
                    "endmonth",
                    "endyear",
                    "rank",
                    "prorank",
                    "boardrank",
                    "proflag",
                    "currentproflag",
                    "boardflag",
                    "currentboardflag",
                    "currentflag",
                    "keyexecflag",
                    "topkeyexecflag",
                    "advisorflag",
                    "graduateflag",
                    "dealmakerflag",
                    "sponsorflag",
                    "undergraduateflag",
                    "onlyoneflag",
                    "companyflag",
                    "hideflag",
                    "committeeid",
                    "gvkey",
                ],
            ),
            "boardex_ciq_link_table": WrdsTableConfig(
                enabled=True,
                required=False,
                library="wrdsapps_plink_boardex_ciq",
                table="boardex_ciq",
                columns=[
                    "directorid",
                    "directorname",
                    "boardid",
                    "boardname",
                    "bd_ticker",
                    "isin",
                    "cikcode",
                    "title",
                    "personid",
                    "companyid",
                    "companyname",
                    "firstname",
                    "middlename",
                    "lastname",
                    "prefix",
                    "suffix",
                    "salutation",
                    "score",
                    "matchstyle",
                ],
            ),
        }
    )

    config = load_config("examples/minimal_config.yml").model_copy(update={"wrds": wrds_config})
    manifest = pull_wrds_bundle(config, logger=configure_logging(), client=FakeWrdsClient(tables))

    canonical_dir = tmp_path / "canonical"
    assert (canonical_dir / "capiq_people_analytics.parquet").exists()
    assert (canonical_dir / "boardex_capiq_bridge.parquet").exists()
    assert (canonical_dir / "boardex_people.parquet").exists()
    assert (canonical_dir / "boardex_board_roles.parquet").exists()
    assert (canonical_dir / "boardex_employment.parquet").exists()

    bridge = pl.read_parquet(canonical_dir / "boardex_capiq_bridge.parquet")
    assert bridge.filter(pl.col("gvkey") == "001001").height == 1

    board_roles = pl.read_parquet(canonical_dir / "boardex_board_roles.parquet")
    assert board_roles.filter(pl.col("gvkey") == "001001").height == 1
    assert "boardex_capiq_bridge" in manifest["optional_outputs"]
