"""Typed configuration models governing data access, identification, and estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from whogetsconsidered.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_OUTPUT_DIR


class InputPaths(BaseModel):
    """Canonical cleaned input extracts used by the package."""

    model_config = ConfigDict(extra="forbid")

    cri_exec_panel: Path
    compustat_firm_year: Path
    hq_history: Path
    release_events: Path
    noncompete_state_year: Path
    ff_industry_map: Path
    boardex_people: Path | None = None
    boardex_board_roles: Path | None = None
    boardex_employment: Path | None = None
    execucomp: Path | None = None
    travel_time_shocks: Path | None = None
    blm_bridge: Path | None = None
    crsp_daily: Path | None = None
    ceo_announcement_dates: Path | None = None
    tfp_inputs: Path | None = None
    reviewed_person_crosswalk: Path | None = None


class WrdsTableConfig(BaseModel):
    """Configuration for a single WRDS extraction target."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    required: bool = False
    library: str
    table: str
    columns: list[str] = Field(default_factory=list)
    sql: str | None = None
    date_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query_mode(self) -> "WrdsTableConfig":
        """Require either explicit columns or SQL for a WRDS pull."""

        if not self.columns and self.sql is None:
            raise ValueError("WRDS table config requires either columns or sql")
        return self


def _default_wrds_compustat_fundamentals() -> WrdsTableConfig:
    """Default WRDS pull for Compustat annual fundamentals."""

    return WrdsTableConfig(
        enabled=True,
        required=True,
        library="comp",
        table="funda",
        columns=[
            "gvkey",
            "fyear",
            "datadate",
            "at",
            "ebit",
            "xrd",
            "capx",
            "dltt",
            "dlc",
            "dv",
            "prcc_f",
            "csho",
            "sale",
            "ceq",
            "indfmt",
            "datafmt",
            "consol",
            "popsrc",
        ],
        date_columns=["datadate"],
    )


def _default_wrds_company_reference() -> WrdsTableConfig:
    """Default WRDS pull for firm reference and address metadata."""

    return WrdsTableConfig(
        enabled=True,
        required=True,
        library="comp",
        table="company",
        columns=["gvkey", "conm", "city", "state", "addzip", "sic"],
    )


def _default_wrds_execucomp() -> WrdsTableConfig:
    """Default WRDS pull for Execucomp annual compensation / executive roster data."""

    return WrdsTableConfig(
        enabled=True,
        required=False,
        library="execcomp",
        table="anncomp",
        columns=["gvkey", "year", "execid", "exec_fullname", "titleann", "execrank", "pceo"],
    )


def _default_wrds_ccm_link() -> WrdsTableConfig:
    """Default WRDS pull for CRSP-Compustat linking."""

    return WrdsTableConfig(
        enabled=True,
        required=False,
        library="crsp",
        table="ccmxpf_linktable",
        sql=(
            "select gvkey, lpermno as permno, lpermco as permco, "
            "linktype, linkprim, linkdt, linkenddt, usedflag "
            "from crsp.ccmxpf_linktable"
        ),
        date_columns=["linkdt", "linkenddt"],
    )


def _default_wrds_crsp_delist() -> WrdsTableConfig:
    """Default WRDS pull for CRSP delisting events."""

    return WrdsTableConfig(
        enabled=True,
        required=False,
        library="crsp",
        table="dsedelist",
        columns=["permno", "dlstdt", "dlstcd", "nwperm", "nextdt"],
        date_columns=["dlstdt", "nextdt"],
    )


def _default_wrds_boardex_people() -> WrdsTableConfig:
    """Default WRDS pull for BoardEx people metadata when licensed."""

    return WrdsTableConfig(
        enabled=False,
        required=False,
        library="boardex",
        table="na_wrds_dir_profile_all",
        columns=["directorid", "directorname"],
    )


def _default_wrds_boardex_board_roles() -> WrdsTableConfig:
    """Default WRDS pull for BoardEx board-role records when licensed."""

    return WrdsTableConfig(
        enabled=False,
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
    )


def _default_wrds_boardex_employment() -> WrdsTableConfig:
    """Default WRDS pull for BoardEx employment histories when licensed."""

    return WrdsTableConfig(
        enabled=False,
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
    )


def _default_wrds_capiq_people_analytics() -> WrdsTableConfig:
    """Default WRDS pull for Capital IQ People Intelligence analytics."""

    return WrdsTableConfig(
        enabled=False,
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
    )


def _default_wrds_boardex_ciq_link() -> WrdsTableConfig:
    """Default WRDS pull for the WRDS BoardEx-to-Capital IQ people bridge."""

    return WrdsTableConfig(
        enabled=False,
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
    )


class WrdsPullConfig(BaseModel):
    """Optional WRDS bootstrap settings for staging and canonical input generation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    username: str | None = None
    pgpass_file: Path | None = None
    row_limit: int | None = Field(default=None, ge=1)
    staging_dir: Path = Path("artifacts/wrds/raw")
    canonical_dir: Path = Path("artifacts/wrds/canonical")
    manifest_path: Path = Path("artifacts/wrds/manifest.json")
    hq_geocode_crosswalk: Path | None = None
    build_cri_proxy_from_execucomp: bool = True
    build_release_events_from_crsp: bool = True
    compustat_fundamentals: WrdsTableConfig = Field(default_factory=_default_wrds_compustat_fundamentals)
    company_reference: WrdsTableConfig = Field(default_factory=_default_wrds_company_reference)
    execucomp_anncomp: WrdsTableConfig = Field(default_factory=_default_wrds_execucomp)
    ccm_link_table: WrdsTableConfig = Field(default_factory=_default_wrds_ccm_link)
    crsp_delist_table: WrdsTableConfig = Field(default_factory=_default_wrds_crsp_delist)
    boardex_people_table: WrdsTableConfig = Field(default_factory=_default_wrds_boardex_people)
    boardex_board_roles_table: WrdsTableConfig = Field(default_factory=_default_wrds_boardex_board_roles)
    boardex_employment_table: WrdsTableConfig = Field(default_factory=_default_wrds_boardex_employment)
    capiq_people_analytics_table: WrdsTableConfig = Field(default_factory=_default_wrds_capiq_people_analytics)
    boardex_ciq_link_table: WrdsTableConfig = Field(default_factory=_default_wrds_boardex_ciq_link)


class MarketConfig(BaseModel):
    """Geographic feasible-set settings used for candidate access."""

    model_config = ConfigDict(extra="forbid")

    definition: str = Field(default="radius")
    radius_miles: float = Field(default=60.0, gt=0)
    robustness_radii_miles: list[float] = Field(default_factory=lambda: [100.0])
    same_state_fallback: bool = False

    @field_validator("definition")
    @classmethod
    def validate_definition(cls, value: str) -> str:
        """Restrict market definitions to supported options."""

        supported = {"msa", "radius", "state"}
        if value not in supported:
            raise ValueError(f"market definition must be one of {sorted(supported)}")
        return value


class TitlesConfig(BaseModel):
    """Rules that determine which titles count as CEO-ready or CEO-identifying."""

    model_config = ConfigDict(extra="forbid")

    ceo_rule: str = "ceo_or_chair_president"
    baseline_ceo_ready_titles: list[str] = Field(
        default_factory=lambda: ["ceo", "president", "coo"]
    )
    include_cfo_in_robustness: bool = True
    include_high_seniority_regex: list[str] = Field(default_factory=list)
    exclude_interim_in_outcomes: bool = True


class FeatureFlags(BaseModel):
    """Optional modules that can be switched on without blocking the core pipeline."""

    model_config = ConfigDict(extra="forbid")

    boardex_enabled: bool = False
    predictive_fit_enabled: bool = False
    iv_enabled: bool = False
    travel_time_enabled: bool = False
    conditional_logit_enabled: bool = False
    blm_bridge_enabled: bool = False
    optional_external_active_enabled: bool = False


class ResidualizationConfig(BaseModel):
    """Residualization options for firm outcomes and post-succession performance."""

    model_config = ConfigDict(extra="forbid")

    year_industry: bool = True
    year_size_quartile: bool = False
    year_state: bool = False


class RegressionConfig(BaseModel):
    """Econometric specification controls."""

    model_config = ConfigDict(extra="forbid")

    release_window_years: int = Field(default=2, ge=1, le=5)
    fixed_effects: list[str] = Field(default_factory=lambda: ["market_year", "industry_year"])
    cluster_by: list[str] = Field(default_factory=lambda: ["market"])
    outcome_horizons: list[int] = Field(default_factory=lambda: [1, 2, 3])
    use_time_ordered_folds: bool = False
    crossfit_folds: int = Field(default=5, ge=2, le=10)
    random_seed: int = 20260414


class PathsConfig(BaseModel):
    """Filesystem locations for artifacts and rendered outputs."""

    model_config = ConfigDict(extra="forbid")

    artifacts_dir: Path = Path(DEFAULT_ARTIFACT_DIR)
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)


class SampleConfig(BaseModel):
    """Base sample restrictions for firm-year and event-level analysis."""

    model_config = ConfigDict(extra="forbid")

    start_year: int = 1990
    end_year: int = 2018
    min_assets: float = 10.0
    exclude_financials: bool = True
    exclude_utilities: bool = True
    exclude_public_sector: bool = True
    require_hq_coordinates: bool = True


class WhoGetsConsideredConfig(BaseModel):
    """Top-level package configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "whogetsconsidered"
    inputs: InputPaths
    paths: PathsConfig = Field(default_factory=PathsConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    titles: TitlesConfig = Field(default_factory=TitlesConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    residualization: ResidualizationConfig = Field(default_factory=ResidualizationConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    wrds: WrdsPullConfig = Field(default_factory=WrdsPullConfig)

    @model_validator(mode="after")
    def validate_optional_dependencies(self) -> "WhoGetsConsideredConfig":
        """Require paths for enabled optional modules."""

        if self.features.boardex_enabled:
            required = [
                self.inputs.boardex_people,
                self.inputs.boardex_board_roles,
                self.inputs.boardex_employment,
            ]
            if any(path is None for path in required):
                raise ValueError("BoardEx paths must be supplied when boardex_enabled=true")
        if self.features.travel_time_enabled and self.inputs.travel_time_shocks is None:
            raise ValueError("travel_time_shocks path is required when travel_time_enabled=true")
        if self.features.blm_bridge_enabled and self.inputs.blm_bridge is None:
            raise ValueError("blm_bridge path is required when blm_bridge_enabled=true")
        return self


def load_config(path: str | Path) -> WhoGetsConsideredConfig:
    """Load a YAML config into a strongly typed configuration object."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return WhoGetsConsideredConfig.model_validate(payload)
