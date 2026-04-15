"""Project-wide constants used across the research pipeline."""

from __future__ import annotations

from enum import StrEnum


class ArtifactName(StrEnum):
    """Canonical artifact names written by pipeline stages."""

    FIRM_YEAR_PANEL = "firm_year_panel"
    EXECUTIVE_YEAR_PANEL = "executive_year_panel"
    CEO_YEAR_PANEL = "ceo_year_panel"
    SUCCESSION_EVENTS = "succession_events"
    INTERNAL_BENCH = "internal_bench"
    RELEASED_CANDIDATES = "released_candidates"
    RELEASE_SUPPLY_METRICS = "release_supply_metrics"
    CANDIDATE_UNIVERSE = "candidate_universe"
    ACCESSIBLE_CANDIDATE_SET = "event_candidate_set"
    FIT_EVENT_SUMMARY = "fit_event_summary"
    EVENT_ANALYSIS_PANEL = "event_analysis_panel"
    MAIN_RESULTS = "regression_results"
    IV_RESULTS = "iv_results"
    CHOICE_RESULTS = "choice_results"
    MODEL_METADATA = "model_metadata"
    TABLE_SUMMARIES = "table_summaries"


CORE_INPUT_TABLES: tuple[str, ...] = (
    "cri_exec_panel",
    "compustat_firm_year",
    "hq_history",
    "release_events",
    "noncompete_state_year",
    "ff_industry_map",
)

OPTIONAL_INPUT_TABLES: tuple[str, ...] = (
    "boardex_people",
    "boardex_board_roles",
    "boardex_employment",
    "execucomp",
    "travel_time_shocks",
    "blm_bridge",
    "crsp_daily",
    "ceo_announcement_dates",
    "tfp_inputs",
    "reviewed_person_crosswalk",
)

DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_OUTPUT_DIR = "artifacts"

DEFAULT_CEO_READY_REGEX: tuple[str, ...] = (
    r"\bchief executive officer\b",
    r"\bceo\b",
    r"\bpresident\b",
    r"\bchief operating officer\b",
    r"\bcoo\b",
)

DEFAULT_ROBUSTNESS_REGEX: tuple[str, ...] = (
    r"\bchief financial officer\b",
    r"\bcfo\b",
    r"\bchairman\b.*\bpresident\b",
)

TASK_ALIGNMENT_DIMENSIONS: tuple[str, ...] = (
    "innovation",
    "turnaround",
    "scale_complexity",
    "operational_discipline",
    "capital_discipline",
    "industry_familiarity",
)
