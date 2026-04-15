"""Deterministic title parsing for CEO identification and CEO-ready status."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import polars as pl

from whogetsconsidered.config import TitlesConfig


CEO_RE = re.compile(r"\bchief executive officer\b|\bceo\b")
CHAIR_RE = re.compile(r"\bchair(man|woman|person)?\b")
PRESIDENT_RE = re.compile(r"\bpresident\b")
COO_RE = re.compile(r"\bchief operating officer\b|\bcoo\b")
CFO_RE = re.compile(r"\bchief financial officer\b|\bcfo\b")
INTERIM_RE = re.compile(r"\binterim\b|\bacting\b")
DIVISION_RE = re.compile(r"\bdivision\b|\bsegment\b|\bgroup president\b")
FOUNDER_RE = re.compile(r"\bfounder\b|\bco-founder\b")


@dataclass(frozen=True)
class TitleFlags:
    """Economic role indicators derived from a raw title string."""

    normalized_title: str
    is_ceo: bool
    is_chair: bool
    is_president: bool
    is_coo: bool
    is_cfo: bool
    is_interim: bool
    is_division_head: bool
    is_founder: bool


def parse_title_flags(title_raw: str) -> TitleFlags:
    """Parse a raw title into deterministic executive-role flags."""

    normalized = " ".join(title_raw.casefold().split())
    return TitleFlags(
        normalized_title=normalized,
        is_ceo=bool(CEO_RE.search(normalized)),
        is_chair=bool(CHAIR_RE.search(normalized)),
        is_president=bool(PRESIDENT_RE.search(normalized)),
        is_coo=bool(COO_RE.search(normalized)),
        is_cfo=bool(CFO_RE.search(normalized)),
        is_interim=bool(INTERIM_RE.search(normalized)),
        is_division_head=bool(DIVISION_RE.search(normalized)),
        is_founder=bool(FOUNDER_RE.search(normalized)),
    )


def _extra_regex_match(title: str, extra_patterns: list[str]) -> bool:
    return any(re.search(pattern, title) is not None for pattern in extra_patterns)


def title_seniority_score(flags: TitleFlags) -> int:
    """Assign a transparent title-seniority score used for CEO fallback ordering."""

    if flags.is_ceo:
        return 5
    if flags.is_chair and flags.is_president:
        return 4
    if flags.is_president:
        return 3
    if flags.is_coo:
        return 2
    if flags.is_cfo:
        return 1
    return 0


def add_title_features(df: pl.DataFrame, config: TitlesConfig) -> pl.DataFrame:
    """Add deterministic title flags and CEO-ready indicators to an executive panel."""

    unique_titles = df.select("title_raw").unique().sort("title_raw")
    parsed_rows: list[dict[str, object]] = []
    for title in unique_titles["title_raw"].to_list():
        flags = parse_title_flags(str(title))
        parsed = asdict(flags)
        parsed["title_raw"] = title
        parsed["title_seniority_score"] = title_seniority_score(flags)
        parsed["is_ceo_rule_candidate"] = flags.is_ceo or (flags.is_chair and flags.is_president)
        parsed["is_ceo_ready"] = flags.is_ceo or flags.is_president or flags.is_coo
        parsed["is_ceo_ready_robust"] = (
            parsed["is_ceo_ready"]
            or (config.include_cfo_in_robustness and flags.is_cfo)
            or (flags.is_chair and flags.is_president)
            or _extra_regex_match(flags.normalized_title, config.include_high_seniority_regex)
        )
        parsed_rows.append(parsed)
    parsed_df = pl.DataFrame(parsed_rows)
    return df.join(parsed_df, on="title_raw", how="left")
