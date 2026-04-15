"""Accessible candidate-set construction for event-level successor choice."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import polars as pl

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.geography.distance import haversine_km


def _latest_universe_row(
    universe_by_person: dict[str, list[dict[str, object]]],
    person_id: str,
    *,
    max_year: int,
    preferred_gvkey: str | None = None,
) -> dict[str, object] | None:
    rows = [row for row in universe_by_person.get(person_id, []) if int(row["candidate_year"]) <= max_year]
    if preferred_gvkey is not None:
        preferred = [row for row in rows if row["current_gvkey"] == preferred_gvkey]
        if preferred:
            rows = preferred
    if not rows:
        return None
    return max(rows, key=lambda row: int(row["candidate_year"]))


def _build_board_tie_pairs(
    succession_events: pl.DataFrame,
    board_roles: pl.DataFrame,
) -> set[tuple[str, str]]:
    if board_roles.height == 0:
        return set()
    pairs: set[tuple[str, str]] = set()
    for event in succession_events.to_dicts():
        cutoff = event["announcement_date"] or date(int(event["succession_year"]), 6, 30)
        candidate_rows = board_roles.filter(
            (pl.col("gvkey") == event["gvkey"])
            & (pl.col("role_start_date").is_null() | (pl.col("role_start_date") <= cutoff))
            & (pl.col("role_end_date").is_null() | (pl.col("role_end_date") >= cutoff))
        )
        for row in candidate_rows.to_dicts():
            pairs.add((str(event["event_id"]), str(row["person_id"])))
    return pairs


def _build_employment_tie_pairs(
    succession_events: pl.DataFrame,
    board_roles: pl.DataFrame,
    board_employment: pl.DataFrame,
) -> set[tuple[str, str]]:
    if board_roles.height == 0 or board_employment.height == 0:
        return set()
    pairs: set[tuple[str, str]] = set()
    board_role_rows = board_roles.to_dicts()
    employment_rows = board_employment.to_dicts()
    for event in succession_events.to_dicts():
        cutoff = event["announcement_date"] or date(int(event["succession_year"]), 6, 30)
        active_directors = [
            row
            for row in board_role_rows
            if row["gvkey"] == event["gvkey"]
            and (row["role_start_date"] is None or row["role_start_date"] <= cutoff)
            and (row["role_end_date"] is None or row["role_end_date"] >= cutoff)
        ]
        director_ids = {str(row["person_id"]) for row in active_directors}
        director_employment = [row for row in employment_rows if str(row["person_id"]) in director_ids]
        for candidate_row in employment_rows:
            candidate_person_id = str(candidate_row["person_id"])
            if candidate_person_id in director_ids:
                continue
            if candidate_row["start_date"] is not None and candidate_row["start_date"] > cutoff:
                continue
            for director_row in director_employment:
                if candidate_row["gvkey"] != director_row["gvkey"] or candidate_row["gvkey"] is None:
                    continue
                candidate_end = candidate_row["end_date"] or cutoff
                director_end = director_row["end_date"] or cutoff
                candidate_start = candidate_row["start_date"] or date(1900, 1, 1)
                director_start = director_row["start_date"] or date(1900, 1, 1)
                if max(candidate_start, director_start) <= min(candidate_end, director_end, cutoff):
                    pairs.add((str(event["event_id"]), candidate_person_id))
                    break
    return pairs


def _event_reference_date(event: dict[str, object]) -> date:
    announcement_date = event.get("announcement_date")
    if announcement_date is not None:
        return announcement_date
    return date(int(event["succession_year"]), 6, 30)


def _distance_miles(source: dict[str, object], target: dict[str, object]) -> float | None:
    if None in (source["lat"], source["lon"], target["lat"], target["lon"]):
        return None
    return haversine_km(
        float(source["lat"]),
        float(source["lon"]),
        float(target["lat"]),
        float(target["lon"]),
    ) / 1.609344


def build_accessible_candidate_set(
    succession_events: pl.DataFrame,
    executive_year_panel: pl.DataFrame,
    released_candidates: pl.DataFrame,
    candidate_universe: pl.DataFrame,
    firm_year_panel: pl.DataFrame,
    config: WhoGetsConsideredConfig,
    *,
    boardex_board_roles: pl.DataFrame | None = None,
    boardex_employment: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Construct the accessible candidate set A_e = insiders U released outsiders."""

    universe_by_person: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidate_universe.to_dicts():
        universe_by_person[str(row["candidate_person_id"])].append(row)
    board_roles = boardex_board_roles if boardex_board_roles is not None else pl.DataFrame()
    board_employment = boardex_employment if boardex_employment is not None else pl.DataFrame()
    board_tie_pairs = _build_board_tie_pairs(succession_events, board_roles) if config.features.boardex_enabled else set()
    employment_tie_pairs = (
        _build_employment_tie_pairs(succession_events, board_roles, board_employment)
        if config.features.boardex_enabled
        else set()
    )
    event_locations = {
        (row["gvkey"], int(row["fyear"])): row
        for row in firm_year_panel.select(
            "gvkey",
            "fyear",
            "lat",
            "lon",
            "state",
            "msa_code",
            "ff10",
            "ff49",
        ).to_dicts()
    }
    exec_rows = executive_year_panel.to_dicts()
    rows: list[dict[str, object]] = []

    for event in succession_events.to_dicts():
        event_id = str(event["event_id"])
        succession_year = int(event["succession_year"])
        event_date = _event_reference_date(event)
        focal = event_locations[(str(event["gvkey"]), succession_year)]
        candidate_map: dict[str, dict[str, object]] = {}

        for record in exec_rows:
            if (
                record["gvkey"] == event["gvkey"]
                and int(record["fyear"]) == succession_year - 1
                and bool(record["is_ceo_ready"])
                and record["person_id"] != event["predecessor_person_id"]
            ):
                person_id = str(record["person_id"])
                universe_row = _latest_universe_row(
                    universe_by_person,
                    person_id,
                    max_year=succession_year - 1,
                    preferred_gvkey=str(event["gvkey"]),
                )
                if universe_row is None:
                    continue
                board_tie_flag = int((event_id, person_id) in board_tie_pairs)
                employment_tie_flag = int((event_id, person_id) in employment_tie_pairs)
                candidate_map[person_id] = {
                    "event_id": event_id,
                    "candidate_person_id": person_id,
                    "internal_flag": 1,
                    "internal_candidate_flag": 1,
                    "released_flag": 0,
                    "released_candidate_flag": 0,
                    "source_market": "internal",
                    "distance_miles_if": 0.0,
                    "chosen_flag": int(person_id == event["successor_person_id"]),
                    "board_tie_flag": board_tie_flag,
                    "employment_tie_flag": employment_tie_flag,
                    "known_to_board_flag": int(board_tie_flag or employment_tie_flag),
                    **universe_row,
                }

        for record in released_candidates.to_dicts():
            day_gap = (event_date - record["release_event_date"]).days
            if day_gap <= 0 or day_gap > 730:
                continue
            if record["source_ff10"] == focal["ff10"]:
                continue
            source = {
                "lat": record["source_hq_lat"],
                "lon": record["source_hq_lon"],
            }
            target = {"lat": focal["lat"], "lon": focal["lon"]}
            distance_miles = _distance_miles(source, target)
            if distance_miles is None or distance_miles > 60.0:
                continue
            person_id = str(record["candidate_person_id"])
            universe_row = _latest_universe_row(
                universe_by_person,
                person_id,
                max_year=int(record["candidate_year"]),
                preferred_gvkey=str(record["source_gvkey"]),
            )
            if universe_row is None:
                continue
            board_tie_flag = int((event_id, person_id) in board_tie_pairs)
            employment_tie_flag = int((event_id, person_id) in employment_tie_pairs)
            candidate_map[person_id] = {
                **candidate_map.get(person_id, {}),
                "event_id": event_id,
                "candidate_person_id": person_id,
                "internal_flag": int(candidate_map.get(person_id, {}).get("internal_flag", 0)),
                "internal_candidate_flag": int(candidate_map.get(person_id, {}).get("internal_candidate_flag", 0)),
                "released_flag": 1,
                "released_candidate_flag": 1,
                "source_market": "released_60mi_outind",
                "distance_miles_if": distance_miles,
                "chosen_flag": int(person_id == event["successor_person_id"]),
                "board_tie_flag": board_tie_flag,
                "employment_tie_flag": employment_tie_flag,
                "known_to_board_flag": int(board_tie_flag or employment_tie_flag),
                **universe_row,
            }

        if config.features.optional_external_active_enabled:
            for record in exec_rows:
                if (
                    int(record["fyear"]) != succession_year - 1
                    or record["gvkey"] == event["gvkey"]
                    or not bool(record["is_ceo_ready"])
                ):
                    continue
                source = event_locations.get((str(record["gvkey"]), succession_year - 1))
                if source is None:
                    continue
                distance_miles = _distance_miles(source, focal)
                if distance_miles is None or distance_miles > 60.0:
                    continue
                person_id = str(record["person_id"])
                universe_row = _latest_universe_row(
                    universe_by_person,
                    person_id,
                    max_year=succession_year - 1,
                    preferred_gvkey=str(record["gvkey"]),
                )
                if universe_row is None:
                    continue
                board_tie_flag = int((event_id, person_id) in board_tie_pairs)
                employment_tie_flag = int((event_id, person_id) in employment_tie_pairs)
                candidate_map[person_id] = {
                    **candidate_map.get(person_id, {}),
                    "event_id": event_id,
                    "candidate_person_id": person_id,
                    "internal_flag": int(candidate_map.get(person_id, {}).get("internal_flag", 0)),
                    "internal_candidate_flag": int(candidate_map.get(person_id, {}).get("internal_candidate_flag", 0)),
                    "released_flag": int(candidate_map.get(person_id, {}).get("released_flag", 0)),
                    "released_candidate_flag": int(candidate_map.get(person_id, {}).get("released_candidate_flag", 0)),
                    "source_market": "active_external_60mi",
                    "distance_miles_if": distance_miles,
                    "chosen_flag": int(person_id == event["successor_person_id"]),
                    "board_tie_flag": board_tie_flag,
                    "employment_tie_flag": employment_tie_flag,
                    "known_to_board_flag": int(board_tie_flag or employment_tie_flag),
                    **universe_row,
                }

        chosen_id = str(event["successor_person_id"])
        if chosen_id not in candidate_map:
            chosen_universe = _latest_universe_row(universe_by_person, chosen_id, max_year=succession_year - 1)
            if chosen_universe is None:
                chosen_universe = {
                    "candidate_person_id": chosen_id,
                    "candidate_year": succession_year - 1,
                    "current_gvkey": event["gvkey"],
                    "current_title_raw": event.get("successor_source_title"),
                    "current_title_seniority_score": 0,
                    "current_state": focal["state"],
                    "current_msa_code": focal["msa_code"],
                    "current_lat": focal["lat"],
                    "current_lon": focal["lon"],
                    "is_ceo": 0,
                    "is_president": 0,
                    "is_coo": 0,
                    "is_cfo": 0,
                    "prior_public_ceo_flag": 0,
                    "years_as_public_ceo": 0,
                    "num_prior_public_firms": 0,
                    "num_prior_industries": 0,
                    "mover_flag": 0,
                    "avg_prior_firm_log_assets": 0.0,
                    "avg_prior_firm_roa": 0.0,
                    "avg_prior_firm_tobin_q": 0.0,
                    "avg_prior_firm_rd_intensity": 0.0,
                    "avg_prior_firm_leverage": 0.0,
                    "avg_prior_firm_capital_intensity": 0.0,
                    "public_market_tenure_years": 0,
                    "portable_quality_score": 0.0,
                }
            source_loc = event_locations.get(
                (str(chosen_universe["current_gvkey"]), int(chosen_universe["candidate_year"]))
            )
            distance_miles = _distance_miles(source_loc or focal, focal) if source_loc is not None else None
            board_tie_flag = int((event_id, chosen_id) in board_tie_pairs)
            employment_tie_flag = int((event_id, chosen_id) in employment_tie_pairs)
            candidate_map[chosen_id] = {
                "event_id": event_id,
                "candidate_person_id": chosen_id,
                "internal_flag": int(chosen_universe["current_gvkey"] == event["gvkey"]),
                "internal_candidate_flag": int(chosen_universe["current_gvkey"] == event["gvkey"]),
                "released_flag": 0,
                "released_candidate_flag": 0,
                "source_market": "chosen_injected",
                "distance_miles_if": distance_miles,
                "chosen_flag": 1,
                "board_tie_flag": board_tie_flag,
                "employment_tie_flag": employment_tie_flag,
                "known_to_board_flag": int(board_tie_flag or employment_tie_flag),
                "chosen_out_of_market_flag": 1,
                **chosen_universe,
            }

        rows.extend(candidate_map.values())

    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("internal_flag").fill_null(0).cast(pl.Int8),
            pl.col("internal_candidate_flag").fill_null(0).cast(pl.Int8),
            pl.col("released_flag").fill_null(0).cast(pl.Int8),
            pl.col("released_candidate_flag").fill_null(0).cast(pl.Int8),
            pl.col("chosen_flag").fill_null(0).cast(pl.Int8),
            pl.col("board_tie_flag").fill_null(0).cast(pl.Int8),
            pl.col("employment_tie_flag").fill_null(0).cast(pl.Int8),
            pl.col("known_to_board_flag").fill_null(0).cast(pl.Int8),
            pl.col("chosen_out_of_market_flag").fill_null(0).cast(pl.Int8),
            pl.col("distance_miles_if").fill_null(0.0),
            pl.col("portable_quality_score").fill_null(0.0).alias("portable_quality_z_i"),
        )
        .unique(subset=["event_id", "candidate_person_id"], keep="first")
    )
