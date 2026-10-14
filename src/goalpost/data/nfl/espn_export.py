# Purpose: validate saved ESPN identities before matching imported NFL simulations.
# Public API: normalize(payload) returns the established schedule DataFrame columns.
# Consumers: WeeklyExport.load; this module reads supplied data and never fetches it.

from typing import Any

import pandas as pd

TEAMS: set[str] = set(
    (
        "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV "
        "MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS"
    ).split()
)
ALIASES: dict[str, str] = {"LAR": "LA", "WSH": "WAS"}
POSTSEASON_ROUNDS: dict[int, str] = {1: "WC", 2: "DIV", 3: "CON", 5: "SB"}
COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "espn_event_id",
    "source_status",
    "kickoff_utc",
]


# Purpose: reject ambiguous competitions before assigning home and away identities.
# Callers: _normalize_event; values remain provider JSON until validation below.
def _competitors_by_side(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    competitions = event["competitions"]
    if len(competitions) != 1:
        raise ValueError("Ambiguous ESPN competition")

    competitors = competitions[0]["competitors"]
    if len(competitors) != 2 or {team["homeAway"] for team in competitors} != {"home", "away"}:
        raise ValueError("Expected exact home and away competitors")

    return {competitor["homeAway"]: competitor for competitor in competitors}


# Purpose: translate ESPN postseason weeks into the export's NFL season numbering.
# Callers: _normalize_event; unsupported rounds must fail rather than become NFL games.
def _season_identity(event: dict[str, Any], payload: dict[str, Any]) -> tuple[int, int, str]:
    season_metadata = event.get("season", payload.get("season", {}))
    season = int(season_metadata["year"])
    season_type = int(season_metadata["type"])
    week = int(event.get("week", payload.get("week", {}))["number"])

    if season_type == 2:
        return season, week, "REG"
    if season_type == 1:
        return season, week, "PRE"
    if season_type != 3 or week not in POSTSEASON_ROUNDS:
        raise ValueError("Unsupported ESPN season/round")

    round_name = POSTSEASON_ROUNDS[week]
    regular_season_weeks = 18 if season >= 2021 else 17
    round_offset = ["WC", "DIV", "CON", "SB"].index(round_name) + 1
    return season, regular_season_weeks + round_offset, round_name


# Purpose: preserve provider orientation, kickoff timezone and scheduled-score semantics.
# Callers: normalize; returns one row without fetching or inferring missing identities.
def _normalize_event(event: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    competitors = _competitors_by_side(event)
    teams = {
        side: ALIASES.get(competitor["team"]["abbreviation"], competitor["team"]["abbreviation"])
        for side, competitor in competitors.items()
    }
    if any(team not in TEAMS for team in teams.values()) or teams["home"] == teams["away"]:
        raise ValueError("Unrecognized NFL team identity")

    season, week, game_type = _season_identity(event, payload)
    kickoff = pd.Timestamp(event["date"])
    if pd.isna(kickoff) or kickoff.tzinfo is None:
        raise ValueError("Missing timezone-aware kickoff")
    kickoff = kickoff.tz_convert("UTC")
    eastern_kickoff = kickoff.tz_convert("America/New_York")

    status = event["status"]["type"]
    status_name = status["name"]
    scheduled = (
        status_name == "STATUS_SCHEDULED"
        and status.get("state") == "pre"
        and status.get("completed") is False
    )
    # Scheduled 0-0 placeholders are not completed results.
    scores = {
        side: None if scheduled else pd.to_numeric(competitor.get("score"), errors="coerce")
        for side, competitor in competitors.items()
    }
    return {
        "game_id": f"{season}_{week:02d}_{teams['away']}_{teams['home']}",
        "season": season,
        "week": week,
        "game_type": game_type,
        "gameday": eastern_kickoff.strftime("%Y-%m-%d"),
        "gametime": eastern_kickoff.strftime("%H:%M"),
        "home_team": teams["home"],
        "away_team": teams["away"],
        "home_score": scores["home"],
        "away_score": scores["away"],
        "espn_event_id": str(event["id"]),
        "source_status": "scheduled" if scheduled else status_name,
        "kickoff_utc": kickoff.isoformat(),
    }


# Purpose: reject duplicate provider and normalized game identities in a saved slate.
# Callers: WeeklyExport.load and offline export validation tests.
def normalize(payload: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("ESPN response must contain an events list")

    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for event in payload["events"]:
        event_id = str(event["id"])
        if not event_id.isdigit() or event_id in event_ids:
            raise ValueError("Invalid or duplicate ESPN event ID")
        event_ids.add(event_id)
        rows.append(_normalize_event(event, payload))

    schedule = pd.DataFrame(rows, columns=COLUMNS)
    if schedule.game_id.duplicated().any():
        raise ValueError("Multiple ESPN events map to the same NFL game ID")
    return schedule
