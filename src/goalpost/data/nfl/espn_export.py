"""Normalize saved ESPN identity evidence for imported simulation exports; no fetching."""

import pandas as pd

TEAMS = set(
    (
        "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV "
        "MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS"
    ).split()
)
ALIASES = {"LAR": "LA", "WSH": "WAS"}
POST = {1: "WC", 2: "DIV", 3: "CON", 5: "SB"}
COLUMNS = [
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


def normalize(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("ESPN response must contain an events list")
    rows = []
    ids = set()
    for event in payload["events"]:
        eid = str(event["id"])
        if not eid.isdigit() or eid in ids:
            raise ValueError("Invalid or duplicate ESPN event ID")
        ids.add(eid)
        competitions = event["competitions"]
        if len(competitions) != 1:
            raise ValueError("Ambiguous ESPN competition")
        competitors = competitions[0]["competitors"]
        if len(competitors) != 2 or {x["homeAway"] for x in competitors} != {"home", "away"}:
            raise ValueError("Expected exact home and away competitors")
        by_side = {x["homeAway"]: x for x in competitors}
        teams = {
            side: ALIASES.get(x["team"]["abbreviation"], x["team"]["abbreviation"])
            for side, x in by_side.items()
        }
        if any(t not in TEAMS for t in teams.values()) or teams["home"] == teams["away"]:
            raise ValueError("Unrecognized NFL team identity")
        season = int(event.get("season", payload.get("season", {}))["year"])
        st = int(event.get("season", payload.get("season", {}))["type"])
        week = int(event.get("week", payload.get("week", {}))["number"])
        if st == 2:
            kind = "REG"
        elif st == 3 and week in POST:
            kind = POST[week]
            week = (18 if season >= 2021 else 17) + ["WC", "DIV", "CON", "SB"].index(kind) + 1
        elif st == 1:
            kind = "PRE"
        else:
            # Pro Bowl and unknown season types are not silently NFL matchups.
            raise ValueError("Unsupported ESPN season/round")
        kickoff = pd.Timestamp(event["date"])
        if pd.isna(kickoff) or kickoff.tzinfo is None:
            raise ValueError("Missing timezone-aware kickoff")
        kickoff = kickoff.tz_convert("UTC")
        local = kickoff.tz_convert("America/New_York")
        status = event["status"]["type"]
        name = status["name"]
        # Only explicitly scheduled events are candidates, including 0-0 placeholders.
        scheduled = (
            name == "STATUS_SCHEDULED"
            and status.get("state") == "pre"
            and status.get("completed") is False
        )
        scores = {
            side: None if scheduled else pd.to_numeric(x.get("score"), errors="coerce")
            for side, x in by_side.items()
        }
        rows.append(
            dict(
                game_id=f"{season}_{week:02d}_{teams['away']}_{teams['home']}",
                season=season,
                week=week,
                game_type=kind,
                gameday=local.strftime("%Y-%m-%d"),
                gametime=local.strftime("%H:%M"),
                home_team=teams["home"],
                away_team=teams["away"],
                home_score=scores["home"],
                away_score=scores["away"],
                espn_event_id=eid,
                source_status="scheduled" if scheduled else name,
                kickoff_utc=kickoff.isoformat(),
            )
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.game_id.duplicated().any():
        raise ValueError("Multiple ESPN events map to the same NFL game ID")
    return frame
