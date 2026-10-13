"""Load standard weekly exports into NFLGame simulation outputs."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from goalpost.transitions.espn_schedule import normalize

from .nfl import NFLGame, ScoreDistribution


def timestamp(value: str) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return instant


@dataclass
class WeeklyExport:
    games: list[NFLGame]
    report: dict
    unavailable: list[dict]

    @classmethod
    def load(cls, directory: Path) -> "WeeklyExport":
        directory = directory.resolve()
        report = json.loads((directory / "report.json").read_text())
        if report.get("regulation_only") is not True:
            raise ValueError("Only explicitly regulation-scoped exports are supported")
        if report.get("status") not in ("complete", "incomplete", "no_scheduled_matchups"):
            raise ValueError("Simulation export is unfinished or failed")
        with (directory / "schedule.csv").open(newline="") as handle:
            schedule = list(csv.DictReader(handle))
        raw_schedule = json.loads((directory / "schedule-espn.json").read_text())
        normalized_schedule = normalize(raw_schedule).to_dict("records")
        events = raw_schedule["events"]
        event_ids = [str(event["id"]) for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Duplicate ESPN event identity")
        by_event = dict(zip(event_ids, events))
        games, unavailable, seen = [], [], set()
        for result in report["results"]:
            identity = result["game"]
            game_id = identity["game_id"]
            if not isinstance(game_id, str) or game_id in ("", ".", ".."):
                raise ValueError("Invalid game directory")
            if Path(game_id).name != game_id or "\\" in game_id:
                raise ValueError("Unsafe game directory")
            if game_id in seen:
                raise ValueError("Duplicate simulation game identity")
            seen.add(game_id)
            matches = [row for row in schedule if row["game_id"] == game_id]
            if len(matches) != 1:
                raise ValueError("Simulation game needs one schedule identity")
            scheduled = matches[0]
            if type(identity.get("season")) is not int or identity["season"] != int(
                scheduled["season"]
            ):
                raise ValueError("Simulation and schedule seasons differ")
            provider_matches = [
                row
                for row in normalized_schedule
                if row["espn_event_id"] == scheduled["espn_event_id"]
            ]
            if len(provider_matches) != 1 or any(
                provider_matches[0][field] != scheduled[field]
                for field in ("game_id", "home_team", "away_team")
            ):
                raise ValueError("Provider event and exported schedule identity differ")
            kickoff = timestamp(identity["kickoff_utc"])
            if (identity["home"], identity["away"], kickoff) != (
                scheduled["home_team"],
                scheduled["away_team"],
                timestamp(scheduled["kickoff_utc"]),
            ):
                raise ValueError("Simulation and schedule orientation/kickoff differ")
            # Join through the exported provider event ID, not hard-coded abbreviation aliases.
            event = by_event[scheduled["espn_event_id"]]
            if len(event["competitions"]) != 1 or timestamp(event["date"]) != kickoff:
                raise ValueError("Ambiguous competition or mismatched kickoff")
            competitors = event["competitions"][0]["competitors"]
            if len(competitors) != 2 or {team["homeAway"] for team in competitors} != {
                "home",
                "away",
            }:
                raise ValueError("Missing or ambiguous home/away identity")
            names = {team["homeAway"]: team["team"]["displayName"] for team in competitors}
            attempted, retained = result["attempted"], result.get("n", 0)
            approximate = result.get("diagnostics", {}).get("grace_completed_games", 0)
            counts = (attempted, retained, approximate)
            if (
                any(type(count) is not int for count in counts)
                or not 0 <= approximate <= retained <= attempted
                or attempted <= 0
            ):
                raise ValueError("Invalid simulation denominators")
            if result["status"] != "simulated":
                unavailable.append(
                    {
                        "game_id": game_id,
                        "status": result["status"],
                        "attempted": attempted,
                        "retained": retained,
                        "approximate": approximate,
                    }
                )
                continue
            folder = directory / game_id
            paths = [folder / "total-pmf.csv", folder / "home_minus_away-pmf.csv"]
            if any(not path.resolve().is_relative_to(directory) for path in paths):
                raise ValueError("PMF outside export directory")
            games.append(
                NFLGame(
                    game_id=game_id,
                    home=names["home"],
                    away=names["away"],
                    kickoff=kickoff,
                    total=ScoreDistribution.from_csv(paths[0], retained),
                    home_margin=ScoreDistribution.from_csv(paths[1], retained),
                    attempted=attempted,
                    retained=retained,
                    approximate=approximate,
                )
            )
        return cls(games, report, unavailable)
