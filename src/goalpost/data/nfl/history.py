# Purpose: check reported simulation donors against the maintained NFLverse schedule.
# Public API: NFLHistory.check loads evidence; assess accepts an existing DataFrame.
# Consumers: MarketTableJob gates quote acquisition on completed-game donor coverage.

from datetime import datetime
from typing import Any

import nflreadpy
import pandas as pd

from goalpost.data.base.odds import timestamp


# Purpose: keep stale donor coverage from being mistaken for current-season modeling.
# Consumers: the shared CLI/TUI report job; this is a coverage check, not calibration.
class NFLHistory:
    # Purpose: fetch schedule evidence only when imported simulations are still upcoming.
    # Callers: MarketTableJob.run; failures stop the job before a paid odds request.
    def check(self, report: dict[str, Any], now: datetime) -> dict[str, Any]:
        results = [
            result
            for result in report["results"]
            if result["status"] == "simulated" and timestamp(result["game"]["kickoff_utc"]) > now
        ]
        if not results:
            return {"status": "not_required", "reason": "no_upcoming_simulations", "errors": []}
        seasons = sorted({int(result["game"]["season"]) for result in results})
        # Use the maintained library's downloader/cache rather than a separate raw CSV client.
        schedule = nflreadpy.load_schedules(seasons=seasons).to_pandas()
        return self.assess(schedule, results, now)

    # Purpose: compare reported donors with completed games while preserving unknown finals.
    # Callers: check and offline tests; supplied schedule evidence is never regraded.
    def assess(
        self, schedule: pd.DataFrame, results: list[dict[str, Any]], now: datetime
    ) -> dict[str, Any]:
        required = {
            "season",
            "game_id",
            "gameday",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        }
        if not required.issubset(schedule) or schedule.game_id.duplicated().any():
            raise ValueError("NFLverse schedule identity/schema is invalid")
        at = pd.Timestamp(now)
        if at.tzinfo is None:
            raise ValueError("History check time must include a timezone")
        dates = pd.to_datetime(schedule.gameday, utc=True, errors="raise")
        prior = schedule[(dates < at.floor("D")) & schedule.game_type.ne("PRE")]
        if prior.empty or pd.to_datetime(prior.gameday, utc=True).max() < at - pd.Timedelta(
            days=14
        ):
            raise ValueError("No recent NFL schedule evidence")
        completed = prior[prior.home_score.notna() & prior.away_score.notna()]
        errors: list[dict[str, Any]] = []
        not_due: list[str] = []
        pending = prior[prior.home_score.isna() | prior.away_score.isna()]
        for game in pending.to_dict("records"):
            kickoff = pd.NaT
            try:
                kickoff = (
                    pd.Timestamp(str(game["gameday"]) + " " + str(game.get("gametime")))
                    .tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                    .tz_convert("UTC")
                )
            except (ValueError, TypeError):
                pass
            if pd.notna(kickoff) and kickoff >= at.floor("D"):
                not_due.append(game["game_id"])
            else:
                errors.append({"reason": "prior_date_result_unknown", "game_id": game["game_id"]})
        for result in results:
            game = result["game"]
            donors = result.get("provenance", {}).get("teams", {})
            for team in (game["home"], game["away"]):
                expected = completed[
                    (completed.season == game["season"])
                    & ((completed.home_team == team) | (completed.away_team == team))
                ]
                used = {entry["game_id"] for entry in donors.get(team, {}).get("games", [])}
                missing = sorted(set(expected.game_id) - used)
                if missing:
                    errors.append(
                        {
                            "reason": "simulation_omits_completed_history",
                            "game_id": game["game_id"],
                            "team": team,
                            "missing_games": missing,
                        }
                    )
        return {
            "status": "blocked" if errors else "passed",
            "completed_games": len(completed),
            "source": "nflreadpy.load_schedules",
            "errors": errors,
            "unresolved_not_due": not_due,
            "scope": "Reported donor coverage only; not independent data or calibration validation",
        }
