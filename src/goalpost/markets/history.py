"""Check reported simulation donors against the NFLverse Python schedule loader."""

from datetime import datetime

import nflreadpy
import pandas as pd

from .export import timestamp


class NFLHistory:
    """Current pricing requires completed current-season games in the export's donors."""

    def check(self, report: dict, now: datetime) -> dict:
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

    def assess(self, schedule: pd.DataFrame, results: list[dict], now: datetime) -> dict:
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
        errors, not_due = [], []
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
