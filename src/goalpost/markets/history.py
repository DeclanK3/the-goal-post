"""Fail closed when current NFL history is missing from weekly model inputs."""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def assess(schedule, meta, now, simulation=None):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("Freshness time needs timezone")
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
    if not required.issubset(schedule.columns):
        raise ValueError("Schedule schema incomplete")
    if schedule.game_id.duplicated().any():
        raise ValueError("Duplicate schedule game identity")
    days = pd.to_datetime(schedule.gameday, utc=True, errors="raise")
    prior = schedule[days < now.floor("D")]
    prior = prior[prior.game_type.ne("PRE")]
    if prior.empty:
        raise ValueError("Schedule contains no prior season")
    season = int(prior.season.max())
    prior = prior[prior.season.eq(season)]
    if pd.to_datetime(prior.gameday, utc=True).max() < now - pd.Timedelta(days=14):
        raise ValueError("No recent NFL schedule evidence; season/source requires review")
    completed = prior[prior.home_score.notna() & prior.away_score.notna()]
    unresolved = []
    not_due = []
    for row in prior[prior.home_score.isna() | prior.away_score.isna()].to_dict("records"):
        # nflverse dates/times are Eastern, as in the weekly schedule adapter.
        # A Monday night kickoff can be Tuesday UTC; it is not a prior-UTC-day
        # missing final. Invalid/absent kickoff evidence still fails closed.
        kickoff = pd.NaT
        try:
            kickoff = (
                pd.Timestamp(str(row["gameday"]) + " " + str(row.get("gametime")))
                .tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                .tz_convert("UTC")
            )
        except (ValueError, TypeError):
            pass
        if pd.notna(kickoff) and kickoff >= now.floor("D"):
            not_due.append(
                dict(
                    game_id=row["game_id"],
                    kickoff_utc=kickoff.isoformat(),
                    reason="not_prior_utc_kickoff_date",
                )
            )
        else:
            unresolved.append(row["game_id"])
    if meta.game_id.duplicated().any():
        raise ValueError("Duplicate dataset identity")
    indexed = meta.set_index("game_id")
    teams = []
    errors = []
    if unresolved:
        errors.append(dict(reason="prior_date_schedule_result_unresolved", game_ids=unresolved))
    for team in sorted(set(completed.home_team) | set(completed.away_team)):
        expected = completed[
            completed.home_team.eq(team) | completed.away_team.eq(team)
        ].sort_values(["gameday", "game_id"])
        missing = []
        rejected = []
        mismatch = []
        for r in expected.itertuples():
            if r.game_id not in indexed.index:
                missing.append(r.game_id)
                continue
            actual = indexed.loc[r.game_id]
            if actual.matrix_status != "parsed_regulation":
                rejected.append(
                    dict(game_id=r.game_id, reason=str(actual.get("matrix_issue", "unparsed")))
                )
            if (actual.home_team_id, actual.away_team_id, str(actual.game_date)[:10]) != (
                r.home_team,
                r.away_team,
                str(r.gameday)[:10],
            ):
                mismatch.append(r.game_id)
        evidence = dict(
            team=team,
            expected_games=expected.game_id.tolist(),
            latest_expected_game_date=expected.gameday.max(),
            missing_games=missing,
            rejected_games=rejected,
            identity_mismatches=mismatch,
        )
        teams.append(evidence)
        if missing or rejected or mismatch:
            errors.append(dict(reason="incomplete_team_history", **evidence))
    if simulation is not None:
        for result in simulation.get("results", []):
            game = result["game"]
            provenance = result.get("provenance", {}).get("teams", {})
            for team in (game["home"], game["away"]):
                expected = set(
                    completed[completed.home_team.eq(team) | completed.away_team.eq(team)].game_id
                )
                used = {r["game_id"] for r in provenance.get(team, {}).get("games", [])}
                missing = sorted(expected - used)
                if missing:
                    errors.append(
                        dict(
                            reason="simulation_omits_completed_history",
                            game_id=game["game_id"],
                            team=team,
                            missing_games=missing,
                        )
                    )
    return dict(
        status="passed" if not errors else "blocked",
        checked_at=now.isoformat(),
        season=season,
        policy="All completed prior-UTC-date current-season games must be parsed and represented; unknown prior-date results require review.",
        completed_games=len(completed),
        teams=teams,
        errors=errors,
        unresolved_not_due=not_due,
    )


def check(data, output, simulation=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {}
    try:
        raw = urllib.request.urlopen(URL, timeout=30).read()
        (output / "nfl-schedule.csv").write_bytes(raw)
        evidence = dict(
            url=URL,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(raw).hexdigest(),
            paid_requests=0,
        )
        meta = pd.read_parquet(
            Path(data) / "nfl_games.parquet",
            columns=[
                "game_id",
                "game_date",
                "home_team_id",
                "away_team_id",
                "matrix_status",
                "matrix_issue",
            ],
        )
        report = assess(pd.read_csv(BytesIO(raw)), meta, datetime.now(timezone.utc), simulation)
    except Exception as exc:
        report = dict(
            status="blocked",
            checked_at=datetime.now(timezone.utc).isoformat(),
            errors=[dict(reason="freshness_check_failed", detail=f"{type(exc).__name__}: {exc}")],
        )
    report["schedule_source"] = evidence
    report["dataset"] = str(Path(data).resolve())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise ValueError("Model data freshness blocked; inspect " + str(output / "report.json"))
    return report
