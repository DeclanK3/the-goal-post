"""Standard weekly PMF export -> existing regulation EV/Kelly table. No sender."""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import comparison, history, pricing

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_export(path):
    path = Path(path).resolve()
    report = json.loads((path / "report.json").read_text())
    if report.get("regulation_only") is not True or report.get("betting_admitted") is not False:
        raise ValueError("unsupported_or_unspecified_contract_scope")
    schedule = json.loads((path / "schedule-espn.json").read_text())
    names = {}
    identities = []
    for event in schedule["events"]:
        if len(event["competitions"]) != 1:
            raise ValueError("ambiguous_schedule_competition")
        competitors = event["competitions"][0]["competitors"]
        if len(competitors) != 2 or {c["homeAway"] for c in competitors} != {"home", "away"}:
            raise ValueError("ambiguous_schedule_home_away")
        sides = {}
        for competitor in event["competitions"][0]["competitors"]:
            team = competitor["team"]
            key = {"LAR": "LA", "WSH": "WAS"}.get(team["abbreviation"], team["abbreviation"])
            if key in names and names[key] != team["displayName"]:
                raise ValueError("conflicting_schedule_team_name:" + key)
            names[key] = team["displayName"]
            sides[competitor["homeAway"]] = key
        identities.append((sides["home"], sides["away"], pricing.stamp(event["date"])))
    ids = set()
    files = {name: digest(path / name) for name in ("report.json", "schedule-espn.json")}
    summaries = []
    for result in report["results"]:
        game = result["game"]
        gid = game["game_id"]
        if (
            not isinstance(gid, str)
            or gid in ("", ".", "..")
            or Path(gid).name != gid
            or "\\" in gid
        ):
            raise ValueError("unsafe_game_directory")
        if gid in ids:
            raise ValueError("duplicate_game_id")
        ids.add(gid)
        if game["home"] == game["away"] or any(
            team not in names for team in (game["home"], game["away"])
        ):
            raise ValueError("unknown_or_duplicate_home_away_team")
        identity = (game["home"], game["away"], pricing.stamp(game["kickoff_utc"]))
        if identities.count(identity) != 1:
            raise ValueError("export_schedule_identity_mismatch_or_ambiguous")
        attempted, included = result["attempted"], result.get("n", 0)
        if (
            type(attempted) is not int
            or type(included) is not int
            or attempted <= 0
            or not 0 <= included <= attempted
        ):
            raise ValueError("invalid_simulation_denominators")
        approximate = result.get("diagnostics", {}).get("grace_completed_games", 0)
        if type(approximate) is not int or not 0 <= approximate <= included:
            raise ValueError("invalid_approximate_denominator")
        if result["status"] == "simulated":
            frames = {}
            for name in ("total", "home_minus_away"):
                rel = gid + "/" + name + "-pmf.csv"
                file = path / rel
                if not file.resolve().is_relative_to(path):
                    raise ValueError("pmf_outside_export")
                frames[name] = pd.read_csv(file)
                files[rel] = digest(file)
            pricing.validate_pmfs(frames, included)
        summaries.append(
            dict(
                game_id=gid,
                home=names[game["home"]],
                away=names[game["away"]],
                status=result["status"],
                attempted=attempted,
                retained=included,
            )
        )
    return dict(
        status="validated_input_schema_and_pmf_counts",
        games=summaries,
        files=files,
        model_calibration_verified=False,
        live_freshness_checked=False,
    ), report


def verify_dataset(data, report):
    hashes = report.get("dataset_hashes", {})
    for name in ("transition_set.json", "nfl_games.parquet", "parsed_segments.parquet"):
        if not isinstance(hashes.get(name), str) or len(hashes[name]) != 64:
            raise ValueError("missing_export_dataset_hash:" + name)
        if digest(Path(data) / name) != hashes[name]:
            raise ValueError("dataset_does_not_match_export:" + name)


def demo_database(path):
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE games(id TEXT,league TEXT,home_team TEXT,away_team TEXT,commence_time_utc TEXT,status TEXT);
          CREATE TABLE flags(key TEXT,value TEXT);
          CREATE TABLE markets(id INTEGER,game_id TEXT,market_type TEXT,participant TEXT,line REAL);
          CREATE TABLE odds_snapshots(id INTEGER,market_id INTEGER,book TEXT,source TEXT,odds_american INTEGER,taken_at TEXT,provider_updated_at TEXT,event_context_json TEXT);""")
        db.execute(
            "INSERT INTO games VALUES(?,?,?,?,?,?)",
            ("demo", "nfl", "Example Home", "Example Away", "2000-01-02T18:00:00Z", "scheduled"),
        )
        db.execute("INSERT INTO flags VALUES(?,?)", ("nfl_manual_review_hold", "enabled"))
        for i, row in enumerate(
            csv.DictReader((HERE / "example/quotes.csv").read_text().splitlines()), 1
        ):
            line = float(row["line"]) if row["line"] else None
            db.execute(
                "INSERT INTO markets VALUES(?,?,?,?,?)",
                (i, "demo", row["market"], row["participant"], line),
            )
            db.execute(
                "INSERT INTO odds_snapshots VALUES(?,?,?,?,?,?,?,?)",
                (
                    i,
                    i,
                    "fanduel",
                    None,
                    int(row["american_odds"]),
                    "2000-01-01T11:59:00Z",
                    "2000-01-01T11:59:00Z",
                    None,
                ),
            )


def run(export, out, data=None, database=None, bankroll=None, demo=False, validate_only=False):
    if demo and (data is not None or database is not None or export is not None):
        raise ValueError("demo_cannot_use_live_inputs")
    if bankroll is not None and (not math.isfinite(bankroll) or bankroll <= 0):
        raise ValueError("bankroll_must_be_positive")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    export = HERE / "example/export" if demo else Path(export).resolve()
    validation, report = validate_export(export)
    (out / "input-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if validate_only:
        return dict(
            status=validation["status"],
            output=str(out),
            simulation_run=False,
            paid_requests=0,
            messages_sent=0,
        )
    freshness = None
    if demo:
        database = out / "synthetic-quotes.sqlite"
        demo_database(database)
        demo_time = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    else:
        if report.get("synthetic_demo"):
            raise ValueError("synthetic_export_requires_demo_mode")
        if data is None or database is None:
            raise ValueError("live_comparison_requires_dataset_and_readonly_quote_database")
        verify_dataset(data, report)
        # Same coverage and actual team-donor check as the existing weekly CLI.
        # This fetches one PUBLIC nflverse schedule, never paid odds.
        freshness = history.check(Path(data), out / "data-freshness", report)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        comparison.compare(
            export,
            out / "comparison",
            db=database,
            bankroll=bankroll,
            now=demo_time if demo else None,
        )
    (out / "comparison.log").write_text(log.getvalue())
    summary_path = out / "comparison/report.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        synthetic_demo=demo,
        input_validation=validation,
        model_data_freshness=freshness,
        calibration_verified=False,
        export_dataset_binding="synthetic_demo" if demo else "all_three_dataset_hashes_match",
        bridge_code_sha256={p.name: digest(p) for p in HERE.glob("*.py")},
        simulated_again=False,
    )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if demo:
        note = "# SYNTHETIC EXAMPLE — NOT REAL TEAMS, PRICES, PROBABILITIES OR A BETTING PICK\n\n"
        p = out / "comparison/README.md"
        p.write_text(note + p.read_text())
    return dict(
        status="synthetic_demo_complete" if demo else "comparison_complete",
        output=str(out),
        table=str(out / "comparison/current-market-sizing.csv"),
        all_rows=str(out / "comparison/ev-kelly.csv"),
        simulation_run=False,
        paid_requests=0,
        messages_sent=0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--export",
        type=Path,
        help="Directory containing report.json, schedule-espn.json and game PMFs",
    )
    source.add_argument(
        "--demo", action="store_true", help="Only synthetic local example inputs; no network"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Exact dataset used for this export; required for a live comparison",
    )
    parser.add_argument(
        "--database", type=Path, help="Existing read-only engine quote SQLite database"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check export schema and PMFs without prices or network",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        help="Optional explicit bankroll for hypothetical dollar arithmetic",
    )
    args = parser.parse_args(argv)
    out = args.output or Path.cwd() / "artifacts/market-sizing" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")
    existed = Path(out).exists()
    try:
        result = run(
            args.export, out, args.data, args.database, args.bankroll, args.demo, args.validate_only
        )
    except Exception as exc:
        if not existed and Path(out).is_dir() and not isinstance(exc, FileExistsError):
            (Path(out) / "failure.json").write_text(
                json.dumps(
                    dict(
                        status="failed",
                        type=type(exc).__name__,
                        error=str(exc),
                        old_report_substituted=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
