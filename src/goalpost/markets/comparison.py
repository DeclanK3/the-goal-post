"""Read stored FanDuel main-market quotes and retain reproducible comparisons."""

import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .pricing import mass, ranked_current, sizing, stamp, validate_pmfs


def select_main_quotes(quotes, home, away):
    """Select a complete pair only at the newest family pull; account for exclusions."""
    kept, excluded = [], []
    families = ("h2h", "spreads", "totals")
    excluded.extend(
        dict(q, disposition="alternate_or_unsupported_market")
        for q in quotes
        if q["market_type"] not in families
    )
    for family in families:
        rows = [q for q in quotes if q["market_type"] == family]
        if not rows:
            continue
        latest = max(stamp(q["taken_at"]) for q in rows)
        excluded.extend(
            dict(q, disposition="superseded_main_line")
            for q in rows
            if stamp(q["taken_at"]) != latest
        )
        rows = [q for q in rows if stamp(q["taken_at"]) == latest]
        expected = {"Over", "Under"} if family == "totals" else {home, away}
        lines = [q["line"] for q in rows]
        valid = len(rows) == 2 and {q["participant"] for q in rows} == expected
        if family == "h2h":
            valid = valid and all(v is None for v in lines)
        else:
            valid = valid and all(v is not None and math.isfinite(v) for v in lines)
            if valid:
                valid = (
                    lines[0] == lines[1]
                    if family == "totals"
                    else math.isclose(sum(lines), 0, abs_tol=1e-8)
                )
        if valid:
            kept.extend(rows)
        else:
            excluded.extend(
                dict(q, disposition="latest_main_pair_incomplete_or_invalid") for q in rows
            )
    return kept, excluded


def latest_main_quotes(quotes, home, away):
    return select_main_quotes(quotes, home, away)[0]


def compare(simulation, output, db, bankroll=None, max_quote_age_seconds=1800, *, now=None):
    if not math.isfinite(max_quote_age_seconds) or max_quote_age_seconds <= 0:
        raise ValueError("Quote age limit must be positive")
    if bankroll is not None and (not math.isfinite(bankroll) or bankroll <= 0):
        raise ValueError("Bankroll must be positive")
    simulation = Path(simulation)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((simulation / "report.json").read_text())
    if not report.get("regulation_only") or report.get("betting_admitted") is not False:
        raise ValueError("Unknown simulation scope")
    raw = json.loads((simulation / "schedule-espn.json").read_text())
    names = {}
    for event in raw["events"]:
        for c in event["competitions"][0]["competitors"]:
            t = c["team"]
            names[{"LAR": "LA", "WSH": "WAS"}.get(t["abbreviation"], t["abbreviation"])] = t[
                "displayName"
            ]
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Comparison time needs timezone")
    rows = []
    game_rows = []
    quote_evidence = []
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        hold = conn.execute("select value from flags where key='nfl_manual_review_hold'").fetchone()
        for result in report["results"]:
            game = result["game"]
            home = names[game["home"]]
            away = names[game["away"]]
            base = dict(
                game_id=game["game_id"],
                home=home,
                away=away,
                kickoff=game["kickoff_utc"],
                attempted=result["attempted"],
                included=result.get("n", 0),
                simulation_status=result["status"],
                approximate=result.get("diagnostics", {}).get("grace_completed_games", 0),
            )
            base["incomplete"] = base["attempted"] - base["included"]
            if base["attempted"] <= 0 or not 0 <= base["included"] <= base["attempted"]:
                raise ValueError("Invalid simulation denominator")
            base["completion_rate"] = base["included"] / base["attempted"]
            matches = [
                dict(r)
                for r in conn.execute(
                    "select * from games where league='nfl' and home_team=? and away_team=?",
                    (home, away),
                )
                if abs((stamp(r["commence_time_utc"]) - stamp(game["kickoff_utc"])).total_seconds())
                <= 600
            ]
            if (
                len(matches) != 1
                or matches[0]["status"] != "scheduled"
                or min(stamp(game["kickoff_utc"]), stamp(matches[0]["commence_time_utc"])) <= now
            ):
                game_rows.append(dict(base, quote_status="unmatched_ambiguous_or_not_pregame"))
                continue
            event = matches[0]
            base["provider_game_id"] = event["id"]
            base["kickoff_difference_seconds"] = (
                stamp(event["commence_time_utc"]) - stamp(game["kickoff_utc"])
            ).total_seconds()
            if result["status"] != "simulated":
                game_rows.append(dict(base, quote_status="simulation_unavailable"))
                continue
            folder = simulation / game["game_id"]
            pmfs = {k: pd.read_csv(folder / (k + "-pmf.csv")) for k in ("total", "home_minus_away")}
            try:
                validate_pmfs(pmfs, base["included"])
            except ValueError as exc:
                game_rows.append(dict(base, quote_status="invalid_simulation_pmf", reason=str(exc)))
                continue
            quotes = [
                dict(r)
                for r in conn.execute(
                    """WITH ordered AS (
              SELECT m.id market_id,m.market_type,m.participant,m.line,s.id snapshot_id,s.odds_american,s.taken_at,s.provider_updated_at,s.event_context_json,
              ROW_NUMBER() OVER (PARTITION BY m.id ORDER BY julianday(s.taken_at) DESC,s.id DESC) rn
              FROM markets m JOIN odds_snapshots s ON s.market_id=m.id
              WHERE m.game_id=? AND s.book='fanduel' AND s.source IS NULL
              AND m.market_type IN ('h2h','spreads','alternate_spreads','totals','alternate_totals'))
              SELECT * FROM ordered WHERE rn=1""",
                    (event["id"],),
                )
            ]
            quotes, excluded = select_main_quotes(quotes, home, away)
            quote_evidence.extend(dict(game_id=game["game_id"], **q) for q in excluded)
            base["missing_main_families"] = [
                f
                for f in ("h2h", "spreads", "totals")
                if not any(q["market_type"] == f for q in quotes)
            ]
            game_rows.append(
                dict(
                    base,
                    quote_status="stored_quotes" if quotes else "missing_quotes",
                    quote_count=len(quotes),
                )
            )
            for q in quotes:
                quote_evidence.append(
                    dict(game_id=game["game_id"], disposition="selected_latest_main_pair", **q)
                )
                row = dict(
                    base,
                    book="fanduel",
                    market=q["market_type"],
                    participant=q["participant"],
                    line=q["line"],
                    odds=q["odds_american"],
                    snapshot_id=q["snapshot_id"],
                    quote_taken_at=q["taken_at"],
                    provider_updated_at=q["provider_updated_at"],
                    actual_contract_ev=None,
                    actual_contract_kelly=None,
                    betting_admitted=False,
                )
                try:
                    age = (now - stamp(q["taken_at"])).total_seconds()
                    provider_age = (now - stamp(q["provider_updated_at"])).total_seconds()
                    row.update(quote_age_minutes=age / 60, provider_age_minutes=provider_age / 60)
                    if age < -60 or provider_age < -60:
                        raise ValueError("Future quote")
                    cutoff = min(stamp(game["kickoff_utc"]), stamp(event["commence_time_utc"]))
                    if stamp(q["taken_at"]) >= cutoff or stamp(q["provider_updated_at"]) >= cutoff:
                        raise ValueError("Quote timestamp at or after kickoff")
                    if q["event_context_json"]:
                        context = json.loads(q["event_context_json"])
                        if any(
                            context.get(k) != event[k]
                            for k in ("home_team", "away_team", "commence_time_utc")
                        ):
                            raise ValueError("Quote context mismatch")
                    else:
                        row["identity_evidence"] = "current_database_identity_only"
                    if q["event_context_json"]:
                        row["identity_evidence"] = "captured_quote_context"
                    frame = pmfs[
                        "total"
                        if q["market_type"] in ("totals", "alternate_totals")
                        else "home_minus_away"
                    ]
                    win, push, loss = mass(
                        frame, q["market_type"], q["participant"], q["line"], home, away
                    )
                    row.update(
                        p_win=win,
                        p_push_or_regulation_tie=push,
                        p_loss=loss,
                        **sizing(win, push, loss, q["odds_american"]),
                    )
                    kept = base["completion_rate"]
                    row.update(
                        p_win_all_attempts_lower=kept * win,
                        p_win_all_attempts_upper=kept * win + 1 - kept,
                    )
                    row["pmf_sha256"] = hashlib.sha256(
                        (
                            folder
                            / (
                                "total-pmf.csv"
                                if q["market_type"] == "totals"
                                else "home_minus_away-pmf.csv"
                            )
                        ).read_bytes()
                    ).hexdigest()
                    row["quote_status"] = (
                        "fresh_stored_quote"
                        if max(age, provider_age) <= max_quote_age_seconds
                        else "stale_quote"
                    )
                    row["calculation_scope"] = (
                        "hypothetical_regulation_contract_ties_void"
                        if q["market_type"] == "h2h"
                        else "hypothetical_regulation_contract_pushes_void"
                    )
                    if bankroll is not None:
                        row["hypothetical_quarter_kelly_dollars"] = (
                            bankroll * row["quarter_kelly_pct"] / 100
                        )
                except (ValueError, TypeError, KeyError) as exc:
                    row["quote_status"] = "unpriced: " + str(exc)
                rows.append(row)
    frame = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "game_id",
                "home",
                "away",
                "market",
                "participant",
                "line",
                "odds",
                "p_win",
                "p_push_or_regulation_tie",
                "p_loss",
                "hypothetical_ev_pct",
                "full_kelly_pct",
                "quarter_kelly_pct",
                "quote_status",
            ]
        )
    )
    frame.to_csv(output / "ev-kelly.csv", index=False)
    frame[frame["quote_status"].eq("fresh_stored_quote")].to_csv(
        output / "fresh-markets-only.csv", index=False
    ) if len(frame) else frame.to_csv(output / "fresh-markets-only.csv", index=False)
    current = ranked_current(rows)
    pd.DataFrame(current, columns=frame.columns).to_csv(
        output / "current-market-sizing.csv", index=False
    )
    (output / "current-market-sizing.json").write_text(
        json.dumps(current, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(game_rows).to_csv(output / "matchup-summary.csv", index=False)
    (output / "quote-evidence.json").write_text(json.dumps(quote_evidence, indent=2) + "\n")
    summary = dict(
        as_of_utc=now.isoformat(),
        simulation_directory=str(simulation.resolve()),
        simulation_report_sha256=hashlib.sha256(
            (simulation / "report.json").read_bytes()
        ).hexdigest(),
        comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        nfl_hold=hold["value"] if hold else "unknown",
        selection="Latest complete main-market pair per game/family; no older lines or alternates; latest is not necessarily fresh",
        games=len(game_rows),
        markets=len(rows),
        priced=sum("hypothetical_ev_pct" in r for r in rows),
        fresh_priced=sum(r["quote_status"] == "fresh_stored_quote" for r in rows),
        bankroll=bankroll,
        kelly_fraction=0.25,
        max_quote_age_seconds=max_quote_age_seconds,
        simulation_dataset_hashes=report.get("dataset_hashes"),
        simulation_limitations=report.get("limitations", []),
        simulation_code_sha256=report.get("code_sha256"),
        betting_admitted=False,
        actual_contract_ev_established=False,
        limitations=[
            "Regulation only; sportsbook overtime-inclusive contract probabilities are not modeled.",
            "Probabilities condition on included simulations, including approximate completions; missing paths may bias probabilities.",
            "Model probabilities are uncalibrated; dataset provenance is copied from the input weekly report. A new schedule does not refresh model data.",
            "Stale quotes remain explicitly labeled hypothetical arithmetic, not current offers.",
            "Moneyline calculation assumes regulation ties void; this is not an overtime moneyline prediction.",
            "Single-bet Kelly fractions are correlated and must not be added into a portfolio stake.",
            "Rows without captured event context use current database team/kickoff identity only; no quote-time identity certification.",
            "No accepted-bet receipts, paid odds requests, betting texts or production changes.",
        ],
        matchups=game_rows,
    )
    summary["quote_exclusions"] = dict(
        Counter(
            q["disposition"]
            for q in quote_evidence
            if q["disposition"] != "selected_latest_main_pair"
        )
    )
    summary["quote_status_counts"] = dict(Counter(r["quote_status"] for r in rows))
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Weekly model / FanDuel comparison",
        "",
        f"{summary['games']} matchups; {summary['priced']} calculated contracts; {summary['fresh_priced']} fresh-priced contracts.",
        "",
        "Current fresh main-market contracts only, sorted by hypothetical EV. All evaluated/stale rows remain in ev-kelly.csv. Research arithmetic only. Regulation distributions do not establish actual overtime-inclusive contract EV or stakes. See report.json for completion counts and limitations.",
        "",
        "| Away @ home | Market | Selection | Odds | Win % | Push/tie % | EV % | Quarter Kelly % | Quote age (hours) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in current:
        lines.append(
            f"| {r['away']} @ {r['home']} | {r['market']} | {r['participant']} {r['line'] if r['line'] is not None else ''} | {r['odds']:+d} | {100 * r['p_win']:.2f} | {100 * r['p_push_or_regulation_tie']:.2f} | {r['hypothetical_ev_pct']:+.2f} | {r['quarter_kelly_pct']:.2f} | {r['quote_age_minutes'] / 60:.1f} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "matchups"}, indent=2))
