"""CLI orchestration for exported NFL distributions and current quote sources."""

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from goalpost.data.base.odds import QuoteSnapshot, QuoteSource
from goalpost.data.nfl.history import NFLHistory
from goalpost.data.nfl.odds import OddsAPISource, SQLiteQuoteSource
from goalpost.simulation.nfl.comparison import ComparisonSettings, MarketRow, NFLComparison
from goalpost.simulation.nfl.export import WeeklyExport


def safe_error(error: Exception) -> str:
    message = str(error)
    key = os.environ.get("ODDS_API", "").strip()
    return message.replace(key, "[redacted]") if key else message


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class MarketTableJob:
    export_path: Path
    output_path: Path
    source: QuoteSource | None
    settings: ComparisonSettings
    validate_only: bool = False

    def run(self) -> dict:
        # Never overwrite an earlier comparison or present its quotes as a new run.
        self.output_path.mkdir(parents=True, exist_ok=False)
        try:
            export = WeeklyExport.load(self.export_path)
            validation = {
                "status": "validated",
                "games": len(export.games),
                "unavailable": export.unavailable,
            }
            save_json(self.output_path / "input-validation.json", validation)
            if self.validate_only:
                return validation
            if self.source is None:
                raise ValueError("Select a live or local quote source")
            now = datetime.now(timezone.utc)
            history = NFLHistory().check(export.report, now)
            save_json(self.output_path / "history.json", history)
            if history["status"] == "blocked":
                raise ValueError("Simulation history is incomplete; inspect history.json")
            # Empty or already-started slates do not spend an odds request.
            if any(game.kickoff > now for game in export.games):
                snapshot = self.source.load()
            else:
                snapshot = QuoteSnapshot((), "not_requested")
            now = datetime.now(timezone.utc)
            comparison = NFLComparison(self.settings).run(export.games, snapshot, now)
            rows, current = comparison["rows"], comparison["current"]
            columns = [field.name for field in fields(MarketRow)]
            for name, values in (("ev-kelly", rows), ("current-market-sizing", current)):
                save_table(self.output_path / (name + ".csv"), values, columns)
                save_json(self.output_path / (name + ".json"), values)
            save_json(self.output_path / "quote-evidence.json", comparison["quote_evidence"])
            matchups = comparison["matchups"] + export.unavailable
            save_json(self.output_path / "matchups.json", matchups)
            summary = {
                "status": "complete",
                "as_of_utc": now.isoformat(),
                "source": snapshot.source,
                "book": self.settings.book,
                "settings": asdict(self.settings),
                "games": len(matchups),
                "attempted": sum(game["attempted"] for game in matchups),
                "retained": sum(game["retained"] for game in matchups),
                "approximate": sum(game["approximate"] for game in matchups),
                "evaluated_rows": len(rows),
                "fresh_rows": len(current),
                "quote_status_counts": dict(Counter(row["quote_status"] for row in rows)),
                "odds_requests": snapshot.requests,
                "credits_last_request": snapshot.credits_last_request,
                "credits_used": snapshot.credits_used,
                "credits_remaining": snapshot.credits_remaining,
                "history": history,
                "calculation_scope": "hypothetical_regulation",
                "limitations": [
                    "Uncalibrated regulation model; actual sportsbook EV/Kelly remains unset.",
                    "Mass uses retained simulations; missing paths can bias probabilities.",
                    "Regulation moneyline ties are treated as void, not modeled through overtime.",
                    "Stale and invalid rows are retained but excluded from the current table.",
                    "Correlated single-contract Kelly fractions must not be added together.",
                ],
            }
            save_json(self.output_path / "report.json", summary)
            return summary
        except Exception as error:
            # Never serialize a credential-bearing URL from a transport or downstream exception.
            message = safe_error(error)
            save_json(self.output_path / "failure.json", {"status": "failed", "error": message})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Refresh once using ODDS_API")
    mode.add_argument(
        "--database", type=Path, help="Existing read-only store; local DB env opt-in required"
    )
    mode.add_argument(
        "--validate-only", action="store_true", help="Validate exported PMFs without network"
    )
    parser.add_argument("--book", default="fanduel")
    parser.add_argument("--bankroll", type=float)
    parser.add_argument("--max-quote-age-minutes", type=float, default=30)
    parser.add_argument("--kickoff-tolerance-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    if not (args.live or args.database or args.validate_only):
        parser.error("Select --live, --database or --validate-only")
    output = args.output or Path("artifacts/market-sizing") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    try:
        settings = ComparisonSettings(
            book=args.book,
            max_quote_age_seconds=args.max_quote_age_minutes * 60,
            kickoff_tolerance_seconds=args.kickoff_tolerance_seconds,
            bankroll=args.bankroll,
        )
        source = (
            OddsAPISource(args.book)
            if args.live
            else SQLiteQuoteSource(args.database, args.book)
            if args.database
            else None
        )
        result = MarketTableJob(
            export_path=args.export,
            output_path=output,
            source=source,
            settings=settings,
            validate_only=args.validate_only,
        ).run()
    except Exception as error:
        message = safe_error(error)
        parser.exit(1, "Market comparison failed: " + message + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
