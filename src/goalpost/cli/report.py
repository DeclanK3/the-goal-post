# Purpose: orchestrate validation, quote acquisition and reproducible market reports.
# Public API: MarketTableJob.run is the shared service; main is the command-line adapter.
# Consumers: the root --report command, goalpost.cli.report and the Generate Report TUI.

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from goalpost.data.base.odds import QuoteSnapshot, QuoteSource
from goalpost.data.nfl.history import NFLHistory
from goalpost.data.nfl.odds import OddsAPISource, SQLiteQuoteSource
from goalpost.simulation.nfl.comparison import ComparisonSettings, MarketRow, NFLComparison
from goalpost.simulation.nfl.export import WeeklyExport


# Purpose: remove the configured credential before an error reaches a report or terminal.
# Callers: MarketTableJob.run, main and the report TUI failure adapter.
def safe_error(error: Exception) -> str:
    message = str(error)
    api_key = os.environ.get("ODDS_API", "").strip()
    return message.replace(api_key, "[redacted]") if api_key else message


# Purpose: write strict JSON so non-finite results cannot masquerade as usable evidence.
# Callers: MarketTableJob's validation, report and failure paths.
def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


# Purpose: keep CSV headers stable even when a valid run has no evaluated rows.
# Callers: MarketTableJob._write_comparison for the full and fresh-only tables.
def save_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# Purpose: give CLI and TUI callers the same validation order and output contract.
# Consumers: main and goalpost.tui.report.generate_report; output directories are unique.
@dataclass
class MarketTableJob:
    export_path: Path
    output_path: Path
    source: QuoteSource | None
    settings: ComparisonSettings
    validate_only: bool = False

    # Purpose: validate inputs and donor history before allowing a quote request.
    # Callers: both user interfaces; failures retain sanitized evidence and are reraised.
    def run(self) -> dict[str, Any]:
        self.output_path.mkdir(parents=True, exist_ok=False)
        try:
            export = WeeklyExport.load(self.export_path)
            validation = self._write_validation(export)
            if self.validate_only:
                return validation
            if self.source is None:
                raise ValueError("Select a live or local quote source")

            history = NFLHistory().check(export.report, datetime.now(timezone.utc))
            save_json(self.output_path / "history.json", history)
            if history["status"] == "blocked":
                raise ValueError("Simulation history is incomplete; inspect history.json")

            snapshot = self._load_snapshot(export)
            compared_at = datetime.now(timezone.utc)
            comparison = NFLComparison(self.settings).run(export.games, snapshot, compared_at)
            return self._write_comparison(export, comparison, snapshot, history, compared_at)
        except Exception as error:
            save_json(
                self.output_path / "failure.json",
                {"status": "failed", "error": safe_error(error)},
            )
            raise

    # Purpose: persist input acceptance independently of subsequent quote availability.
    # Callers: run; validate-only exits immediately after this artifact is written.
    def _write_validation(self, export: WeeklyExport) -> dict[str, Any]:
        validation = {
            "status": "validated",
            "games": len(export.games),
            "unavailable": export.unavailable,
        }
        save_json(self.output_path / "input-validation.json", validation)
        return validation

    # Purpose: avoid spending a request on an empty or already-started slate.
    # Callers: run, only after export and history validation have succeeded.
    def _load_snapshot(self, export: WeeklyExport) -> QuoteSnapshot:
        now = datetime.now(timezone.utc)
        if not any(game.kickoff > now for game in export.games):
            return QuoteSnapshot((), "not_requested")
        if self.source is None:
            raise ValueError("Select a live or local quote source")
        return self.source.load()

    # Purpose: preserve every evaluated/excluded row alongside the fresh comparison.
    # Callers: run after pricing; the report retains simulation and request denominators.
    def _write_comparison(
        self,
        export: WeeklyExport,
        comparison: dict[str, Any],
        snapshot: QuoteSnapshot,
        history: dict[str, Any],
        compared_at: datetime,
    ) -> dict[str, Any]:
        evaluated_rows = comparison["rows"]
        current_rows = comparison["current"]
        columns = [column.name for column in fields(MarketRow)]
        tables = (("ev-kelly", evaluated_rows), ("current-market-sizing", current_rows))
        for name, rows in tables:
            save_table(self.output_path / f"{name}.csv", rows, columns)
            save_json(self.output_path / f"{name}.json", rows)

        save_json(self.output_path / "quote-evidence.json", comparison["quote_evidence"])
        matchups = comparison["matchups"] + export.unavailable
        save_json(self.output_path / "matchups.json", matchups)

        summary = {
            "status": "complete",
            "as_of_utc": compared_at.isoformat(),
            "source": snapshot.source,
            "book": self.settings.book,
            "settings": asdict(self.settings),
            "games": len(matchups),
            "attempted": sum(game["attempted"] for game in matchups),
            "retained": sum(game["retained"] for game in matchups),
            "approximate": sum(game["approximate"] for game in matchups),
            "evaluated_rows": len(evaluated_rows),
            "fresh_rows": len(current_rows),
            "quote_status_counts": dict(Counter(row["quote_status"] for row in evaluated_rows)),
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


# Purpose: define explicit input and source choices without fetching any data.
# Callers: main; the TUI constructs the same service without parsing command-line options.
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exported NFL regulation distributions with current market quotes."
    )
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
    return parser


# Purpose: translate explicit CLI options into one shared report job and exit status.
# Callers: the root --report dispatch and python -m goalpost.cli.report.
def main(argv: list[str] | None = None) -> None:
    parser = _parser()
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
        source: QuoteSource | None = None
        if args.live:
            source = OddsAPISource(args.book)
        elif args.database:
            source = SQLiteQuoteSource(args.database, args.book)

        result = MarketTableJob(
            export_path=args.export,
            output_path=output,
            source=source,
            settings=settings,
            validate_only=args.validate_only,
        ).run()
    except Exception as error:
        parser.exit(1, "Market comparison failed: " + safe_error(error) + "\n")

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
