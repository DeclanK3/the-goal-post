# Purpose: Collect explicit report inputs and invoke the shared research report job.
# Public API: PROMPTS, generate_report, run_report_screen.
# Consumers: The Generate Report action in tui.main and interface parity tests.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import ratatui as r

from goalpost.cli.report import MarketTableJob
from goalpost.data.nfl.odds import OddsAPISource, SQLiteQuoteSource
from goalpost.simulation.nfl.comparison import ComparisonSettings

PROMPTS: list[str] = [
    "Simulation export directory",
    "Quote source (live / database / validate)",
    "Database path (database mode only)",
    "Output directory (optional)",
    "Book (default fanduel)",
    "Bankroll (optional; hypothetical dollars)",
]


# Purpose: Validate explicit TUI source choices and invoke the same report job as the CLI.
# Callers: run_report_screen and CLI/TUI parity tests.
def generate_report(values: dict[str, str]) -> str:
    export = values.get(PROMPTS[0], "").strip()
    if not export:
        raise ValueError("Simulation export directory is required")
    mode = values.get(PROMPTS[1], "").strip().lower()
    if mode not in ("live", "database", "validate"):
        raise ValueError("Choose live, database or validate explicitly")
    database = values.get(PROMPTS[2], "").strip()
    if mode == "database" and not database:
        raise ValueError("Database path is required in database mode")
    if mode != "database" and database:
        raise ValueError("Database path only applies in database mode")
    book = values.get(PROMPTS[4], "").strip() or "fanduel"
    bankroll = values.get(PROMPTS[5], "").strip()
    settings = ComparisonSettings(book=book, bankroll=float(bankroll) if bankroll else None)
    source = (
        OddsAPISource(book)
        if mode == "live"
        else SQLiteQuoteSource(Path(database).expanduser(), book)
        if mode == "database"
        else None
    )
    output = values.get(PROMPTS[3], "").strip()
    output_path = (
        Path(output).expanduser()
        if output
        else Path("artifacts/market-sizing")
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    result = MarketTableJob(
        Path(export).expanduser(), output_path, source, settings, validate_only=mode == "validate"
    ).run()
    if mode == "validate":
        return f"Validated {result['games']} games. Output: {output_path}"
    return (
        f"Research/regulation-only: {result['fresh_rows']} fresh rows, "
        f"{result['evaluated_rows']} evaluated. EV/Kelly table: "
        f"{output_path / 'current-market-sizing.csv'}; full report: {output_path / 'report.json'}"
    )


# Purpose: Collect report inputs without prompting for or persisting API credentials.
# Callers: tui.main Generate Report action; cancellation does not create a job.
def run_report_screen(
    term: r.Terminal, prompt: Callable[[r.Terminal, str, list[str]], dict[str, str] | None]
) -> str:
    # No key prompt, saved key or implicit paid request. Selecting live is explicit.
    values = prompt(
        term, "Generate Report — research/regulation only; live makes one paid request", PROMPTS
    )
    if values is None:
        return "Canceled"
    return generate_report(values)
