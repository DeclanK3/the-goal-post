# Purpose: Verify report sequencing, artifacts, denominators and safe failures.
# Public API: pytest tests and local fixtures in this module.
# Consumers: The offline unit suite; temporary files never access production stores.

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from pathlib import Path
from unittest.mock import Mock

import pytest

from goalpost.cli import report as bridge
from goalpost.cli.report import MarketTableJob
from goalpost.data.base.odds import BookEvent
from goalpost.data.nfl.history import NFLHistory
from goalpost.data.nfl.odds import QuoteSnapshot
from goalpost.simulation.nfl.comparison import ComparisonSettings


# Purpose: Replace report time with the shared fixture instant for reproducible output.
# Callers: Bridge tests requesting this fixture.
@pytest.fixture
def fixed_clock(now: datetime, monkeypatch: pytest.MonkeyPatch) -> None:
    # Purpose: Provide the datetime.now surface needed by a controlled report run.
    # Consumers: Tests monkeypatch the report module datetime binding.
    class Clock:
        # Purpose: Supply a fixed aware instant so freshness tests do not depend on wall time.
        # Callers: pytest fixtures or the local monkeypatched report clock.
        @staticmethod
        def now(zone: tzinfo | None = None) -> datetime:
            return now

    monkeypatch.setattr(bridge, "datetime", Clock)


# Purpose: Verify validate only does not load quotes or history.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_validate_only_does_not_load_quotes_or_history(
    export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Mock()
    history = Mock(side_effect=AssertionError("No public requests"))
    monkeypatch.setattr(NFLHistory, "check", history)
    result = MarketTableJob(export_path, tmp_path / "out", source, ComparisonSettings(), True).run()
    assert result["status"] == "validated" and result["games"] == 1
    source.load.assert_not_called()
    history.assert_not_called()


# Purpose: Verify full job writes all rows and sorted current.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_full_job_writes_all_rows_and_sorted_current(
    export_path: Path,
    event: BookEvent,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: None,
) -> None:
    monkeypatch.setattr(NFLHistory, "check", Mock(return_value={"status": "passed"}))
    source = Mock()
    source.load.return_value = QuoteSnapshot((event,), "odds_api", 1, 10, 990, 3)
    output = tmp_path / "out"
    result = MarketTableJob(export_path, output, source, ComparisonSettings()).run()
    assert result["status"] == "complete" and result["fresh_rows"] == 2
    assert result["attempted"] == 120 and result["retained"] == 100 and result["approximate"] == 2
    assert result["credits_last_request"] == 3
    current = json.loads((output / "current-market-sizing.json").read_text())
    assert current[0]["hypothetical_ev_pct"] > current[1]["hypothetical_ev_pct"]
    assert len(json.loads((output / "ev-kelly.json").read_text())) == 2
    assert not list(output.glob("*.md"))
    assert all(row["actual_contract_ev"] is None for row in current)
    source.load.assert_called_once()


# Purpose: Verify incomplete history blocks before paid request.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_incomplete_history_blocks_before_paid_request(
    export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_clock: None
) -> None:
    monkeypatch.setattr(
        NFLHistory, "check", Mock(return_value={"status": "blocked", "errors": ["missing"]})
    )
    source = Mock()
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="history is incomplete"):
        MarketTableJob(export_path, output, source, ComparisonSettings()).run()
    source.load.assert_not_called()
    assert (output / "history.json").exists() and (output / "failure.json").exists()
    assert not (output / "current-market-sizing.csv").exists()


# Purpose: Verify old output is never overwritten.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_old_output_is_never_overwritten(export_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        MarketTableJob(export_path, output, None, ComparisonSettings(), True).run()
    assert (output / "report.json").read_text() == "preserve"
    assert not (output / "failure.json").exists()


# Purpose: Verify empty export is not simulated and spends no request.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_empty_export_is_not_simulated_and_spends_no_request(
    export_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_clock: None
) -> None:
    report = json.loads((export_path / "report.json").read_text())
    report.update(status="no_scheduled_matchups", results=[])
    (export_path / "report.json").write_text(json.dumps(report))
    source = Mock()
    output = tmp_path / "out"
    result = MarketTableJob(export_path, output, source, ComparisonSettings()).run()
    assert result["games"] == 0 and result["attempted"] == 0 and result["odds_requests"] == 0
    assert (output / "current-market-sizing.csv").read_text().startswith("game_id,")
    source.load.assert_not_called()


# Purpose: Verify source error is sanitized in artifact and cli.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_source_error_is_sanitized_in_artifact_and_cli(
    export_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_clock: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ODDS_API", "unit-test-only")
    monkeypatch.setattr(NFLHistory, "check", Mock(return_value={"status": "passed"}))
    monkeypatch.setattr(
        bridge.OddsAPISource, "load", Mock(side_effect=ValueError("bad apiKey=unit-test-only"))
    )
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        bridge.main(["--export", str(export_path), "--output", str(output), "--live"])
    assert error.value.code == 1
    assert "unit-test-only" not in (output / "failure.json").read_text() + capsys.readouterr().err


# Purpose: Verify removed demo cli option rejected.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_removed_demo_cli_option_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        bridge.main(["--demo"])
    assert "--export" in capsys.readouterr().err


# Purpose: Verify cli requires explicit quote source.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_cli_requires_explicit_quote_source(
    export_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        bridge.main(["--export", str(export_path)])
    assert "Select --live" in capsys.readouterr().err
