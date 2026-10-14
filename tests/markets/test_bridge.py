import json
from unittest.mock import Mock

import pytest

from goalpost.cli import report as bridge
from goalpost.cli.report import MarketTableJob
from goalpost.data.nfl.history import NFLHistory
from goalpost.data.nfl.odds import QuoteSnapshot
from goalpost.simulation.nfl.comparison import ComparisonSettings


@pytest.fixture
def fixed_clock(now, monkeypatch):
    class Clock:
        @staticmethod
        def now(zone=None):
            return now

    monkeypatch.setattr(bridge, "datetime", Clock)


def test_validate_only_does_not_load_quotes_or_history(export_path, tmp_path, monkeypatch):
    source = Mock()
    history = Mock(side_effect=AssertionError("No public requests"))
    monkeypatch.setattr(NFLHistory, "check", history)
    result = MarketTableJob(export_path, tmp_path / "out", source, ComparisonSettings(), True).run()
    assert result["status"] == "validated" and result["games"] == 1
    source.load.assert_not_called()
    history.assert_not_called()


def test_full_job_writes_all_rows_and_sorted_current(
    export_path, event, tmp_path, monkeypatch, fixed_clock
):
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


def test_incomplete_history_blocks_before_paid_request(
    export_path, tmp_path, monkeypatch, fixed_clock
):
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


def test_old_output_is_never_overwritten(export_path, tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        MarketTableJob(export_path, output, None, ComparisonSettings(), True).run()
    assert (output / "report.json").read_text() == "preserve"
    assert not (output / "failure.json").exists()


def test_empty_export_is_not_simulated_and_spends_no_request(
    export_path, tmp_path, monkeypatch, fixed_clock
):
    report = json.loads((export_path / "report.json").read_text())
    report.update(status="no_scheduled_matchups", results=[])
    (export_path / "report.json").write_text(json.dumps(report))
    source = Mock()
    output = tmp_path / "out"
    result = MarketTableJob(export_path, output, source, ComparisonSettings()).run()
    assert result["games"] == 0 and result["attempted"] == 0 and result["odds_requests"] == 0
    assert (output / "current-market-sizing.csv").read_text().startswith("game_id,")
    source.load.assert_not_called()


def test_source_error_is_sanitized_in_artifact_and_cli(
    export_path, tmp_path, monkeypatch, fixed_clock, capsys
):
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


def test_removed_demo_cli_option_rejected(capsys):
    with pytest.raises(SystemExit):
        bridge.main(["--demo"])
    assert "--export" in capsys.readouterr().err


def test_cli_requires_explicit_quote_source(export_path, capsys):
    with pytest.raises(SystemExit):
        bridge.main(["--export", str(export_path)])
    assert "Select --live" in capsys.readouterr().err
