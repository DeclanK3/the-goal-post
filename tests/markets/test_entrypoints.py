"""Both interfaces invoke the same report job; all quotes and schedules are mocked."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ratatui as r

from goalpost.cli import report
from goalpost.data.base.odds import QuoteSnapshot
from goalpost.data.nfl.history import NFLHistory
from goalpost.data.nfl.odds import OddsAPISource
from goalpost.tui.report import PROMPTS, generate_report, run_report_screen


def values(export_path, output, mode="validate"):
    return dict(zip(PROMPTS, [str(export_path), mode, "", str(output), "", ""]))


def test_cli_root_report_help_does_not_import_tui():
    result = subprocess.run(
        [sys.executable, "-m", "goalpost", "--report", "--help"],
        capture_output=True,
        text=True,
        env=os.environ,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--live" in result.stdout and "--export" in result.stdout


def test_tui_validation_runs_shared_job_without_quotes_or_history(
    export_path, tmp_path, monkeypatch
):
    history, odds = Mock(side_effect=AssertionError), Mock(side_effect=AssertionError)
    monkeypatch.setattr(NFLHistory, "check", history)
    monkeypatch.setattr(OddsAPISource, "load", odds)
    output = tmp_path / "validated"
    assert "Validated 1 games" in generate_report(values(export_path, output))
    assert json.loads((output / "input-validation.json").read_text())["status"] == "validated"
    history.assert_not_called()
    odds.assert_not_called()


def test_tui_live_uses_same_pricing_table_as_cli(export_path, tmp_path, event, now, monkeypatch):
    class Clock:
        @staticmethod
        def now(tz=None):
            return now

    monkeypatch.setattr(report, "datetime", Clock)
    monkeypatch.setattr(NFLHistory, "check", Mock(return_value={"status": "passed"}))
    odds = Mock(return_value=QuoteSnapshot((event,), "test", requests=1))
    monkeypatch.setattr(OddsAPISource, "load", odds)
    tui_output, cli_output = tmp_path / "tui", tmp_path / "cli"
    message = generate_report(values(export_path, tui_output, "live"))
    report.main(["--export", str(export_path), "--live", "--output", str(cli_output)])
    assert "Research/regulation-only" in message
    assert (tui_output / "ev-kelly.csv").read_bytes() == (cli_output / "ev-kelly.csv").read_bytes()
    assert odds.call_count == 2  # Exactly one for each explicit action.
    rows = json.loads((tui_output / "ev-kelly.json").read_text())
    assert rows and all(row["actual_contract_kelly"] is None for row in rows)


@pytest.mark.parametrize("mode", ["", "demo", "auto"])
def test_tui_requires_explicit_source(export_path, tmp_path, mode):
    with pytest.raises(ValueError, match="Choose live"):
        generate_report(values(export_path, tmp_path / "out", mode))


def test_cancel_never_constructs_a_job(monkeypatch):
    job = Mock(side_effect=AssertionError)
    monkeypatch.setattr("goalpost.tui.report.MarketTableJob", job)
    assert run_report_screen(None, lambda *a: None) == "Canceled"
    job.assert_not_called()


def test_menu_dispatches_generate_report(monkeypatch):
    from goalpost.tui import main as app

    term = Mock()
    enter = SimpleNamespace(kind="key", code=r.KeyCode.Enter, ch=0, mods=0)
    down = SimpleNamespace(kind="key", code=r.KeyCode.Down, ch=0, mods=0)
    quit_key = SimpleNamespace(kind="key", code=r.KeyCode.Char, ch=ord("q"), mods=0)
    index = app._MENU_OPTIONS.index("Generate Report")
    term.next_event_typed.side_effect = [down] * index + [enter, quit_key]
    monkeypatch.setattr(app, "_draw_main_menu", Mock())
    action = Mock(return_value="research report")
    monkeypatch.setattr("goalpost.tui.report.run_report_screen", action)
    app._run_app(term)
    action.assert_called_once_with(term, app._run_prompt_flow)


def test_ui_failure_does_not_echo_key(monkeypatch):
    from goalpost.tui import main as app

    term = Mock()

    def key(code, ch=0):
        return SimpleNamespace(kind="key", code=code, ch=ch, mods=0)

    term.next_event_typed.side_effect = [key(r.KeyCode.Down)] * 4 + [
        key(r.KeyCode.Enter),
        key(r.KeyCode.Char, ord("q")),
    ]
    monkeypatch.setenv("ODDS_API", "test-only-secret")
    draw = Mock()
    monkeypatch.setattr(app, "_draw_main_menu", draw)
    monkeypatch.setattr(
        "goalpost.tui.report.run_report_screen", Mock(side_effect=RuntimeError("test-only-secret"))
    )
    app._run_app(term)
    messages = [call.args[2] for call in draw.call_args_list]
    assert any("[redacted]" in message for message in messages)
    assert all("test-only-secret" not in message for message in messages)
