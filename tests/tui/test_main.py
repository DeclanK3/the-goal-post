"""Exercise application flows offline without a native terminal or live data."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ratatui as r

from goalpost.tui import main as app


def key(code, ch=0, mods=0):
    return SimpleNamespace(kind="key", code=code, ch=ch, mods=mods)


def text(value):
    return [key(r.KeyCode.Char, ord(c)) for c in value]


ENTER = key(r.KeyCode.Enter)
ESC = key(r.KeyCode.Esc)
QUIT = key(r.KeyCode.Char, ord("q"))


@pytest.fixture
def terminal(monkeypatch, tmp_path):
    term = MagicMock()
    term.draw_paragraph.return_value = True
    monkeypatch.setattr(app.r, "Terminal", lambda: term)
    paragraph = MagicMock()
    monkeypatch.setattr(app.r.Paragraph, "from_text", lambda value: paragraph)
    monkeypatch.setenv("GOALPOST_LOG_DIR", str(tmp_path / "logs"))
    return term


@pytest.mark.parametrize("index", range(5))
def test_every_menu_option_can_return_and_exit(terminal, index):
    events = [key(r.KeyCode.Down)] * index + [ENTER]
    if index < 3:
        events += [None, SimpleNamespace(kind="resize"), ESC, QUIT]
    elif index == 3:
        events += [None, ENTER, QUIT]
    terminal.next_event_typed.side_effect = events
    app.run_tui()
    terminal.disable_raw.assert_called_once_with()
    terminal.show_cursor.assert_called_once_with()
    terminal.leave_alt.assert_called_once_with()
    terminal.close.assert_called_once_with()


def test_prompt_uses_events_not_input_and_accepts_q_in_paths(terminal, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("raw-mode input()"))
    terminal.next_event_typed.side_effect = (
        [None]
        + text("squadx")
        + [
            key(r.KeyCode.Backspace),
            ENTER,
            ENTER,
        ]
    )
    assert app._run_prompt_flow(terminal, "Test", ["Path", "Optional"]) == {
        "Path": "squad",
        "Optional": "",
    }


@pytest.mark.parametrize("index,action", [(0, "_generate"), (1, "_simulate"), (2, "_inspect")])
def test_operation_failure_is_logged_and_menu_remains_usable(
    terminal, monkeypatch, tmp_path, index, action
):
    def fail(term):
        raise ValueError("fixture failure")

    monkeypatch.setattr(app, action, fail)
    terminal.next_event_typed.side_effect = [key(r.KeyCode.Down)] * index + [ENTER, QUIT]
    app.run_tui()
    log = (tmp_path / "logs/tui.log").read_text()
    assert "action failed" in log and "Traceback" in log and "fixture failure" in log
    terminal.close.assert_called_once()


def test_prompt_failure_is_contained(terminal, monkeypatch):
    monkeypatch.setattr(app, "_run_prompt_flow", MagicMock(side_effect=OSError("prompt failure")))
    terminal.next_event_typed.side_effect = [ENTER, QUIT]
    app.run_tui()
    terminal.close.assert_called_once()


def test_generate_uses_shared_builder(terminal, monkeypatch, tmp_path):
    prompts = {
        "Sources directory (or download)": str(tmp_path),
        "Output directory": str(tmp_path / "out"),
        "Seasons (optional)": "",
    }
    monkeypatch.setattr(app, "_run_prompt_flow", lambda *a: prompts)
    build = MagicMock(return_value=SimpleNamespace(observation_count=9, game_count=1))
    monkeypatch.setattr(app, "generate_dataset", build)
    assert app._generate(terminal) == "Built 9 observations from 1 games"
    assert build.call_args.args[0].sources_dir == tmp_path


def test_simulation_does_not_report_empty_result_as_success(terminal, monkeypatch):
    monkeypatch.setattr(
        app,
        "_run_prompt_flow",
        lambda *a: {
            "Dataset directory": "/missing",
            "Week number": "5",
            "Season year": "2026",
        },
    )
    monkeypatch.setattr(app, "run_simulation", lambda *a: SimpleNamespace(game_results=[]))
    with pytest.raises(ValueError, match="No simulation results"):
        app._simulate(terminal)


def test_inspect_success(terminal, monkeypatch, tmp_path):
    import polars as pl

    path = tmp_path / "game.parquet"
    pl.DataFrame({"game_id": ["fixture"]}).write_parquet(path)
    monkeypatch.setattr(app, "_run_prompt_flow", lambda *a: {"Parquet path": str(path)})
    assert "Loaded 1 rows" in app._inspect(terminal)


def test_invalid_input_is_recoverable(terminal):
    terminal.next_event_typed.side_effect = (
        [key(r.KeyCode.Down), ENTER]
        + text("dataset")
        + [
            ENTER,
            *text("not-a-week"),
            ENTER,
            *text("2026"),
            ENTER,
            QUIT,
        ]
    )
    app.run_tui()
    terminal.close.assert_called_once()


def test_ctrl_c_restores_terminal(terminal):
    terminal.next_event_typed.side_effect = [ENTER, key(r.KeyCode.Char, ord("c"), r.KeyMods.CTRL)]
    app.run_tui()
    terminal.disable_raw.assert_called_once()
    terminal.close.assert_called_once()


def test_fatal_error_restores_terminal_and_reports_log(terminal, tmp_path, capsys):
    terminal.next_event_typed.side_effect = RuntimeError("event failure")
    with pytest.raises(SystemExit) as error:
        app.run_tui()
    assert error.value.code == 1
    assert str(tmp_path / "logs/tui.log") in capsys.readouterr().err
    terminal.disable_raw.assert_called_once()
    terminal.close.assert_called_once()


def test_cleanup_attempts_all_steps_even_if_one_fails(terminal):
    terminal.next_event_typed.side_effect = [QUIT]
    terminal.disable_raw.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(SystemExit):
        app.run_tui()
    terminal.show_cursor.assert_called_once()
    terminal.leave_alt.assert_called_once()
    terminal.close.assert_called_once()


def test_native_startup_failure_is_logged(terminal, monkeypatch, tmp_path):
    monkeypatch.setattr(app.r, "Terminal", MagicMock(side_effect=RuntimeError("missing backend")))
    with pytest.raises(SystemExit):
        app.run_tui()
    assert "missing backend" in (tmp_path / "logs/tui.log").read_text()


def test_module_cli_entry_point():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "goalpost", "--build", "--help"],
        capture_output=True,
        text=True,
        env=os.environ,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--sources" in result.stdout
