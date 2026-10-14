# Purpose: Verify menu recovery and terminal cleanup using an offline event stream.
# Public API: pytest tests and local fixtures in this module.
# Consumers: The offline unit suite; temporary files never access production stores.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ratatui as r

from goalpost.tui import main as app


# Purpose: Construct a key event matching the fields consumed by the TUI loop.
# Callers: Offline menu, prompt and failure-path tests.
def key(code: int, ch: int = 0, mods: int = 0) -> SimpleNamespace:
    return SimpleNamespace(kind="key", code=code, ch=ch, mods=mods)


# Purpose: Translate a string into character events without native terminal input.
# Callers: Prompt and input-recovery tests.
def text(value: str) -> list[SimpleNamespace]:
    return [key(r.KeyCode.Char, ord(c)) for c in value]


ENTER: SimpleNamespace = key(r.KeyCode.Enter)
ESC: SimpleNamespace = key(r.KeyCode.Esc)
QUIT: SimpleNamespace = key(r.KeyCode.Char, ord("q"))


# Purpose: Replace native rendering and input with an isolated event-driven test double.
# Callers: pytest TUI flows; log output stays in tmp_path.
@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    term = MagicMock()
    term.draw_paragraph.return_value = True
    monkeypatch.setattr(app.r, "Terminal", lambda: term)
    paragraph = MagicMock()
    monkeypatch.setattr(app.r.Paragraph, "from_text", lambda value: paragraph)
    monkeypatch.setenv("GOALPOST_LOG_DIR", str(tmp_path / "logs"))
    return term


# Purpose: Verify every menu option can return and exit.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize("index", range(len(app._MENU_OPTIONS)))
def test_every_menu_option_can_return_and_exit(terminal: MagicMock, index: int) -> None:
    events = [key(r.KeyCode.Down)] * index + [ENTER]
    if app._MENU_OPTIONS[index] in (
        "Generate Dataset",
        "Run Simulation",
        "Inspect Game",
        "Generate Report",
    ):
        events += [None, SimpleNamespace(kind="resize"), ESC, QUIT]
    elif index == 3:
        events += [None, ENTER, QUIT]
    terminal.next_event_typed.side_effect = events
    app.run_tui()
    terminal.disable_raw.assert_called_once_with()
    terminal.show_cursor.assert_called_once_with()
    terminal.leave_alt.assert_called_once_with()
    terminal.close.assert_called_once_with()


# Purpose: Verify prompt uses events not input and accepts q in paths.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_prompt_uses_events_not_input_and_accepts_q_in_paths(
    terminal: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
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


# Purpose: Verify operation failure is logged and menu remains usable.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize("index,action", [(0, "_generate"), (1, "_simulate"), (2, "_inspect")])
def test_operation_failure_is_logged_and_menu_remains_usable(
    terminal: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, index: int, action: str
) -> None:
    # Purpose: Inject an operation error to verify recovery and diagnostics.
    # Callers: The patched action selected by a TUI failure-path test.
    def fail(term: MagicMock) -> None:
        raise ValueError("fixture failure")

    monkeypatch.setattr(app, action, fail)
    terminal.next_event_typed.side_effect = [key(r.KeyCode.Down)] * index + [ENTER, QUIT]
    app.run_tui()
    log = (tmp_path / "logs/tui.log").read_text()
    assert "action failed" in log and "Traceback" in log and "fixture failure" in log
    terminal.close.assert_called_once()


# Purpose: Verify prompt failure is contained.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_prompt_failure_is_contained(terminal: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_run_prompt_flow", MagicMock(side_effect=OSError("prompt failure")))
    terminal.next_event_typed.side_effect = [ENTER, QUIT]
    app.run_tui()
    terminal.close.assert_called_once()


# Purpose: Verify generate uses shared builder.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_generate_uses_shared_builder(
    terminal: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


# Purpose: Verify simulation does not report empty result as success.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_simulation_does_not_report_empty_result_as_success(
    terminal: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
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


# Purpose: Verify inspect success.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_inspect_success(
    terminal: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import polars as pl

    path = tmp_path / "game.parquet"
    pl.DataFrame({"game_id": ["fixture"]}).write_parquet(path)
    monkeypatch.setattr(app, "_run_prompt_flow", lambda *a: {"Parquet path": str(path)})
    assert "Loaded 1 rows" in app._inspect(terminal)


# Purpose: Verify invalid input is recoverable.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_invalid_input_is_recoverable(terminal: MagicMock) -> None:
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


# Purpose: Verify ctrl c restores terminal.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_ctrl_c_restores_terminal(terminal: MagicMock) -> None:
    terminal.next_event_typed.side_effect = [ENTER, key(r.KeyCode.Char, ord("c"), r.KeyMods.CTRL)]
    app.run_tui()
    terminal.disable_raw.assert_called_once()
    terminal.close.assert_called_once()


# Purpose: Verify fatal error restores terminal and reports log.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_fatal_error_restores_terminal_and_reports_log(
    terminal: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal.next_event_typed.side_effect = RuntimeError("event failure")
    with pytest.raises(SystemExit) as error:
        app.run_tui()
    assert error.value.code == 1
    assert str(tmp_path / "logs/tui.log") in capsys.readouterr().err
    terminal.disable_raw.assert_called_once()
    terminal.close.assert_called_once()


# Purpose: Verify cleanup attempts all steps even if one fails.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_cleanup_attempts_all_steps_even_if_one_fails(terminal: MagicMock) -> None:
    terminal.next_event_typed.side_effect = [QUIT]
    terminal.disable_raw.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(SystemExit):
        app.run_tui()
    terminal.show_cursor.assert_called_once()
    terminal.leave_alt.assert_called_once()
    terminal.close.assert_called_once()


# Purpose: Verify native startup failure is logged.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_native_startup_failure_is_logged(
    terminal: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app.r, "Terminal", MagicMock(side_effect=RuntimeError("missing backend")))
    with pytest.raises(SystemExit):
        app.run_tui()
    assert "missing backend" in (tmp_path / "logs/tui.log").read_text()


# Purpose: Verify module cli entry point.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_module_cli_entry_point() -> None:
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
