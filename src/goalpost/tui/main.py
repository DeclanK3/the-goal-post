"""Interactive TUI with one event reader, recoverable actions and local diagnostics."""

import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import polars as pl
import ratatui as r

from goalpost.cli.build import DatasetRequest, generate_dataset
from goalpost.cli.simulate import SimulationRequest, run_simulation
from goalpost.data.nfl.fetcher import fetch_nflverse
from goalpost.tui.diagnostics import LOG, LogStream, configure_logging, redact

_MENU_OPTIONS = [
    "Generate Dataset",
    "Run Simulation",
    "Inspect Game",
    "Settings",
    "Exit",
]


def _draw_text(term, lines):
    term.clear()
    paragraph = r.Paragraph.from_text("\n".join(lines))
    paragraph.set_alignment("left")
    if not term.draw_paragraph(paragraph):
        raise RuntimeError("Terminal could not render the screen")


def _draw_main_menu(term, selected, status=""):
    lines = ["", "  GoalPost — Transition Modeling", ""]
    lines.extend(f"  {'>' if i == selected else ' '} {opt}" for i, opt in enumerate(_MENU_OPTIONS))
    lines.extend(["", "  j/k or ↑/↓ to navigate, Enter to select; q to exit", "", status])
    _draw_text(term, lines)


def _next_key(term):
    # The binding returns None on timeout/no event. Negative timeouts wrap to u64.
    while True:
        event = term.next_event_typed(100)
        if event is None or event.kind != "key":
            continue
        if event.code == r.KeyCode.Char and event.mods & r.KeyMods.CTRL:
            if chr(event.ch).lower() in ("c", "d"):
                raise KeyboardInterrupt
        return event


def _run_prompt_flow(term, title, prompt_labels):
    values = {}
    for label in prompt_labels:
        buffer = ""
        while True:
            _draw_text(
                term,
                [
                    "",
                    f"  {title}",
                    "",
                    *[f"  {key}: {value}" for key, value in values.items()],
                    f"  {label}: {buffer}▏",
                    "",
                    "  Type a value; Enter advances, Esc cancels",
                ],
            )
            event = _next_key(term)
            if event.code == r.KeyCode.Esc:
                return None
            if event.code == r.KeyCode.Enter:
                values[label] = buffer.strip()
                break
            if event.code == r.KeyCode.Backspace:
                buffer = buffer[:-1]
            elif event.code == r.KeyCode.Char:
                character = chr(event.ch)
                if character.isprintable():
                    buffer += character
    return values


def _required(values, label):
    value = values.get(label, "").strip()
    if not value:
        raise ValueError(f"{label} is required")
    return value


def _generate(term):
    labels = ["Sources directory (or download)", "Output directory", "Seasons (optional)"]
    values = _run_prompt_flow(term, "Generate Dataset", labels)
    if values is None:
        return "Canceled"
    source = _required(values, labels[0])
    output = Path(_required(values, labels[1])).expanduser()
    if source.lower() == "download":
        seasons = [int(s.strip()) for s in _required(values, labels[2]).split(",")]
        source_path = Path.home() / ".cache/goalpost/nflverse"
        fetch_nflverse(seasons, source_path)
    else:
        source_path = Path(source).expanduser()
    result = generate_dataset(DatasetRequest(source_path, output))
    return f"Built {result.observation_count} observations from {result.game_count} games"


def _simulate(term):
    labels = ["Dataset directory", "Week number", "Season year"]
    values = _run_prompt_flow(term, "Run Simulation", labels)
    if values is None:
        return "Canceled"
    request = SimulationRequest(
        Path(_required(values, labels[0])).expanduser(),
        int(_required(values, labels[1])),
        int(_required(values, labels[2])),
    )
    result = run_simulation(request)
    if not result.game_results:
        raise ValueError("No simulation results; check the dataset and matchup coverage")
    return f"Simulated {len(result.game_results)} games"


def _inspect(term):
    values = _run_prompt_flow(term, "Inspect Game", ["Parquet path"])
    if values is None:
        return "Canceled"
    frame = pl.read_parquet(Path(_required(values, "Parquet path")).expanduser())
    return f"Loaded {len(frame)} rows, columns: {frame.columns[:5]}"


def _settings(term):
    _draw_text(term, ["", "  Settings", "", "  No settings yet", "", "  Press any key to return"])
    _next_key(term)
    return ""


def _run_action(term, action, title, log_path):
    LOG.info("action started: %s", title)
    try:
        with redirect_stdout(LogStream()), redirect_stderr(LogStream()):
            result = action(term)
        LOG.info("action finished: %s", title)
        return result
    except Exception as error:
        LOG.exception("action failed: %s", title)
        return f"Error: {redact(str(error))}. Details: {log_path or 'tui.log'}"


def _run_app(term: r.Terminal, log_path=None):
    selected = 0
    status = ""
    screen = "main"

    while True:
        if screen == "main":
            _draw_main_menu(term, selected, status)
            evt = _next_key(term)
            if evt.code == r.KeyCode.Esc:
                return
            if evt.code == r.KeyCode.Char:
                ch = chr(evt.ch).lower()
                if ch == "j":
                    selected = (selected + 1) % len(_MENU_OPTIONS)
                elif ch == "k":
                    selected = (selected - 1) % len(_MENU_OPTIONS)
                elif ch == "q":
                    return
            elif evt.code == r.KeyCode.Down:
                selected = (selected + 1) % len(_MENU_OPTIONS)
            elif evt.code == r.KeyCode.Up:
                selected = (selected - 1) % len(_MENU_OPTIONS)
            elif evt.code == r.KeyCode.Enter:
                if selected == 0:
                    screen = "generate"
                elif selected == 1:
                    screen = "simulate"
                elif selected == 2:
                    screen = "inspect"
                elif selected == 3:
                    screen = "settings"
                elif selected == 4:
                    return

        elif screen == "generate":
            status = _run_action(term, _generate, "Generate Dataset", log_path)
            screen = "main"
        elif screen == "simulate":
            status = _run_action(term, _simulate, "Run Simulation", log_path)
            screen = "main"
        elif screen == "inspect":
            status = _run_action(term, _inspect, "Inspect Game", log_path)
            screen = "main"
        elif screen == "settings":
            status = _run_action(term, _settings, "Settings", log_path)
            screen = "main"


def run_tui():
    # Logging starts before terminal creation so startup failures are also recorded.
    log_path = configure_logging()
    LOG.info("session started")
    term = None
    failed = False
    try:
        term = r.Terminal()
        term.enable_raw()
        term.enter_alt()
        _run_app(term, log_path)
    except KeyboardInterrupt:
        LOG.info("session interrupted")
    except Exception:
        failed = True
        LOG.exception("terminal session failed")
    finally:
        if term is not None:
            # Attempt every restoration step even if an earlier one fails.
            for method in (term.disable_raw, term.show_cursor, term.leave_alt, term.close):
                try:
                    method()
                except Exception:
                    failed = True
                    LOG.exception("terminal restoration failed")
        LOG.info("session finished")
    if failed:
        print(f"GoalPost TUI failed. Details: {log_path}", file=sys.stderr)
        raise SystemExit(1)


def main():
    run_tui()


if __name__ == "__main__":
    main()
