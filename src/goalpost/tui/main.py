"""Interactive TUI entry point for GoalPost using ratatui."""
import sys
from pathlib import Path

import ratatui as r

from goalpost.tui.app_state import AppState
from goalpost.app.dataset_service import generate_dataset, DatasetRequest
from goalpost.app.simulation_service import run_simulation, SimulationRequest


_MENU_OPTIONS = [
    "Generate Dataset",
    "Run Simulation",
    "Inspect Game",
    "Settings",
    "Exit",
]


def _draw_main_menu(term: r.Terminal, selected: int, status: str = ""):
    term.clear()
    lines = [
        "",
        "        GoalPost",
        "  NFL Transition Modeling",
        "",
    ]
    for i, opt in enumerate(_MENU_OPTIONS):
        marker = ">" if i == selected else " "
        lines.append(f"  {marker} {opt}")
    lines.append("")
    lines.append("  j/k or \u2191/\u2193 to navigate, Enter to select")
    if status:
        lines.append("")
        lines.append(f"  {status}")

    p = r.Paragraph.from_text("\n".join(lines))
    p.set_alignment("left")
    term.draw_paragraph(p)


def _draw_prompt(term: r.Terminal, title: str, prompts: list, values: dict):
    term.clear()
    lines = ["", f"  {title}", "  " + "-" * len(title), ""]
    for label in prompts:
        lines.append(f"  {label}: {values.get(label, '')}")
    lines.append("")
    lines.append("  Press Enter on empty line to confirm, 'q' to go back")
    p = r.Paragraph.from_text("\n".join(lines))
    p.set_alignment("left")
    term.draw_paragraph(p)


def _draw_result(term: r.Terminal, message: str):
    term.clear()
    lines = ["", "  Result", "  ------", ""]
    for line in message.splitlines():
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  Press any key to return")
    p = r.Paragraph.from_text("\n".join(lines))
    p.set_alignment("left")
    term.draw_paragraph(p)


def _run_prompt_flow(term: r.Terminal, title: str, prompt_labels: list) -> dict | None:
    values = {}
    current = 0
    while current < len(prompt_labels):
        _draw_prompt(term, title, prompt_labels, values)
        term.show_cursor()
        evt = term.next_event_typed(-1)
        if evt.kind != "key":
            continue
        if evt.code == r.KeyCode.Esc or (evt.code == r.KeyCode.Char and chr(evt.ch).lower() == "q"):
            return None
        if evt.code == r.KeyCode.Enter:
            # read line from terminal
            term.show_cursor()
            buf = input(f"  {prompt_labels[current]}: ").strip()
            if buf:
                values[prompt_labels[current]] = buf
            current += 1
    return values


def run_tui():
    term = r.Terminal()
    try:
        term.enable_raw()
        _run_app(term)
    finally:
        term.disable_raw()
        term.show_cursor()
        term.clear()


def _run_app(term: r.Terminal):
    selected = 0
    status = ""
    screen = "main"

    while True:
        if screen == "main":
            _draw_main_menu(term, selected, status)
            status = ""
            evt = term.next_event_typed(-1)
            if evt.kind != "key":
                continue
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
            values = _run_prompt_flow(term, "Generate Dataset", [
                "Sources directory",
                "Output directory",
            ])
            if values is None:
                screen = "main"
                continue
            try:
                result = generate_dataset(DatasetRequest(
                    sources_dir=Path(values["Sources directory"]),
                    output_dir=Path(values["Output directory"]),
                ))
                status = f"Built {result.observation_count} observations from {result.game_count} games"
            except Exception as e:
                status = f"Error: {e}"
            screen = "main"

        elif screen == "simulate":
            values = _run_prompt_flow(term, "Run Simulation", [
                "Dataset directory",
                "Week number",
                "Season year",
            ])
            if values is None:
                screen = "main"
                continue
            try:
                result = run_simulation(SimulationRequest(
                    dataset_dir=Path(values["Dataset directory"]),
                    week=int(values["Week number"]),
                    season=int(values["Season year"]),
                ))
                lines = [f"Simulated {len(result.game_results)} games"]
                for res in result.game_results[:3]:
                    lines.append(f"  {res['home_team']} vs {res['away_team']}: spread={res['spread']}")
                status = "\n".join(lines)
            except Exception as e:
                status = f"Error: {e}"
            screen = "main"

        elif screen == "inspect":
            values = _run_prompt_flow(term, "Inspect Game", [
                "Parquet path",
            ])
            if values is None:
                screen = "main"
                continue
            try:
                import polars as pl
                df = pl.read_parquet(values["Parquet path"])
                status = f"Loaded {len(df)} rows, columns: {df.columns[:5]}..."
            except Exception as e:
                status = f"Error: {e}"
            screen = "main"

        elif screen == "settings":
            term.clear()
            p = r.Paragraph.from_text("\n".join([
                "", "  Settings", "  --------", "", "  (No settings yet)", "", "  Press any key to go back"
            ]))
            p.set_alignment("left")
            term.draw_paragraph(p)
            term.next_event_typed(-1)
            screen = "main"


def main():
    run_tui()


if __name__ == "__main__":
    main()
