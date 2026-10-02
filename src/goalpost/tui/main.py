"""Interactive TUI entry point for GoalPost."""
import sys
import os

from goalpost.tui.app_state import AppState
from goalpost.tui.screens import (
    draw_main_menu,
    draw_generate_dataset,
    draw_run_simulation,
    draw_inspect_game,
    draw_settings,
    draw_result,
)


def clear():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def prompt(text: str) -> str:
    return input(f"  {text}: ").strip()


def run_tui():
    state = AppState()
    screen = "main"
    selected = 0
    status = ""

    while True:
        clear()
        if screen == "main":
            print(draw_main_menu(selected))
            if status:
                print(f"\n  {status}")
                status = ""

            try:
                key = input("\n  ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)

            if key in ("j", "J"):
                selected = (selected + 1) % 5
            elif key in ("k", "K"):
                selected = (selected - 1) % 5
            elif key == "":
                if selected == 0:
                    screen = "generate"
                elif selected == 1:
                    screen = "simulate"
                elif selected == 2:
                    screen = "inspect"
                elif selected == 3:
                    screen = "settings"
                elif selected == 4:
                    sys.exit(0)

        elif screen == "generate":
            print(draw_generate_dataset())
            src = prompt("Sources directory")
            if src.lower() == "b":
                screen = "main"
                continue
            out = prompt("Output directory")
            if out.lower() == "b":
                screen = "main"
                continue

            from pathlib import Path
            from goalpost.app.dataset_service import generate_dataset, DatasetRequest

            try:
                result = generate_dataset(DatasetRequest(
                    sources_dir=Path(src),
                    output_dir=Path(out),
                ))
                status = f"Built {result.observation_count} observations from {result.game_count} games"
            except Exception as e:
                status = f"Error: {e}"
            screen = "main"

        elif screen == "simulate":
            print(draw_run_simulation())
            dataset_dir = prompt("Dataset directory")
            if dataset_dir.lower() == "b":
                screen = "main"
                continue
            week = prompt("Week number")
            if week.lower() == "b":
                screen = "main"
                continue
            season = prompt("Season year")
            if season.lower() == "b":
                screen = "main"
                continue

            from pathlib import Path
            from goalpost.app.simulation_service import run_simulation, SimulationRequest

            try:
                result = run_simulation(SimulationRequest(
                    dataset_dir=Path(dataset_dir),
                    week=int(week),
                    season=int(season),
                ))
                lines = [f"Simulated {len(result.game_results)} games"]
                for r in result.game_results[:3]:
                    lines.append(f"  {r['home_team']} vs {r['away_team']}: spread={r['spread']}")
                status = "\n".join(lines)
            except Exception as e:
                status = f"Error: {e}"
            screen = "main"

        elif screen == "inspect":
            print(draw_inspect_game())
            path = prompt("Parquet path")
            if path.lower() == "b":
                screen = "main"
                continue
            try:
                import polars as pl
                df = pl.read_parquet(path)
                status = f"Loaded {len(df)} rows, columns: {df.columns[:5]}..."
            except Exception as e:
                status = f"Error: {e}"
            screen = "main"

        elif screen == "settings":
            print(draw_settings())
            input("\n  Press any key to go back...")
            screen = "main"


def main():
    run_tui()


if __name__ == "__main__":
    main()
