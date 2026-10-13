"""TUI screen renderers."""
import os


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def draw_main_menu(selected: int = 0) -> str:
    options = [
        "Generate Dataset",
        "Run Simulation",
        "Inspect Game",
        "Settings",
        "Exit",
    ]
    lines = [
        "",
        "  ┌──────────────────────────────┐",
        "  │        GoalPost               │",
        "  │  NFL Transition Modeling      │",
        "  └──────────────────────────────┘",
        "",
    ]
    for i, opt in enumerate(options):
        marker = ">" if i == selected else " "
        lines.append(f"  {marker} {opt}")
    lines.append("")
    lines.append("  Use arrow keys or j/k to navigate, Enter to select")
    return "\n".join(lines)


def draw_generate_dataset() -> str:
    return "\n".join([
        "",
        "  Generate VAE Dataset",
        "  ────────────────────",
        "",
        "  1. Enter sources directory (path to dataset parquets)",
        "     OR type 'download' to fetch from nflverse",
        "  2. Enter output directory",
        "  3. If downloading, enter seasons (e.g. 2023,2024,2025)",
        "  4. Press Enter to build",
        "",
        "  Press 'b' to go back",
    ])


def draw_run_simulation() -> str:
    return "\n".join([
        "",
        "  Run Simulation",
        "  ───────────────",
        "",
        "  1. Enter dataset directory",
        "  2. Enter week number",
        "  3. Enter season year",
        "  4. Press Enter to run",
        "",
        "  Press 'b' to go back",
    ])


def draw_inspect_game() -> str:
    return "\n".join([
        "",
        "  Inspect Game",
        "  ─────────────",
        "",
        "  Enter parquet file path:",
        "",
        "  Press 'b' to go back",
    ])


def draw_settings() -> str:
    return "\n".join([
        "",
        "  Settings",
        "  ────────",
        "",
        "  (No settings yet)",
        "",
        "  Press 'b' to go back",
    ])


def draw_result(message: str) -> str:
    return "\n".join([
        "",
        "  Result",
        "  ──────",
        "",
        f"  {message}",
        "",
        "  Press any key to return to menu",
    ])
