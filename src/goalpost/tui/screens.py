# Purpose: Build text-only menu and help screens without starting terminal event handling.
# Public API: clear and draw_* renderers returning screen text.
# Consumers: Screen tests and callers that need plain text; tui.main owns the active UI.

from __future__ import annotations

import os


# Purpose: Clear a conventional console for callers using the text screen helpers.
# Callers: Optional text-screen consumers; the active ratatui loop owns its own clearing.
def clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


# Purpose: Render the ordered menu with one selected item as plain text.
# Callers: Screen tests and text menu consumers.
def draw_main_menu(selected: int = 0) -> str:
    options = [
        "Generate Dataset",
        "Run Simulation",
        "Inspect Game",
        "Settings",
        "Generate Report",
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


# Purpose: Describe dataset input fields without starting a download or build.
# Callers: Text-screen consumers; the interactive loop handles actual prompts.
def draw_generate_dataset() -> str:
    return "\n".join(
        [
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
        ]
    )


# Purpose: Describe required simulation inputs without executing a simulation.
# Callers: Text-screen consumers; tui.main owns interactive execution.
def draw_run_simulation() -> str:
    return "\n".join(
        [
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
        ]
    )


# Purpose: Describe the parquet inspection input without reading a file.
# Callers: Text-screen consumers.
def draw_inspect_game() -> str:
    return "\n".join(
        [
            "",
            "  Inspect Game",
            "  ─────────────",
            "",
            "  Enter parquet file path:",
            "",
            "  Press 'b' to go back",
        ]
    )


# Purpose: Show that configurable settings are not yet exposed by this text screen.
# Callers: Text-screen consumers.
def draw_settings() -> str:
    return "\n".join(
        [
            "",
            "  Settings",
            "  ────────",
            "",
            "  (No settings yet)",
            "",
            "  Press 'b' to go back",
        ]
    )


# Purpose: Wrap an operation result in a plain-text result screen.
# Callers: Text-screen consumers and renderer tests.
def draw_result(message: str) -> str:
    return "\n".join(
        [
            "",
            "  Result",
            "  ──────",
            "",
            f"  {message}",
            "",
            "  Press any key to return to menu",
        ]
    )
