"""Interactive TUI entry point for GoalPost using ratatui."""
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import ratatui as r

from goalpost.tui.app_state import AppState
from goalpost.data.nfl.fetcher import fetch_nflverse
from goalpost.representation.nfl.extractor import extract_game, NFLExtractedGame
from goalpost.representation.nfl.builder import NFLTransitionBuilder
from goalpost.representation.nfl.context import extract_paired_observations
from goalpost.representation.nfl.flatten import NFLFlattener
from goalpost.representation.nfl.dataset import VAEDatasetBuilder
from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.simulation.nfl.weekly import weekly_runner


_MENU_OPTIONS = [
    "Generate Dataset",
    "Run Simulation",
    "Inspect Game",
    "Settings",
    "Exit",
]


@dataclass
class _DatasetRequest:
    sources_dir: Path
    output_dir: Path
    season: int = 2024


@dataclass
class _DatasetResult:
    game_count: int
    observation_count: int
    row_count: int
    dataset_path: Path
    schema_path: Path


def _generate_dataset(request: _DatasetRequest) -> _DatasetResult:
    if not request.sources_dir.exists():
        raise FileNotFoundError(f"Sources directory does not exist: {request.sources_dir}")

    parquet_paths = sorted(request.sources_dir.glob("*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No .parquet files found in {request.sources_dir}")

    output = request.output_dir
    output.mkdir(parents=True, exist_ok=True)

    builder = NFLTransitionBuilder()
    games = []
    for parquet_path in parquet_paths:
        df = pl.read_parquet(parquet_path)
        for key_tuple, g in df.group_by("game_id"):
            try:
                game = extract_game(g)
                games.append(game)
            except ValueError:
                continue

    table = builder.build(games)

    flattener = NFLFlattener()
    dataset_builder = VAEDatasetBuilder(flattener)

    print("games", games)
    
    observations = []
    for game in games:
        if not isinstance(game, NFLExtractedGame):
            continue
        home_key = (game.home, game.game_id)
        away_key = (game.away, game.game_id)
        home_mat = table.matrices.get(home_key)
        away_mat = table.matrices.get(away_key)
        if home_mat is None or away_mat is None:
            continue
        observations.extend(extract_paired_observations(game, home_mat, away_mat, season=request.season))

    df = dataset_builder.build(observations)
    dataset_path = output / "vae_dataset.parquet"
    schema_path = output / "vae_schema.json"
    dataset_builder.write(df, dataset_path)
    dataset_builder.write_schema(schema_path)

    return _DatasetResult(
        game_count=len(games),
        observation_count=len(observations),
        row_count=len(df),
        dataset_path=dataset_path,
        schema_path=schema_path,
    )


@dataclass
class _SimulationRequest:
    dataset_dir: Path
    week: int
    season: int
    n_sims: int = 1000
    seed: int = 42


@dataclass
class _SimulationResult:
    game_results: list[dict]
    week: int
    season: int


def _load_dataset_as_models(dataset_dir: Path) -> dict:
    parquet_path = dataset_dir / "vae_dataset.parquet"
    if not parquet_path.exists():
        return {}

    df = pl.read_parquet(parquet_path)
    models = {}
    for row in df.to_dicts():
        game_id = row["game_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_key = (home_team, game_id)
        away_key = (away_team, game_id)

        if home_key not in models:
            models[home_key] = NFLTransitionModel(home_team, game_id, away_team)
        if away_key not in models:
            models[away_key] = NFLTransitionModel(away_team, game_id, home_team)

    return models


def _run_simulation(request: _SimulationRequest) -> _SimulationResult:
    models = _load_dataset_as_models(request.dataset_dir)
    if not models:
        return _SimulationResult(game_results=[], week=request.week, season=request.season)

    game_ids = {key[1] for key in models.keys()}
    games = [{"game_id": gid, "home_team": "", "away_team": ""} for gid in game_ids]

    results = weekly_runner(games, models, n_sims=request.n_sims, seed=request.seed)
    return _SimulationResult(
        game_results=results,
        week=request.week,
        season=request.season,
    )


def _draw_main_menu(term: r.Terminal, selected: int, status: str = ""):
    term.clear()
    lines = [
        "",
        "        GoalPost",
        "  Transition Modeling",
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
                "Sources directory (or 'download' to fetch nflverse)",
                "Output directory",
                "Seasons (comma-separated, e.g. 2023,2024,2025) (optional)",
            ])
            if values is None:
                screen = "main"
                continue
            try:
                sources_input = values.get("Sources directory (or 'download' to fetch nflverse)", "").strip()
                output_dir = Path(values["Output directory"])
                seasons_input = values.get("Seasons (comma-separated, e.g. 2023,2024,2025) (optional)", "").strip()

                if sources_input.lower() == "download":
                    if not seasons_input:
                        raise ValueError("Seasons are required when using 'download' as sources")
                    seasons = [int(s.strip()) for s in seasons_input.split(",") if s.strip()]
                    cache_dir = Path.home() / ".cache" / "goalpost" / "nflverse"
                    print(f"Downloading nflverse seasons {seasons} to {cache_dir} ...")
                    fetch_nflverse(seasons, cache_dir)
                    print("Download complete.")
                    sources_dir = cache_dir
                else:
                    sources_dir = Path(sources_input)

                result = _generate_dataset(_DatasetRequest(
                    sources_dir=sources_dir,
                    output_dir=output_dir,
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
                result = _run_simulation(_SimulationRequest(
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
