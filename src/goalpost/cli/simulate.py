"""Simulation CLI entry point."""
import argparse
import json
from pathlib import Path

import numpy as np

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.simulation.nfl.weekly import weekly_runner


def _load_manifest(models_dir: Path):
    manifest_path = models_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text())


def _load_models(models_dir: Path, manifest: list):
    models = {}
    for entry in manifest:
        key = (entry["team"], entry["game_id"])
        matrix = np.load(models_dir / entry["file"])
        model = NFLTransitionModel(entry["team"], entry["game_id"], "")
        model.load_matrix(matrix)
        models[key] = model
    return models


def main():
    parser = argparse.ArgumentParser(description="Run GoalPost simulations")
    parser.add_argument("--week", type=int, required=True, help="NFL week number")
    parser.add_argument("--season", type=int, required=True, help="NFL season year")
    parser.add_argument("--models", type=Path, required=True, help="Models directory")
    parser.add_argument("--schedule", type=Path, required=True, help="Schedule CSV")
    parser.add_argument("--n-sims", type=int, default=1000, help="Simulations per game")
    args = parser.parse_args()

    models_dir = Path(args.models)
    manifest = _load_manifest(models_dir)
    models = _load_models(models_dir, manifest)

    # Build games list from manifest (all games in models)
    game_ids = {entry["game_id"] for entry in manifest}
    games = [{"game_id": gid, "home_team": "", "away_team": ""} for gid in game_ids]

    results = weekly_runner(games, models, n_sims=args.n_sims, seed=42)
    for r in results:
        print(f"{r['home_team']} vs {r['away_team']}: spread={r['spread']}, total={r['total']}, ml={r['moneyline']}")
    print(f"Simulated {len(results)} games")


if __name__ == "__main__":
    main()
