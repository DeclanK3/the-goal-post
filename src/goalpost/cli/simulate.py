"""Simulation CLI entry point."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import polars as pl

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.simulation.nfl.weekly import weekly_runner


@dataclass
class SimulationRequest:
    dataset_dir: Path
    week: int
    season: int
    n_sims: int = 1000
    seed: int = 42


@dataclass
class SimulationResult:
    game_results: List[dict]
    week: int
    season: int


def _load_dataset_as_models(dataset_dir: Path) -> dict:
    parquet_path = dataset_dir / "vae_dataset.parquet"
    if not parquet_path.exists():
        return {}

    df = pl.read_parquet(parquet_path)

    # Feature columns are prefixed by team and exclude metadata columns
    home_cols = [
        c for c in df.columns
        if c.startswith("home_") and c not in ("home_team", "home_score", "home_available")
    ]
    away_cols = [
        c for c in df.columns
        if c.startswith("away_") and c not in ("away_team", "away_score", "away_available")
    ]

    models = {}
    for game_id, group in df.group_by("game_id"):
        # The dataset stores one row per context_index; we need all 9 to
        # reconstruct the full flat vector expected by from_flat().
        rows = sorted(group.to_dicts(), key=lambda r: r["context_index"])
        if len(rows) != 9:
            continue

        home_flat = np.concatenate(
            [np.array([r[c] for c in home_cols], dtype=np.float32) for r in rows]
        )
        away_flat = np.concatenate(
            [np.array([r[c] for c in away_cols], dtype=np.float32) for r in rows]
        )

        home_team = rows[0]["home_team"]
        away_team = rows[0]["away_team"]

        home_model = NFLTransitionModel.from_flat(
            home_flat, home_team, game_id, away_team
        )
        away_model = NFLTransitionModel.from_flat(
            away_flat, away_team, game_id, home_team
        )

        # Lightweight per-game fallback: average of both team matrices
        blend = (home_model._matrix + away_model._matrix) / 2.0
        home_model._fallback_matrix = blend.astype(np.float32)
        away_model._fallback_matrix = blend.astype(np.float32)

        models[(home_team, game_id)] = home_model
        models[(away_team, game_id)] = away_model

    return models


def run_simulation(request: SimulationRequest) -> SimulationResult:
    models = _load_dataset_as_models(request.dataset_dir)
    if not models:
        return SimulationResult(game_results=[], week=request.week, season=request.season)

    game_ids = {key[1] for key in models.keys()}
    games = [{"game_id": gid, "home_team": "", "away_team": ""} for gid in game_ids]

    results = weekly_runner(games, models, n_sims=request.n_sims, seed=request.seed)
    return SimulationResult(
        game_results=results,
        week=request.week,
        season=request.season,
    )


def main():
    parser = argparse.ArgumentParser(description="Run GoalPost simulations")
    parser.add_argument("--week", type=int, required=True, help="NFL week number")
    parser.add_argument("--season", type=int, required=True, help="NFL season year")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--n-sims", type=int, default=1000, help="Simulations per game")
    args = parser.parse_args()

    result = run_simulation(SimulationRequest(
        dataset_dir=Path(args.dataset),
        week=args.week,
        season=args.season,
        n_sims=args.n_sims,
    ))
    for r in result.game_results:
        print(f"{r['home_team']} vs {r['away_team']}: spread={r['spread']}, total={r['total']}, ml={r['moneyline']}")
    print(f"Simulated {len(result.game_results)} games")


if __name__ == "__main__":
    main()
