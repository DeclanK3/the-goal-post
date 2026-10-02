"""Application service for running NFL simulations."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

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
    """Load vae_dataset.parquet and reconstruct transition models."""
    # For the TUI, we read the dataset and construct models from it
    import polars as pl

    parquet_path = dataset_dir / "vae_dataset.parquet"
    if not parquet_path.exists():
        return {}

    df = pl.read_parquet(parquet_path)
    models = {}
    # Group by game and context, reconstruct per-team models
    for row in df.to_dicts():
        game_id = row["game_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        ctx = row["context_index"]

        # Reconstruct matrices from flattened features
        # This is a simplified version - in practice you'd need to unflatten
        home_key = (home_team, game_id)
        away_key = (away_team, game_id)

        if home_key not in models:
            models[home_key] = NFLTransitionModel(home_team, game_id, away_team)
        if away_key not in models:
            models[away_key] = NFLTransitionModel(away_team, game_id, home_team)

    return models


def run_simulation(request: SimulationRequest) -> SimulationResult:
    models = _load_dataset_as_models(request.dataset_dir)
    if not models:
        return SimulationResult(game_results=[], week=request.week, season=request.season)

    # Build games list from available models
    game_ids = {key[1] for key in models.keys()}
    games = [{"game_id": gid, "home_team": "", "away_team": ""} for gid in game_ids]

    results = weekly_runner(games, models, n_sims=request.n_sims, seed=request.seed)
    return SimulationResult(
        game_results=results,
        week=request.week,
        season=request.season,
    )
