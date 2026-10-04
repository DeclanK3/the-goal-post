"""Simulation CLI entry point."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import polars as pl

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.builder import N_CONTEXTS, N_PHYSICAL_STATES
from goalpost.representation.nfl.timing import load_timing_for_game
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
    from goalpost.representation.nfl.builder import N_CONTEXTS, N_STATES

    parquet_path = dataset_dir / "vae_dataset.parquet"
    if not parquet_path.exists():
        return {}

    df = pl.read_parquet(parquet_path)

    # Feature columns are prefixed by team and exclude metadata columns
    home_cols = [
        c for c in df.columns
        if c.startswith("home_") and c not in (
            "home_team", "home_score", "home_available",
            "home_prior_available",
        ) and not c.startswith("home_prior_")
    ]
    away_cols = [
        c for c in df.columns
        if c.startswith("away_") and c not in (
            "away_team", "away_score", "away_available",
            "away_prior_available",
        ) and not c.startswith("away_prior_")
    ]
    # Prior columns (may not exist in older datasets)
    home_prior_cols = [c for c in df.columns if c.startswith("home_prior_") and c != "home_prior_available"]
    away_prior_cols = [c for c in df.columns if c.startswith("away_prior_") and c != "away_prior_available"]
    global_prior_cols = [c for c in df.columns if c.startswith("global_prior_")]

    models = {}
    for game_id_tuple, group in df.group_by("game_id"):
        # game_id may be wrapped in a tuple by polars group_by
        game_id = game_id_tuple[0] if isinstance(game_id_tuple, tuple) else game_id_tuple
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

        def _reconstruct_matrix(prior_cols, rows, flattener):
            """Unflatten per-context columns into a full (N_CONTEXTS, N, N) matrix."""
            mats = []
            for r in rows:
                flat = np.array([r[c] for c in prior_cols], dtype=np.float32)
                mats.append(flattener.unflatten(flat))
            return np.stack(mats, axis=0)

        # Load layered priors (layer 2 = team prior, layer 1 = global prior)
        if home_prior_cols:
            home_model.load_prior(_reconstruct_matrix(home_prior_cols, rows, home_model._flattener))
        if away_prior_cols:
            away_model.load_prior(_reconstruct_matrix(away_prior_cols, rows, away_model._flattener))
        if global_prior_cols:
            gp = _reconstruct_matrix(global_prior_cols, rows, home_model._flattener)
            home_model.load_matrix(home_model._matrix, fallback_matrix=gp)
            away_model.load_matrix(away_model._matrix, fallback_matrix=gp)

        # Load start-state probabilities
        start_cols = [c for c in rows[0].keys() if c.startswith("start_ctx_")]
        if start_cols:
            start_probs = np.zeros((N_CONTEXTS, N_PHYSICAL_STATES), dtype=np.float32)
            for r in rows:
                ctx = r["context_index"]
                for c in start_cols:
                    parts = c.split("_")
                    if len(parts) == 5 and parts[1] == "ctx" and parts[3] == "state":
                        s = int(parts[4])
                        start_probs[ctx, s] = float(r[c])
            home_model.load_start_states(start_probs)
            away_model.load_start_states(start_probs)

        # Load timing
        timing_mean = rows[0].get("global_timing_mean")
        if timing_mean is not None:
            home_model.load_timing(float(timing_mean))
            away_model.load_timing(float(timing_mean))

        # Load empirical timing from raw nflverse pbp (separate from VAE dataset)
        gid_str = game_id[0] if isinstance(game_id, tuple) else game_id
        timing = load_timing_for_game(gid_str)
        if timing is not None:
            home_model._timing = timing
            away_model._timing = timing

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
