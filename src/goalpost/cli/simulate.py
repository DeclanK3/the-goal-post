"""Simulation CLI entry point."""
import argparse
from pathlib import Path

from goalpost.app.simulation_service import run_simulation, SimulationRequest


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
