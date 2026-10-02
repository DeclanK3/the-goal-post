"""Simulation CLI entry point."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run GoalPost simulations")
    parser.add_argument("--week", type=int, required=True, help="NFL week number")
    parser.add_argument("--season", type=int, required=True, help="NFL season year")
    parser.add_argument("--models", type=Path, required=True, help="Models directory")
    args = parser.parse_args()
    print(f"Simulating week {args.week} of season {args.season}")


if __name__ == "__main__":
    main()
