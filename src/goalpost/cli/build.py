"""Dataset build CLI entry point."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build GoalPost transition datasets")
    parser.add_argument("--sources", type=Path, required=True, help="Source data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    print(f"Building from {args.sources} to {args.output}")


if __name__ == "__main__":
    main()
