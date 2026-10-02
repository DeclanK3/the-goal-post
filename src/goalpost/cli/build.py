"""Dataset build CLI entry point."""
import argparse
from pathlib import Path

from goalpost.app.dataset_service import generate_dataset, DatasetRequest


def main():
    parser = argparse.ArgumentParser(description="Build GoalPost transition datasets")
    parser.add_argument("--sources", type=Path, required=True, help="Source data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    result = generate_dataset(DatasetRequest(
        sources_dir=Path(args.sources),
        output_dir=Path(args.output),
    ))
    print(f"Built {result.observation_count} context observations from {result.game_count} games -> {result.row_count} rows")


if __name__ == "__main__":
    main()
