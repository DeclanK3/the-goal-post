"""Dataset build CLI entry point."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from goalpost.data.nfl.fetcher import fetch_nflverse
from goalpost.representation.nfl.extractor import extract_game, NFLExtractedGame
from goalpost.representation.nfl.builder import NFLTransitionBuilder
from goalpost.representation.nfl.context import extract_paired_observations
from goalpost.representation.nfl.flatten import NFLFlattener
from goalpost.representation.nfl.dataset import VAEDatasetBuilder


@dataclass
class DatasetRequest:
    sources_dir: Path
    output_dir: Path
    season: int = 2024


@dataclass
class DatasetResult:
    game_count: int
    observation_count: int
    row_count: int
    dataset_path: Path
    schema_path: Path


def generate_dataset(request: DatasetRequest) -> DatasetResult:
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

    return DatasetResult(
        game_count=len(games),
        observation_count=len(observations),
        row_count=len(df),
        dataset_path=dataset_path,
        schema_path=schema_path,
    )


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "goalpost" / "nflverse"


def main():
    parser = argparse.ArgumentParser(description="Build GoalPost transition datasets")
    parser.add_argument(
        "--sources",
        type=Path,
        help="Source data directory (required unless --seasons is used)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--seasons",
        type=lambda s: [int(x.strip()) for x in s.split(",")],
        help="Comma-separated list of NFL seasons to download from nflverse (e.g. 2023,2024,2025)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="Directory to cache downloaded nflverse parquet files (default: ~/.cache/goalpost/nflverse)",
    )
    args = parser.parse_args()

    if args.seasons and args.sources:
        parser.error("Cannot specify both --seasons and --sources")
    if not args.seasons and not args.sources:
        parser.error("Must specify either --seasons or --sources")

    sources_dir = Path(args.sources) if args.sources else args.cache_dir

    if args.seasons:
        print(f"Downloading nflverse seasons {args.seasons} to {args.cache_dir} ...")
        fetch_nflverse(args.seasons, args.cache_dir)
        print("Download complete.")

    result = generate_dataset(DatasetRequest(
        sources_dir=sources_dir,
        output_dir=Path(args.output),
    ))
    print(f"Built {result.observation_count} context observations from {result.game_count} games -> {result.row_count} rows")


if __name__ == "__main__":
    main()
