"""Dataset build CLI entry point."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from goalpost.representation.nfl.extractor import extract_game, NFLExtractedGame
from goalpost.representation.nfl.builder import NFLTransitionBuilder
from goalpost.representation.nfl.context import extract_paired_observations
from goalpost.representation.nfl.flatten import NFLFlattener
from goalpost.representation.nfl.dataset import VAEDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Build GoalPost transition datasets")
    parser.add_argument("--sources", type=Path, required=True, help="Source data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    sources = Path(args.sources)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    builder = NFLTransitionBuilder()
    games = []
    for parquet_path in sorted(sources.glob("*.parquet")):
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
        observations.extend(extract_paired_observations(game, home_mat, away_mat, season=2024))

    df = dataset_builder.build(observations)
    dataset_builder.write(df, output / "vae_dataset.parquet")
    dataset_builder.write_schema(output / "vae_schema.json")
    print(f"Built {len(observations)} context observations from {len(games)} games -> {len(df)} rows")


if __name__ == "__main__":
    main()
