"""Dataset build CLI entry point."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from goalpost.data.base.paths import ARTIFACTS
from goalpost.representation.nfl.extractor import extract_game
from goalpost.representation.nfl.builder import NFLTransitionBuilder


def _write_matrix(path: Path, key: str, matrix: np.ndarray):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"{key}.npy", matrix)


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
    manifest = []
    for (team, game_id), matrix in table.matrices.items():
        key = f"{team}_{game_id}"
        _write_matrix(output, key, matrix)
        manifest.append({"team": team, "game_id": game_id, "file": f"{key}.npy"})

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Built {len(manifest)} matrices from {len(games)} games")


if __name__ == "__main__":
    main()
