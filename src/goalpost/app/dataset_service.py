"""Application service for generating VAE-ready datasets from NFL pbp data."""
from dataclasses import dataclass
from pathlib import Path

import polars as pl

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
    output = request.output_dir
    output.mkdir(parents=True, exist_ok=True)

    builder = NFLTransitionBuilder()
    games = []
    for parquet_path in sorted(request.sources_dir.glob("*.parquet")):
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
