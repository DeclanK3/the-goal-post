"""VAE dataset construction from paired context observations."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import polars as pl

from .context import PairedContextObservation
from .flatten import NFLFlattener
from .builder import N_CONTEXTS


@dataclass
class VAEDatasetBuilder:
    flattener: NFLFlattener

    def build(self, observations: List[PairedContextObservation]) -> pl.DataFrame:
        if not observations:
            return pl.DataFrame()

        rows = []
        for obs in observations:
            ctx = obs.context
            home_flat = self.flattener.flatten(obs.home_matrix)
            away_flat = self.flattener.flatten(obs.away_matrix)

            row = {
                "game_id": ctx.game_id,
                "season": ctx.season,
                "week": ctx.week,
                "date": ctx.date,
                "home_team": ctx.home_team,
                "away_team": ctx.away_team,
                "context_index": ctx.context_index,
                "time_bin": ctx.time_bin,
                "score_bin": ctx.score_bin,
                "home_score": ctx.home_score,
                "away_score": ctx.away_score,
                "home_available": obs.home_available,
                "away_available": obs.away_available,
            }

            home_names = self.flattener.feature_names("home")
            away_names = self.flattener.feature_names("away")
            for name, val in zip(home_names, home_flat):
                row[name] = float(val)
            for name, val in zip(away_names, away_flat):
                row[name] = float(val)

            rows.append(row)

        schema = self._build_schema()
        return pl.DataFrame(rows, schema=schema)

    def _build_schema(self) -> dict:
        base = {
            "game_id": pl.Utf8,
            "season": pl.Int32,
            "week": pl.Int32,
            "date": pl.Utf8,
            "home_team": pl.Utf8,
            "away_team": pl.Utf8,
            "context_index": pl.Int32,
            "time_bin": pl.Int32,
            "score_bin": pl.Int32,
            "home_score": pl.Int32,
            "away_score": pl.Int32,
            "home_available": pl.Boolean,
            "away_available": pl.Boolean,
        }
        for name in self.flattener.feature_names("home"):
            base[name] = pl.Float32
        for name in self.flattener.feature_names("away"):
            base[name] = pl.Float32
        return base

    def write(self, df: pl.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd")

    def write_schema(self, path: Path):
        schema = self._build_schema()
        manifest = {
            "context_columns": [
                "game_id", "season", "week", "date",
                "home_team", "away_team",
                "context_index", "time_bin", "score_bin",
                "home_score", "away_score",
                "home_available", "away_available",
            ],
            "home_feature_columns": self.flattener.feature_names("home"),
            "away_feature_columns": self.flattener.feature_names("away"),
            "total_columns": len(schema),
            "n_contexts": N_CONTEXTS,
        }
        path.write_text(json.dumps(manifest, indent=2))
