import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from goalpost.cli.simulate import _load_dataset_as_models

SRC = Path(__file__).resolve().parents[3] / "src"


def test_simulate_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "goalpost.cli.simulate", "--help"],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode == 0
    assert "Run GoalPost" in result.stdout


def test_load_dataset_reconstructs_nonzero_matrices(tmp_path: Path):
    # Build a minimal VAE dataset row-per-context layout
    home_cols = [f"home_1d_short_goal_to_1d_short_goal_{i}" for i in range(5476)]
    away_cols = [f"away_1d_short_goal_to_1d_short_goal_{i}" for i in range(5476)]
    # Use real feature names from the schema — for the test we only need
    # a nonzero flat vector that passes from_flat validation.
    rows = []
    for ctx in range(9):
        row = {
            "game_id": "test-g1",
            "season": 2024,
            "week": 1,
            "date": "2024-01-01",
            "home_team": "A",
            "away_team": "B",
            "context_index": ctx,
            "time_bin": ctx // 3,
            "score_bin": ctx % 3,
            "home_score": 21,
            "away_score": 14,
            "home_available": True,
            "away_available": True,
        }
        # Put an absorbing TD transition on state 16 for every context
        for c in home_cols:
            row[c] = 0.0
        for c in away_cols:
            row[c] = 0.0
        # The exact column name doesn't matter for from_flat as long as
        # the flat vector has the right length; but we must give a valid
        # probability distribution that sums to 1.
        # Put mass on the first physical-to-physical transition.
        row[home_cols[0]] = 1.0
        row[away_cols[0]] = 1.0
        rows.append(row)

    df = pl.DataFrame(rows)
    parquet_path = tmp_path / "vae_dataset.parquet"
    df.write_parquet(parquet_path)

    models = _load_dataset_as_models(tmp_path)
    assert len(models) == 2
    for model in models.values():
        assert model._matrix.sum() > 0
