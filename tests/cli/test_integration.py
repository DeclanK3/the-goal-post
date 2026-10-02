import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _make_synthetic_parquet(path: Path):
    rows = [
        {"play_id": 1, "game_id": "test-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 10, "yardline_100": 25,
         "half_seconds_remaining": 1800, "qtr": 1,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 2, "game_id": "test-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 2, "ydstogo": 5, "yardline_100": 30,
         "half_seconds_remaining": 1750, "qtr": 1,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "run",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 3, "game_id": "test-g1", "home_team": "A", "away_team": "B",
         "posteam": "B", "down": 1, "ydstogo": 10, "yardline_100": 50,
         "half_seconds_remaining": 900, "qtr": 2,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 4, "game_id": "test-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 10, "yardline_100": 75,
         "half_seconds_remaining": 1800, "qtr": 3,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 5, "game_id": "test-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 8, "yardline_100": 72,
         "half_seconds_remaining": 900, "qtr": 4,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "run",
         "extra_point_attempt": 0, "two_point_attempt": 0},
    ]
    df = pl.DataFrame(rows)
    df.write_parquet(path)


def test_build_cli_runs():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sources"
        out = Path(tmp) / "output"
        src.mkdir()
        out.mkdir()
        _make_synthetic_parquet(src / "test.parquet")

        result = subprocess.run(
            [sys.executable, "-m", "goalpost.cli.build", "--sources", str(src), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (out / "vae_dataset.parquet").exists()
        assert (out / "vae_schema.json").exists()
        schema = json.loads((out / "vae_schema.json").read_text())
        assert schema["total_columns"] > 0
        df = pl.read_parquet(out / "vae_dataset.parquet")
        assert len(df) > 0


def test_simulate_cli_runs():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sources"
        out = Path(tmp) / "output"
        src.mkdir()
        out.mkdir()
        _make_synthetic_parquet(src / "test.parquet")

        subprocess.run(
            [sys.executable, "-m", "goalpost.cli.build", "--sources", str(src), "--output", str(out)],
            capture_output=True,
            text=True,
            check=True,
        )

        sched = Path(tmp) / "schedule.csv"
        sched.write_text("game_id,season,week,home_team,away_team\ntest-g1,2024,1,A,B\n")

        result = subprocess.run(
            [sys.executable, "-m", "goalpost.cli.simulate",
             "--week", "1", "--season", "2024", "--models", str(out),
             "--schedule", str(sched), "--n-sims", "10"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Simulated" in result.stdout
