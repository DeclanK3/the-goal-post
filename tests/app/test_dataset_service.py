import tempfile
from pathlib import Path

import polars as pl
import pytest

from goalpost.app.dataset_service import generate_dataset, DatasetRequest


def _make_parquet(path: Path):
    rows = [
        {"play_id": 1, "game_id": "svc-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 10, "yardline_100": 25,
         "half_seconds_remaining": 1800, "qtr": 1,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 2, "game_id": "svc-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 2, "ydstogo": 5, "yardline_100": 30,
         "half_seconds_remaining": 1750, "qtr": 1,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "run",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 3, "game_id": "svc-g1", "home_team": "A", "away_team": "B",
         "posteam": "B", "down": 1, "ydstogo": 10, "yardline_100": 50,
         "half_seconds_remaining": 900, "qtr": 2,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 4, "game_id": "svc-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 10, "yardline_100": 75,
         "half_seconds_remaining": 1800, "qtr": 3,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass",
         "extra_point_attempt": 0, "two_point_attempt": 0},
        {"play_id": 5, "game_id": "svc-g1", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 8, "yardline_100": 72,
         "half_seconds_remaining": 900, "qtr": 4,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "run",
         "extra_point_attempt": 0, "two_point_attempt": 0},
    ]
    pl.DataFrame(rows).write_parquet(path)


def test_generate_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        out = Path(tmp) / "out"
        src.mkdir()
        out.mkdir()
        _make_parquet(src / "test.parquet")

        req = DatasetRequest(sources_dir=src, output_dir=out, season=2024)
        result = generate_dataset(req)

        assert result.game_count == 1
        assert result.observation_count > 0
        assert result.dataset_path.exists()
        assert result.schema_path.exists()
