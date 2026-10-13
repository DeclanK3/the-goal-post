import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from goalpost.cli.build import generate_dataset, DatasetRequest

SRC = Path(__file__).resolve().parents[3] / "src"


def test_build_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "goalpost.cli.build", "--help"],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode == 0
    assert "Build GoalPost" in result.stdout


@pytest.mark.parametrize("source_name", ["nonexistent_dir", "existing_empty_dir"])
def test_build_zero_games_when_no_source_parquet(source_name, tmp_path):
    """
    When the sources directory contains no *.parquet files (either because
    it does not exist or because it is empty), the pipeline should raise a
    clear error rather than silently produce zero rows.
    """
    if source_name == "nonexistent_dir":
        src = tmp_path / "nonexistent"
    else:
        src = tmp_path / "empty"
        src.mkdir()

    out = tmp_path / "output"
    out.mkdir()

    if source_name == "nonexistent_dir":
        with pytest.raises(FileNotFoundError):
            generate_dataset(DatasetRequest(sources_dir=src, output_dir=out, season=2024))
    else:
        with pytest.raises(ValueError):
            generate_dataset(DatasetRequest(sources_dir=src, output_dir=out, season=2024))


def test_build_cli_requires_seasons_or_sources():
    """CLI must reject invocations that specify neither --seasons nor --sources."""
    result = subprocess.run(
        [sys.executable, "-m", "goalpost.cli.build", "--output", "/tmp/fake_out"],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode != 0
    assert "Must specify either --seasons or --sources" in result.stderr


def test_build_cli_rejects_both_seasons_and_sources():
    """CLI must reject invocations that specify both --seasons and --sources."""
    result = subprocess.run(
        [
            sys.executable, "-m", "goalpost.cli.build",
            "--seasons", "2024",
            "--sources", "/tmp/fake_src",
            "--output", "/tmp/fake_out",
        ],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode != 0
    assert "Cannot specify both --seasons and --sources" in result.stderr


def test_build_cli_downloads_before_build(tmp_path):
    """With --seasons, the CLI should call fetch_nflverse and use the cache dir."""
    import polars as pl
    from goalpost.cli.build import main

    out = tmp_path / "output"
    out.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # Create a minimal synthetic parquet *in the cache* that matches the nflverse schema.
    # This is a test fixture, not production synthetic data.
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
    df.write_parquet(cache / "nfl-2024.parquet")

    with patch.object(sys, "argv", [
        "build", "--seasons", "2024", "--output", str(out), "--cache-dir", str(cache)
    ]):
        with patch("goalpost.cli.build.fetch_nflverse") as mock_fetch:
            main()
            mock_fetch.assert_called_once_with([2024], cache)

    assert (out / "vae_dataset.parquet").exists()
    assert (out / "vae_schema.json").exists()
    dataset = pl.read_parquet(out / "vae_dataset.parquet")
    assert len(dataset) > 0, "Dataset should contain rows from real extracted games"
