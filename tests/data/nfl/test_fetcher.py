"""Tests for the NFLverse fetcher — production path must never use synthetic data."""
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from goalpost.data.nfl.fetcher import (
    download_season,
    fetch_nflverse,
    NFLverseError,
    REQUIRED_COLUMNS,
)


def _make_valid_nflverse_df() -> pl.DataFrame:
    """Create a DataFrame with all required nflverse columns."""
    return pl.DataFrame({
        "game_id": ["g1"],
        "home_team": ["A"],
        "away_team": ["B"],
        "posteam": ["A"],
        "down": [1],
        "ydstogo": [10],
        "yardline_100": [25],
        "half_seconds_remaining": [1800],
        "qtr": [1],
        "play_type": ["pass"],
        "play_id": [1],
        "posteam_score_post": [0],
        "defteam_score_post": [0],
        "home_score": [0],
        "away_score": [0],
        "week": [1],
        "game_date": ["2024-01-01"],
    })


def test_download_season_success(tmp_path):
    df = _make_valid_nflverse_df()
    with patch("goalpost.data.nfl.fetcher.pl.read_parquet", return_value=df):
        out = download_season(2024, tmp_path)
    assert out.exists()
    read_back = pl.read_parquet(out)
    assert len(read_back) == 1


def test_download_season_fails_on_missing_season():
    """Must raise NFLverseError when the release asset does not exist."""
    with patch(
        "goalpost.data.nfl.fetcher.pl.read_parquet",
        side_effect=Exception("HTTP 404"),
    ):
        with pytest.raises(NFLverseError, match="Failed to download nflverse data"):
            download_season(1900, MagicMock())


def test_download_season_fails_on_bad_schema(tmp_path):
    """Must raise NFLverseError when required columns are missing."""
    bad_df = pl.DataFrame({"game_id": ["g1"]})
    with patch("goalpost.data.nfl.fetcher.pl.read_parquet", return_value=bad_df):
        with pytest.raises(NFLverseError, match="missing required columns"):
            download_season(2024, tmp_path)


def test_download_season_schema_validation_includes_all_required():
    """Ensure REQUIRED_COLUMNS covers every field used by the extractor."""
    extractor_fields = {
        "game_id",
        "home_team",
        "away_team",
        "posteam",
        "down",
        "ydstogo",
        "yardline_100",
        "half_seconds_remaining",
        "qtr",
        "play_type",
        "play_id",
        "posteam_score_post",
        "defteam_score_post",
        "home_score",
        "away_score",
        "week",
        "game_date",
    }
    assert REQUIRED_COLUMNS == extractor_fields, (
        f"REQUIRED_COLUMNS mismatch: {REQUIRED_COLUMNS ^ extractor_fields}"
    )


def test_fetch_nflverse_downloads_multiple_seasons(tmp_path):
    df = _make_valid_nflverse_df()
    with patch("goalpost.data.nfl.fetcher.pl.read_parquet", return_value=df):
        paths = fetch_nflverse([2023, 2024], tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_fetch_nflverse_fails_fast_on_unavailable_season(tmp_path):
    """If one season fails, the whole batch must fail with a clear error."""
    df = _make_valid_nflverse_df()
    call_count = 0

    def side_effect(url):
        nonlocal call_count
        call_count += 1
        if "play_by_play_2024" in str(url):
            raise Exception("HTTP 404")
        return df

    with patch("goalpost.data.nfl.fetcher.pl.read_parquet", side_effect=side_effect):
        with pytest.raises(NFLverseError):
            fetch_nflverse([2023, 2024], tmp_path)
