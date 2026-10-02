import numpy as np
import polars as pl
import pytest

from goalpost.representation.nfl.extractor import extract_game, state_index, context_index, NFLExtractedGame


def test_state_index():
    assert state_index(1, 2, 10) == 0 * 18 + 0 * 6 + 1  # 1, short, red
    assert state_index(4, 15, 95) == 3 * 18 + 2 * 6 + 5  # 4, long, own20


def test_context_index():
    assert context_index(200, -7) == 0 * 3 + 0  # time_bin 0, losing
    assert context_index(60, 0) == 1 * 3 + 1  # time_bin 1, tied
    assert context_index(10, 3) == 2 * 3 + 2  # time_bin 2, winning


def test_extract_game_basic():
    # Need at least one decision play in each of Q1, Q2, Q3, Q4
    rows = [
        # Q1 (half=1)
        {"play_id": 1, "game_id": "test", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 10, "yardline_100": 25,
         "half_seconds_remaining": 1800, "qtr": 1,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass"},
        {"play_id": 2, "game_id": "test", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 2, "ydstogo": 5, "yardline_100": 30,
         "half_seconds_remaining": 1750, "qtr": 1,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "run"},
        # Q2 (half=1)
        {"play_id": 3, "game_id": "test", "home_team": "A", "away_team": "B",
         "posteam": "B", "down": 1, "ydstogo": 10, "yardline_100": 50,
         "half_seconds_remaining": 900, "qtr": 2,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass"},
        # Q3 (half=2)
        {"play_id": 4, "game_id": "test", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 10, "yardline_100": 75,
         "half_seconds_remaining": 1800, "qtr": 3,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "pass"},
        # Q4 (half=2)
        {"play_id": 5, "game_id": "test", "home_team": "A", "away_team": "B",
         "posteam": "A", "down": 1, "ydstogo": 8, "yardline_100": 72,
         "half_seconds_remaining": 900, "qtr": 4,
         "posteam_score_post": 0, "defteam_score_post": 0,
         "home_score": 0, "away_score": 0,
         "game_date": "2024-01-01", "week": 1, "play_type": "run"},
    ]
    df = pl.DataFrame(rows)
    game = extract_game(df)
    assert isinstance(game, NFLExtractedGame)
    assert game.home == "A"
    assert game.away == "B"
    assert len(game.events) > 0
