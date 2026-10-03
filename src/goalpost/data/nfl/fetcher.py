from pathlib import Path
from typing import List

import polars as pl

NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

REQUIRED_COLUMNS = {
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


class NFLverseError(Exception):
    """Raised when nflverse data cannot be downloaded or validated."""


def _validate_nflverse_schema(df: pl.DataFrame) -> None:
    """Raise NFLverseError if the DataFrame is missing required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise NFLverseError(
            f"Downloaded nflverse data is missing required columns: {sorted(missing)}"
        )


def download_season(season: int, dest: Path) -> Path:
    url = NFLVERSE_URL.format(season=season)
    try:
        df = pl.read_parquet(url)
    except Exception as e:
        raise NFLverseError(
            f"Failed to download nflverse data for season {season} from {url}: {e}"
        ) from e
    _validate_nflverse_schema(df)
    out = dest / f"nfl-{season}.parquet"
    df.write_parquet(out, compression="zstd")
    return out


def fetch_nflverse(seasons: List[int], dest: Path) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    return [download_season(s, dest) for s in seasons]
