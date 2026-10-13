from pathlib import Path
from typing import List

import polars as pl


class NFLSchedule:
    def __init__(self, path: Path):
        self.df = pl.read_csv(path)

    def completed_games(self, season: int) -> List[dict]:
        d = self.df.filter(
            (pl.col("season") == season)
            & (pl.col("home_score").is_not_null())
            & (pl.col("away_score").is_not_null())
            & (pl.col("game_type") != "PRE")
        )
        return d.to_dicts()

    def upcoming(self, as_of) -> List[dict]:
        d = self.df.filter(
            (pl.col("gameday") > as_of)
            & (pl.col("home_score").is_null())
        )
        return d.to_dicts()
