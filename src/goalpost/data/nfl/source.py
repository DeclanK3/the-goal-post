from datetime import datetime
from pathlib import Path
from typing import List

import polars as pl

from ..base.source import DataSource
from ..base.models import Game, Play, Possession
from .fetcher import fetch_nflverse
from .schedule import NFLSchedule


class NFLDataSource(DataSource):
    def __init__(self, schedule_path: Path, cache_dir: Path):
        self.schedule = NFLSchedule(schedule_path)
        self.cache_dir = cache_dir
        self._raw = []

    def fetch(self, seasons: List[int]) -> None:
        fetch_nflverse(seasons, self.cache_dir)

    def fetch_upcoming(self, as_of: datetime) -> List[Game]:
        rows = self.schedule.upcoming(as_of)
        return [
            Game(
                game_id=str(r["game_id"]),
                season=int(r["season"]),
                week=int(r["week"]),
                home_team=str(r["home_team"]),
                away_team=str(r["away_team"]),
                date=datetime.fromisoformat(r["gameday"]),
            )
            for r in rows
        ]

    def parse(self) -> List[Game]:
        games = []
        for path in self.cache_dir.glob("nfl-*.parquet"):
            df = pl.read_parquet(path)
            for gid, g in df.group_by("game_id"):
                first = g.row(0, named=True)
                possessions = []
                for _, plays in g.group_by("drive"):
                    possession_plays = [
                        Play(play_id=str(p["play_id"]), play_type=str(p["play_type"]))
                        for p in plays.to_dicts()
                    ]
                    possessions.append(
                        Possession(
                            team=str(first["posteam"]),
                            plays=possession_plays,
                            game_id=str(gid),
                        )
                    )
                games.append(
                    Game(
                        game_id=str(gid),
                        season=int(first["season"]),
                        week=int(first["week"]),
                        home_team=str(first["home_team"]),
                        away_team=str(first["away_team"]),
                        possessions=possessions,
                        home_score=int(first["home_score"]),
                        away_score=int(first["away_score"]),
                    )
                )
        return games
