from dataclasses import dataclass, field
from typing import List
from datetime import datetime

import numpy as np


@dataclass
class HistoricalMatrix:
    date: datetime
    team_id: str
    season: int
    game_id: str
    context_count: int
    state_space_size: int
    matrix: np.ndarray  # shape (C, N, N)


@dataclass
class History:
    entries: List[HistoricalMatrix] = field(default_factory=list)

    def add(self, entry: HistoricalMatrix):
        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.date)

    def by_team(self, team_id: str) -> List[HistoricalMatrix]:
        return [e for e in self.entries if e.team_id == team_id]

    def by_team_season(self, team_id: str, season: int) -> List[HistoricalMatrix]:
        return [e for e in self.entries if e.team_id == team_id and e.season == season]

    def by_team_game(self, team_id: str, game_id: str) -> HistoricalMatrix | None:
        for e in self.entries:
            if e.team_id == team_id and e.game_id == game_id:
                return e
        return None

    def global_aggregate(self, context: int, state: int, context_count: int, state_space_size: int) -> np.ndarray | None:
        agg = np.zeros(state_space_size, dtype=np.float32)
        count = 0
        for entry in self.entries:
            m = entry.matrix
            if (
                m.shape == (context_count, state_space_size, state_space_size)
                and m[context, state].sum() > 0
            ):
                agg += m[context, state]
                count += 1
        if count == 0:
            return None
        return agg / count
