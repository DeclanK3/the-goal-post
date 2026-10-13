from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Play:
    play_id: str = ""
    play_type: str = ""
    points_scored: int = 0


@dataclass
class Possession:
    possession_id: str = ""
    team: str = ""
    plays: List[Play] = field(default_factory=list)
    points_scored: int = 0
    game_id: Optional[str] = None


@dataclass
class Game:
    game_id: str = ""
    season: int = 0
    week: int = 0
    home_team: str = ""
    away_team: str = ""
    possessions: List[Possession] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0
    date: Optional[datetime] = None
