from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class GameOutcome:
    home_score: int = 0
    away_score: int = 0
    total: int = 0
    margin: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.total = self.home_score + self.away_score
        self.margin = self.home_score - self.away_score
