from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class ExtractedGame:
    game_id: str = ""
    events: list = field(default_factory=list)


class TransitionModel(ABC):
    def __init__(self, team_id: str, game_id: str, opponent_id: str):
        self.team_id = team_id
        self.game_id = game_id
        self.opponent_id = opponent_id

    @property
    @abstractmethod
    def state_space_size(self) -> int:
        pass

    @property
    @abstractmethod
    def context_count(self) -> int:
        pass

    @property
    @abstractmethod
    def validity_mask(self) -> np.ndarray:
        pass

    @abstractmethod
    def extract_from_game(self, game) -> None:
        pass

    @abstractmethod
    def get_probabilities(self, context: int, state: int) -> Dict[int, float]:
        pass

    @abstractmethod
    def flatten(self) -> np.ndarray:
        pass

    @classmethod
    @abstractmethod
    def from_flat(cls, array: np.ndarray, team_id: str, game_id: str, opponent_id: str):
        pass
