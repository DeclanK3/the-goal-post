from abc import ABC, abstractmethod
from typing import List

from .models import ExtractedGame


class GameTable:
    pass


class TransitionBuilder(ABC):
    @abstractmethod
    def build(self, games: List[ExtractedGame]) -> GameTable:
        pass
