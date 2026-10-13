from abc import ABC, abstractmethod
from typing import List

from ...representation.base.models import TransitionModel
from .models import GameOutcome


class Simulator(ABC):
    @abstractmethod
    def simulate(self, home_model: TransitionModel, away_model: TransitionModel, n_sims: int = 10000, seed: int = 0) -> List[GameOutcome]:
        pass
