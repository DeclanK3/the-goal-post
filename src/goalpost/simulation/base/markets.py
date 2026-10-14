"""Contracts for probabilistic simulation outputs and sport-specific markets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np


@dataclass(frozen=True)
class OutcomeMass:
    """Win, push and loss probabilities for one precisely specified contract."""

    win: float
    push: float
    loss: float

    def __post_init__(self):
        values = (self.win, self.push, self.loss)
        if not all(isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("Probabilities must be finite and between zero and one")
        if not isclose(sum(values), 1, abs_tol=1e-8):
            raise ValueError("Win, push and loss probabilities must sum to one")

    def reversed(self) -> "OutcomeMass":
        return OutcomeMass(self.loss, self.push, self.win)


class SimulationGame(ABC):
    """A game's simulation output, rather than its play-by-play input data."""

    @abstractmethod
    def mass(self, market: "Market") -> OutcomeMass:
        """Evaluate a market's line against the simulated score distribution."""


class Market(ABC):
    """A selection and its settlement rule; it does not contain a book's price."""

    @abstractmethod
    def mass(self, game: SimulationGame) -> OutcomeMass:
        """Return win/push/loss mass using the supplied simulation output."""


def over_under_probability(total_dist: np.ndarray, line: float):
    over = np.mean(total_dist > line)
    under = np.mean(total_dist < line)
    push = np.mean(total_dist == line)
    return {"over": over, "under": under, "push": push}


def spread_cover_probability(spread_dist: np.ndarray, line: float):
    home_covers = np.mean(spread_dist > line)
    away_covers = np.mean(spread_dist < line)
    push = np.mean(spread_dist == line)
    return {"home_covers": home_covers, "away_covers": away_covers, "push": push}


def moneyline_probability(spread_dist: np.ndarray):
    home_win = np.mean(spread_dist > 0)
    away_win = np.mean(spread_dist < 0)
    tie = np.mean(spread_dist == 0)
    return {"home_win": home_win, "away_win": away_win, "tie": tie}
