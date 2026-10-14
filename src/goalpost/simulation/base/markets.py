# Purpose: Separate contract settlement probabilities from sportsbook payout arithmetic.
# Public API: OutcomeMass, SimulationGame, Market and empirical probability helpers.
# Consumers: Sport-specific contracts, existing simulation summaries and generic pricing.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np


# Purpose: Validate win/push/loss probabilities for one exact settlement contract.
# Consumers: Sport-specific market evaluators and AmericanOdds.size.
@dataclass(frozen=True)
class OutcomeMass:
    win: float
    push: float
    loss: float

    # Purpose: Reject nonfinite, out-of-range or non-normalized settlement mass.
    # Callers: Dataclass construction by market evaluators.
    def __post_init__(self) -> None:
        values = (self.win, self.push, self.loss)
        if not all(isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("Probabilities must be finite and between zero and one")
        if not isclose(sum(values), 1, abs_tol=1e-8):
            raise ValueError("Win, push and loss probabilities must sum to one")

    # Purpose: Swap win/loss for the opposing selection while preserving push mass.
    # Callers: Under totals and away-side spread/moneyline evaluation.
    def reversed(self) -> "OutcomeMass":
        return OutcomeMass(self.loss, self.push, self.win)


# Purpose: Expose simulation settlement separately from play-by-play input models.
# Consumers: Sport-specific simulation outputs and contract evaluators.
class SimulationGame(ABC):
    # Purpose: Evaluate an exact contract against this game output.
    # Callers: Comparison clients; concrete sports implement the delegation.
    @abstractmethod
    def mass(self, market: "Market") -> OutcomeMass:
        pass


# Purpose: Represent a selection and settlement rule without a sportsbook price.
# Consumers: NFLTotal, NFLSpread and NFLMoneyline; evaluated by simulation games.
class Market(ABC):
    # Purpose: Return win/push/loss probabilities under the supplied game distribution.
    # Callers: SimulationGame.mass and sport-specific overrides.
    @abstractmethod
    def mass(self, game: SimulationGame) -> OutcomeMass:
        pass


# Purpose: Count empirical totals above, below and exactly on the specified threshold.
# Callers: Existing simulation summaries and base probability tests.
def over_under_probability(total_dist: np.ndarray, line: float) -> dict[str, np.floating]:
    over = np.mean(total_dist > line)
    under = np.mean(total_dist < line)
    push = np.mean(total_dist == line)
    return {"over": over, "under": under, "push": push}


# Purpose: Count home-margin samples relative to a threshold, including equality.
# Callers: Existing simulation summaries; callers supply the margin threshold convention.
def spread_cover_probability(spread_dist: np.ndarray, line: float) -> dict[str, np.floating]:
    home_covers = np.mean(spread_dist > line)
    away_covers = np.mean(spread_dist < line)
    push = np.mean(spread_dist == line)
    return {"home_covers": home_covers, "away_covers": away_covers, "push": push}


# Purpose: Partition home-minus-away samples into home wins, away wins and ties.
# Callers: Existing simulation summaries and base probability tests.
def moneyline_probability(spread_dist: np.ndarray) -> dict[str, np.floating]:
    home_win = np.mean(spread_dist > 0)
    away_win = np.mean(spread_dist < 0)
    tie = np.mean(spread_dist == 0)
    return {"home_win": home_win, "away_win": away_win, "tie": tie}
