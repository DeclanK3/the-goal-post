# Purpose: Evaluate exact regulation-only contracts against validated discrete score mass.
# Public API: ScoreDistribution, NFLGame, NFLTotal, NFLSpread, NFLMoneyline.
# Consumers: WeeklyExport, NFLComparison and settlement regression tests.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path

import pandas as pd

from goalpost.simulation.base.markets import Market, OutcomeMass, SimulationGame


# Purpose: Keep normalized discrete support for regulation totals or home margins.
# Consumers: WeeklyExport loading and NFL contract evaluation.
@dataclass(frozen=True)
class ScoreDistribution:
    points: tuple[int, ...]
    probabilities: tuple[float, ...]

    # Purpose: Reject duplicate/noninteger support and invalid probability mass.
    # Callers: Distribution construction, including from_csv.
    def __post_init__(self) -> None:
        if not self.points or len(self.points) != len(self.probabilities):
            raise ValueError("Distribution support and probabilities must align")
        if len(set(self.points)) != len(self.points):
            raise ValueError("Duplicate score support")
        if any(type(point) is not int for point in self.points):
            raise ValueError("Score support must contain integers")
        if not all(isfinite(value) and 0 <= value <= 1 for value in self.probabilities):
            raise ValueError("Invalid probability mass")
        if not isclose(sum(self.probabilities), 1, abs_tol=1e-8):
            raise ValueError("Probability mass must sum to one")

    # Purpose: Bind exported mass to integer counts and the reported retained-path denominator.
    # Callers: WeeklyExport.load for total and home-minus-away PMFs.
    @classmethod
    def from_csv(cls, path: Path, retained: int) -> "ScoreDistribution":
        frame = pd.read_csv(path)
        if not {"points", "count", "probability"}.issubset(frame) or frame.empty:
            raise ValueError("PMF requires points, count and probability columns")
        values = frame[["points", "count", "probability"]]
        if not all(isfinite(float(value)) for value in values.to_numpy().flat):
            raise ValueError("Nonfinite PMF value")
        if retained <= 0 or any(frame["count"] < 0) or any(frame["count"] % 1):
            raise ValueError("Invalid PMF counts")
        if frame["count"].sum() != retained or any(frame["points"] % 1):
            raise ValueError("PMF denominator or integer support mismatch")
        if any(abs(frame["probability"] - frame["count"] / retained) > 1e-8):
            raise ValueError("PMF probabilities disagree with counts")
        return cls(tuple(map(int, frame["points"])), tuple(map(float, frame["probability"])))

    # Purpose: Partition mass above/equal/below the exact line; equality is a push.
    # Callers: NFLTotal.mass and NFLGame.side_mass.
    def above(self, line: float) -> OutcomeMass:
        if not isfinite(line):
            raise ValueError("Line must be finite")
        win = push = loss = 0.0
        for points, probability in zip(self.points, self.probabilities):
            if points > line:
                win += probability
            elif points == line:
                push += probability
            else:
                loss += probability
        return OutcomeMass(win, push, loss)


# Purpose: Keep regulation total and margin PMFs together with simulation attrition counts.
# Consumers: WeeklyExport constructs it; NFLComparison evaluates its contracts.
@dataclass(frozen=True)
class NFLGame(SimulationGame):
    game_id: str
    home: str
    away: str
    kickoff: datetime
    total: ScoreDistribution
    home_margin: ScoreDistribution
    attempted: int
    retained: int
    approximate: int = 0

    # Purpose: Reject ambiguous teams, timezone-free kickoff and inconsistent path counts.
    # Callers: Dataclass construction while loading saved exports.
    def __post_init__(self) -> None:
        if not self.home or not self.away or self.home == self.away:
            raise ValueError("Distinct home and away teams are required")
        if self.kickoff.tzinfo is None:
            raise ValueError("Kickoff must include a timezone")
        if any(
            type(value) is not int for value in (self.attempted, self.retained, self.approximate)
        ):
            raise ValueError("Simulation counts must be integers")
        if not 0 <= self.approximate <= self.retained <= self.attempted or self.retained <= 0:
            raise ValueError("Invalid simulation denominators")
        if any(point < 0 for point in self.total.points):
            raise ValueError("Total points cannot be negative")

    # Purpose: Delegate one exact selection to its market settlement rule.
    # Callers: NFLComparison.price and public contract tests.
    def mass(self, market: Market) -> OutcomeMass:
        return market.mass(self)

    # Purpose: Apply signed handicaps to the home margin and reverse for the away side.
    # Callers: NFLSpread.mass and NFLMoneyline.mass.
    def side_mass(self, team: str, line: float) -> OutcomeMass:
        if team == self.home:
            return self.home_margin.above(-line)
        if team == self.away:
            return self.home_margin.above(line).reversed()
        raise ValueError("Selection is not a team in this game")


# Purpose: Specify one over/under side at an exact total-points line.
# Consumers: NFLComparison builds it from a selected quote.
@dataclass(frozen=True)
class NFLTotal(Market):
    line: float
    side: str

    # Purpose: Evaluate over mass or its under complement without dropping pushes.
    # Callers: NFLGame.mass.
    def mass(self, game: NFLGame) -> OutcomeMass:
        if self.side not in ("Over", "Under"):
            raise ValueError("Total side must be Over or Under")
        mass = game.total.above(self.line)
        return mass if self.side == "Over" else mass.reversed()


# Purpose: Specify a team and signed handicap independently of a book price.
# Consumers: NFLComparison builds it from a selected quote.
@dataclass(frozen=True)
class NFLSpread(Market):
    team: str
    line: float

    # Purpose: Evaluate the selected team using the signed handicap convention.
    # Callers: NFLGame.mass.
    def mass(self, game: NFLGame) -> OutcomeMass:
        return game.side_mass(self.team, self.line)


# Purpose: Represent a regulation-only team outcome with ties treated as void.
# Consumers: NFLComparison; this does not validate overtime-inclusive book contracts.
@dataclass(frozen=True)
class NFLMoneyline(Market):
    team: str

    # Purpose: Evaluate a zero-handicap regulation result, retaining tie mass.
    # Callers: NFLGame.mass; actual-contract EV remains unset downstream.
    def mass(self, game: NFLGame) -> OutcomeMass:
        return game.side_mass(self.team, 0)
