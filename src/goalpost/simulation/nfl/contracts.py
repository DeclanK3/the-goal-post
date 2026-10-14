"""NFL regulation distributions and moneyline, spread and total contracts."""

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path

import pandas as pd

from goalpost.simulation.base.markets import Market, OutcomeMass, SimulationGame


@dataclass(frozen=True)
class ScoreDistribution:
    """Discrete probability mass over integer scores or score differences."""

    points: tuple[int, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self):
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

    def above(self, line: float) -> OutcomeMass:
        """P(X > line), P(X = line), P(X < line); equality is a push."""
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


@dataclass(frozen=True)
class NFLGame(SimulationGame):
    """Exported regulation output; total and margin use the same retained paths."""

    game_id: str
    home: str
    away: str
    kickoff: datetime
    total: ScoreDistribution
    home_margin: ScoreDistribution
    attempted: int
    retained: int
    approximate: int = 0

    def __post_init__(self):
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

    def mass(self, market: Market) -> OutcomeMass:
        """Example: game.mass(NFLTotal(37.5, 'Over')).win is P(total > 37.5)."""
        return market.mass(self)

    def side_mass(self, team: str, line: float) -> OutcomeMass:
        """For a home spread h, home_margin + h must exceed zero to win."""
        if team == self.home:
            return self.home_margin.above(-line)
        if team == self.away:
            return self.home_margin.above(line).reversed()
        raise ValueError("Selection is not a team in this game")


@dataclass(frozen=True)
class NFLTotal(Market):
    line: float
    side: str

    def mass(self, game: NFLGame) -> OutcomeMass:
        if self.side not in ("Over", "Under"):
            raise ValueError("Total side must be Over or Under")
        mass = game.total.above(self.line)
        return mass if self.side == "Over" else mass.reversed()


@dataclass(frozen=True)
class NFLSpread(Market):
    team: str
    line: float

    def mass(self, game: NFLGame) -> OutcomeMass:
        return game.side_mass(self.team, self.line)


@dataclass(frozen=True)
class NFLMoneyline(Market):
    team: str

    def mass(self, game: NFLGame) -> OutcomeMass:
        """Regulation ties are void in this hypothetical calculation, not overtime wins."""
        return game.side_mass(self.team, 0)
