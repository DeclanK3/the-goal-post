"""Push-aware hypothetical EV and single-contract Kelly arithmetic."""

from dataclasses import dataclass
from math import isfinite

from goalpost.abc.markets import OutcomeMass


@dataclass(frozen=True)
class Sizing:
    decimal_odds: float
    hypothetical_ev_pct: float
    full_kelly_pct: float
    quarter_kelly_pct: float


@dataclass(frozen=True)
class AmericanOdds:
    value: int

    def __post_init__(self):
        if type(self.value) is not int or abs(self.value) < 100:
            raise ValueError("American odds must be an integer with magnitude at least 100")

    @property
    def net_payout(self) -> float:
        return self.value / 100 if self.value > 0 else 100 / abs(self.value)

    def size(self, mass: OutcomeMass) -> Sizing:
        """Pushes return the stake. Kelly conditions its denominator on decisive results."""
        payout = self.net_payout
        expected_return = payout * mass.win - mass.loss
        decisive_probability = mass.win + mass.loss
        full_kelly = 0.0
        if decisive_probability:
            full_kelly = max(0.0, expected_return / (payout * decisive_probability))
        if not isfinite(full_kelly):
            raise ValueError("Invalid Kelly calculation")
        return Sizing(1 + payout, 100 * expected_return, 100 * full_kelly, 25 * full_kelly)
