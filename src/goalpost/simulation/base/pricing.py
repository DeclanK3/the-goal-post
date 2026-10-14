# Purpose: Compute hypothetical single-contract returns while treating pushes as stake returns.
# Public API: AmericanOdds, Sizing.
# Consumers: NFLComparison and offline pricing regressions; no betting or portfolio allocation.

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from goalpost.simulation.base.markets import OutcomeMass


# Purpose: Carry payout, expected return and standalone Kelly percentages.
# Consumers: AmericanOdds.size returns it to comparison/report code.
@dataclass(frozen=True)
class Sizing:
    decimal_odds: float
    hypothetical_ev_pct: float
    full_kelly_pct: float
    quarter_kelly_pct: float


# Purpose: Validate a quoted American price before deriving its net payout.
# Consumers: NFLComparison and offline arithmetic tests.
@dataclass(frozen=True)
class AmericanOdds:
    value: int

    # Purpose: Reject booleans, nonintegers and impossible American-odds magnitudes.
    # Callers: Dataclass construction for each quoted selection.
    def __post_init__(self) -> None:
        if type(self.value) is not int or abs(self.value) < 100:
            raise ValueError("American odds must be an integer with magnitude at least 100")

    # Purpose: Convert positive or negative American odds to net profit per unit staked.
    # Callers: AmericanOdds.size; principal is added separately for decimal odds.
    @property
    def net_payout(self) -> float:
        return self.value / 100 if self.value > 0 else 100 / abs(self.value)

    # Purpose: Calculate EV and nonnegative single-bet Kelly with pushes returning the stake.
    # Callers: NFLComparison.price; fractions must not be added across correlated selections.
    def size(self, mass: OutcomeMass) -> Sizing:
        payout = self.net_payout
        expected_return = payout * mass.win - mass.loss
        decisive_probability = mass.win + mass.loss
        full_kelly = 0.0
        if decisive_probability:
            full_kelly = max(0.0, expected_return / (payout * decisive_probability))
        if not isfinite(full_kelly):
            raise ValueError("Invalid Kelly calculation")
        return Sizing(1 + payout, 100 * expected_return, 100 * full_kelly, 25 * full_kelly)
