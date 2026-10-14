# Purpose: Verify settlement mass, signed handicaps and push-aware sizing arithmetic.
# Public API: pytest tests and local fixtures in this module.
# Consumers: The offline unit suite; temporary files never access production stores.

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from goalpost.simulation.base.markets import Market, OutcomeMass, SimulationGame
from goalpost.simulation.base.pricing import AmericanOdds
from goalpost.simulation.nfl.contracts import (
    NFLGame,
    NFLMoneyline,
    NFLSpread,
    NFLTotal,
    ScoreDistribution,
)


# Purpose: Verify interfaces are abstract.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_interfaces_are_abstract() -> None:
    with pytest.raises(TypeError):
        Market()
    with pytest.raises(TypeError):
        SimulationGame()


# Purpose: Verify public mass.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize(
    "contract, expected",
    [
        (NFLTotal(37.5, "Over"), (0.6, 0, 0.4)),
        (NFLTotal(37, "Under"), (0.2, 0.2, 0.6)),
        (NFLSpread("Cleveland Browns", -3), (0.4, 0.3, 0.3)),
        (NFLSpread("Pittsburgh Steelers", 3), (0.3, 0.3, 0.4)),
        (NFLMoneyline("Cleveland Browns"), (0.7, 0.1, 0.2)),
        (NFLMoneyline("Pittsburgh Steelers"), (0.2, 0.1, 0.7)),
    ],
)
def test_public_mass(game: NFLGame, contract: Market, expected: tuple[float, float, float]) -> None:
    mass = game.mass(contract)
    assert (mass.win, mass.push, mass.loss) == pytest.approx(expected)


# Purpose: Verify invalid contracts.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize(
    "contract",
    [
        NFLTotal(float("nan"), "Over"),
        NFLTotal(37, "Up"),
        NFLSpread("wrong", 3),
        NFLMoneyline("wrong"),
    ],
)
def test_invalid_contracts(game: NFLGame, contract: Market) -> None:
    with pytest.raises(ValueError):
        game.mass(contract)


# Purpose: Verify push aware kelly and ev.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_push_aware_kelly_and_ev(game: NFLGame) -> None:
    sized = AmericanOdds(-200).size(game.mass(NFLMoneyline(game.home)))
    assert sized.hypothetical_ev_pct == pytest.approx(15)
    assert sized.full_kelly_pct == pytest.approx(100 / 3)
    assert sized.quarter_kelly_pct == pytest.approx(100 / 12)


# Purpose: Verify no kelly for negative ev or all pushes.
# Callers: pytest; uses synthetic fixtures or mocked services only.
def test_no_kelly_for_negative_ev_or_all_pushes() -> None:
    assert AmericanOdds(-110).size(OutcomeMass(0.4, 0.1, 0.5)).full_kelly_pct == 0
    assert AmericanOdds(100).size(OutcomeMass(0, 1, 0)).quarter_kelly_pct == 0


# Purpose: Verify invalid american odds.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize("odds", [0, 99, -99, 110.5, True, math.inf])
def test_invalid_american_odds(odds: object) -> None:
    with pytest.raises(ValueError):
        AmericanOdds(odds)


# Purpose: Verify invalid distributions.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize(
    "points,probabilities",
    [
        ((1, 1), (0.5, 0.5)),
        ((1, 2), (0.3, 0.3)),
        ((1, 2), (-0.1, 1.1)),
        ((1.5,), (1,)),
        ((1,), (float("nan"),)),
        ((), ()),
    ],
)
def test_invalid_distributions(
    points: tuple[int | float, ...], probabilities: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        ScoreDistribution(points, probabilities)


# Purpose: Verify invalid game denominators and identity.
# Callers: pytest; uses synthetic fixtures or mocked services only.
@pytest.mark.parametrize(
    "changes",
    [
        {"retained": 121},
        {"approximate": 101},
        {"attempted": True},
        {"retained": 0},
        {"home": "Pittsburgh Steelers"},
    ],
)
def test_invalid_game_denominators_and_identity(game: NFLGame, changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(game, **changes)
