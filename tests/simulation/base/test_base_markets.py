import numpy as np

from goalpost.simulation.base.markets import over_under_probability, spread_cover_probability, moneyline_probability


def test_over_under_sums_to_one():
    dist = np.array([40, 42, 44, 45])
    probs = over_under_probability(dist, 42.5)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_spread_sums_to_one():
    dist = np.array([-3, 0, 3, 7])
    probs = spread_cover_probability(dist, 2.5)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_moneyline_sums_to_one():
    dist = np.array([-7, 0, 3, 14])
    probs = moneyline_probability(dist)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
