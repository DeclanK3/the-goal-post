import numpy as np

from goalpost.simulation.nfl.markets import over_under_probability, spread_cover_probability, moneyline_probability


def test_over_under_probabilities_sum():
    dist = np.array([35, 38, 42, 45, 48])
    probs = over_under_probability(dist, 42.5)
    assert abs(probs["over"] + probs["under"] + probs["push"] - 1.0) < 1e-9


def test_spread_probabilities_sum():
    dist = np.array([-7, -3, 0, 3, 7])
    probs = spread_cover_probability(dist, -2.5)
    assert abs(probs["home_covers"] + probs["away_covers"] + probs["push"] - 1.0) < 1e-9


def test_moneyline_probabilities_sum():
    dist = np.array([-14, -7, 0, 7, 14])
    probs = moneyline_probability(dist)
    assert abs(probs["home_win"] + probs["away_win"] + probs["tie"] - 1.0) < 1e-9
