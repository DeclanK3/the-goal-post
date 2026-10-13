"""NFL market probability functions."""
import numpy as np


def over_under_probability(total_dist: np.ndarray, line: float):
    over = float(np.mean(total_dist > line))
    under = float(np.mean(total_dist < line))
    push = float(np.mean(total_dist == line))
    return {"over": over, "under": under, "push": push}


def spread_cover_probability(spread_dist: np.ndarray, line: float):
    home_covers = float(np.mean(spread_dist > line))
    away_covers = float(np.mean(spread_dist < line))
    push = float(np.mean(spread_dist == line))
    return {"home_covers": home_covers, "away_covers": away_covers, "push": push}


def moneyline_probability(spread_dist: np.ndarray):
    home_win = float(np.mean(spread_dist > 0))
    away_win = float(np.mean(spread_dist < 0))
    tie = float(np.mean(spread_dist == 0))
    return {"home_win": home_win, "away_win": away_win, "tie": tie}
