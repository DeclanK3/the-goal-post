"""Weekly slate runner for NFL simulations."""
from datetime import datetime, timedelta

import numpy as np

from .engine import NFLEngine
from .markets import over_under_probability, spread_cover_probability, moneyline_probability
from ...representation.nfl.model import NFLTransitionModel


def week_for_date(date: datetime) -> int:
    """NFL week: Thursday starts a new week; Tuesday/Wednesday belong to preceding week."""
    weekday = date.weekday()
    if weekday < 3:
        date = date - timedelta(days=weekday + 4)
    else:
        date = date - timedelta(days=weekday - 3)
    return date.isocalendar()[1]


def weekly_runner(games: list, models: dict, n_sims: int = 10000, seed: int = 0) -> list[dict]:
    """Run simulations for a weekly slate.

    games: list of dicts with home_team, away_team, game_id
    models: dict mapping (team_id, game_id) -> NFLTransitionModel
    """
    engine = NFLEngine()
    results = []
    for game in games:
        home_model = models.get((game["home_team"], game["game_id"]))
        away_model = models.get((game["away_team"], game["game_id"]))
        if home_model is None or away_model is None:
            continue
        outcomes = engine.simulate(home_model, away_model, n_sims=n_sims, seed=seed)
        home_scores = np.array([o.home_score for o in outcomes])
        away_scores = np.array([o.away_score for o in outcomes])
        spreads = home_scores - away_scores
        totals = home_scores + away_scores
        results.append(
            {
                "game_id": game["game_id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "spread": spread_cover_probability(spreads, 0.0),
                "total": over_under_probability(totals, 42.5),
                "moneyline": moneyline_probability(spreads),
            }
        )
    return results
