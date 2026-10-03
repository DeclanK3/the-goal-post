"""NFL elapsed-clock simulation engine."""
import numpy as np

from ..base.engine import Simulator
from ..base.models import GameOutcome
from ...representation.base.models import TransitionModel

N_PHYSICAL_STATES = 72
ABSORBING_FG = 72
ABSORBING_TD = 73
ABSORBING_TO = 74
ABSORBING_PUNT = 75


def _context_index(remaining, lead):
    time_bin = 0 if remaining > 120 else 1 if remaining > 30 else 2
    score_bin = 0 if lead < 0 else 1 if lead == 0 else 2
    return time_bin * 3 + score_bin


class NFLEngine(Simulator):
    def simulate(self, home_model: TransitionModel, away_model: TransitionModel, n_sims: int = 10000, seed: int = 0) -> list[GameOutcome]:
        rng = np.random.default_rng(seed)
        outcomes = []
        for _ in range(n_sims):
            home_score, away_score = self._simulate_game(home_model, away_model, rng)
            outcomes.append(GameOutcome(home_score=home_score, away_score=away_score))
        return outcomes

    def _simulate_game(self, home_model: TransitionModel, away_model: TransitionModel, rng) -> tuple[int, int]:
        home = away = 0
        starter = int(rng.integers(2))
        for half in range(2):
            offense = starter if half == 0 else 1 - starter
            remaining = 1800.0
            state = 16  # typical starting state
            while remaining > 0:
                lead = (home - away) if offense == 0 else (away - home)
                ctx = _context_index(remaining, lead)
                # Use the correct team's model depending on who has the ball
                model = home_model if offense == 0 else away_model
                probs = model.get_probabilities(ctx, state)
                # If the primary row is empty, try the global fallback matrix
                if not probs and hasattr(model, '_fallback_matrix') and model._fallback_matrix is not None:
                    fb = model._fallback_matrix[ctx, state]
                    nz = np.flatnonzero(fb)
                    probs = {int(j): float(fb[j]) for j in nz}
                if not probs:
                    break
                states = list(probs.keys())
                weights = np.array(list(probs.values()), dtype=np.float64)
                weights /= weights.sum()
                next_state = rng.choice(states, p=weights)
                if next_state >= N_PHYSICAL_STATES:
                    own_pts, opp_pts = model.get_point_values(ctx, state, next_state)
                    if offense == 0:
                        home += own_pts
                        away += opp_pts
                    else:
                        away += own_pts
                        home += opp_pts
                    offense = 1 - offense
                    state = 16
                else:
                    state = next_state
                remaining -= rng.exponential(30.0)
            if remaining <= 0:
                remaining = 0
        return home, away
