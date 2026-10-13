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


def _sample_duration(model, ctx, state, next_state, rng):
    """Sample elapsed time from empirical timing, falling back to global mean."""
    # Try empirical per-transition timing first
    if hasattr(model, '_timing') and model._timing is not None:
        try:
            return model._timing.sample_duration(ctx, state, next_state, rng)
        except ValueError:
            pass
    # Fall back to global empirical mean (I8: fixed from extractor, never fitted)
    if hasattr(model, '_global_timing_mean') and model._global_timing_mean is not None:
        return float(model._global_timing_mean)
    raise ValueError(
        f"No timing data attached to model for game_id={model.game_id}, "
        f"team_id={model.team_id}. Empirical timing required; no constant fallback."
    )


# Cache for pre-sampled durations per transition to avoid repeated random draws
# Format: dict[(model_id, ctx, src, dst)] -> list of pre-sampled durations
_DURATION_CACHE: dict = {}


class NFLEngine(Simulator):
    def simulate(self, home_model: TransitionModel, away_model: TransitionModel, n_sims: int = 10000, seed: int = 0) -> list[GameOutcome]:
        rng = np.random.default_rng(seed)
        outcomes = []
        for _ in range(n_sims):
            home_score, away_score = self._simulate_game(home_model, away_model, rng)
            outcomes.append(GameOutcome(home_score=home_score, away_score=away_score))
        return outcomes

    def simulate_instrumented(self, home_model: TransitionModel, away_model: TransitionModel, n_sims: int = 10000, seed: int = 0) -> list[dict]:
        """Return full possession-level metrics for each simulation."""
        rng = np.random.default_rng(seed)
        results = []
        for _ in range(n_sims):
            result = self._simulate_game_instrumented(home_model, away_model, rng)
            results.append(result)
        return results

    def _simulate_game(self, home_model: TransitionModel, away_model: TransitionModel, rng) -> tuple[int, int]:
        result = self._simulate_game_instrumented(home_model, away_model, rng)
        return result["home_score"], result["away_score"]

    def _simulate_game_instrumented(self, home_model: TransitionModel, away_model: TransitionModel, rng) -> dict:
        """Instrumented simulation returning scores and possession-level metrics."""
        home = away = 0
        possessions = 0
        plays = 0
        drive_ends: dict[int, int] = {ABSORBING_FG: 0, ABSORBING_TD: 0, ABSORBING_TO: 0, ABSORBING_PUNT: 0}
        breaks = 0
        elapsed_sum = 0.0

        starter = int(rng.integers(2))
        for half in range(2):
            offense = starter if half == 0 else 1 - starter
            remaining = 1800.0
            lead = (home - away) if offense == 0 else (away - home)
            ctx = _context_index(remaining, lead)
            model = home_model if offense == 0 else away_model
            state = self._sample_start_state(model, ctx, rng)
            possessions += 1
            while remaining > 0:
                lead = (home - away) if offense == 0 else (away - home)
                ctx = _context_index(remaining, lead)
                model = home_model if offense == 0 else away_model
                probs = self._get_probs(model, ctx, state)
                if not probs:
                    breaks += 1
                    raise RuntimeError(
                        f"Empty row: ctx={ctx}, state={state}, "
                        f"model={model.team_id}/{model.game_id}"
                    )
                states = list(probs.keys())
                weights = np.array(list(probs.values()), dtype=np.float64)
                weights /= weights.sum()
                next_state = rng.choice(states, p=weights)
                src_state = state
                elapsed = _sample_duration(model, ctx, src_state, next_state, rng)
                if remaining - elapsed <= 0:
                    remaining = 0
                    break
                remaining -= elapsed
                elapsed_sum += elapsed
                plays += 1
                if next_state >= N_PHYSICAL_STATES:
                    own_pts, opp_pts = model.get_point_values(ctx, state, next_state)
                    if offense == 0:
                        home += own_pts
                        away += opp_pts
                    else:
                        away += own_pts
                        home += opp_pts
                    drive_ends[int(next_state)] = drive_ends.get(int(next_state), 0) + 1
                    offense = 1 - offense
                    lead = (home - away) if offense == 0 else (away - home)
                    ctx = _context_index(remaining, lead)
                    model = home_model if offense == 0 else away_model
                    state = self._sample_start_state(model, ctx, rng)
                    possessions += 1
                else:
                    state = next_state
        return {
            "home_score": int(home),
            "away_score": int(away),
            "possessions": possessions,
            "plays": plays,
            "drive_ends": drive_ends,
            "breaks": breaks,
            "elapsed_sum": elapsed_sum,
        }

    @staticmethod
    def _get_probs(model, ctx, state):
        """Return probability dict, falling back from team prior to global prior."""
        # Try team-specific prior first (layer 2)
        if hasattr(model, '_team_prior_matrix') and model._team_prior_matrix is not None:
            row = model._team_prior_matrix[ctx, state]
            nz = np.flatnonzero(row)
            if len(nz) > 0:
                return {int(j): float(row[j]) for j in nz}
        # Fall back to global prior (layer 1)
        if hasattr(model, '_fallback_matrix') and model._fallback_matrix is not None:
            fb = model._fallback_matrix[ctx, state]
            nz = np.flatnonzero(fb)
            if len(nz) > 0:
                return {int(j): float(fb[j]) for j in nz}
        # Last resort: the game matrix itself (layer 3)
        return model.get_probabilities(ctx, state)

    @staticmethod
    def _sample_start_state(model, ctx, rng):
        """Sample a drive-start physical state from empirical distribution."""
        if hasattr(model, '_start_state_probs') and model._start_state_probs is not None:
            probs = model._start_state_probs[ctx]
            nz = np.flatnonzero(probs)
            if len(nz) > 0:
                weights = probs[nz] / probs[nz].sum()
                return int(rng.choice(nz, p=weights))
        return 16  # historical hard-coded fallback
