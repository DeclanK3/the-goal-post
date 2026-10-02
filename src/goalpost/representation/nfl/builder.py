"""Build NFL transition matrices from extracted games."""
import numpy as np

from ..base.builder import TransitionBuilder, GameTable
from .extractor import NFLExtractedGame, N_PHYSICAL_STATES

N_STATES = 76
N_CONTEXTS = 9

# 72 physical + 4 absorbing
ABSORBING_FG = 72
ABSORBING_TD = 73
ABSORBING_TO = 74
ABSORBING_PUNT = 75


def build_validity_mask():
    """Return boolean mask (N, N) of structurally possible transitions."""
    mask = np.zeros((N_STATES, N_STATES), dtype=bool)
    # All physical states can transition to any physical state
    mask[:N_PHYSICAL_STATES, :N_PHYSICAL_STATES] = True
    # Absorbing states self-loop
    for s in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
        mask[s, s] = True
    return mask


VALIDITY_MASK = build_validity_mask()


def _map_destination(event) -> int:
    """Map an event destination to a Markov state index."""
    if event.destination == -1:
        # Half-ending event - infer from points
        if event.own_points == 3:
            return ABSORBING_FG
        elif event.own_points == 6 or event.own_points == 7:
            return ABSORBING_TD
        elif event.opponent_points > 0:
            return ABSORBING_TO
        elif event.own_points == 0:
            return ABSORBING_PUNT
        return ABSORBING_TO
    return event.destination


class NFLGameTable(GameTable):
    def __init__(self, matrices: dict, keys: list):
        # matrices: {(team, game): (C, N, N) array}
        self.matrices = matrices
        self.keys = keys


class NFLTransitionBuilder(TransitionBuilder):
    def build(self, games) -> NFLGameTable:
        matrices = {}
        for game in games:
            if not isinstance(game, NFLExtractedGame):
                continue
            # Build per-team matrices
            for team in (game.home, game.away):
                counts = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.int32)
                totals = np.zeros((N_CONTEXTS, N_STATES), dtype=np.int32)
                for e in game.events:
                    is_home = e.home_offense
                    offense = game.home if is_home else game.away
                    if offense != team:
                        continue
                    ctx = e.ctx
                    src = e.source
                    dst = _map_destination(e)
                    if src >= N_PHYSICAL_STATES:
                        continue
                    counts[ctx, src, dst] += 1
                    totals[ctx, src] += 1
                # Convert counts to probabilities
                probs = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                for c in range(N_CONTEXTS):
                    for s in range(N_STATES):
                        if totals[c, s] > 0:
                            probs[c, s, :] = counts[c, s, :] / totals[c, s]
                # Enforce absorbing states
                for s in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
                    probs[:, s, :] = 0
                    probs[:, s, s] = 1.0
                matrices[(team, game.game_id)] = probs
        return NFLGameTable(matrices, list(matrices.keys()))
