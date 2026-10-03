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
    mask = np.zeros((N_STATES, N_STATES), dtype=bool)
    # Physical states can reach any physical state and any absorbing outcome
    mask[:N_PHYSICAL_STATES, :N_PHYSICAL_STATES] = True
    for dst in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
        mask[:N_PHYSICAL_STATES, dst] = True
    # Absorbing states self-loop
    for s in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
        mask[s, s] = True
    return mask


VALIDITY_MASK = build_validity_mask()


def _map_destination(event) -> int:
    if event.destination == -1:
        return -1  # half-ending; no transition recorded
    if event.own_points == 3:
        return ABSORBING_FG
    if event.own_points in (6, 7, 8):
        return ABSORBING_TD
    if event.opponent_points > 0:
        return ABSORBING_TO
    if event.play_type == "punt":
        return ABSORBING_PUNT
    if event.switch:
        return ABSORBING_TO
    return event.destination


class NFLGameTable(GameTable):
    def __init__(self, matrices: dict, keys: list, point_values: dict | None = None, global_matrix: np.ndarray | None = None):
        # matrices: {(team, game): (C, N, N) array}
        self.matrices = matrices
        self.keys = keys
        self.point_values = point_values or {}
        self.global_matrix = global_matrix  # fallback for unobserved rows


class NFLTransitionBuilder(TransitionBuilder):
    def build(self, games) -> NFLGameTable:
        matrices = {}
        point_values = {}
        # Global counts accumulate across all teams and games for fallback
        global_counts = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.int32)
        global_totals = np.zeros((N_CONTEXTS, N_STATES), dtype=np.int32)

        for game in games:
            if not isinstance(game, NFLExtractedGame):
                continue
            for team in (game.home, game.away):
                counts = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.int32)
                own_pts_sum = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                opp_pts_sum = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                totals = np.zeros((N_CONTEXTS, N_STATES), dtype=np.int32)
                for e in game.events:
                    is_home = e.home_offense
                    offense = game.home if is_home else game.away
                    if offense != team:
                        continue
                    ctx = e.ctx
                    src = e.source
                    dst = _map_destination(e)
                    if dst == -1:
                        continue
                    if src >= N_PHYSICAL_STATES:
                        continue
                    counts[ctx, src, dst] += 1
                    own_pts_sum[ctx, src, dst] += e.own_points
                    opp_pts_sum[ctx, src, dst] += e.opponent_points
                    totals[ctx, src] += 1
                    # Also feed the global accumulator
                    global_counts[ctx, src, dst] += 1
                    global_totals[ctx, src] += 1
                probs = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                own_pts = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                opp_pts = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                for c in range(N_CONTEXTS):
                    for s in range(N_STATES):
                        if totals[c, s] > 0:
                            probs[c, s, :] = counts[c, s, :] / totals[c, s]
                            nz = counts[c, s, :] > 0
                            own_pts[c, s, nz] = own_pts_sum[c, s, nz] / counts[c, s, nz]
                            opp_pts[c, s, nz] = opp_pts_sum[c, s, nz] / counts[c, s, nz]
                for s in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
                    probs[:, s, :] = 0
                    probs[:, s, s] = 1.0
                matrices[(team, game.game_id)] = probs
                point_values[(team, game.game_id)] = (own_pts, opp_pts)

        # Build global fallback matrix from all accumulated events
        global_probs = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
        for c in range(N_CONTEXTS):
            for s in range(N_STATES):
                if global_totals[c, s] > 0:
                    global_probs[c, s, :] = global_counts[c, s, :] / global_totals[c, s]
        for s in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
            global_probs[:, s, :] = 0
            global_probs[:, s, s] = 1.0

        return NFLGameTable(matrices, list(matrices.keys()), point_values, global_probs)
