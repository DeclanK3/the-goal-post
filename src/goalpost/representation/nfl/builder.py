"""Build NFL transition matrices from extracted games."""
import numpy as np

from ..base.builder import TransitionBuilder, GameTable
from .extractor import NFLExtractedGame, N_PHYSICAL_STATES, context_index

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
    # I11: safeties are not modeled; treat as drive-ending TO with zero points
    if event.own_points == 2:
        return ABSORBING_TO
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
    def __init__(
        self,
        matrices: dict,
        keys: list,
        point_values: dict | None = None,
        global_matrix: np.ndarray | None = None,
        team_priors: dict | None = None,
        start_state_probs: np.ndarray | None = None,
        global_timing_mean: float | None = None,
        global_timing_std: float | None = None,
    ):
        # matrices: {(team, game): (C, N, N) array}
        self.matrices = matrices
        self.keys = keys
        self.point_values = point_values or {}
        self.global_matrix = global_matrix  # fallback for unobserved rows
        self.team_priors = team_priors or {}
        self.start_state_probs = start_state_probs
        self.global_timing_mean = global_timing_mean
        self.global_timing_std = global_timing_std


def _counts_to_probs(counts, totals, own_pts_sum, opp_pts_sum):
    """Convert raw counts to probability matrix and point-value matrices."""
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
    return probs, own_pts, opp_pts


class NFLTransitionBuilder(TransitionBuilder):
    def build(self, games) -> NFLGameTable:
        # Sort games chronologically for team-prior accumulation (I5)
        sorted_games = []
        for game in games:
            if isinstance(game, NFLExtractedGame):
                sorted_games.append(game)
        sorted_games.sort(key=lambda g: (g.week, g.date))

        matrices: dict = {}
        point_values: dict = {}
        team_priors: dict = {}

        # Per-team running accumulators for chronological prior (layer 2)
        team_counts: dict[str, np.ndarray] = {}
        team_totals: dict[str, np.ndarray] = {}
        team_own_sum: dict[str, np.ndarray] = {}
        team_opp_sum: dict[str, np.ndarray] = {}

        # Global counts accumulate across ALL teams and games (layer 1)
        global_counts = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.int32)
        global_totals = np.zeros((N_CONTEXTS, N_STATES), dtype=np.int32)

        # Start-state and timing accumulators (I8, I10)
        start_counts = np.zeros((N_CONTEXTS, N_PHYSICAL_STATES), dtype=np.int32)
        elapsed_times: list[float] = []

        for game in sorted_games:
            # Accumulate start-state distribution from game.starts
            for st in game.starts:
                ctx = context_index(st["remaining"], st["home_points"] - st["away_points"])
                if 0 <= st["state"] < N_PHYSICAL_STATES:
                    start_counts[ctx, st["state"]] += 1

            # Accumulate global timing
            for e in game.events:
                elapsed_times.append(float(e.elapsed))
                ctx = e.ctx
                src = e.source
                dst = _map_destination(e)
                if dst == -1 or src >= N_PHYSICAL_STATES:
                    continue
                global_counts[ctx, src, dst] += 1
                global_totals[ctx, src] += 1

            for team in (game.home, game.away):
                # --- Compute team prior from accumulator BEFORE this game ---
                prior_counts = team_counts.get(team)
                if prior_counts is not None and prior_counts.sum() > 0:
                    prior_totals = team_totals[team]
                    prior_own = team_own_sum[team]
                    prior_opp = team_opp_sum[team]
                    prior_probs, prior_own_pts, prior_opp_pts = _counts_to_probs(
                        prior_counts, prior_totals, prior_own, prior_opp
                    )
                    team_priors[(team, game.game_id)] = (prior_probs, prior_own_pts, prior_opp_pts)

                # --- Build this game's per-team matrix ---
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
                    # I11: safeties and defensive/special-teams scores are not modeled
                    if e.own_points == 2 or e.opponent_points > 0:
                        own_pts_sum[ctx, src, dst] += 0.0
                        opp_pts_sum[ctx, src, dst] += 0.0
                    else:
                        own_pts_sum[ctx, src, dst] += e.own_points
                        opp_pts_sum[ctx, src, dst] += e.opponent_points
                    totals[ctx, src] += 1

                probs, own_pts, opp_pts = _counts_to_probs(counts, totals, own_pts_sum, opp_pts_sum)
                matrices[(team, game.game_id)] = probs
                point_values[(team, game.game_id)] = (own_pts, opp_pts)

                # --- Update team accumulator AFTER storing this game ---
                if team not in team_counts:
                    team_counts[team] = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.int32)
                    team_totals[team] = np.zeros((N_CONTEXTS, N_STATES), dtype=np.int32)
                    team_own_sum[team] = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                    team_opp_sum[team] = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
                team_counts[team] += counts
                team_totals[team] += totals
                team_own_sum[team] += own_pts_sum
                team_opp_sum[team] += opp_pts_sum

        # Build global fallback matrix from all accumulated events
        global_probs = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
        for c in range(N_CONTEXTS):
            for s in range(N_STATES):
                if global_totals[c, s] > 0:
                    global_probs[c, s, :] = global_counts[c, s, :] / global_totals[c, s]
        for s in (ABSORBING_FG, ABSORBING_TD, ABSORBING_TO, ABSORBING_PUNT):
            global_probs[:, s, :] = 0
            global_probs[:, s, s] = 1.0

        # Normalise start-state distribution per context
        start_state_probs = np.zeros((N_CONTEXTS, N_PHYSICAL_STATES), dtype=np.float32)
        for c in range(N_CONTEXTS):
            total = start_counts[c].sum()
            if total > 0:
                start_state_probs[c, :] = start_counts[c, :] / total

        global_timing_mean = float(np.mean(elapsed_times)) if elapsed_times else 30.0
        global_timing_std = float(np.std(elapsed_times)) if elapsed_times else 0.0

        return NFLGameTable(
            matrices,
            list(matrices.keys()),
            point_values,
            global_probs,
            team_priors,
            start_state_probs,
            global_timing_mean,
            global_timing_std,
        )
