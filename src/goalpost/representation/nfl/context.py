"""Game-context observations and paired team-context data structures."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .builder import N_CONTEXTS, N_STATES


@dataclass
class GameContextFeatures:
    game_id: str
    season: int
    week: int
    date: str
    home_team: str
    away_team: str
    context_index: int          # 0-8
    time_bin: int               # 0 (>120s), 1 (30-120s), 2 (<=30s)
    score_bin: int              # 0 (losing), 1 (tied), 2 (winning)
    home_score: int
    away_score: int


@dataclass
class PairedContextObservation:
    context: GameContextFeatures
    home_matrix: np.ndarray     # (N, N)
    away_matrix: np.ndarray     # (N, N)
    home_available: bool
    away_available: bool
    home_prior: np.ndarray | None = None   # layer 2 team prior (N, N)
    away_prior: np.ndarray | None = None
    global_prior: np.ndarray | None = None   # layer 1 global prior (N, N)
    start_state_probs: np.ndarray | None = None  # (N_CONTEXTS, N_PHYSICAL_STATES)
    global_timing_mean: float | None = None
    global_timing_std: float | None = None


def _is_available(matrix: np.ndarray) -> bool:
    """A context matrix is available if at least one row is observed (sums to 1)."""
    row_sums = matrix.sum(axis=-1)
    return bool(np.any(np.isclose(row_sums, 1.0, atol=1e-6)))


def extract_paired_observations(
    game,
    home_matrix: np.ndarray,
    away_matrix: np.ndarray,
    season: int = 0,
    home_prior: np.ndarray | None = None,
    away_prior: np.ndarray | None = None,
    global_prior: np.ndarray | None = None,
    start_state_probs: np.ndarray | None = None,
    global_timing_mean: float | None = None,
    global_timing_std: float | None = None,
) -> list[PairedContextObservation]:
    """Extract PairedContextObservation for each of the 9 contexts."""
    observations = []
    for ctx in range(N_CONTEXTS):
        time_bin = ctx // 3
        score_bin = ctx % 3
        home_ctx = home_matrix[ctx]
        away_ctx = away_matrix[ctx]
        observations.append(
            PairedContextObservation(
                context=GameContextFeatures(
                    game_id=game.game_id,
                    season=season,
                    week=game.week,
                    date=game.date,
                    home_team=game.home,
                    away_team=game.away,
                    context_index=ctx,
                    time_bin=time_bin,
                    score_bin=score_bin,
                    home_score=game.regulation[0],
                    away_score=game.regulation[1],
                ),
                home_matrix=home_ctx,
                away_matrix=away_ctx,
                home_available=_is_available(home_ctx),
                away_available=_is_available(away_ctx),
                home_prior=home_prior[ctx] if home_prior is not None else None,
                away_prior=away_prior[ctx] if away_prior is not None else None,
                global_prior=global_prior[ctx] if global_prior is not None else None,
                start_state_probs=start_state_probs,
                global_timing_mean=global_timing_mean,
                global_timing_std=global_timing_std,
            )
        )
    return observations
