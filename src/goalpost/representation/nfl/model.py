"""NFL-specific transition model."""
import numpy as np

from ..base.models import TransitionModel
from .builder import N_STATES, N_CONTEXTS, build_validity_mask
from .flatten import NFLFlattener


class NFLTransitionModel(TransitionModel):
    def __init__(self, team_id: str, game_id: str, opponent_id: str):
        super().__init__(team_id, game_id, opponent_id)
        self._matrix = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
        self._flat = None
        self._flattener = NFLFlattener()

    @property
    def state_space_size(self) -> int:
        return N_STATES

    @property
    def context_count(self) -> int:
        return N_CONTEXTS

    @property
    def validity_mask(self) -> np.ndarray:
        return build_validity_mask()

    def extract_from_game(self, game) -> None:
        pass

    def get_probabilities(self, context: int, state: int) -> dict[int, float]:
        row = self._matrix[context, state]
        nz = np.flatnonzero(row)
        return {int(j): float(row[j]) for j in nz}

    def flatten(self) -> np.ndarray:
        if self._flat is not None:
            return self._flat
        flat = np.concatenate([self._flattener.flatten(self._matrix[c]) for c in range(N_CONTEXTS)])
        self._flat = flat
        return flat

    @classmethod
    def from_flat(cls, array: np.ndarray, team_id: str, game_id: str, opponent_id: str):
        obj = cls(team_id, game_id, opponent_id)
        mask = build_validity_mask()
        n_masked = mask.sum()
        expected_len = N_CONTEXTS * n_masked
        if len(array) != expected_len:
            raise ValueError(f"flat length mismatch: {len(array)} != {expected_len}")
        obj._matrix = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
        flat_idx = 0
        for c in range(N_CONTEXTS):
            vals = array[flat_idx:flat_idx + n_masked]
            obj._matrix[c] = obj._flattener.unflatten(vals)
            flat_idx += n_masked
        return obj

    def load_matrix(self, matrix: np.ndarray):
        if matrix.shape != (N_CONTEXTS, N_STATES, N_STATES):
            raise ValueError(f"matrix shape mismatch: {matrix.shape}")
        self._matrix = matrix.astype(np.float32)
        self._flat = None
