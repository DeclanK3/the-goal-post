"""Deterministic flattening of NFL transition matrices into feature vectors."""
import numpy as np

from .builder import N_STATES, build_validity_mask, N_PHYSICAL_STATES


def _physical_state_name(s: int) -> str:
    down = s // 18 + 1
    rem = s % 18
    distance = ["short", "medium", "long"][rem // 6]
    yardline = ["goal", "red", "opp40", "mid", "own40", "own20"][rem % 6]
    return f"{down}d_{distance}_{yardline}"


_ABSORBING_NAMES = {72: "fg", 73: "td", 74: "to", 75: "punt"}


def _state_name(s: int) -> str:
    if s < N_PHYSICAL_STATES:
        return _physical_state_name(s)
    return _ABSORBING_NAMES[s]


class NFLFlattener:
    def __init__(self):
        self.mask = build_validity_mask()
        self._base_names: list[str] | None = None

    def flatten(self, matrix: np.ndarray) -> np.ndarray:
        """Flatten one (N, N) matrix through the validity mask."""
        return matrix[self.mask].astype(np.float32)

    def feature_names(self, prefix: str) -> list[str]:
        """Return stable feature names like ['home_1d_short_goal_to_1d_short_goal', ...]."""
        if self._base_names is None:
            self._base_names = [
                f"{_state_name(src)}_to_{_state_name(dst)}"
                for src in range(N_STATES)
                for dst in range(N_STATES)
                if self.mask[src, dst]
            ]
        return [f"{prefix}_{n}" for n in self._base_names]

    def unflatten(self, flat: np.ndarray) -> np.ndarray:
        """Reconstruct (N, N) from flat vector."""
        n_masked = self.mask.sum()
        if len(flat) != n_masked:
            raise ValueError(f"flat length mismatch: {len(flat)} != {n_masked}")
        matrix = np.zeros((N_STATES, N_STATES), dtype=np.float32)
        matrix[self.mask] = flat
        return matrix
