import numpy as np
import pytest

from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.builder import N_STATES, N_CONTEXTS, build_validity_mask


def test_state_space_size():
    model = NFLTransitionModel("A", "g1", "B")
    assert model.state_space_size == 76
    assert model.context_count == 9


def test_flatten_roundtrip():
    model = NFLTransitionModel("A", "g1", "B")
    matrix = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    # Populate some values within validity mask
    mask = build_validity_mask()
    for c in range(N_CONTEXTS):
        for s in range(N_STATES):
            if mask[s].any():
                matrix[c, s, mask[s]] = 1.0 / mask[s].sum()
    # Fix absorbing states
    for s in (72, 73, 74, 75):
        matrix[:, s, :] = 0
        matrix[:, s, s] = 1.0
    model.load_matrix(matrix)
    flat = model.flatten()
    rebuilt = NFLTransitionModel.from_flat(flat, "A", "g1", "B")
    np.testing.assert_allclose(rebuilt.flatten(), flat)
