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


def test_from_flat_default_point_values():
    model = NFLTransitionModel("A", "g1", "B")
    # Build a flat vector that lands on absorbing transitions so defaults matter
    mask = build_validity_mask()
    n_masked = mask.sum()
    flat = np.zeros(N_CONTEXTS * n_masked, dtype=np.float32)
    # Put a single probability on FG and TD for context 0, state 16
    for c in range(N_CONTEXTS):
        base = c * n_masked
        flat[base] = 1.0  # first entry in mask is physical->physical (0,0)
    rebuilt = NFLTransitionModel.from_flat(flat, "A", "g1", "B")
    own_fg, opp_fg = rebuilt.get_point_values(0, 16, 72)
    own_td, opp_td = rebuilt.get_point_values(0, 16, 73)
    assert own_fg == 3.0
    assert opp_fg == 0.0
    assert own_td == 7.0  # empirical mode from 2024 data (395 7-pt vs 44 6-pt)
    assert opp_td == 0.0
