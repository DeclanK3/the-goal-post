import numpy as np

from goalpost.simulation.nfl.engine import NFLEngine
from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.builder import N_STATES, N_CONTEXTS, build_validity_mask


def _make_identity_matrix():
    m = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    for c in range(N_CONTEXTS):
        for s in range(N_STATES):
            m[c, s, s] = 1.0
    return m


def test_simulation_deterministic_with_seed():
    model = NFLTransitionModel("A", "g1", "B")
    model.load_matrix(_make_identity_matrix())
    engine = NFLEngine()
    out1 = engine.simulate(model, n_sims=100, seed=42)
    out2 = engine.simulate(model, n_sims=100, seed=42)
    scores1 = [(o.home_score, o.away_score) for o in out1]
    scores2 = [(o.home_score, o.away_score) for o in out2]
    assert scores1 == scores2
