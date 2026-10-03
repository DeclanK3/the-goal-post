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
    # Pass the same model for both sides; scores should still be deterministic
    out1 = engine.simulate(model, model, n_sims=100, seed=42)
    out2 = engine.simulate(model, model, n_sims=100, seed=42)
    scores1 = [(o.home_score, o.away_score) for o in out1]
    scores2 = [(o.home_score, o.away_score) for o in out2]
    assert scores1 == scores2


def test_dual_model_simulation_balanced_scores():
    """Each team should score using its own model's transitions."""
    # Home model: high TD probability (7 pts for offense)
    home_mat = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    home_mat[:, 16, 73] = 1.0  # always TD from state 16
    home_model = NFLTransitionModel("A", "g1", "B")
    home_model.load_matrix(home_mat)

    # Away model: high FG probability (3 pts for offense)
    away_mat = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    away_mat[:, 16, 72] = 1.0  # always FG from state 16
    away_model = NFLTransitionModel("B", "g1", "A")
    away_model.load_matrix(away_mat)

    engine = NFLEngine()
    outcomes = engine.simulate(home_model, away_model, n_sims=500, seed=42)
    home_scores = np.array([o.home_score for o in outcomes])
    away_scores = np.array([o.away_score for o in outcomes])

    assert home_scores.mean() > away_scores.mean(), (
        f"Expected home mean > away mean because TD (7) > FG (3), "
        f"got home={home_scores.mean():.1f}, away={away_scores.mean():.1f}"
    )


def test_global_fallback_prevents_early_termination():
    """An empty primary row with a populated fallback should continue the game."""
    # Primary model knows nothing about state 16 (the start state).
    primary = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    primary[:, 0, 0] = 1.0  # only state 0 is observed
    primary_model = NFLTransitionModel("A", "g1", "B")
    primary_model.load_matrix(primary)

    # Fallback: from state 16, always score a TD (state 73).
    # This lets the simulation proceed instead of breaking on the empty row.
    fallback = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    fallback[:, 16, 73] = 1.0
    primary_model._fallback_matrix = fallback.astype(np.float32)

    # Opponent also uses the same fallback-driven model
    opp_model = NFLTransitionModel("B", "g1", "A")
    opp_model.load_matrix(primary)
    opp_model._fallback_matrix = fallback.astype(np.float32)

    engine = NFLEngine()
    outcomes = engine.simulate(primary_model, opp_model, n_sims=100, seed=42)
    home_scores = np.array([o.home_score for o in outcomes])
    away_scores = np.array([o.away_score for o in outcomes])

    # If the fallback works, both teams should score TDs repeatedly.
    assert home_scores.mean() > 0, (
        f"Expected home scores > 0 with fallback TD, got mean {home_scores.mean():.1f}"
    )
    assert away_scores.mean() > 0, (
        f"Expected away scores > 0 with fallback TD, got mean {away_scores.mean():.1f}"
    )

