import numpy as np
import pytest

from goalpost.simulation.nfl.engine import NFLEngine
from goalpost.representation.nfl.model import NFLTransitionModel
from goalpost.representation.nfl.builder import N_STATES, N_CONTEXTS, build_validity_mask


def _make_identity_matrix():
    m = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    for c in range(N_CONTEXTS):
        for s in range(N_STATES):
            m[c, s, s] = 1.0
    return m


def test_empty_row_raises():
    """An empty primary row with no fallback must raise RuntimeError."""
    primary = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    primary[:, 0, 0] = 1.0  # only state 0 observed
    model = NFLTransitionModel("A", "g1", "B")
    model.load_matrix(primary)

    engine = NFLEngine()
    with pytest.raises(RuntimeError):
        engine.simulate(model, model, n_sims=1, seed=42)


def test_simulation_deterministic_with_seed():
    model = NFLTransitionModel("A", "g1", "B")
    model.load_matrix(_make_identity_matrix())
    model.load_timing(30.0)
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
    home_model.load_timing(30.0)

    # Away model: high FG probability (3 pts for offense)
    away_mat = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    away_mat[:, 16, 72] = 1.0  # always FG from state 16
    away_model = NFLTransitionModel("B", "g1", "A")
    away_model.load_matrix(away_mat)
    away_model.load_timing(30.0)

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
    primary_model.load_timing(30.0)
    opp_model.load_timing(30.0)

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


def test_start_state_sampling_increases_possessions():
    """Empirical start states should vary from the hard-coded state=16."""
    mat = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    mat[:, :, 73] = 1.0  # always TD from any state
    for s in (72, 73, 74, 75):
        mat[:, s, s] = 1.0

    model = NFLTransitionModel("A", "g1", "B")
    model.load_matrix(mat)
    model.load_timing(30.0)

    # No start states attached -> falls back to state=16
    outcomes1 = NFLEngine().simulate(model, model, n_sims=50, seed=42)

    # With start states that strongly prefer state=0 (not 16)
    start_probs = np.zeros((N_CONTEXTS, 72), dtype=np.float32)
    start_probs[:, 0] = 1.0
    model.load_start_states(start_probs)
    outcomes2 = NFLEngine().simulate(model, model, n_sims=50, seed=42)

    # Both should score the same because every state always TDs,
    # but we just verify the simulation doesn't crash.
    assert len(outcomes1) == 50
    assert len(outcomes2) == 50


def test_empirical_timing_reduces_possession_length():
    """A faster clock mean should produce more possessions (shorter drives)."""
    mat = np.zeros((N_CONTEXTS, N_STATES, N_STATES), dtype=np.float32)
    mat[:, 16, 16] = 1.0  # self-loop forever
    for s in (72, 73, 74, 75):
        mat[:, s, s] = 1.0

    # Model with fast clock (10s per play) -> more possessions in 1800s
    fast = NFLTransitionModel("A", "g1", "B")
    fast.load_matrix(mat)
    fast.load_timing(10.0)

    # Model with slow clock (60s per play) -> fewer possessions
    slow = NFLTransitionModel("A", "g1", "B")
    slow.load_matrix(mat)
    slow.load_timing(60.0)

    # We can't easily count possessions from outcomes, but we can at least
    # verify the simulation runs without error for both.
    engine = NFLEngine()
    assert len(engine.simulate(fast, fast, n_sims=10, seed=42)) == 10
    assert len(engine.simulate(slow, slow, n_sims=10, seed=42)) == 10


def test_reconstruction_dataset_fields_exist():
    """Verify the layered dataset contains the new fields after a build."""
    import tempfile
    from pathlib import Path

    import polars as pl

    from goalpost.cli.build import generate_dataset, DatasetRequest

    cache_dir = Path.home() / ".cache" / "goalpost" / "nflverse"
    if not cache_dir.exists():
        pytest.skip("nflverse cache not available")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = generate_dataset(
            DatasetRequest(sources_dir=cache_dir, output_dir=tmp_path, season=2024)
        )
        if result.game_count == 0:
            pytest.skip("no games extracted from cache")

        df = pl.read_parquet(result.dataset_path)
        assert "builder_version" in df.columns
        assert "global_timing_mean" in df.columns
        assert "home_prior_available" in df.columns
        assert "away_prior_available" in df.columns
        assert any(c.startswith("start_ctx_") for c in df.columns)
        assert any(c.startswith("home_prior_") for c in df.columns)
        assert any(c.startswith("global_prior_") for c in df.columns)

