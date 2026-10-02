import numpy as np
import polars as pl

from goalpost.representation.nfl.flatten import NFLFlattener
from goalpost.representation.nfl.dataset import VAEDatasetBuilder
from goalpost.representation.nfl.context import (
    GameContextFeatures, PairedContextObservation, extract_paired_observations,
)
from goalpost.representation.nfl.builder import N_STATES


def _make_observation(ctx_idx=0):
    ctx = GameContextFeatures(
        game_id="g1", season=2024, week=1, date="2024-01-01",
        home_team="A", away_team="B",
        context_index=ctx_idx, time_bin=0, score_bin=1,
        home_score=7, away_score=3,
    )
    mat = np.eye(N_STATES, dtype=np.float32)
    return PairedContextObservation(
        context=ctx,
        home_matrix=mat,
        away_matrix=mat,
        home_available=True,
        away_available=True,
    )


def test_dataset_schema():
    builder = VAEDatasetBuilder(NFLFlattener())
    obs = [_make_observation(0), _make_observation(1)]
    df = builder.build(obs)
    assert "game_id" in df.columns
    assert "home_available" in df.columns
    assert "away_available" in df.columns
    assert any(c.startswith("home_") for c in df.columns)
    assert any(c.startswith("away_") for c in df.columns)


def test_dataset_row_count():
    builder = VAEDatasetBuilder(NFLFlattener())
    obs = [_make_observation(i) for i in range(2)]
    df = builder.build(obs)
    assert len(df) == 2


def test_dataset_availability_flags():
    builder = VAEDatasetBuilder(NFLFlattener())
    obs = _make_observation(0)
    obs.home_available = False
    df = builder.build([obs])
    assert df["home_available"][0] == False
    assert df["away_available"][0] == True


def test_dataset_serialization(tmp_path):
    builder = VAEDatasetBuilder(NFLFlattener())
    df = builder.build([_make_observation(0)])
    out = tmp_path / "test.parquet"
    builder.write(df, out)
    df2 = pl.read_parquet(out)
    assert len(df2) == 1
    assert df2.columns == df.columns
