import numpy as np
import pytest

from goalpost.representation.nfl.flatten import NFLFlattener
from goalpost.representation.nfl.builder import N_STATES


def test_flatten_and_unflatten_roundtrip():
    flattener = NFLFlattener()
    matrix = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    for s in range(N_STATES):
        if s < 72:
            matrix[s, s] = 1.0
        else:
            matrix[s, s] = 1.0
    flat = flattener.flatten(matrix)
    assert flat.dtype == np.float32
    rebuilt = flattener.unflatten(flat)
    np.testing.assert_allclose(rebuilt, matrix)


def test_feature_names_deterministic():
    flattener = NFLFlattener()
    names1 = flattener.feature_names("home")
    names2 = flattener.feature_names("home")
    assert names1 == names2


def test_feature_names_unique():
    flattener = NFLFlattener()
    names = flattener.feature_names("home")
    assert len(names) == len(set(names))


def test_feature_names_include_prefix():
    flattener = NFLFlattener()
    names = flattener.feature_names("away")
    assert all(n.startswith("away_") for n in names)


def test_unflatten_length_mismatch():
    flattener = NFLFlattener()
    with pytest.raises(ValueError):
        flattener.unflatten(np.array([1.0, 2.0]))
